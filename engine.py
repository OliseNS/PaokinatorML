import pandas as pd
import numpy as np

class AkinatorEngine:
    """
    Highly optimized NumPy Akinator engine with a robust "minimax" gain
    strategy for resilience to wrong answers, blended with expected gain
    for late-game speed.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map

        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        # Softened penalties for better niche animal handling
        self.definite_exp = -3.5
        self.uncertain_exp = -2.0

        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'usually': 0.75,
            'sometimes': 0.5, 'maybe': 0.5, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        # Single sparse question at random position 5-10
        self.sparse_question_position = np.random.randint(5, 11)
        self.sparse_question_asked = False
        
        self.sorted_initial_feature_indices = []
        self._precompute_likelihood_tables()
        self._build_arrays()
        
    def _precompute_likelihood_tables(self):
        """Precompute all possible likelihoods with gentler penalties."""
        # Increase resolution for better accuracy
        feature_grid = np.linspace(0.0, 1.0, 41, dtype=np.float32) # 41 for finer granularity
        n_grid = len(feature_grid)
        n_answers = len(self.answer_values)
        
        self.likelihood_table_definite = np.zeros((n_grid, n_answers), dtype=np.float32)
        self.likelihood_table_uncertain = np.zeros((n_grid, n_answers), dtype=np.float32)
        
        for i, fval in enumerate(feature_grid):
            for j, aval in enumerate(self.answer_values):
                dist = abs(fval - aval)
                
                like_def = np.exp(self.definite_exp * dist)
                # Even gentler penalties for niche animals
                if dist > 0.7:
                    like_def *= 0.05
                elif dist > 0.3:
                    like_def *= 0.5
                self.likelihood_table_definite[i, j] = np.clip(like_def, 0.001, 1.0)
                
                like_unc = np.exp(self.uncertain_exp * dist)
                self.likelihood_table_uncertain[i, j] = np.clip(like_unc, 0.001, 1.0)
        
        self.feature_grid = feature_grid
        
    def _build_arrays(self):
        """Builds all arrays from the DataFrame - optimized for NumPy."""
        self.animals = self.df['animal_name'].values
        features_np = self.df[self.feature_cols].values.astype(np.float32)
        
        self.features = features_np
        self.nan_mask = np.isnan(self.features)
        self.features_filled = np.where(self.nan_mask, 0.5, self.features)
        
        # Calculate NaN fractions and variance for each column
        col_nan_frac = np.isnan(features_np).mean(axis=0)
        col_var = np.nanvar(features_np, axis=0)
        col_var = np.where(np.isnan(col_var), 0.0, col_var)
        
        self.col_nan_frac = col_nan_frac
        
        # More lenient feature selection for better coverage
        self.allowed_feature_mask = (col_nan_frac < 0.95) & (col_var > 5e-5)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].tolist()
        
        # Sparse features (50-100% NaN) for data population
        self.sparse_feature_mask = (col_nan_frac >= 0.50) & (col_nan_frac < 1.0) & (col_var > 1e-6)
        self.sparse_feature_indices = np.where(self.sparse_feature_mask)[0].tolist()
        
        N = len(self.animals)
        if N > 0 and len(self.allowed_feature_indices) > 0:
            uniform_prior_np = np.ones(N, dtype=np.float32) / N
            gains = self._fast_info_gain_numpy(uniform_prior_np, self.allowed_feature_indices)
            
            self.feature_importance = {
                self.feature_cols[idx]: float(gains[i])
                for i, idx in enumerate(self.allowed_feature_indices)
            }
            
            sorted_gains_indices = np.argsort(gains)[::-1]
            self.sorted_initial_feature_indices = [
                self.allowed_feature_indices[i] for i in sorted_gains_indices
            ]
        else:
            self.feature_importance = {f: 0.0 for f in self.feature_cols}
            self.sorted_initial_feature_indices = []

    def _fast_info_gain_numpy(self, prior_np, feature_indices):
        """Fast info gain using precomputed likelihood tables - batched."""
        curr_entropy = self._entropy_numpy(prior_np)
        if curr_entropy < 0.01:
            return np.zeros(len(feature_indices))
        
        features_np = self.features_filled[:, feature_indices]
        nan_mask_np = self.nan_mask[:, feature_indices]
        
        # Higher resolution quantization
        quantized = np.clip(np.round(features_np * 40).astype(int), 0, 40)
        
        k = len(feature_indices)
        A = len(self.answer_values)
        n = len(features_np)
        
        # Vectorized likelihood computation
        likelihoods = np.zeros((n, k, A), dtype=np.float32)
        for a_idx, aval in enumerate(self.answer_values):
            is_definite = abs(aval - 0.5) > 0.3
            table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
            likelihoods[:, :, a_idx] = table[quantized, a_idx]
        
        likelihoods = np.where(nan_mask_np[:, :, None], 1.0, likelihoods)
        
        # Vectorized probability computation
        prob_answer = prior_np @ likelihoods.reshape(n, -1)
        prob_answer = prob_answer.reshape(k, A)
        
        # Vectorized entropy computation
        gains = np.zeros(k)
        for f_idx in range(k):
            posteriors = prior_np[:, None] * likelihoods[:, f_idx, :]
            post_sums = posteriors.sum(axis=0)
            
            exp_ent = 0.0
            for a_idx in range(A):
                if post_sums[a_idx] > 1e-10:
                    posterior = posteriors[:, a_idx] / post_sums[a_idx]
                    exp_ent += prob_answer[f_idx, a_idx] * self._entropy_numpy(posterior)
            
            gains[f_idx] = curr_entropy - exp_ent
        
        return gains
    
    @staticmethod
    def _entropy_numpy(probs):
        """Fast numpy entropy calculation."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs_safe * np.log(probs_safe))

    def calc_likelihood(self, feature_vec: np.ndarray, target: float, 
                        definite_exp: float, uncertain_exp: float) -> np.ndarray:
        """Calculate likelihood using precomputed table - ultra fast."""
        quantized = np.clip(np.round(feature_vec * 40).astype(int), 0, 40)
        target_idx = np.argmin(np.abs(self.answer_values - target))
        
        is_definite = abs(target - 0.5) > 0.3
        table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
        
        return table[quantized, target_idx]
    
    @staticmethod
    def entropy(probs: np.ndarray) -> float:
        """Calculate entropy."""
        probs_safe = np.clip(probs, 1e-10, None)
        return -np.sum(probs_safe * np.log(probs_safe))
    
    def get_prior(self, rejected_mask):
        prior = np.ones(len(self.animals), dtype=np.float32)
        prior[rejected_mask] = 0.0
        return prior / (prior.sum() + 1e-10)
    
    def update(self, prior, feature_idx, answer):
        """Update beliefs via Bayesian inference - optimized."""
        fuzzy_val = self.fuzzy_map.get(answer.lower().strip())
        if fuzzy_val is None:
            return prior
        
        feature_vec = self.features_filled[:, feature_idx]
        likelihood = self.calc_likelihood(feature_vec, fuzzy_val, 
                                          self.definite_exp, self.uncertain_exp)
        
        likelihood = np.where(self.nan_mask[:, feature_idx], 
                              np.ones_like(likelihood), likelihood)
        
        posterior = prior * likelihood
        return posterior / (posterior.sum() + 1e-10)

    def info_gain_batch(self, prior, feature_indices):
        """Compute information gain (Expected Value) with active set optimization."""
        curr_entropy = self.entropy(prior)
        if curr_entropy < 0.01 or not feature_indices:
            return np.zeros(len(feature_indices))

        # More aggressive active set optimization
        active_thresh = 1e-9
        active_idx = np.where(prior > active_thresh)[0]
        
        if len(active_idx) == 0:
            active_idx = np.arange(len(prior))
            prior = np.ones_like(prior) / len(prior)
            curr_entropy = self.entropy(prior)

        # Use active subset for faster computation
        if len(active_idx) < len(prior) * 0.9:
            prior_sub = prior[active_idx]
            prior_sub = prior_sub / (prior_sub.sum() + 1e-12)
            feature_batch = self.features_filled[active_idx][:, feature_indices]
            nan_mask_batch = self.nan_mask[active_idx][:, feature_indices]
        else:
            prior_sub = prior
            feature_batch = self.features_filled[:, feature_indices]
            nan_mask_batch = self.nan_mask[:, feature_indices]

        k = len(feature_indices)
        A = len(self.answer_values)
        n = len(prior_sub)
        
        quantized = np.clip(np.round(feature_batch * 40).astype(int), 0, 40)
        likelihoods = np.zeros((n, k, A), dtype=np.float32)
        
        # Vectorized likelihood lookup
        for a_idx, aval in enumerate(self.answer_values):
            is_definite = abs(aval.item() - 0.5) > 0.3
            table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
            likelihoods[:, :, a_idx] = table[quantized, a_idx]
        
        likelihoods = np.where(nan_mask_batch[:, :, None], 
                               np.ones_like(likelihoods), likelihoods)
        
        # Fast matmul for probability computation
        prior_2d = prior_sub.reshape(1, -1)
        like_2d = likelihoods.reshape(n, k * A)
        prob_answer = (prior_2d @ like_2d).reshape(k, A)
        
        # Vectorized gain computation (Expected Gain)
        gains = np.zeros(k)
        for f_idx in range(k):
            expected_ent = 0.0
            for a_idx in range(A):
                posterior = prior_sub * likelihoods[:, f_idx, a_idx]
                post_sum = posterior.sum()
                
                if post_sum > 1e-10:
                    posterior = posterior / post_sum
                    ent = self.entropy(posterior)
                    expected_ent += prob_answer[f_idx, a_idx] * ent
            
            gains[f_idx] = curr_entropy - expected_ent
        
        return gains

    # --- NEW METHOD ---
    def info_gain_robust_batch(self, prior, feature_indices):
        """
        Compute information gain using a "Max-Min" strategy (like minimax).
        Picks the question that maximizes the *minimum* information gain
        (i.e., minimizes the *maximum* possible posterior entropy).
        This makes the engine highly robust to confusing or "worst-case" answers.
        """
        curr_entropy = self.entropy(prior)
        if curr_entropy < 0.01 or not feature_indices:
            return np.zeros(len(feature_indices))

        # Active set optimization (from your original function)
        active_thresh = 1e-9
        active_idx = np.where(prior > active_thresh)[0]
        
        if len(active_idx) == 0:
            active_idx = np.arange(len(prior))
            prior = np.ones_like(prior) / len(prior)
            curr_entropy = self.entropy(prior)

        if len(active_idx) < len(prior) * 0.9:
            prior_sub = prior[active_idx]
            prior_sub = prior_sub / (prior_sub.sum() + 1e-12)
            feature_batch = self.features_filled[active_idx][:, feature_indices]
            nan_mask_batch = self.nan_mask[active_idx][:, feature_indices]
        else:
            prior_sub = prior
            feature_batch = self.features_filled[:, feature_indices]
            nan_mask_batch = self.nan_mask[:, feature_indices]

        k = len(feature_indices)
        A = len(self.answer_values)
        n = len(prior_sub)
        
        quantized = np.clip(np.round(feature_batch * 40).astype(int), 0, 40)
        likelihoods = np.zeros((n, k, A), dtype=np.float32)
        
        for a_idx, aval in enumerate(self.answer_values):
            is_definite = abs(aval.item() - 0.5) > 0.3
            table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
            likelihoods[:, :, a_idx] = table[quantized, a_idx]
        
        likelihoods = np.where(nan_mask_batch[:, :, None], 
                               np.ones_like(likelihoods), likelihoods)
        
        # Fast matmul for probability computation
        prior_2d = prior_sub.reshape(1, -1)
        like_2d = likelihoods.reshape(n, k * A)
        prob_answer = (prior_2d @ like_2d).reshape(k, A)
        
        # === ROBUST GAIN CALCULATION (THE MINIMAX PART) ===
        gains = np.zeros(k)
        for f_idx in range(k):
            max_posterior_entropy = 0.0  # Find the entropy of the "worst" answer
            
            for a_idx in range(A):
                posterior = prior_sub * likelihoods[:, f_idx, a_idx]
                post_sum = posterior.sum()
                
                ent = 0.0
                if post_sum > 1e-10:
                    posterior = posterior / post_sum
                    ent = self.entropy(posterior)
                
                # Use current entropy as upper bound if something goes wrong
                # (e.g., if post_sum is 0, ent will be 0, which is good)
                
                if ent > max_posterior_entropy:
                    max_posterior_entropy = ent
            
            # Gain is the reduction from current entropy to the *worst-case* posterior entropy
            gains[f_idx] = curr_entropy - max_posterior_entropy
            
        return gains
    # --- END OF NEW METHOD ---

    def select_sparse_question(self, asked):
        """Select a sparse feature (50-100% NaN) to populate missing data."""
        available_sparse = [
            idx for idx in self.sparse_feature_indices
            if self.feature_cols[idx] not in asked
        ]
        
        if not available_sparse:
            return None, None
        
        # Weight by NaN ratio - prefer higher NaN
        nan_ratios = [self.col_nan_frac[idx] for idx in available_sparse]
        weights = []
        for ratio in nan_ratios:
            if 0.80 <= ratio < 1.0:
                weights.append(3.0)
            elif 0.70 <= ratio < 0.80:
                weights.append(2.0)
            elif 0.60 <= ratio < 0.70:
                weights.append(1.5)
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        chosen_idx = np.random.choice(available_sparse, p=weights)
        feature = self.feature_cols[chosen_idx]
        question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        
        return feature, question

    # --- UPDATED METHOD ---
    def select_question(self, prior, asked, question_count):
        """
        Strategic question selection with:
        1. Blended Gain Strategy: Uses "Robust/Minimax" gain in early game (high entropy)
           and "Expected" gain in late game (low entropy) for speed.
        2. Sparse Question: Injects a data-collection question at a random point.
        3. Enhanced Variety: Uses a larger random pool for early questions and
           weighted random choice for later questions.
        """
        
        # === SPARSE QUESTION AT RANDOM POSITION 5-10 ===
        if question_count == self.sparse_question_position and not self.sparse_question_asked:
            feature, question = self.select_sparse_question(asked)
            if feature is not None:
                self.sparse_question_asked = True
                return feature, question
        
        # === Q0-2: RANDOM FROM TOP 10 FOR MORE VARIETY ===
        if question_count < 3:
            if self.sorted_initial_feature_indices:
                available_top = [
                    idx for idx in self.sorted_initial_feature_indices[:10]  # Was 7, now 10
                    if self.feature_cols[idx] not in asked
                ]
                
                if available_top:
                    chosen_idx = np.random.choice(available_top)
                    feature = self.feature_cols[chosen_idx]
                    question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
                    return feature, question
        
        # === PREPARE AVAILABLE FEATURES ===
        all_available_indices = [
            i for i, f in enumerate(self.feature_cols)
            if f not in asked and self.allowed_feature_mask[i]
        ]
        
        if not all_available_indices:
            return None, None
        
        # === ADAPTIVE FEATURE POOL SIZE ===
        if question_count < 5:
            MAX_FEATURES_TO_CHECK = 30
        elif question_count < 8:
            MAX_FEATURES_TO_CHECK = 50
        elif question_count < 12:
            MAX_FEATURES_TO_CHECK = 75
        else:
            MAX_FEATURES_TO_CHECK = 100
        
        available_set = set(all_available_indices)
        top_available_features = [
            idx for idx in self.sorted_initial_feature_indices
            if idx in available_set
        ][:MAX_FEATURES_TO_CHECK]

        if len(top_available_features) < MAX_FEATURES_TO_CHECK and len(all_available_indices) > len(top_available_features):
            remaining = list(available_set - set(top_available_features))
            np.random.shuffle(remaining)
            top_available_features.extend(remaining[:MAX_FEATURES_TO_CHECK - len(top_available_features)])
        
        available_features_to_check = top_available_features[:MAX_FEATURES_TO_CHECK]
        if not available_features_to_check: # Handle edge case
             return None, None
             
        sampled_indices_map = {
            new_idx: old_idx for new_idx, old_idx in enumerate(available_features_to_check)
        }
        
        # === BLENDED GAIN STRATEGY ===
        curr_entropy = self.entropy(prior)
        
        # Tune this entropy threshold as needed.
        # 2.5 is a good starting point (high confusion)
        if curr_entropy > 2.5:
            # High entropy: Be robust. Use the minimax strategy.
            gains_array = self.info_gain_robust_batch(prior, available_features_to_check)
        else:
            # Low entropy: Be fast. Use the expected value strategy.
            gains_array = self.info_gain_batch(prior, available_features_to_check)
        # === END OF BLENDED STRATEGY ===
            
        sorted_indices_of_gains = np.argsort(gains_array)[::-1]

        if len(sorted_indices_of_gains) == 0:
            return None, None

        # === WEIGHTED RANDOM CHOICE FROM TOP 3 ===
        if question_count >= 3 and len(sorted_indices_of_gains) >= 3:
            # 70% pick best, 20% pick 2nd best, 10% pick 3rd best
            top_3_choices = sorted_indices_of_gains[:3]
            chosen_local_idx = np.random.choice(top_3_choices, p=[0.7, 0.2, 0.1])
        else:
            # Always pick the best feature
            chosen_local_idx = sorted_indices_of_gains[0]
        # === END OF WEIGHTED CHOICE ===
            
        idx = sampled_indices_map[int(chosen_local_idx)]
        
        feature = self.feature_cols[idx]
        question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        return feature, question
    # --- END OF UPDATED METHOD ---
    
    def should_guess(self, prior, question_count):
        """
        Adaptive guessing thresholds that consider entropy and separation.
        """
        top_idx = np.argmax(prior)
        top_prob = prior[top_idx]
        
        # Get second highest probability
        prior_copy = prior.copy()
        prior_copy[top_idx] = 0.0
        second_prob = np.max(prior_copy)
        
        # Calculate separation ratio
        separation = top_prob / (second_prob + 1e-10)
        
        # Calculate entropy for additional confidence measure
        entropy = self.entropy(prior)
        
        # Adaptive thresholds based on question count and entropy
        if question_count < 10:
            # Need very high confidence early
            should_guess = (top_prob > 0.80 and separation > 6.0) or (top_prob > 0.90 and separation > 4.0)
        elif question_count < 15:
            # High confidence needed
            should_guess = (top_prob > 0.70 and separation > 5.0) or (top_prob > 0.85 and entropy < 0.5)
        elif question_count < 25:
            # Moderate confidence
            should_guess = (top_prob > 0.60 and separation > 4.0) or (top_prob > 0.75 and entropy < 0.8)
        else:
            # Still conservative late game
            should_guess = (top_prob > 0.50 and separation > 3.0) or (top_prob > 0.65 and entropy < 1.2)
        
        return should_guess, top_prob, int(top_idx)
    
    def get_discriminative_question(self, top_idx, prior, asked):
        """Find question that best separates top candidate from others."""
        top_prob = prior[top_idx]
        # More lenient similarity threshold
        similar_mask = (prior > top_prob * 0.05) # Was 0.1, now 0.05
        similar_mask[top_idx] = False
        similar_indices = np.where(similar_mask)[0]
        
        if len(similar_indices) == 0:
            return None, None
        
        available_indices = [
            i for i, f in enumerate(self.feature_cols)
            if f not in asked and self.allowed_feature_mask[i]
        ]
        if not available_indices:
            return None, None
        
        top_features = self.features[top_idx]
        similar_features = self.features[similar_indices]
        diffs = np.abs(top_features[None, :] - similar_features)
        avg_diffs = np.nanmean(diffs, axis=0)
        
        mask = np.full_like(avg_diffs, -1.0)
        mask[available_indices] = avg_diffs[available_indices]
        mask[np.isnan(top_features)] = -1.0
        
        best_idx = np.argmax(mask)
        best_diff = mask[best_idx]
        
        # Lower threshold for more aggressive discrimination
        if best_diff > 0.25: # Was 0.3, now 0.25
            feature = self.feature_cols[best_idx]
            question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
            return feature, question

        return None, None

    def get_features_for_data_collection(self, item_name: str, num_features: int = 5) -> list[dict]:
        """
        Gets a list of features for data collection for a specific item.
        Prioritizes NULL features for the item, then sparse features, then any allowed.
        """
        final_feature_indices = []
        
        # Find item-specific NULLs
        item_idx_list = np.where(self.animals == item_name)[0]
        
        if len(item_idx_list) > 0:
            item_idx = item_idx_list[0]
            item_row = self.features[item_idx]
            null_indices = np.where(np.isnan(item_row))[0]
            
            allowed_indices_set = set(self.allowed_feature_indices)
            item_null_features = list(set(null_indices.tolist()).intersection(allowed_indices_set))
            
            np.random.shuffle(item_null_features)
            final_feature_indices = item_null_features[:num_features]
        
        # Pad with globally sparse features
        num_needed = num_features - len(final_feature_indices)
        if num_needed > 0:
            sparse_pool = set(self.sparse_feature_indices) - set(final_feature_indices)
            padding_pool = list(sparse_pool)
            np.random.shuffle(padding_pool)
            final_feature_indices.extend(padding_pool[:num_needed])

        # Pad with any allowed feature
        num_needed = num_features - len(final_feature_indices)
        if num_needed > 0:
            all_allowed_pool = set(self.allowed_feature_indices) - set(final_feature_indices)
            padding_pool = list(all_allowed_pool)
            np.random.shuffle(padding_pool)
            final_feature_indices.extend(padding_pool[:num_needed])
            
        # Format output
        output_list = []
        for idx in final_feature_indices:
            feature_name = self.feature_cols[idx]
            q_text = self.questions_map.get(feature_name, f"Does it have {feature_name.replace('_', ' ')}?")
            
            output_list.append({
                "feature_name": feature_name,
                "question": q_text
            })
        
        return output_list
