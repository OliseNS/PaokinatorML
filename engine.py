import pandas as pd
import numpy as np

class AkinatorEngine:
    """Enhanced Akinator with improved accuracy through better question selection and confidence calibration."""
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        self.definite_exp = -6.0  # Sharper penalties for contradictions
        self.uncertain_exp = -2.0
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'no': 0.0, 'n': 0.0,
            'mostly': 0.75, 'probably': 0.75, 'usually': 0.75,
            'sort of': 0.5, 'sometimes': 0.5, 'maybe': 0.5,
            'not really': 0.25, 'rarely': 0.25
        }
        
        self.MAX_QUESTIONS = 25  
        
        # FIXED: More conservative confidence schedule for better accuracy
        self.confidence_schedule = {
            range(0, 8): (0.95, 20.0),   # Q1-7: Very conservative early
            range(8, 12): (0.93, 18.0),  # Q8-11: Still conservative
            range(12, 16): (0.90, 15.0), # Q12-15: Moderate
            range(16, 20): (0.87, 12.0), # Q16-19: More willing
            range(20, 25): (0.83, 9.0),  # Q20-24: Lenient at end
        }
        
        # Track question history for adaptive strategy
        self.answer_history = []
        self.error_dampening = 0.92  # FIXED: Less aggressive dampening to preserve signal
        
        # Improved sparse question strategy
        self.sparse_entropy_threshold = 4.0  # Higher threshold
        self.sparse_questions_asked = set()
        self.max_sparse_questions = 3  # Fewer sparse questions
        
        # Track consistency for error detection
        self.contradiction_count = 0
        self.max_contradictions = 3
        
        self._precompute_likelihood_tables()
        self._build_arrays()
        
    def _precompute_likelihood_tables(self):
        """Precompute likelihoods with sharper penalties for definite answers."""
        feature_grid = np.linspace(0.0, 1.0, 41, dtype=np.float32)
        n_grid, n_answers = len(feature_grid), len(self.answer_values)
        
        self.likelihood_table_definite = np.zeros((n_grid, n_answers), dtype=np.float32)
        self.likelihood_table_uncertain = np.zeros((n_grid, n_answers), dtype=np.float32)
        
        for i, fval in enumerate(feature_grid):
            for j, aval in enumerate(self.answer_values):
                dist = abs(fval - aval)
                like_def = np.exp(self.definite_exp * dist)
                
                # Sharper penalties for definite contradictions
                if dist > 0.7: like_def *= 0.005
                elif dist > 0.5: like_def *= 0.05
                elif dist > 0.3: like_def *= 0.3
                
                self.likelihood_table_definite[i, j] = np.clip(like_def, 0.0001, 1.0)
                self.likelihood_table_uncertain[i, j] = np.clip(
                    np.exp(self.uncertain_exp * dist), 0.01, 1.0)
        
        self.feature_grid = feature_grid
        
    def _build_arrays(self):
        """Build arrays from DataFrame with improved feature selection."""
        self.animals = self.df['animal_name'].values
        features = self.df[self.feature_cols].values.astype(np.float32)
        
        self.features = features
        self.nan_mask = np.isnan(features)
        self.features_filled = np.where(self.nan_mask, 0.5, features)
        
        col_nan_frac = self.nan_mask.mean(axis=0)
        col_var = np.nanvar(features, axis=0)
        col_var = np.where(np.isnan(col_var), 0.0, col_var)
        self.col_nan_frac = col_nan_frac
        
        # More selective feature filtering
        self.allowed_feature_mask = (col_nan_frac < 0.90) & (col_var > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].tolist()
        
        # Sparse features (50-100% NaN)
        self.data_collection_sparse_mask = (col_nan_frac >= 0.50)
        self.data_collection_sparse_indices = np.where(self.data_collection_sparse_mask)[0].tolist()
        
        print(f"   Feature stats: {len(self.allowed_feature_indices)} allowed, "
              f"{len(self.data_collection_sparse_indices)} sparse")
        
        # Enhanced feature ranking with diversity
        if len(self.animals) > 0 and self.allowed_feature_indices:
            uniform_prior = np.ones(len(self.animals), dtype=np.float32) / len(self.animals)
            gains = self._compute_gains(uniform_prior, self.allowed_feature_indices, robust=False)
            
            # Add diversity bonus
            diversity_scores = self._compute_feature_diversity(self.allowed_feature_indices)
            combined_scores = gains + 0.1 * diversity_scores  # 10% diversity bonus
            
            self.feature_importance = {self.feature_cols[idx]: float(combined_scores[i]) 
                                       for i, idx in enumerate(self.allowed_feature_indices)}
            sorted_idx = np.argsort(combined_scores)[::-1]
            self.sorted_initial_feature_indices = [self.allowed_feature_indices[i] for i in sorted_idx]
        else:
            self.feature_importance = {f: 0.0 for f in self.feature_cols}
            self.sorted_initial_feature_indices = []

    def _compute_feature_diversity(self, feature_indices):
        """Calculate diversity score based on feature distribution balance."""
        diversity = np.zeros(len(feature_indices))
        for i, idx in enumerate(feature_indices):
            feature_vals = self.features[:, idx]
            valid_vals = feature_vals[~np.isnan(feature_vals)]
            if len(valid_vals) > 0:
                # Higher diversity for balanced distributions
                hist, _ = np.histogram(valid_vals, bins=5, range=(0, 1))
                hist_norm = hist / (hist.sum() + 1e-10)
                # Entropy of distribution
                diversity[i] = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        return diversity

    def _compute_gains(self, prior, feature_indices, robust=False):
        """FIXED: Removed variance bonus that prioritized noisy questions."""
        curr_entropy = self._entropy(prior)
        if curr_entropy < 0.01 or not feature_indices:
            return np.zeros(len(feature_indices))
        
        # Active set optimization
        active_idx = np.where(prior > 1e-9)[0]
        if len(active_idx) == 0:
            active_idx = np.arange(len(prior))
            prior = np.ones_like(prior) / len(prior)
        
        # Use subset if beneficial
        if len(active_idx) < len(prior) * 0.9:
            prior_sub = prior[active_idx] / (prior[active_idx].sum() + 1e-12)
            feature_batch = self.features_filled[active_idx][:, feature_indices]
            nan_mask_batch = self.nan_mask[active_idx][:, feature_indices]
        else:
            prior_sub = prior
            feature_batch = self.features_filled[:, feature_indices]
            nan_mask_batch = self.nan_mask[:, feature_indices]
        
        k, A, n = len(feature_indices), len(self.answer_values), len(prior_sub)
        quantized = np.clip(np.round(feature_batch * 40).astype(int), 0, 40)
        likelihoods = np.zeros((n, k, A), dtype=np.float32)
        
        # Vectorized likelihood lookup
        for a_idx, aval in enumerate(self.answer_values):
            table = (self.likelihood_table_definite if abs(aval.item() - 0.5) > 0.3 
                     else self.likelihood_table_uncertain)
            likelihoods[:, :, a_idx] = table[quantized, a_idx]
        
        likelihoods = np.where(nan_mask_batch[:, :, None], 1.0, likelihoods)
        
        # Compute gains - pure information gain without variance bonus
        gains = np.zeros(k)
        for f_idx in range(k):
            if robust:
                # Minimax: worst-case entropy
                max_ent = max((self._entropy(prior_sub * likelihoods[:, f_idx, a_idx] / 
                                (prior_sub * likelihoods[:, f_idx, a_idx]).sum()) 
                                if (prior_sub * likelihoods[:, f_idx, a_idx]).sum() > 1e-10 else 0.0)
                               for a_idx in range(A))
                gains[f_idx] = curr_entropy - max_ent
            else:
                # Pure expected information gain
                prob_answer = prior_sub @ likelihoods[:, f_idx, :]
                exp_ent = sum(prob_answer[a_idx] * self._entropy(
                    prior_sub * likelihoods[:, f_idx, a_idx] / 
                    (prior_sub * likelihoods[:, f_idx, a_idx]).sum())
                    if (prior_sub * likelihoods[:, f_idx, a_idx]).sum() > 1e-10 else 0.0
                    for a_idx in range(A))
                
                gains[f_idx] = curr_entropy - exp_ent
        
        return gains
    
    @staticmethod
    def _entropy(probs):
        """Calculate entropy."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs_safe * np.log(probs_safe))
    
    def entropy(self, probs):
        return self._entropy(probs)
    
    def get_prior(self, rejected_mask):
        prior = np.ones(len(self.animals), dtype=np.float32)
        prior[rejected_mask] = 0.0
        return prior / (prior.sum() + 1e-10)
    
    def update(self, prior, feature_idx, answer):
        """Enhanced Bayesian update with contradiction tracking."""
        fuzzy_val = self.fuzzy_map.get(answer.lower().strip())
        if fuzzy_val is None:
            return prior
        
        # Get likelihood from precomputed table
        quantized = np.clip(np.round(self.features_filled[:, feature_idx] * 40).astype(int), 0, 40)
        target_idx = np.argmin(np.abs(self.answer_values - fuzzy_val))
        is_definite = abs(fuzzy_val - 0.5) > 0.3
        table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
        likelihood = table[quantized, target_idx]
        
        # FIXED: Less aggressive error dampening to preserve signal
        likelihood = 1.0 + (likelihood - 1.0) * self.error_dampening
        
        # Track contradictions
        if fuzzy_val in [1.0, 0.0]:
            contradictions = np.abs(self.features[:, feature_idx] - fuzzy_val)
            hard_contradiction = (~self.nan_mask[:, feature_idx]) & (contradictions > 0.7)
            
            # Count if this eliminates top candidates
            top_candidates = prior > 0.1
            if np.any(hard_contradiction & top_candidates):
                self.contradiction_count += 1
            
            # Sharper penalty for definite contradictions
            likelihood = np.where(hard_contradiction, 0.001, likelihood)
        
        # Only treat NaN as neutral for uncertain answers
        if not is_definite:
            likelihood = np.where(self.nan_mask[:, feature_idx], 1.0, likelihood)
        
        # Record answer history
        self.answer_history.append({
            'feature_idx': feature_idx,
            'answer': answer,
            'prior_entropy': self._entropy(prior)
        })
            
        posterior = prior * likelihood
        posterior_norm = posterior / (posterior.sum() + 1e-10)
        
        # FIXED: More conservative early smoothing - only first 3 questions
        if len(self.answer_history) <= 3:
            posterior_norm = 0.97 * posterior_norm + 0.03 * prior
        
        return posterior_norm

    def select_sparse_question(self, asked, question_count):
        """Select sparse question for data collection (only when very confused)."""
        available = [idx for idx in self.data_collection_sparse_indices 
                     if self.feature_cols[idx] not in asked]
        
        if not available:
            return None, None
        
        # Weight heavily by NaN ratio
        nan_ratios = np.array([self.col_nan_frac[idx] for idx in available])
        weights = np.exp(12 * (nan_ratios - 0.5))
        weights = weights / weights.sum()
        
        chosen_idx = np.random.choice(available, p=weights)
        feature = self.feature_cols[chosen_idx]
        
        print(f"   [Q{question_count}] SPARSE: '{feature}' (NaN: {self.col_nan_frac[chosen_idx]:.1%})")
        
        return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")

    def select_question(self, prior, asked, question_count):
        """
        Improved question selection with better early-game strategy.
        """
        current_entropy = self._entropy(prior)
        
        # Sparse questions only when very confused
        if (current_entropy > self.sparse_entropy_threshold and 
            len(self.sparse_questions_asked) < self.max_sparse_questions and
            question_count >= 8):
            
            feature, question = self.select_sparse_question(asked, question_count)
            if feature:
                self.sparse_questions_asked.add(feature)
                return feature, question
        
        # *** FIX 1: Make "early game" logic *only* apply to the very first question. ***
        # For Q2+, we MUST use the updated prior (via _compute_gains) to "divide fast".
        # Original was `if question_count < 4:`, which ignored user's first answers.
        if question_count <= 1 and self.sorted_initial_feature_indices:
            available_top = [idx for idx in self.sorted_initial_feature_indices[:8]
                             if self.feature_cols[idx] not in asked]
            if available_top:
                # Weighted toward top
                weights = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.01, 0.01][:len(available_top)])
                weights = weights / weights.sum()
                idx = np.random.choice(available_top, p=weights)
                feature = self.feature_cols[idx]
                return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        
        # Get available features
        available = [i for i, f in enumerate(self.feature_cols)
                     if f not in asked and self.allowed_feature_mask[i]]
        if not available:
            return None, None
        
        # FIXED: More selective discriminative questions - only when very confident
        sorted_prior = np.argsort(prior)[::-1]
        top_prob = prior[sorted_prior[0]]
        
        if top_prob > 0.50 and question_count >= 6:  # Higher threshold, later in game
            top_idx = sorted_prior[0]
            feature, question = self.get_discriminative_question(top_idx, prior, asked)
            if feature:
                print(f"   [Q{question_count}] DISCRIMINATIVE for {self.animals[top_idx]}")
                return feature, question
        
        # Adaptive pool size - smaller for focused search
        max_features = {
            range(0,4): 20,
            range(4,8): 40,
            range(8,12): 60
        }.get(next((r for r in [range(0,4), range(4,8), range(8,12)] if question_count in r), None), 80)
        
        # Build candidate pool
        available_set = set(available)
        candidates = [idx for idx in self.sorted_initial_feature_indices 
                      if idx in available_set][:max_features]
        
        if len(candidates) < max_features:
            remaining = list(available_set - set(candidates))
            np.random.shuffle(remaining) # This shuffle is good!
            candidates.extend(remaining[:max_features - len(candidates)])
        
        if not candidates:
            return None, None
        
        # Compute gains
        robust = current_entropy > 3.5
        gains = self._compute_gains(prior, candidates, robust=robust)
        sorted_idx = np.argsort(gains)[::-1]
        
        # *** FIX 2: Implement "smarter randomization" using softmax. ***
        # This dynamically creates probabilities based on *how good* the
        # questions are. If gains are [1.5, 1.49], they get ~equal chance.
        # If gains are [1.5, 0.5], the 1.5 gets a *much* higher chance.
        # Original was `p=[0.6, 0.2, 0.1, 0.1]`
        if question_count >= 3 and len(sorted_idx) >= 4:
            top_n_indices = sorted_idx[:4]
            top_n_gains = gains[top_n_indices]
            
            # Use softmax with a "temperature" (T)
            T = 2.0  # Controls "sharpness" of probability distribution
            exp_gains = np.exp(top_n_gains * T)
            exp_sum = exp_gains.sum()

            if exp_sum < 1e-10:
                # Fallback if gains are all effectively zero
                probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
            else:
                probs = exp_gains / exp_sum
                # CRITICAL FIX: Force the sum to be exactly 1.0 for np.random.choice
                probs = probs / probs.sum()
            
            chosen_local_idx = np.random.choice(np.arange(4), p=probs)
            chosen = top_n_indices[chosen_local_idx]
            
        else:
            # For early questions, just be greedy
            chosen = sorted_idx[0]
        
        idx = candidates[int(chosen)]
        feature = self.feature_cols[idx]
        return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
    
    def should_guess(self, prior, question_count):
        """
        FIXED: More conservative guessing strategy for better accuracy.
        """
        # Force guess at maximum questions
        if question_count >= self.MAX_QUESTIONS:
            sorted_idx = np.argsort(prior)[::-1]
            top_idx = sorted_idx[0]
            print(f"   [Q{question_count}] FORCED GUESS at max questions")
            return {
                'guess': True,
                'animal': self.animals[top_idx],
                'confidence': float(prior[top_idx])
            }
        
        # Get confidence thresholds for current question count
        confidence_threshold = 0.95
        ratio_threshold = 20.0
        
        for q_range, (conf, ratio) in self.confidence_schedule.items():
            if question_count in q_range:
                confidence_threshold = conf
                ratio_threshold = ratio
                break
        
        # Get top candidates
        sorted_idx = np.argsort(prior)[::-1]
        top_idx = sorted_idx[0]
        top_prob = prior[top_idx]
        second_prob = prior[sorted_idx[1]] if len(prior) > 1 else 0.0
        
        ratio = top_prob / (second_prob + 1e-12)
        current_entropy = self._entropy(prior)
        
        # FIXED: More conservative entropy check - don't guess when confused
        if current_entropy > 2.5 and question_count < 15:
            return {'guess': False, 'animal': None, 'confidence': float(top_prob)}
        
        # Check both conditions with AND (more conservative)
        is_absolutely_sure = top_prob > confidence_threshold
        is_relatively_sure = ratio > ratio_threshold
        
        # REMOVED: Mid-game heuristic that caused premature guessing
        
        if is_absolutely_sure and is_relatively_sure:
            print(f"   [Q{question_count}] Confident: {top_prob:.1%} confidence, "
                  f"{ratio:.1f}x ratio (thresholds: {confidence_threshold:.1%}, {ratio_threshold:.1f}x)")
            return {
                'guess': True,
                'animal': self.animals[top_idx],
                'confidence': float(top_prob)
            }
        
        return {'guess': False, 'animal': None, 'confidence': float(top_prob)}
    
    def get_discriminative_question(self, top_idx, prior, asked):
        """Find question that best separates top candidate from competitors."""
        # FIXED: Tighter competitor threshold for more relevant discrimination
        competitor_threshold = max(0.08, prior[top_idx] * 0.15)  # Stricter threshold
        similar_mask = (prior > competitor_threshold)
        similar_mask[top_idx] = False
        similar_indices = np.where(similar_mask)[0]
        
        if len(similar_indices) == 0:
            return None, None
        
        available = [i for i, f in enumerate(self.feature_cols)
                     if f not in asked and self.allowed_feature_mask[i]]
        if not available:
            return None, None
        
        # Find features where top candidate differs most from competitors
        top_features = self.features[top_idx]
        competitor_features = self.features[similar_indices]
        
        # Weighted by competitor probabilities
        weights = prior[similar_indices] / prior[similar_indices].sum()
        weighted_avg = np.average(competitor_features, axis=0, weights=weights)
        
        diffs = np.abs(top_features - weighted_avg)
        
        # Mask unavailable features and NaNs
        mask = np.full_like(diffs, -1.0)
        mask[available] = diffs[available]
        mask[np.isnan(top_features)] = -1.0
        mask[np.isnan(weighted_avg)] = -1.0
        
        # Get feature with maximum difference
        best_idx = np.argmax(mask)
        if mask[best_idx] > 0.4:  # FIXED: Higher threshold for significant difference
            feature = self.feature_cols[best_idx]
            return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        
        return None, None

    def get_features_for_data_collection(self, item_name: str, num_features: int = 5) -> list[dict]:
        """
        *** FIX 3: Collect data with randomized sampling to avoid "always same questions". ***
        We create a pool of good candidates (e.g., top 15) and randomly 
        sample from that pool, rather than just taking the deterministic top 5.
        """
        final_indices = []
        pool_multiplier = 3 # Create a pool 3x larger than needed
        
        # Find item
        item_idx_list = np.where(self.animals == item_name)[0]
        if len(item_idx_list) == 0:
            print(f"   Warning: Item '{item_name}' not found")
            item_idx = None
        else:
            item_idx = item_idx_list[0]
        
        # Strategy 1: Item-specific NULLs
        if item_idx is not None:
            null_indices = np.where(np.isnan(self.features[item_idx]))[0]
            # Prioritize nulls that are also globally sparse
            item_nulls_sorted = sorted(null_indices.tolist(), 
                                      key=lambda idx: self.col_nan_frac[idx], 
                                      reverse=True)
            
            # Create a pool of good candidates
            pool_size = min(len(item_nulls_sorted), num_features * pool_multiplier)
            pool = item_nulls_sorted[:pool_size]
            
            # Sample from the pool
            sample_size = min(len(pool), num_features)
            if sample_size > 0:
                final_indices = np.random.choice(pool, size=sample_size, replace=False).tolist()
        
        # Strategy 2: Globally sparse features
        needed = num_features - len(final_indices)
        if needed > 0:
            sparse_candidates = [idx for idx in self.data_collection_sparse_indices 
                                if idx not in final_indices]
            sparse_sorted = sorted(sparse_candidates, 
                                  key=lambda idx: self.col_nan_frac[idx], 
                                  reverse=True)
            
            pool_size = min(len(sparse_sorted), needed * pool_multiplier)
            pool = sparse_sorted[:pool_size]
            
            sample_size = min(len(pool), needed)
            if sample_size > 0:
                chosen = np.random.choice(pool, size=sample_size, replace=False).tolist()
                final_indices.extend(chosen)
        
        # Strategy 3: High-variance features
        needed = num_features - len(final_indices)
        if needed > 0:
            variance_candidates = [idx for idx in self.allowed_feature_indices 
                                  if idx not in final_indices]
            if variance_candidates:
                variances = np.nanvar(self.features[:, variance_candidates], axis=0)
                sorted_var_idx = np.argsort(variances)[::-1]
                high_var_indices = [variance_candidates[i] for i in sorted_var_idx]

                pool_size = min(len(high_var_indices), needed * pool_multiplier)
                pool = high_var_indices[:pool_size]
                
                sample_size = min(len(pool), needed)
                if sample_size > 0:
                    chosen = np.random.choice(pool, size=sample_size, replace=False).tolist()
                    final_indices.extend(chosen)
        
        # Ensure we return the correct number, just in case
        final_indices_unique = list(dict.fromkeys(final_indices)) # Preserve order, remove duplicates
        
        return [{"feature_name": self.feature_cols[idx],
                 "question": self.questions_map.get(self.feature_cols[idx], 
                                     f"Does it have {self.feature_cols[idx].replace('_', ' ')}?"),
                 "nan_percentage": float(self.col_nan_frac[idx])}
                for idx in final_indices_unique[:num_features]]