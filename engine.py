import pandas as pd
import numpy as np

class AkinatorEngine:
    """Improved Akinator with adaptive confidence and error tolerance."""
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        self.definite_exp = -5.0
        self.uncertain_exp = -2.5
        self.fuzzy_map = {'yes': 1.0, 'y': 1.0, 'usually': 0.75, 'sometimes': 0.5, 
                          'maybe': 0.5, 'rarely': 0.25, 'no': 0.0, 'n': 0.0}
        
        
        # *** IMPROVED: Adaptive Guessing Strategy ***
        self.MAX_QUESTIONS = 25  # Hard cap before final guess
        
        # Dynamic confidence requirements based on question count
        # Early game: very strict, late game: more lenient
        self.confidence_schedule = {
            range(0, 8): (0.95, 20.0),   # Q1-7: 95% + 20x ratio (very strict)
            range(8, 12): (0.92, 15.0),  # Q8-11: 92% + 15x ratio
            range(12, 16): (0.88, 12.0), # Q12-15: 88% + 12x ratio
            range(16, 20): (0.85, 10.0), # Q16-19: 85% + 10x ratio
            range(20, 25): (0.80, 8.0),  # Q20-24: 80% + 8x ratio
        }
        
        # *** NEW: Error Tolerance ***
        # Track answer history to detect potential user errors
        self.answer_history = []  # List of (feature, answer, prior_entropy)
        self.error_dampening = 0.85  # Dampen likelihood impact to 85% to allow recovery
        
        # *** IMPROVED: Smart Sparse Question Strategy ***
        # Ask sparse questions opportunistically when entropy is high
        self.sparse_entropy_threshold = 3.5  # Only ask sparse when confused
        self.sparse_questions_asked = set()
        self.max_sparse_questions = 4  # Limit total sparse questions
        
        self._precompute_likelihood_tables()
        self._build_arrays()
        
    def _precompute_likelihood_tables(self):
        """Precompute likelihoods with moderate penalties for error tolerance."""
        feature_grid = np.linspace(0.0, 1.0, 41, dtype=np.float32)
        n_grid, n_answers = len(feature_grid), len(self.answer_values)
        
        self.likelihood_table_definite = np.zeros((n_grid, n_answers), dtype=np.float32)
        self.likelihood_table_uncertain = np.zeros((n_grid, n_answers), dtype=np.float32)
        
        for i, fval in enumerate(feature_grid):
            for j, aval in enumerate(self.answer_values):
                dist = abs(fval - aval)
                like_def = np.exp(self.definite_exp * dist)
                
                # *** SOFTENED: Less aggressive penalties for error recovery ***
                if dist > 0.7: like_def *= 0.01    # Was 0.001
                elif dist > 0.5: like_def *= 0.1   # Was 0.05
                elif dist > 0.3: like_def *= 0.4   # Was 0.3
                
                self.likelihood_table_definite[i, j] = np.clip(like_def, 0.001, 1.0)
                self.likelihood_table_uncertain[i, j] = np.clip(
                    np.exp(self.uncertain_exp * dist), 0.01, 1.0)
        
        self.feature_grid = feature_grid
        
    def _build_arrays(self):
        """Build arrays from DataFrame."""
        self.animals = self.df['animal_name'].values
        features = self.df[self.feature_cols].values.astype(np.float32)
        
        self.features = features
        self.nan_mask = np.isnan(features)
        self.features_filled = np.where(self.nan_mask, 0.5, features)
        
        col_nan_frac = self.nan_mask.mean(axis=0)
        col_var = np.nanvar(features, axis=0)
        col_var = np.where(np.isnan(col_var), 0.0, col_var)
        self.col_nan_frac = col_nan_frac
        
        # Feature masks
        self.allowed_feature_mask = (col_nan_frac < 0.95) & (col_var > 5e-5)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].tolist()
        
        # Sparse features for data collection (40-100% NaN)
        self.data_collection_sparse_mask = (col_nan_frac >= 0.40)
        self.data_collection_sparse_indices = np.where(self.data_collection_sparse_mask)[0].tolist()
        
        print(f"   Feature stats: {len(self.allowed_feature_indices)} allowed, "
              f"{len(self.data_collection_sparse_indices)} sparse")
        
        # Initial feature ranking
        if len(self.animals) > 0 and self.allowed_feature_indices:
            uniform_prior = np.ones(len(self.animals), dtype=np.float32) / len(self.animals)
            gains = self._compute_gains(uniform_prior, self.allowed_feature_indices, robust=False)
            self.feature_importance = {self.feature_cols[idx]: float(gains[i]) 
                                       for i, idx in enumerate(self.allowed_feature_indices)}
            sorted_idx = np.argsort(gains)[::-1]
            self.sorted_initial_feature_indices = [self.allowed_feature_indices[i] for i in sorted_idx]
        else:
            self.feature_importance = {f: 0.0 for f in self.feature_cols}
            self.sorted_initial_feature_indices = []

    def _compute_gains(self, prior, feature_indices, robust=False):
        """Unified gain computation (expected or minimax)."""
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
        
        # Compute gains
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
                # Expected gain
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
        """Bayesian update with error tolerance and dampening."""
        fuzzy_val = self.fuzzy_map.get(answer.lower().strip())
        if fuzzy_val is None:
            return prior
        
        # Get likelihood from precomputed table
        quantized = np.clip(np.round(self.features_filled[:, feature_idx] * 40).astype(int), 0, 40)
        target_idx = np.argmin(np.abs(self.answer_values - fuzzy_val))
        is_definite = abs(fuzzy_val - 0.5) > 0.3
        table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
        likelihood = table[quantized, target_idx]
        
        # *** NEW: Error Dampening ***
        # Soften the likelihood impact to allow recovery from user errors
        # Move likelihood values closer to 1.0 (neutral) by dampening factor
        likelihood = 1.0 + (likelihood - 1.0) * self.error_dampening
        
        # Hard constraint for definite answers (but softened)
        if fuzzy_val in [1.0, 0.0]:
            contradictions = np.abs(self.features[:, feature_idx] - fuzzy_val)
            hard_contradiction = (~self.nan_mask[:, feature_idx]) & (contradictions > 0.6)
            likelihood = np.where(hard_contradiction, 0.01, likelihood)  # Was 0.0001
        
        # Only treat NaN as neutral for uncertain answers
        if not is_definite:
            likelihood = np.where(self.nan_mask[:, feature_idx], 1.0, likelihood)
        
        # Record answer history for potential error detection
        self.answer_history.append({
            'feature_idx': feature_idx,
            'answer': answer,
            'prior_entropy': self._entropy(prior)
        })
            
        posterior = prior * likelihood
        return posterior / (posterior.sum() + 1e-10)

    def select_sparse_question(self, asked, question_count):
        """Select sparse question for data collection (only when confused)."""
        available = [idx for idx in self.data_collection_sparse_indices 
                     if self.feature_cols[idx] not in asked]
        
        if not available:
            return None, None
        
        # Weight heavily by NaN ratio
        nan_ratios = np.array([self.col_nan_frac[idx] for idx in available])
        weights = np.exp(10 * (nan_ratios - 0.5))
        weights = weights / weights.sum()
        
        chosen_idx = np.random.choice(available, p=weights)
        feature = self.feature_cols[chosen_idx]
        
        print(f"   [Q{question_count}] SPARSE: '{feature}' (NaN: {self.col_nan_frac[chosen_idx]:.1%})")
        
        return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")

    def select_question(self, prior, asked, question_count):
        """
        *** IMPROVED: Adaptive Question Selection ***
        
        Prioritizes discriminative questions and only asks sparse questions
        when entropy is high (confused state).
        """
        # *** NEW: Opportunistic Sparse Questions (only when confused) ***
        current_entropy = self._entropy(prior)
        if (current_entropy > self.sparse_entropy_threshold and 
            len(self.sparse_questions_asked) < self.max_sparse_questions and
            question_count >= 5):  # Not too early
            
            feature, question = self.select_sparse_question(asked, question_count)
            if feature:
                self.sparse_questions_asked.add(feature)
                return feature, question
        
        # Early game: random from top 10 for variety
        if question_count < 3 and self.sorted_initial_feature_indices:
            available_top = [idx for idx in self.sorted_initial_feature_indices[:10]
                             if self.feature_cols[idx] not in asked]
            if available_top:
                idx = np.random.choice(available_top)
                feature = self.feature_cols[idx]
                return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        
        # Get available features
        available = [i for i, f in enumerate(self.feature_cols)
                     if f not in asked and self.allowed_feature_mask[i]]
        if not available:
            return None, None
        
        # *** IMPROVED: Discriminative question when narrowing down ***
        # When we have a clear frontrunner but not confident enough to guess,
        # ask questions that separate it from similar candidates
        sorted_prior = np.argsort(prior)[::-1]
        top_prob = prior[sorted_prior[0]]
        
        if top_prob > 0.4 and question_count >= 5:  # Potential frontrunner
            top_idx = sorted_prior[0]
            feature, question = self.get_discriminative_question(top_idx, prior, asked)
            if feature:
                print(f"   [Q{question_count}] DISCRIMINATIVE for {self.animals[top_idx]}")
                return feature, question
        
        # Adaptive pool size
        max_features = {range(0,5): 30, range(5,8): 50, range(8,12): 75}.get(
            next((r for r in [range(0,5), range(5,8), range(8,12)] if question_count in r), None), 100)
        
        # Build candidate pool
        available_set = set(available)
        candidates = [idx for idx in self.sorted_initial_feature_indices 
                      if idx in available_set][:max_features]
        
        if len(candidates) < max_features:
            remaining = list(available_set - set(candidates))
            np.random.shuffle(remaining)
            candidates.extend(remaining[:max_features - len(candidates)])
        
        if not candidates:
            return None, None
        
        # Compute gains (robust if high entropy)
        robust = current_entropy > 3.0
        gains = self._compute_gains(prior, candidates, robust=robust)
        sorted_idx = np.argsort(gains)[::-1]
        
        # Weighted random from top 3 for variety
        if question_count >= 3 and len(sorted_idx) >= 3:
            chosen = np.random.choice(sorted_idx[:3], p=[0.7, 0.2, 0.1])
        else:
            chosen = sorted_idx[0]
        
        idx = candidates[int(chosen)]
        feature = self.feature_cols[idx]
        return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
    
    def should_guess(self, prior, question_count):
        """
        *** IMPROVED: Adaptive Confidence Guessing ***
        
        Uses dynamic confidence thresholds that become more lenient as
        questions progress. Forces guess at MAX_QUESTIONS.
        
        Early game: Very strict (avoid premature guessing)
        Late game: More lenient (must make a guess eventually)
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
        confidence_threshold = 0.95  # Default (very strict)
        ratio_threshold = 20.0
        
        for q_range, (conf, ratio) in self.confidence_schedule.items():
            if question_count in q_range:
                confidence_threshold = conf
                ratio_threshold = ratio
                break
        
        # Get top 2 candidates
        sorted_idx = np.argsort(prior)[::-1]
        top_idx = sorted_idx[0]
        top_prob = prior[top_idx]
        second_prob = prior[sorted_idx[1]] if len(prior) > 1 else 0.0
        
        # Calculate ratio
        ratio = top_prob / (second_prob + 1e-12)
        
        # *** NEW: Additional check for entropy ***
        # Don't guess if entropy is still very high (still confused)
        current_entropy = self._entropy(prior)
        if current_entropy > 2.5 and question_count < 15:
            return {'guess': False, 'animal': None, 'confidence': float(top_prob)}
        
        # Check both conditions
        is_absolutely_sure = top_prob > confidence_threshold
        is_relatively_sure = ratio > ratio_threshold
        
        if is_absolutely_sure and is_relatively_sure:
            print(f"   [Q{question_count}] Confident enough: {top_prob:.1%} confidence, "
                  f"{ratio:.1f}x ratio (thresholds: {confidence_threshold:.1%}, {ratio_threshold:.1f}x)")
            return {
                'guess': True,
                'animal': self.animals[top_idx],
                'confidence': float(top_prob)
            }
        
        return {'guess': False, 'animal': None, 'confidence': float(top_prob)}
    
    def get_discriminative_question(self, top_idx, prior, asked):
        """
        *** IMPROVED: Find question that best separates top candidate from similar ones ***
        """
        # Find candidates similar to the top one (within 10% probability)
        similar_mask = (prior > prior[top_idx] * 0.1)
        similar_mask[top_idx] = False
        similar_indices = np.where(similar_mask)[0]
        
        if len(similar_indices) == 0:
            return None, None
        
        available = [i for i, f in enumerate(self.feature_cols)
                     if f not in asked and self.allowed_feature_mask[i]]
        if not available:
            return None, None
        
        # Find features where top candidate differs most from similar ones
        diffs = np.abs(self.features[top_idx] - self.features[similar_indices])
        avg_diffs = np.nanmean(diffs, axis=0)
        
        # Mask unavailable features
        mask = np.full_like(avg_diffs, -1.0)
        mask[available] = avg_diffs[available]
        mask[np.isnan(self.features[top_idx])] = -1.0
        
        # Get feature with maximum difference
        best_idx = np.argmax(mask)
        if mask[best_idx] > 0.25:  # Significant difference
            feature = self.feature_cols[best_idx]
            return feature, self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        
        return None, None

    def get_features_for_data_collection(self, item_name: str, num_features: int = 5) -> list[dict]:
        """Collect data for sparse and missing features."""
        final_indices = []
        
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
            item_nulls_sorted = sorted(null_indices.tolist(), 
                                      key=lambda idx: self.col_nan_frac[idx], 
                                      reverse=True)
            final_indices = item_nulls_sorted[:num_features]
        
        # Strategy 2: Globally sparse features
        if len(final_indices) < num_features:
            needed = num_features - len(final_indices)
            sparse_candidates = [idx for idx in self.data_collection_sparse_indices 
                                if idx not in final_indices]
            sparse_sorted = sorted(sparse_candidates, 
                                  key=lambda idx: self.col_nan_frac[idx], 
                                  reverse=True)
            final_indices.extend(sparse_sorted[:needed])
        
        # Strategy 3: High-variance features
        if len(final_indices) < num_features:
            needed = num_features - len(final_indices)
            variance_candidates = [idx for idx in self.allowed_feature_indices 
                                  if idx not in final_indices]
            if variance_candidates:
                variances = np.nanvar(self.features[:, variance_candidates], axis=0)
                sorted_var_idx = np.argsort(variances)[::-1]
                high_var_indices = [variance_candidates[i] for i in sorted_var_idx[:needed]]
                final_indices.extend(high_var_indices)
        
        return [{"feature_name": self.feature_cols[idx],
                 "question": self.questions_map.get(self.feature_cols[idx], 
                                     f"Does it have {self.feature_cols[idx].replace('_', ' ')}?"),
                 "nan_percentage": float(self.col_nan_frac[idx])}
                for idx in final_indices[:num_features]]