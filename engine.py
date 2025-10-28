import pandas as pd
import numpy as np
import torch

class AkinatorEngine:
    """
    Highly optimized PyTorch Akinator engine with improved question variance
    and more conservative guessing strategy.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map

        self.answer_values = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float32)
        self.definite_exp = -6.0
        self.uncertain_exp = -3.0

        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'usually': 0.75,
            'sometimes': 0.5, 'maybe': 0.5, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        self.sorted_initial_feature_indices = []
        self._precompute_likelihood_tables()
        self._build_tensors()
        
    def _precompute_likelihood_tables(self):
        """Precompute all possible likelihoods for instant lookup."""
        feature_grid = torch.linspace(0.0, 1.0, 21)
        n_grid = len(feature_grid)
        n_answers = len(self.answer_values)
        
        self.likelihood_table_definite = torch.zeros(n_grid, n_answers)
        self.likelihood_table_uncertain = torch.zeros(n_grid, n_answers)
        
        for i, fval in enumerate(feature_grid):
            for j, aval in enumerate(self.answer_values):
                dist = abs(fval - aval)
                
                like_def = np.exp(self.definite_exp * dist)
                if dist > 0.7:
                    like_def *= 0.001
                elif dist > 0.3:
                    like_def *= 0.2
                self.likelihood_table_definite[i, j] = np.clip(like_def, 0.0001, 1.0)
                
                like_unc = np.exp(self.uncertain_exp * dist)
                self.likelihood_table_uncertain[i, j] = np.clip(like_unc, 0.0001, 1.0)
        
        self.feature_grid = feature_grid
        
    def _build_tensors(self):
        """Builds all tensors from the DataFrame - CPU optimized."""
        self.animals = self.df['animal_name'].values
        features_np = self.df[self.feature_cols].values.astype(np.float32)
        
        self.features = torch.from_numpy(features_np)
        self.nan_mask = torch.isnan(self.features)
        self.features_filled = torch.nan_to_num(self.features, nan=0.5)
        
        col_nan_frac = np.isnan(features_np).mean(axis=0)
        col_var = np.nanvar(features_np, axis=0)
        col_var = np.where(np.isnan(col_var), 0.0, col_var)
        
        self.allowed_feature_mask = (col_nan_frac < 0.8) & (col_var > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].tolist()
        
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
        """Fast info gain using precomputed likelihood tables."""
        curr_entropy = self._entropy_numpy(prior_np)
        if curr_entropy < 0.01:
            return np.zeros(len(feature_indices))
        
        features_np = self.features_filled.numpy()[:, feature_indices]
        nan_mask_np = self.nan_mask.numpy()[:, feature_indices]
        
        quantized = np.clip(np.round(features_np * 20).astype(int), 0, 20)
        
        k = len(feature_indices)
        A = len(self.answer_values)
        n = len(features_np)
        
        likelihoods = np.zeros((n, k, A))
        for a_idx, aval in enumerate(self.answer_values):
            is_definite = abs(aval - 0.5) > 0.3
            table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
            
            for f_idx in range(k):
                indices = quantized[:, f_idx]
                likelihoods[:, f_idx, a_idx] = table[indices, a_idx].numpy()
        
        likelihoods = np.where(nan_mask_np[:, :, None], 1.0, likelihoods)
        
        prob_answer = prior_np @ likelihoods.reshape(n, -1)
        prob_answer = prob_answer.reshape(k, A)
        
        gains = np.zeros(k)
        for f_idx in range(k):
            exp_ent = 0.0
            for a_idx in range(A):
                posterior = prior_np * likelihoods[:, f_idx, a_idx]
                post_sum = posterior.sum()
                if post_sum > 1e-10:
                    posterior = posterior / post_sum
                    exp_ent += prob_answer[f_idx, a_idx] * self._entropy_numpy(posterior)
            gains[f_idx] = curr_entropy - exp_ent
        
        return gains
    
    @staticmethod
    def _entropy_numpy(probs):
        """Fast numpy entropy calculation."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs_safe * np.log(probs_safe))

    def calc_likelihood(self, feature_vec: torch.Tensor, target: float, 
                        definite_exp: float, uncertain_exp: float) -> torch.Tensor:
        """Calculate likelihood using precomputed table - ultra fast."""
        quantized = torch.clamp(torch.round(feature_vec * 20).long(), 0, 20)
        target_idx = torch.argmin(torch.abs(self.answer_values - target))
        
        is_definite = abs(target - 0.5) > 0.3
        table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
        
        return table[quantized, target_idx]
    
    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        """Calculate entropy."""
        probs_safe = torch.clamp(probs, min=1e-10)
        return -torch.sum(probs_safe * torch.log(probs_safe))
    
    def get_prior(self, rejected_mask):
        prior = torch.ones(len(self.animals), dtype=torch.float32)
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
        
        likelihood = torch.where(self.nan_mask[:, feature_idx], 
                                 torch.ones_like(likelihood), likelihood)
        
        posterior = prior * likelihood
        return posterior / (posterior.sum() + 1e-10)

    def info_gain_batch(self, prior, feature_indices):
        """Compute information gain using efficient matmul operations."""
        curr_entropy = self.entropy(prior)
        if curr_entropy < 0.01 or not feature_indices:
            return torch.zeros(len(feature_indices))

        active_thresh = 1e-8
        active_idx = torch.where(prior > active_thresh)[0]
        
        if len(active_idx) == 0:
            active_idx = torch.arange(len(prior))
            prior = torch.ones_like(prior) / len(prior)
            curr_entropy = self.entropy(prior)

        if len(active_idx) < len(prior):
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
        
        quantized = torch.clamp(torch.round(feature_batch * 20).long(), 0, 20)
        likelihoods = torch.zeros(n, k, A)
        
        for a_idx, aval in enumerate(self.answer_values):
            is_definite = abs(aval.item() - 0.5) > 0.3
            table = self.likelihood_table_definite if is_definite else self.likelihood_table_uncertain
            likelihoods[:, :, a_idx] = table[quantized, a_idx]
        
        likelihoods = torch.where(nan_mask_batch.unsqueeze(-1), 
                                  torch.ones_like(likelihoods), likelihoods)
        
        prior_2d = prior_sub.unsqueeze(0)
        like_2d = likelihoods.view(n, k * A)
        prob_answer = torch.matmul(prior_2d, like_2d).view(k, A)
        
        gains = torch.zeros(k)
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

    def select_question(self, prior, asked, question_count):
        """
        Select most informative question with variance in early questions.
        
        Q0: Random from top 10 precomputed features (VARIED)
        Q1-4: Random from top 5 of 25 best precomputed features (MORE VARIANCE)
        Q5+: Pick best from 40 features (MORE ACCURATE)
        """
        
        # === Q0: VARIED START - RANDOM FROM TOP 10 ===
        if question_count == 0:
            if self.sorted_initial_feature_indices:
                available_top_10 = [
                    idx for idx in self.sorted_initial_feature_indices[:10]
                    if self.feature_cols[idx] not in asked
                ]
                
                if available_top_10:
                    # RANDOM choice from top 10 for variety
                    chosen_idx = np.random.choice(available_top_10)
                    feature = self.feature_cols[chosen_idx]
                    question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
                    return feature, question
        
        # === Q1-4: MORE VARIANCE - EXPANDED POOL ===
        all_available_indices = [
            i for i, f in enumerate(self.feature_cols)
            if f not in asked and self.allowed_feature_mask[i]
        ]
        
        if not all_available_indices:
            return None, None
        
        if question_count < 5:
            # Expanded from 15 to 25 for more variety
            MAX_FEATURES_TO_CHECK = 25
            
            available_set = set(all_available_indices)
            top_available_features = [
                idx for idx in self.sorted_initial_feature_indices
                if idx in available_set
            ][:MAX_FEATURES_TO_CHECK]

            if not top_available_features:
                top_available_features = all_available_indices[:MAX_FEATURES_TO_CHECK]
            
            available_features_to_check = top_available_features
            sampled_indices_map = {
                new_idx: old_idx for new_idx, old_idx in enumerate(available_features_to_check)
            }

        else:
            # === Q5+: MORE ACCURATE - EXPANDED FROM 30 TO 40 ===
            MAX_FEATURES_TO_CHECK = 40
            
            if len(all_available_indices) > MAX_FEATURES_TO_CHECK:
                sampled = np.random.choice(all_available_indices, MAX_FEATURES_TO_CHECK, replace=False)
                sampled_indices_map = {new_idx: old_idx for new_idx, old_idx in enumerate(sampled)}
                available_features_to_check = sampled.tolist()
            else:
                sampled_indices_map = {idx: old_idx for idx, old_idx in enumerate(all_available_indices)}
                available_features_to_check = all_available_indices
        
        gains_tensor = self.info_gain_batch(prior, available_features_to_check)
        sorted_indices_of_gains = torch.argsort(gains_tensor, descending=True)

        if len(sorted_indices_of_gains) == 0:
            return None, None

        # Q1-4: Random from top 5 (expanded from 3)
        if question_count < 5:
            top_n = min(5, len(sorted_indices_of_gains))
            chosen_local_idx = sorted_indices_of_gains[np.random.randint(top_n)]
        else:
            # Q5+: Always best
            chosen_local_idx = sorted_indices_of_gains[0]
        
        idx = sampled_indices_map[chosen_local_idx.item()]
        
        feature = self.feature_cols[idx]
        question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        return feature, question
    
    def should_guess(self, prior, question_count):
        """
        More conservative guessing strategy to improve accuracy.
        
        Returns: (should_guess, confidence, top_animal_idx)
        """
        top_prob, top_idx = torch.max(prior, dim=0)
        top_prob = top_prob.item()
        
        # Get second highest probability
        prior_copy = prior.clone()
        prior_copy[top_idx] = 0.0
        second_prob = torch.max(prior_copy).item()
        
        # Calculate separation ratio
        separation = top_prob / (second_prob + 1e-10)
        
        # More conservative thresholds
        if question_count < 8:
            # Need very high confidence early on
            should_guess = top_prob > 0.70 and separation > 5.0
        elif question_count < 12:
            # Moderate confidence needed
            should_guess = top_prob > 0.55 and separation > 3.5
        else:
            # More lenient after many questions
            should_guess = top_prob > 0.40 and separation > 2.5
        
        return should_guess, top_prob, top_idx.item()
    
    def get_discriminative_question(self, top_idx, prior, asked):
        """Find question that best separates top candidate from others."""
        top_prob = prior[top_idx].item()
        similar_mask = (prior > top_prob * 0.1)
        similar_mask[top_idx] = False
        similar_indices = torch.where(similar_mask)[0]
        
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
        diffs = torch.abs(top_features.unsqueeze(0) - similar_features)
        avg_diffs = torch.nanmean(diffs, dim=0)
        
        mask = torch.full_like(avg_diffs, -1.0)
        mask[available_indices] = avg_diffs[available_indices]
        mask[torch.isnan(top_features)] = -1.0
        
        best_diff, best_idx = torch.max(mask, dim=0)
        
        if best_diff.item() > 0.3:
            feature = self.feature_cols[best_idx.item()]
            question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
            return feature, question

        return None, None