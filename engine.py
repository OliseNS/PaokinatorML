import pandas as pd
import numpy as np
import torch

class AkinatorEngine:
    """
    Optimized PyTorch-based Akinator engine with FP16 (half precision).
    Uses float16 for 50% memory reduction and better cache efficiency.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map

        # Use float32 for answer values (small array, needs precision)
        self.answer_values = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float32)
        self.definite_exp = -6.0
        self.uncertain_exp = -3.0

        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'usually': 0.75,
            'sometimes': 0.5, 'maybe': 0.5, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        # Performance optimization caches
        self._question_cache = {}  # Cache for precomputed questions
        self._uniform_prior = None  # Precomputed uniform prior
        self.sorted_initial_feature_indices = []
        self._build_tensors()
        
    def _build_tensors(self):
        """(Re)builds all tensors from the DataFrame using FP16."""
        self.animals = self.df['animal_name'].values
        
        features_np = self.df[self.feature_cols].values.astype(np.float32)
        
        # Store features as float16 (half precision) - 50% memory savings
        self.features = torch.from_numpy(features_np).half()
        self.nan_mask = torch.isnan(self.features)
        self.features_filled = torch.nan_to_num(self.features, nan=0.5)
        
        self.specificity = self._compute_specificity_all()
        
        # Feature-level heuristics (keep as float32 for stability)
        try:
            col_nan_frac = np.isnan(features_np).mean(axis=0)
            col_var = np.nanvar(features_np, axis=0)
        except Exception:
            col_nan_frac = np.zeros(len(self.feature_cols))
            col_var = np.ones(len(self.feature_cols))
        
        self.allowed_feature_mask = (col_nan_frac < 0.8) & (col_var > 1e-4)
        self.allowed_feature_indices = [i for i, ok in enumerate(self.allowed_feature_mask) if ok]
        
        # Pre-compute Q0 results and uniform prior (use float32 for precision)
        try:
            N = len(self.animals)
            if N > 0 and len(self.allowed_feature_indices) > 0:
                # Precompute uniform prior for instant Q0 access
                self._uniform_prior = torch.ones(N, dtype=torch.float32) / float(N)
                
                gains = self.info_gain_batch(self._uniform_prior, self.allowed_feature_indices)
                
                self.feature_importance = {self.feature_cols[idx]: float(gains[i].item())
                                           for i, idx in enumerate(self.allowed_feature_indices)}
                
                sorted_gains_indices = torch.argsort(gains, descending=True)
                self.sorted_initial_feature_indices = [self.allowed_feature_indices[i.item()] for i in sorted_gains_indices]
                
                # Precompute Q0 questions for instant response
                self._precompute_q0_questions()
                
            else:
                self.feature_importance = {f: 0.0 for f in self.feature_cols}
                self.sorted_initial_feature_indices = []
                self._uniform_prior = None
        except Exception as e:
            print(f"Warning: failed computing feature importance: {e}")
            self.feature_importance = {f: 0.0 for f in self.feature_cols}
            self.sorted_initial_feature_indices = []
            self._uniform_prior = None

    def _precompute_q0_questions(self):
        """Precompute Q0 questions for instant response."""
        if not self.sorted_initial_feature_indices:
            return
            
        # Precompute top 20 Q0 questions for variety
        top_20_features = self.sorted_initial_feature_indices[:20]
        self._question_cache['q0_questions'] = []
        
        for idx in top_20_features:
            feature = self.feature_cols[idx]
            question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
            self._question_cache['q0_questions'].append((feature, question))
        
        print(f"✅ Precomputed {len(self._question_cache['q0_questions'])} Q0 questions for instant response")

    def _compute_specificity_all(self):
        """Compute specificity score for all animals (batched)."""
        with torch.no_grad():
            # Convert to float32 for exp/log operations
            features_f32 = self.features_filled.float()
            dists = torch.abs(features_f32.unsqueeze(-1) - self.answer_values.view(1, 1, -1))
            nearest = torch.argmin(dists, dim=-1)
            
            A = len(self.answer_values)
            one_hot = torch.nn.functional.one_hot(nearest, num_classes=A)
            bins = one_hot.sum(dim=1).float()
            
            probs = bins / (bins.sum(dim=1, keepdim=True) + 1e-10)
            probs_safe = torch.clamp(probs, min=1e-10)
            entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=1)
            max_entropy = np.log(len(self.answer_values))
            
            # Return as float16
            return (entropy / (max_entropy + 1e-12)).half()

    @staticmethod
    def calc_likelihood(feature_vec: torch.Tensor, target: float, 
                        definite_exp: float, uncertain_exp: float) -> torch.Tensor:
        """Calculate likelihood of answer given features."""
        # Convert to float32 for exp operations (more stable)
        feature_vec = feature_vec.float()
        dist = torch.abs(target - feature_vec)
        
        if abs(target - 0.5) > 0.3:
            likelihood = torch.exp(definite_exp * dist)
            likelihood = torch.where(dist > 0.7, likelihood * 0.001, likelihood)
            likelihood = torch.where((dist > 0.3) & (dist <= 0.7), likelihood * 0.2, likelihood)
        else:
            likelihood = torch.exp(uncertain_exp * dist)
        
        return torch.clamp(likelihood, 0.0001, 1.0)
    
    @staticmethod
    @torch.jit.script
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        probs_safe = torch.clamp(probs, min=1e-10)
        return -torch.sum(probs_safe * torch.log(probs_safe))
    
    @staticmethod
    @torch.jit.script
    def fast_entropy(probs: torch.Tensor) -> torch.Tensor:
        """Optimized entropy calculation for better performance."""
        probs_safe = torch.clamp(probs, min=1e-10)
        log_probs = torch.log(probs_safe)
        return -torch.sum(probs_safe * log_probs)
    
    def get_prior(self, rejected_mask):
        prior = torch.ones(len(self.animals), dtype=torch.float32)
        prior[rejected_mask] = 0.0
        return prior / (prior.sum() + 1e-10)
    
    def update(self, prior, feature_idx, answer):
        """Update beliefs via Bayesian inference."""
        fuzzy_val = self.fuzzy_map.get(answer.lower().strip())
        if fuzzy_val is None:
            return prior
        
        # Features are in FP16, but likelihood calc needs FP32
        feature_vec = self.features_filled[:, feature_idx]
        likelihood = self.calc_likelihood(feature_vec, fuzzy_val, 
                                          self.definite_exp, self.uncertain_exp)
        
        likelihood = torch.where(self.nan_mask[:, feature_idx], 
                                 torch.ones_like(likelihood), likelihood)
        
        posterior = prior * likelihood
        return posterior / (posterior.sum() + 1e-10)

    def info_gain_batch(self, prior, feature_indices):
        """Compute information gain for multiple features efficiently."""
        curr_entropy = self.fast_entropy(prior)
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
            # Convert FP16 to FP32 for calculations
            feature_batch = self.features_filled[active_idx][:, feature_indices].float()
            nan_mask_batch = self.nan_mask[active_idx][:, feature_indices]
        else:
            prior_sub = prior
            feature_batch = self.features_filled[:, feature_indices].float()
            nan_mask_batch = self.nan_mask[:, feature_indices]

        k = len(feature_indices)
        A = len(self.answer_values)
        
        prior_exp = prior_sub.view(-1, 1, 1)
        feat_exp = feature_batch.unsqueeze(-1)
        nan_exp = nan_mask_batch.unsqueeze(-1)
        ans_exp = self.answer_values.view(1, 1, -1)
        
        dist = torch.abs(ans_exp - feat_exp)
        definite_mask = (torch.abs(ans_exp - 0.5) > 0.3)
        
        def_like = torch.exp(self.definite_exp * dist)
        def_like = torch.where(dist > 0.7, def_like * 0.001, def_like)
        def_like = torch.where((dist > 0.3) & (dist <= 0.7), def_like * 0.2, def_like)
        
        unc_like = torch.exp(self.uncertain_exp * dist)
        
        likelihoods = torch.where(definite_mask, def_like, unc_like)
        likelihoods = torch.where(nan_exp, torch.ones_like(likelihoods), likelihoods)
        likelihoods = torch.clamp(likelihoods, 0.0001, 1.0)
        
        prob_answer = torch.einsum('n,nka->ka', prior_sub, likelihoods)
        posterior = prior_exp * likelihoods
        posterior_sum = posterior.sum(dim=0, keepdim=True)
        normalized_posterior = posterior / (posterior_sum + 1e-10)

        probs_safe = torch.clamp(normalized_posterior, min=1e-10)
        posterior_entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=0)
        expected_entropy = torch.sum(prob_answer * posterior_entropy, dim=1)

        gains = curr_entropy - expected_entropy
        return gains

    def select_question(self, prior, asked, question_count):
        """
        Select most informative question using precomputed Q0.
        
        Q0: INSTANT - Randomly pick from precomputed questions (no calculation)
        Q1-4: FAST - Randomly pick from top 3 of 15 best precomputed features
        Q5+: ACCURATE - Pick best from 30 features (accurate, slower)
        """
        
        # === Q0: INSTANT, RANDOM FROM PRECOMPUTED QUESTIONS ===
        if question_count == 0:
            if 'q0_questions' in self._question_cache and self._question_cache['q0_questions']:
                # Get available precomputed questions
                available_questions = [
                    (feature, question) for feature, question in self._question_cache['q0_questions']
                    if feature not in asked
                ]
                
                if available_questions:
                    # Randomly pick one from precomputed questions - INSTANT!
                    chosen_feature, chosen_question = available_questions[np.random.randint(len(available_questions))]
                    return chosen_feature, chosen_question
                
                print("Warning: All precomputed Q0 questions were asked. Using fallback.")
            else:
                print("Warning: No precomputed Q0 questions available. Using fallback.")
                
            # Fallback to old method if precomputed questions exhausted
            if not self.sorted_initial_feature_indices:
                print("Warning: No sorted_initial_feature_indices. Falling back.")
            else:
                # Get available features from top 5
                available_top_5 = [
                    idx for idx in self.sorted_initial_feature_indices[:5]
                    if self.feature_cols[idx] not in asked
                ]
                
                if available_top_5:
                    # Randomly pick one from the top 5
                    chosen_idx = available_top_5[np.random.randint(len(available_top_5))]
                    feature = self.feature_cols[chosen_idx]
                    question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
                    return feature, question
                
                print("Warning: All top 5 Q0 features were asked. Using fallback.")
        
        # === Q1-4: FAST, RANDOM FROM TOP 3 OF 15 BEST ===
        all_available_indices = [i for i, f in enumerate(self.feature_cols)
                                 if f not in asked and (i in self.allowed_feature_indices)]
        if not all_available_indices:
            return None, None
        
        if question_count < 5:
            # Use top 15 from precomputed rankings (no calculation needed)
            MAX_FEATURES_TO_CHECK = 15
            
            available_set = set(all_available_indices)
            top_available_features = [
                idx for idx in self.sorted_initial_feature_indices
                if idx in available_set
            ][:MAX_FEATURES_TO_CHECK]

            if not top_available_features:
                # Fallback if something went wrong
                top_available_features = np.random.choice(
                    all_available_indices, 
                    min(len(all_available_indices), MAX_FEATURES_TO_CHECK), 
                    replace=False
                ).tolist()
            
            available_features_to_check = top_available_features
            sampled_indices_map = {
                new_idx: old_idx for new_idx, old_idx in enumerate(available_features_to_check)
            }

        else:
            # === Q5+: ACCURATE, BEST FROM 30 FEATURES ===
            MAX_FEATURES_TO_CHECK = 30
            
            if len(all_available_indices) > MAX_FEATURES_TO_CHECK:
                sampled_indices_map = {
                    new_idx: old_idx for new_idx, old_idx in enumerate(
                        np.random.choice(all_available_indices, MAX_FEATURES_TO_CHECK, replace=False)
                    )
                }
                available_features_to_check = list(sampled_indices_map.values())
            else:
                sampled_indices_map = {idx: old_idx for idx, old_idx in enumerate(all_available_indices)}
                available_features_to_check = all_available_indices
        
        # Compute gains for Q1+ (Q0 skipped this entirely)
        gains_tensor = self.info_gain_batch(prior, available_features_to_check)
        sorted_indices_of_gains = torch.argsort(gains_tensor, descending=True)

        # Random selection from top candidates
        if question_count < 5:
            # Q1-4: Pick randomly from top 3 (good variety + accuracy)
            top_n = min(3, len(sorted_indices_of_gains))
            chosen_local_idx = sorted_indices_of_gains[np.random.randint(top_n)]
        else:
            # Q5+: Always pick the absolute best (maximum accuracy)
            chosen_local_idx = sorted_indices_of_gains[0]
        
        idx = sampled_indices_map[chosen_local_idx.item()]
        
        feature = self.feature_cols[idx]
        question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
        return feature, question
    
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
            if f not in asked and (i in self.allowed_feature_indices)
        ]
        if not available_indices:
            return None, None
        
        # Convert to float32 for calculations
        top_features = self.features[top_idx].float()
        similar_features = self.features[similar_indices].float()
        diffs = torch.abs(top_features.unsqueeze(0) - similar_features)
        avg_diffs = torch.nanmean(diffs, dim=0)
        
        # Mask to only available features
        mask = torch.full_like(avg_diffs, -1.0)
        mask[available_indices] = avg_diffs[available_indices]
        mask[torch.isnan(top_features)] = -1.0
        
        best_diff, best_idx = torch.max(mask, dim=0)
        
        if best_diff.item() > 0.3:
            feature = self.feature_cols[best_idx.item()]
            question = self.questions_map.get(feature, f"Does it have {feature.replace('_', ' ')}?")
            return feature, question

        return None, None