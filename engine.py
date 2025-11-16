import pandas as pd
import numpy as np
from datetime import datetime
import os

class AkinatorEngine:
    """
    Akinator guessing engine (V6.0 - "Blood Seeking" Mode).

    STRATEGY:
    1. TIERED OPENING: Q1 (Top 50) -> Q2 (Top 10) -> Q3 (Top 5) -> Q4+ (Greedy).
    2. HARD VETO: Massive penalties for contradictions (prevents "Blackboard" scenarios).
    3. FAST CONVERGENCE: Tighter sigma curves to eliminate candidates quickly.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        self.MAX_QUESTIONS = None 
        
        # Granular answer values
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'mostly': 0.75, 'usually': 0.75, 'probably': 0.75,
            'sort of': 0.5, 'sometimes': 0.5, 
            'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0,
        }
        
        self._precompute_likelihood_tables()
        self._build_arrays()
        print(f"✓ Engine initialized: {len(self.animals)} items, {len(self.feature_cols)} features.")

    def _precompute_likelihood_tables(self):
        """
        Likelihood calculation with TIGHTER sigma to force faster convergence.
        """
        steps = 101
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                
                # Tighter sigmas for faster convergence
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.05  # Very sharp for Yes/No
                elif a_val == 0.75 or a_val == 0.25:
                    sigma = 0.10  # Sharp for Probably/Rarely
                else:
                    sigma = 0.15  # Looser for Don't Know/Sometimes

                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                
                # Floor at a small epsilon, but not too small to allow penalties to work
                self.likelihood_table[i, j] = max(likelihood, 0.0001)

    def _build_arrays(self):
        """Converts dataframe to optimized numpy arrays."""
        self.animals = self.df['animal_name'].values
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.nan_mask = np.isnan(self.features)
        
        self.col_nan_frac = np.mean(self.nan_mask, axis=0)
        
        nan_masked_features = np.ma.masked_invalid(self.features)
        col_var = nan_masked_features.var(axis=0).data
        col_mean = nan_masked_features.mean(axis=0).data
        self.col_ambiguity = 1.0 - 2.0 * np.abs(col_mean - 0.5)

        # Filter features that are 100% empty
        self.allowed_feature_mask = (self.col_nan_frac < 1.0)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

        # Identify sparse features for data collection phases
        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        
        self.col_var = col_var
        
        # Precompute initial importance for Q1 heuristic
        if len(self.animals) > 0:
            n = len(self.animals)
            self.uniform_prior = np.ones(n, dtype=np.float32) / n
            
            # Simple variance-based heuristic for fast startup
            initial_gains = col_var[self.allowed_feature_indices]
            
            # Penalize empty columns
            penalty = 1.0 - (0.5 * self.col_nan_frac[self.allowed_feature_indices])
            boosted_gains = initial_gains * penalty
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))

    def _compute_gains_batched(self, prior: np.ndarray, feature_indices: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Computes information gain."""
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._calculate_entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains

        # Optimization: Only look at animals with > 0 probability
        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active < len(prior) * 0.8 and n_active > 0:
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum()
            active_features_subset = self.features[active_mask]
        else:
            active_prior = prior
            active_features_subset = self.features
            active_mask = slice(None)

        if n_active == 0: return gains

        # Iterate in batches
        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            
            # Get features for active animals
            f_batch = active_features_subset[:, batch_indices]
            
            # Quantize features for lookup
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5)
            f_batch_quant = np.clip(np.rint(f_batch_filled * 100), 0, 100).astype(np.int16)

            # Look up likelihoods
            likelihoods = self.likelihood_table[f_batch_quant]

            # Handle NaNs (neutral likelihood)
            nan_batch_mask = np.isnan(f_batch)
            if np.any(nan_batch_mask):
                nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
                likelihoods[nan_expand] = 1.0

            batch_gains = np.zeros(len(batch_indices), dtype=np.float32)
            
            # Vectorized Entropy Calculation
            for f_local_idx in range(len(batch_indices)):
                L_f = likelihoods[:, f_local_idx, :]
                p_answers = active_prior @ L_f # Probability of user giving each answer
                
                posteriors = active_prior[:, np.newaxis] * L_f
                posteriors /= (p_answers + 1e-10)
                
                # Fast entropy
                p_logs = np.log2(np.clip(posteriors, 1e-10, 1.0))
                entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)
                expected_entropy = p_answers @ entropies_per_answer
                
                batch_gains[f_local_idx] = current_entropy - expected_entropy

            gains[i:end] = batch_gains
            
        return gains

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """
        Updates scores with HARD VETO logic for contradictions.
        This fixes the 'Blackboard' vs 'Electronics' issue.
        """
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        # 1. Standard Gaussian Update
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 100), 0, 100).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # 2. THE VETO: Hard Penalty for contradictions
        # If user says "Yes" (1.0) and animal is "No" (0.0) -> Diff is 1.0
        # If user says "No" (0.0) and animal is "Yes" (1.0) -> Diff is 1.0
        diff = np.abs(f_col - answer_val)
        
        # We apply a massive penalty if the difference is huge (> 0.8).
        # This effectively kills the candidate, unless it was the only one left.
        veto_mask = (diff > 0.8) & (~nan_mask)
        
        # Apply standard update
        scores = np.log(likelihoods + 1e-10)
        
        # Apply VETO penalty (approx -25.0 in log space is massive)
        scores[veto_mask] -= 25.0 

        new_cumulative_scores = current_scores + scores
        return new_cumulative_scores
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Tiered selection strategy:
        Q1: Random Top 50
        Q2: Random Top 10
        Q3: Random Top 5
        Q4+: Pure Greedy (Top 1)
        """
        asked_set = set(asked_features)
        
        # 1. Determine Candidate Pool Size based on Game Stage
        if question_count == 0:
            candidates_to_eval = self.sorted_initial_feature_indices[:200] # Eval top 200 to find top 50
        elif question_count < 4:
            candidates_indices = [idx for idx in self.allowed_feature_indices 
                                  if self.feature_cols[idx] not in asked_set]
            # Heuristic trim to speed up calculation, but keep enough for accurate top-k
            candidates_to_eval = self._select_candidate_subset(candidates_indices, limit=150) 
        else:
            # Greedy Mode: Optimize heavily
            candidates_indices = [idx for idx in self.allowed_feature_indices 
                                  if self.feature_cols[idx] not in asked_set]
            candidates_to_eval = self._select_candidate_subset(candidates_indices, limit=100)

        if len(candidates_to_eval) == 0:
            return None, None

        # 2. Compute Information Gain (The "Blood Seeking" part)
        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=128)
        
        # Apply quality penalties (prefer questions with less NaNs)
        penalty_gains = np.ones_like(gains)
        for i, feat_idx in enumerate(candidates_to_eval):
            nan_frac = self.col_nan_frac[feat_idx]
            ambiguity = self.col_ambiguity[feat_idx]
            # Less penalty than before to prioritize pure splitting power
            penalty = 1.0 - (0.3 * nan_frac + 0.2 * ambiguity) 
            penalty_gains[i] = gains[i] * penalty

        # 3. Selection Logic based on Turn
        
        # Sort candidates by gain (Desc)
        sorted_local_indices = np.argsort(penalty_gains)[::-1]
        sorted_real_indices = candidates_to_eval[sorted_local_indices]
        
        # Ensure we have enough candidates to sample from
        n_available = len(sorted_real_indices)
        
        if question_count == 0:
            # Random from Top 50
            top_k = min(50, n_available)
            choice_idx = np.random.choice(top_k)
            best_feat_idx = sorted_real_indices[choice_idx]
            print(f"[Q1] Opening: Selected rank {choice_idx+1}/{top_k} (Gain: {penalty_gains[sorted_local_indices[choice_idx]]:.4f})")
            
        elif question_count == 1:
            # Random from Top 10
            top_k = min(10, n_available)
            choice_idx = np.random.choice(top_k)
            best_feat_idx = sorted_real_indices[choice_idx]
            print(f"[Q2] Narrowing: Selected rank {choice_idx+1}/{top_k}")
            
        elif question_count == 2:
            # Random from Top 5
            top_k = min(5, n_available)
            choice_idx = np.random.choice(top_k)
            best_feat_idx = sorted_real_indices[choice_idx]
            print(f"[Q3] Precision: Selected rank {choice_idx+1}/{top_k}")
            
        else:
            # Q4+: PURE GREEDY BLOOD SEEKING
            best_feat_idx = sorted_real_indices[0]
            # print(f"[Q{question_count+1}] GREEDY: Best Gain {penalty_gains[sorted_local_indices[0]]:.4f}")

        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text

    def _select_candidate_subset(self, candidates_indices, limit=100):
        """Heuristic to select which features to calculate gain for."""
        if len(candidates_indices) <= limit:
            return np.array(candidates_indices, dtype=np.int32)
        
        # Score based on variance and initial rank (pre-computed)
        scored_candidates = []
        for idx in candidates_indices:
            # Using cached variance
            score = self.col_var[idx]
            scored_candidates.append((idx, score))
        
        # Sort by raw variance as a heuristic proxy for information gain
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [idx for idx, _ in scored_candidates[:limit]]
        
        return np.array(top_candidates, dtype=np.int32)

    def get_discriminative_question(self, top_animal_idx: int, prior: np.ndarray, asked_features: list) -> tuple[str, str]:
        """Finds questions that best separate top candidate from rivals."""
        top_prob = prior[top_animal_idx]
        
        rival_threshold = top_prob * 0.20
        rival_mask = (prior > rival_threshold) & (np.arange(len(prior)) != top_animal_idx)
        rival_indices = np.where(rival_mask)[0]
        
        if len(rival_indices) == 0:
            return None, None
            
        asked_set = set(asked_features)
        candidate_indices = [idx for idx in self.allowed_feature_indices 
                                 if self.feature_cols[idx] not in asked_set]
                                
        if not candidate_indices:
            return None, None

        # Calculate average difference between Top Dog and Rivals
        top_feats = self.features[top_animal_idx, candidate_indices]
        rival_feats = self.features[rival_indices][:, candidate_indices]
        rival_weights = prior[rival_indices]
        rival_weights /= rival_weights.sum()
        avg_rival_feats = np.average(rival_feats, axis=0, weights=rival_weights)
        
        diffs = np.abs(np.nan_to_num(top_feats, nan=0.5) - np.nan_to_num(avg_rival_feats, nan=0.5))
        
        # Prioritize features where the Top Dog has a STRONG opinion (0 or 1)
        # We don't want to ask something where the top dog is "Maybe"
        certainty = 1.0 + 2.0 * np.abs(np.nan_to_num(top_feats, nan=0.5) - 0.5)
        weighted_diffs = diffs * certainty
        
        best_local_idx = np.argmax(weighted_diffs)
        
        if weighted_diffs[best_local_idx] > 0.25:
            best_feat_idx = candidate_indices[best_local_idx]
            feature_name = self.feature_cols[best_feat_idx]
            return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                      
        return None, None

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
            """
            Decides if we should stop and guess.
            """
            q_count = game_state['question_count']

            if probs.sum() < 1e-10:
                return False, None, None

            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            top_animal = self.animals[top_idx]

            probs_copy = probs.copy()
            probs_copy[top_idx] = 0.0
            second_prob = np.max(probs_copy)
            
            # Prevent guessing too soon after a user rejection in continue mode
            if game_state.get('continue_mode', False):
                if game_state.get('questions_since_last_guess', 0) < 4:
                    return False, None, None

            # --- 1. Forced Safety Net (Q25) ---
            if q_count >= 25 and not game_state.get('continue_mode', False):
                return True, top_animal, 'final'

            # --- 2. Instant Victory (Early Game) ---
            # If prob > 97% and we have a massive lead
            if top_prob > 0.97 and top_prob > (second_prob * 100) and q_count >= 5:
                return True, top_animal, 'final'

            # --- 3. Standard Win (Late Game) ---
            if top_prob >= 0.94 and top_prob > (second_prob * 10) and q_count >= 10:
                return True, top_animal, 'final'

            return False, None, None

    def get_features_for_data_collection(self, item_name, num_features=5):
        """Gets a list of features for data collection for a specific item."""
        try:
            matches = np.where(self.animals == item_name)[0]
            if len(matches) == 0:
                matches = np.where(np.char.lower(self.animals.astype(str)) == item_name.lower())[0]
            
            if len(matches) == 0:
                return self._get_random_allowed_features(num_features)
            item_idx = matches[0]
            item_feats = self.features[item_idx]
        except Exception:
            return self._get_random_allowed_features(num_features)

        nan_indices = np.where(np.isnan(item_feats))[0]
        useful_nan_indices = np.intersect1d(nan_indices, self.allowed_feature_indices).copy()
        
        if len(useful_nan_indices) < num_features:
            needed = num_features - len(useful_nan_indices)
            extras = np.setdiff1d(self.sparse_indices, useful_nan_indices).copy()
            
            if len(extras) > 0:
                np.random.shuffle(extras)
                selected_indices = np.concatenate((useful_nan_indices, extras[:needed]))
            else:
                remaining_allowed = np.setdiff1d(self.allowed_feature_indices, useful_nan_indices).copy()
                if len(remaining_allowed) > 0:
                    np.random.shuffle(remaining_allowed)
                    needed_more = num_features - len(useful_nan_indices)
                    selected_indices = np.concatenate((useful_nan_indices, remaining_allowed[:needed_more]))
                else:
                    selected_indices = useful_nan_indices
        else:
            np.random.shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
        
        if len(selected_indices) < num_features and len(self.allowed_feature_indices) >= num_features:
            remaining = np.setdiff1d(self.allowed_feature_indices, selected_indices).copy()
            np.random.shuffle(remaining)
            needed = num_features - len(selected_indices)
            selected_indices = np.concatenate((selected_indices, remaining[:needed]))
            
        return self._format_features(selected_indices[:num_features])
    
    def _get_random_allowed_features(self, num_features):
        if len(self.allowed_feature_indices) == 0:
            return []
        num_to_select = min(num_features, len(self.allowed_feature_indices))
        selected = np.random.choice(self.allowed_feature_indices, size=num_to_select, replace=False)
        return self._format_features(selected)
    
    def _format_features(self, indices):
        results = []
        for idx in indices:
            py_idx = int(idx)
            if py_idx >= len(self.feature_cols) or py_idx >= len(self.col_nan_frac):
                continue
            fname = str(self.feature_cols[py_idx])
            results.append({
                "feature_name": fname,
                "question": str(self.questions_map.get(fname, f"Is it {fname}?")),
                "nan_percentage": float(self.col_nan_frac[py_idx])
            })
        return results
    
    def get_all_feature_gains(self, initial_prior: np.ndarray = None) -> list[dict]:
        if initial_prior is None:
            if hasattr(self, 'uniform_prior'):
                initial_prior = self.uniform_prior
            else:
                if len(self.animals) == 0: return []
                initial_prior = np.ones(len(self.animals), dtype=np.float32) / len(self.animals)
        
        indices_to_calc = self.allowed_feature_indices
        if len(indices_to_calc) == 0: return []
        
        gains = self._compute_gains_batched(initial_prior, indices_to_calc, batch_size=128)
        
        results = []
        for i, idx in enumerate(indices_to_calc):
            feature_name = self.feature_cols[idx]
            results.append({
                'feature_name': feature_name,
                'question': self.questions_map.get(feature_name, "N/A"),
                'initial_gain': float(gains[i]),
                'variance': float(self.col_var[idx]) if not np.isnan(self.col_var[idx]) else 0.0,
                'nan_percentage': float(self.col_nan_frac[idx])
            })
        results.sort(key=lambda x: x['initial_gain'], reverse=True)
        return results
 
    def to_delete(self, similarity_threshold=0.85, min_variance=0.01, output_file='questions_to_delete.csv'):
            """Identifies and logs redundant, similar, or low-quality questions."""
            start_time = datetime.now()
            deletion_candidates = []
            
            # 1. Excessive missing data
            high_nan_mask = self.col_nan_frac >= 0.40
            high_nan_indices = np.where(high_nan_mask)[0]
            
            for idx in high_nan_indices:
                fname = self.feature_cols[idx]
                deletion_candidates.append({
                    'feature_name': fname,
                    'question_text': self.questions_map.get(fname, f"Does it have {fname}?"),
                    'reason': 'excessive_missing_data',
                    'nan_percentage': float(self.col_nan_frac[idx]),
                    'variance': float(self.col_var[idx]) if not np.isnan(self.col_var[idx]) else 0.0,
                    'correlation_with': None
                })
            
            # 2. Low variance
            valid_var_mask = ~np.isnan(self.col_var) & (self.col_var < min_variance)
            low_var_indices = np.where(valid_var_mask)[0]
            
            for idx in low_var_indices:
                if idx in high_nan_indices: continue
                fname = self.feature_cols[idx]
                deletion_candidates.append({
                    'feature_name': fname,
                    'question_text': self.questions_map.get(fname, f"Does it have {fname}?"),
                    'reason': 'low_variance',
                    'nan_percentage': float(self.col_nan_frac[idx]),
                    'variance': float(self.col_var[idx]),
                    'correlation_with': None
                })
            
            if deletion_candidates:
                df_delete = pd.DataFrame(deletion_candidates)
                df_delete['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
                df_delete = df_delete.sort_values(['reason', 'nan_percentage'], ascending=[True, False])
                df_delete.to_csv(output_file, index=False)
                return df_delete
            return pd.DataFrame()