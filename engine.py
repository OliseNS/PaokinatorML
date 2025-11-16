import pandas as pd
import numpy as np
from datetime import datetime
import os

class AkinatorEngine:
    """
    Akinator guessing engine (V6.1 - Random Start + Pure Entropy).
    
    UX IMPROVEMENT:
    - Q1 & Q2: Random selection from top candidates to vary gameplay.
    - Q3+: Pure Information Gain (Greedy) for efficiency.
    - No artificial "discriminative" phases.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        # Granular answer values for precision
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
        """Likelihood calculation with adaptive sigma."""
        steps = 1001  # Increased for higher resolution
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                # Tighter sigma for definite answers
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.06
                elif a_val == 0.75 or a_val == 0.25:
                    sigma = 0.12
                else:
                    sigma = 0.18
                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                self.likelihood_table[i, j] = max(likelihood, 0.001)
    
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
        self.allowed_feature_mask = (self.col_nan_frac < 1.0)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)
        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        self.col_var = col_var
        
        if len(self.animals) > 0:
            n = len(self.animals)
            self.uniform_prior = np.ones(n, dtype=np.float32) / n
            
            # Boost initial features based on variance/quality
            initial_gains = self._compute_gains_batched(self.uniform_prior, self.allowed_feature_indices, batch_size=256)
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var[self.allowed_feature_indices]) + 1e-5)
            penalty = 1.0 - (0.4 * self.col_nan_frac[self.allowed_feature_indices] + 0.4 * self.col_ambiguity[self.allowed_feature_indices])
            penalty = np.clip(penalty, 0.3, 1.0)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.6) * penalty
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Robust Shannon entropy calculation."""
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))
    
    def _compute_gains_batched(self, prior: np.ndarray, feature_indices: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Computes information gain in batches."""
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._calculate_entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains
        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active < len(prior) * 0.8 and n_active > 0:
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum()
        else:
            active_prior = prior
            active_mask = slice(None)
        if n_active == 0: return gains
        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            
            f_batch = self.features[active_mask][:, batch_indices]
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5)
            f_batch_quant = np.clip(np.rint(f_batch_filled * 1000), 0, 1000).astype(np.int32)  # Updated for higher resolution
            likelihoods = self.likelihood_table[f_batch_quant]
            nan_batch_mask = np.isnan(f_batch)
            nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
            likelihoods[nan_expand] = 1.0
            batch_gains = np.zeros(len(batch_indices), dtype=np.float32)
            
            for f_local_idx in range(len(batch_indices)):
                L_f = likelihoods[:, f_local_idx, :]
                p_answers = active_prior @ L_f
                posteriors = active_prior[:, np.newaxis] * L_f
                posteriors /= (p_answers + 1e-10)
                posteriors = np.clip(posteriors, 1e-10, 1.0)
                
                p_logs = np.log2(posteriors)
                entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)
                expected_entropy = p_answers @ entropies_per_answer
                batch_gains[f_local_idx] = current_entropy - expected_entropy
            gains[i:end] = batch_gains
            
        return gains
    
    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """Updates probability scores based on answer."""
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 1000), 0, 1000).astype(np.int32)  # Updated for higher resolution
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # Contradiction penalty
        distance = np.abs(f_col - answer_val)
        certainty = np.clip(1.0 - 2.0 * np.abs(f_col - 0.5), 0.0, 1.0)
        severity = distance * certainty
        severity[nan_mask] = 0.0
        
        penalty_factor = 12.0
        penalty_multiplier = np.exp(-penalty_factor * severity)
        likelihoods *= penalty_multiplier
        scores = np.log(likelihoods + 1e-10)
        return current_scores + scores
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Selects the next question.
        - Q0: Random from top 20% of *initial* best features.
        - Q1: Random from top 5 of *current* best features (using new prior).
        - Q2+: Greedy max entropy.
        """
        asked_set = set(asked_features)
        candidates_indices = [idx for idx in self.allowed_feature_indices
                              if self.feature_cols[idx] not in asked_set]
        if not candidates_indices:
            print("[Question] No more features to ask.")
            return None, None
        # --- Q0: Fast Random Start ---
        # Uses pre-computed 'sorted_initial_feature_indices' for speed and variety.
        if question_count == 0:
            top_20_pct = max(1, len(self.sorted_initial_feature_indices) // 5)
            available_top = [idx for idx in self.sorted_initial_feature_indices[:top_20_pct]
                           if idx in candidates_indices]
            if available_top:
                best_feat_idx = np.random.choice(available_top)
                feature_name = self.feature_cols[best_feat_idx]
                question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                print(f"[Q1] START: Random selection from top 20% features.")
                return feature_name, question_text
        # --- Calculate Real Information Gain ---
        candidates_to_eval = self._select_candidate_subset(candidates_indices)
        if len(candidates_to_eval) == 0: return None, None
        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=128)
        
        # Quality Penalties
        penalty_gains = np.ones_like(gains)
        for i, feat_idx in enumerate(candidates_to_eval):
            nan_frac = self.col_nan_frac[feat_idx]
            ambiguity = self.col_ambiguity[feat_idx]
            penalty = 1.0 - (0.6 * nan_frac + 0.5 * ambiguity)
            penalty = np.clip(penalty, 0.15, 1.0)
            penalty_gains[i] = gains[i] * penalty
        gains = penalty_gains
        
        # --- Selection Logic ---
        
        # Q1 (Second Question): Random from top 5 to prevent robotic pathing
        # We check 'question_count < 2' to cover cases where Q0 might have been skipped or manually set
        if question_count < 2 and len(gains) > 1:
             # Get indices of the top 5 gains
             top_n = min(5, len(gains))
             # argpartition puts top_n elements at the end
             top_indices_local = np.argpartition(gains, -top_n)[-top_n:]
             
             # Pick one random index from these top N
             random_choice = np.random.choice(top_indices_local)
             best_feat_idx = candidates_to_eval[random_choice]
             print(f"[Q{question_count+1}] VARIETY: Random selection from top {top_n} gains.")
             
        else:
            # Q2+: Pure Greedy (Max Gain)
            best_local_idx = np.argmax(gains)
            best_feat_idx = candidates_to_eval[best_local_idx]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text
    
    def _select_candidate_subset(self, candidates_indices):
        """Helper to sample feature candidates efficiently."""
        if len(candidates_indices) <= 120:
            return np.array(candidates_indices, dtype=np.int32)
        
        scored_candidates = []
        for idx in candidates_indices:
            initial_rank = np.where(self.sorted_initial_feature_indices == idx)[0]
            rank_score = 1.0 / (initial_rank[0] + 1) if len(initial_rank) > 0 else 0
            var_score = self.col_var[idx] / (np.max(self.col_var[self.allowed_feature_indices]) + 1e-5)
            scored_candidates.append((idx, rank_score + var_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [idx for idx, _ in scored_candidates[:90]]
        
        other_candidates = [idx for idx in candidates_indices if idx not in top_candidates]
        if other_candidates:
            random_extras = np.random.choice(other_candidates, size=min(len(other_candidates), 30), replace=False)
            candidates_to_eval = np.unique(np.concatenate((top_candidates, random_extras))).astype(np.int32)
        else:
            candidates_to_eval = np.array(top_candidates, dtype=np.int32)
        return candidates_to_eval
    
    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        STRICT GUESSING LOGIC:
        1. High Confidence (>98%) after at least 10 questions.
        2. Safety Net (Q35) ONLY if we haven't made a guess yet.
        """
        q_count = game_state['question_count']
        if probs.sum() < 1e-10:
            return False, None, None
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 8:
                return False, None, None
        # --- TIER 1: ABSOLUTE CERTAINTY ---
        if top_prob >= 0.995 and q_count >= 10:
            print(f"[Q{q_count}] CONFIDENT GUESS: {top_animal} (prob={top_prob:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
        # --- TIER 2: SAFETY NET (Q35) ---
        if q_count >= 20 and not game_state.get('continue_mode', False):
            print(f"[Q{q_count}] FORCED GUESS (Limit Reached): {top_animal} (prob={top_prob:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
        return False, None, None
    
    def get_features_for_data_collection(self, item_name, num_features=5):
        """Gets a list of features for data collection."""
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
                selected_indices = useful_nan_indices
        else:
            np.random.shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
            
        return self._format_features(selected_indices[:num_features])
    
    def _get_random_allowed_features(self, num_features):
        if len(self.allowed_feature_indices) == 0: return []
        num_to_select = min(num_features, len(self.allowed_feature_indices))
        selected = np.random.choice(self.allowed_feature_indices, size=num_to_select, replace=False)
        return self._format_features(selected)
    
    def _format_features(self, indices):
        results = []
        for idx in indices:
            py_idx = int(idx)
            if py_idx >= len(self.feature_cols) or py_idx >= len(self.col_nan_frac): continue
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
            return pd.DataFrame()