import pandas as pd
import numpy as np
from datetime import datetime
import os

class AkinatorEngine:
    """
    Akinator guessing engine (V6.4 - Random Start + Pure Entropy + Fixes + Patience + Stricter Elimination).
    
    FIXES & IMPROVEMENTS (V6.4):
    - Extended strict elimination in update() to moderate answers (mostly/not really) for better removal of contradictory items.
    - Added margin requirement (top prob - second prob >= 0.3) for confident guesses to prevent premature wrong guesses.
    - Tightened sigma values further for even stricter fuzzy matching.
    - Increased penalties for sparse/ambiguous features (0.8 * nan_frac + 0.6 * ambiguity) to avoid irrelevant questions like emojis early on.
    - Expanded candidate subset to 250 top + 150 random for broader high-quality question selection.
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
        """Likelihood calculation with even tighter sigma for stricter matching."""
        steps = 1001  # Kept for resolution
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                # Even tighter sigma for stricter elimination
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.025
                elif a_val == 0.75 or a_val == 0.25:
                    sigma = 0.06
                else:
                    sigma = 0.12
                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                self.likelihood_table[i, j] = max(likelihood, 1e-12)  # Lower floor for numerical stability
    
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
            initial_gains = self._compute_gains(self.uniform_prior, self.allowed_feature_indices)
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var[self.allowed_feature_indices]) + 1e-5)
            penalty = 1.0 - (0.4 * self.col_nan_frac[self.allowed_feature_indices] + 0.4 * self.col_ambiguity[self.allowed_feature_indices])
            penalty = np.clip(penalty, 0.3, 1.0)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.6) * penalty
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
            
            # Precompute ranks for vectorized scoring
            max_rank = len(self.feature_cols) + 1
            self.feature_ranks = np.full(len(self.feature_cols), max_rank, dtype=np.float32)
            self.feature_ranks[self.sorted_initial_feature_indices] = np.arange(len(self.sorted_initial_feature_indices)) + 1
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)
            self.feature_ranks = np.array([], dtype=np.float32)
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Robust Shannon entropy calculation."""
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))
    
    def _compute_gains(self, prior: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
        """Fully vectorized information gain computation for speed."""
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
            active_features = self.features[active_mask][:, feature_indices]
            active_nan_mask = self.nan_mask[active_mask][:, feature_indices]
        else:
            active_prior = prior
            active_features = self.features[:, feature_indices]
            active_nan_mask = self.nan_mask[:, feature_indices]
            active_mask = slice(None)
        if n_active == 0: return gains
        
        # Vectorized likelihood computation
        f_filled = np.nan_to_num(active_features, nan=0.5)
        f_quant = np.clip(np.rint(f_filled * 1000), 0, 1000).astype(np.int32)
        flat_quant = f_quant.flatten()
        likelihoods = self.likelihood_table[flat_quant, :].reshape(n_active, n_features, n_answers)
        
        nan_expand = np.repeat(active_nan_mask[:, :, np.newaxis], n_answers, axis=2)
        likelihoods[nan_expand] = 1.0
        
        # Vectorized posterior and entropy computation
        p_answers = np.einsum('i,ijk->jk', active_prior, likelihoods)  # (n_features, n_answers)
        
        posteriors = active_prior[:, np.newaxis, np.newaxis] * likelihoods  # (n_active, n_features, n_answers)
        posteriors /= (p_answers[np.newaxis, :, :] + 1e-10)
        posteriors = np.clip(posteriors, 1e-10, 1.0)
        
        p_logs = np.log2(posteriors)
        entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)  # (n_features, n_answers)
        
        expected_entropy = np.sum(p_answers * entropies_per_answer, axis=1)  # (n_features,)
        
        gains = current_entropy - expected_entropy
        
        return gains
    
    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """Updates probability scores based on answer with fixed certainty and extended strict elimination."""
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 1000), 0, 1000).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # Fixed certainty: high for definite values (close to 0/1), low for ambiguous (close to 0.5)
        certainty = np.clip(2.0 * np.abs(f_col - 0.5), 0.0, 1.0)
        
        # Contradiction penalty
        distance = np.abs(f_col - answer_val)
        severity = distance * certainty
        severity[nan_mask] = 0.0
        
        penalty_factor = 12.0
        penalty_multiplier = np.exp(-penalty_factor * severity)
        likelihoods *= penalty_multiplier
        
        # Extended strict elimination for definite and moderate contradictions
        definite_mismatch = np.zeros_like(nan_mask, dtype=bool)
        if answer_val == 1.0:
            definite_mismatch = (~nan_mask) & (f_col == 0.0)
        elif answer_val == 0.0:
            definite_mismatch = (~nan_mask) & (f_col == 1.0)
        elif answer_val == 0.75:
            definite_mismatch = (~nan_mask) & (f_col <= 0.25)
        elif answer_val == 0.25:
            definite_mismatch = (~nan_mask) & (f_col >= 0.75)
        likelihoods[definite_mismatch] = 0.0
        
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
        gains = self._compute_gains(prior, candidates_to_eval)
        
        # Vectorized Quality Penalties (stronger to avoid irrelevant features)
        nan_fracs = self.col_nan_frac[candidates_to_eval]
        ambiguities = self.col_ambiguity[candidates_to_eval]
        penalties = 1.0 - (0.8 * nan_fracs + 0.6 * ambiguities)
        penalties = np.clip(penalties, 0.1, 1.0)
        gains = gains * penalties
        
        # --- Selection Logic ---
        if question_count < 2 and len(gains) > 1:
            top_n = min(5, len(gains))
            top_indices_local = np.argpartition(gains, -top_n)[-top_n:]
            random_choice = np.random.choice(top_indices_local)
            best_feat_idx = candidates_to_eval[random_choice]
            print(f"[Q{question_count+1}] VARIETY: Random selection from top {top_n} gains.")
        else:
            best_local_idx = np.argmax(gains)
            best_feat_idx = candidates_to_eval[best_local_idx]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text
    
    def _select_candidate_subset(self, candidates_indices):
        """Helper to sample feature candidates efficiently with vectorized scoring."""
        candidates_indices = np.array(candidates_indices, dtype=np.int32)
        if len(candidates_indices) <= 400:
            return candidates_indices
        
        # Vectorized scoring
        ranks = self.feature_ranks[candidates_indices]
        rank_scores = np.where(ranks < self.feature_ranks.size + 1, 1.0 / ranks, 0.0)
        max_var = np.max(self.col_var[self.allowed_feature_indices]) + 1e-5
        var_scores = self.col_var[candidates_indices] / max_var
        scores = rank_scores + var_scores
        
        sorted_local_indices = np.argsort(scores)[::-1]
        top_candidates = candidates_indices[sorted_local_indices[:250]]
        
        other_candidates = candidates_indices[sorted_local_indices[250:]]
        if len(other_candidates) > 0:
            random_extras = np.random.choice(other_candidates, size=min(len(other_candidates), 150), replace=False)
            candidates_to_eval = np.unique(np.concatenate((top_candidates, random_extras))).astype(np.int32)
        else:
            candidates_to_eval = top_candidates
        return candidates_to_eval
    
    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        STRICT GUESSING LOGIC (More Patient + Margin):
        1. High Confidence (>98% with >=0.3 margin over second) after at least 15 questions.
        2. Safety Net (Q35) ONLY if we haven't made a guess yet.
        """
        q_count = game_state['question_count']
        if probs.sum() < 1e-10:
            return False, None, None
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else -1
        top_prob = probs[top_idx]
        second_prob = probs[second_idx] if second_idx != -1 else 0.0
        top_animal = self.animals[top_idx]
        
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 8:
                return False, None, None
        # --- TIER 1: ABSOLUTE CERTAINTY WITH MARGIN ---
        if top_prob >= 0.98 and top_prob - second_prob >= 0.3 and q_count >= 15:
            print(f"[Q{q_count}] CONFIDENT GUESS: {top_animal} (prob={top_prob:.4f}, margin={top_prob - second_prob:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
        # --- TIER 2: SAFETY NET (Q35) ---
        if q_count >= 35 and not game_state.get('continue_mode', False):
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
        gains = self._compute_gains(initial_prior, indices_to_calc)
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