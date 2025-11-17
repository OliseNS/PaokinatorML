# [file: engine.py]

import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Akinator guessing engine (V9.0 - Context-Aware Engine).
    
    IMPROVEMENTS (V9.0):
    - Solves "False Confidence": Stores an 'is_imputed_mask' to 
      track original NaNs. Imputed values are now used for
      gain calculations but are NOT used to penalize or 
      soft-eliminate items during the 'update' step.
    - Solves "Stupid Questions": Computes a feature-correlation
      matrix. 'select_question' now applies an exponential penalty
      to any candidate question that is highly correlated with
      a question that has already been asked, promoting
      more diverse and intelligent questioning.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df # This will be replaced
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        # --- NEW: Store the original sparse dataframe ---
        sparse_df = df.copy()
        
        # Granular answer values for precision
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'mostly': 0.75, 'usually': 0.75, 'probably': 0.75,
            'sort of': 0.5, 'sometimes': 0.5,
            'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0,
        }
        
        # --- Matrix Completion Step ---
        print("🧠 Engine received sparse data, starting matrix completion...")
        try:
            self.df = self._complete_matrix(df, feature_cols)
            print("✅ Matrix completion successful. Engine now using dense data.")
        except Exception as e:
            print(f"❌ WARNING: Matrix completion failed ({e}). Falling back to sparse data.")
            self.df = df # Fallback
        # --- END ---
        
        # --- NEW: Precompute feature correlation matrix (for Problem 2) ---
        # This is done on the DENSE (imputed) matrix
        print("Calculating feature correlation matrix...")
        try:
            # This is the key to solving the "Africa" problem
            self.feature_correlation_matrix = self.df[self.feature_cols].corr().values.astype(np.float32)
            # Fill any NaNs in the correlation matrix itself (e.g., for 0-variance cols)
            self.feature_correlation_matrix = np.nan_to_num(self.feature_correlation_matrix, nan=0.0)
            print("✅ Correlation matrix calculated.")
        except Exception as e:
            print(f"❌ WARNING: Correlation matrix failed ({e}). Redundant questions may occur.")
            # Create a fallback identity matrix (no correlations)
            n_feat = len(self.feature_cols)
            self.feature_correlation_matrix = np.eye(n_feat, dtype=np.float32)
        # --- END NEW ---
        
        
        self._precompute_likelihood_tables()
        
        # --- MODIFIED: Pass sparse_df to _build_arrays ---
        self._build_arrays(sparse_df) 
        print(f"✓ Engine initialized: {len(self.animals)} items, {len(self.feature_cols)} features.")
    
    
    def _complete_matrix(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """(No changes to this method)"""
        if df.empty or not feature_cols:
            print("   Matrix completion skipped: No data or features.")
            return df
            
        print(f"   Imputing {df[feature_cols].isna().sum().sum()} missing values...")
        
        animal_names = df['animal_name'].values
        features_matrix = df[feature_cols].values.astype(np.float32)

        imputer = KNNImputer(n_neighbors=5, weights='distance', missing_values=np.nan)
        
        completed_matrix = imputer.fit_transform(features_matrix)
        completed_matrix = np.clip(completed_matrix, 0.0, 1.0)
        
        print("   Imputation finished.")

        completed_df = pd.DataFrame(completed_matrix, columns=feature_cols)
        completed_df['animal_name'] = animal_names
        
        completed_df = completed_df[['animal_name'] + feature_cols]
        
        return completed_df
    

    def _precompute_likelihood_tables(self):
        """(No changes to this method)"""
        steps = 1001
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        
        f_grid_col = self.feature_grid[:, np.newaxis]
        a_vals_row = self.answer_values[np.newaxis, :]
        diffs = np.abs(f_grid_col - a_vals_row)
        sigmas = np.where(self.answer_values == 0.5, 0.20, 0.15)
        sigmas_row = sigmas[np.newaxis, :]
        likelihoods = np.exp(-0.5 * (diffs / sigmas_row) ** 2)
        self.likelihood_table = np.maximum(likelihoods, 1e-12).astype(np.float32)
    
    # --- MODIFIED: Accepts sparse_df ---
    def _build_arrays(self, sparse_df: pd.DataFrame):
        """
        Converts dataframe to optimized numpy arrays.
        NOW tracks original NaNs to prevent "False Confidence".
        """
        self.animals = self.df['animal_name'].values
        
        # This is the DENSE, imputed feature matrix
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        # --- NEW: Create the 'is_imputed_mask' (for Problem 1) ---
        # This is the key to solving "False Confidence"
        original_features = sparse_df[self.feature_cols].values.astype(np.float32)
        self.is_imputed_mask = np.isnan(original_features)
        # --- END NEW ---
        
        # This mask is based on the *original* data, not the imputed data
        self.col_nan_frac = np.mean(self.is_imputed_mask, axis=0) 
        
        # We can still use the dense matrix for variance/mean calculations
        nan_masked_features = np.ma.masked_invalid(self.features) # This is fine
        col_var = nan_masked_features.var(axis=0).data
        col_mean = nan_masked_features.mean(axis=0).data
        self.col_ambiguity = 1.0 - 2.0 * np.abs(col_mean - 0.5)
        
        # This logic is now based on the ORIGINAL nan fraction
        self.allowed_feature_mask = (self.col_nan_frac < 1.0) 
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)
        
        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask 
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        self.col_var = col_var
        
        if len(self.animals) > 0:
            n = len(self.animals)
            self.uniform_prior = np.ones(n, dtype=np.float32) / n
            
            initial_gains = self._compute_gains(self.uniform_prior, self.allowed_feature_indices)
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var[self.allowed_feature_indices]) + 1e-5)
            
            # This penalty now correctly uses the ORIGINAL nan_frac
            penalty = 1.0 - (0.4 * self.col_nan_frac[self.allowed_feature_indices] + 0.4 * self.col_ambiguity[self.allowed_feature_indices])
            penalty = np.clip(penalty, 0.3, 1.0)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.6) * penalty
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
            
            max_rank = len(self.feature_cols) + 1
            self.feature_ranks = np.full(len(self.feature_cols), max_rank, dtype=np.float32)
            self.feature_ranks[self.sorted_initial_feature_indices] = np.arange(len(self.sorted_initial_feature_indices)) + 1
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)
            self.feature_ranks = np.array([], dtype=np.float32)
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """(No changes)"""
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))
    
    def _compute_gains(self, prior: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
        """
        Fully vectorized information gain computation.
        MODIFIED: Uses 'is_imputed_mask' to prevent false confidence.
        """
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
            # --- MODIFIED: Use the is_imputed_mask ---
            active_nan_mask = self.is_imputed_mask[active_mask][:, feature_indices]
        else:
            active_prior = prior
            active_features = self.features[:, feature_indices]
            # --- MODIFIED: Use the is_imputed_mask ---
            active_nan_mask = self.is_imputed_mask[:, feature_indices]
            active_mask = slice(None)
        
        if active_prior.size == 0:
            return gains
        
        # f_filled is still needed because active_features is the DENSE matrix
        f_filled = np.nan_to_num(active_features, nan=0.5) 
        f_quant = np.clip(np.rint(f_filled * 1000), 0, 1000).astype(np.int32)
        flat_quant = f_quant.flatten()
        
        n_actual_rows = active_features.shape[0]
        if n_actual_rows == 0 or n_features == 0:
            return gains
            
        likelihoods = self.likelihood_table[flat_quant, :].reshape(n_actual_rows, n_features, n_answers)
        
        # --- THIS IS THE KEY (Problem 1) ---
        # We use the ORIGINAL nan_mask (active_nan_mask)
        # If the data was originally NaN, we treat it as "neutral"
        # (likelihood=1.0) for all answers. This prevents the
        # *imputed* value from influencing the gain calculation.
        nan_expand = np.repeat(active_nan_mask[:, :, np.newaxis], n_answers, axis=2)
        likelihoods[nan_expand] = 1.0
        # --- END KEY FIX ---
        
        p_answers = np.einsum('i,ijk->jk', active_prior, likelihoods)
        
        posteriors = active_prior[:, np.newaxis, np.newaxis] * likelihoods
        posteriors /= (p_answers[np.newaxis, :, :] + 1e-10)
        posteriors = np.clip(posteriors, 1e-10, 1.0)
        
        p_logs = np.log2(posteriors)
        entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)
        
        expected_entropy = np.sum(p_answers * entropies_per_answer, axis=1)
        
        gains = current_entropy - expected_entropy
        
        return gains
    
    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """
        Updates probability scores based on answer.
        MODIFIED: Uses 'is_imputed_mask' to prevent false penalties.
        """
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        f_col = self.features[:, feature_idx] # DENSE features
        
        # --- MODIFIED: Use the is_imputed_mask (Problem 1) ---
        # This is the mask of *original* NaNs
        imputed_mask = self.is_imputed_mask[:, feature_idx] 
        
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 1000), 0, 1000).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        certainty = np.clip(2.0 * np.abs(f_col - 0.5), 0.0, 1.0)
        distance = np.abs(f_col - answer_val)
        severity = distance * certainty
        
        # --- THIS IS THE KEY (Problem 1) ---
        # If the value was imputed, DO NOT apply a contradiction penalty.
        severity[imputed_mask] = 0.0 
        # --- END KEY FIX ---
        
        penalty_factor = 8.0
        
        penalty_multiplier = np.exp(-penalty_factor * severity)
        likelihoods *= penalty_multiplier
        
        # Extended strict elimination
        definite_mismatch = np.zeros_like(imputed_mask, dtype=bool)
        if answer_val == 1.0:
            definite_mismatch = (f_col == 0.0)
        elif answer_val == 0.0:
            definite_mismatch = (f_col == 1.0)
        elif answer_val == 0.75:
            definite_mismatch = (f_col <= 0.25)
        elif answer_val == 0.25:
            definite_mismatch = (f_col >= 0.75)
        
        # --- THIS IS THE OTHER KEY (Problem 1) ---
        # If the value was imputed, DO NOT apply soft elimination.
        definite_mismatch[imputed_mask] = False
        # --- END KEY FIX ---
        
        likelihoods[definite_mismatch] = 1e-9
        
        scores = np.log(likelihoods + 1e-10)
        return current_scores + scores
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Selects the next question.
        MODIFIED: Applies a correlation penalty to avoid "stupid questions".
        """
        # Vectorized finding of candidate indices
        asked_mask = np.isin(self.feature_cols, asked_features)
        
        # This mask is based on ORIGINAL nan_frac, which is correct
        candidates_mask = self.allowed_feature_mask & ~asked_mask
        candidates_indices = np.where(candidates_mask)[0].astype(np.int32)

        if not candidates_indices.any():
            print("[Question] No more features to ask.")
            return None, None
            
        # Fast Random Start (No change)
        if question_count == 0:
            # ... (omitted for brevity, no changes) ...
            top_initial = self.sorted_initial_feature_indices
            top_is_available_mask = np.isin(top_initial, candidates_indices)
            available_top = top_initial[top_is_available_mask]
            top_20_pct_len = max(1, len(self.sorted_initial_feature_indices) // 5)
            available_top_20 = available_top[available_top < top_20_pct_len]
            
            if available_top_20.any():
                best_feat_idx = np.random.choice(available_top_20)
                feature_name = self.feature_cols[best_feat_idx]
                question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                print(f"[Q1] START: Random selection from top 20% features.")
                return feature_name, question_text
        
        # Calculate Real Information Gain
        candidates_to_eval = self._select_candidate_subset(candidates_indices)
        if len(candidates_to_eval) == 0: return None, None
        gains = self._compute_gains(prior, candidates_to_eval)
        
        # Vectorized Quality Penalties
        # This uses ORIGINAL nan_fracs, which is correct.
        nan_fracs = self.col_nan_frac[candidates_to_eval]
        ambiguities = self.col_ambiguity[candidates_to_eval]
        
        penalties = 1.0 - (0.8 * nan_fracs + 0.6 * ambiguities)
        penalties = np.clip(penalties, 0.1, 1.0)
        gains = gains * penalties
        
        # --- NEW: CORRELATION PENALTY (Problem 2) ---
        # This is the fix for the "Africa" problem.
        asked_indices = np.where(asked_mask)[0]
        if len(asked_indices) > 0:
            # Get correlations between our candidates and ALL asked features
            # Shape: (n_candidates, n_asked)
            corr_slice = self.feature_correlation_matrix[candidates_to_eval, :][:, asked_indices]
            
            # Find the *single highest* correlation for each candidate
            # Shape: (n_candidates,)
            max_correlations = np.abs(corr_slice).max(axis=1)
            
            # Apply an exponential penalty.
            # If max_corr = 0.1 (weak), penalty = exp(-0.025) ~= 0.975 (small)
            # If max_corr = 0.5 (medium), penalty = exp(-0.625) ~= 0.535 (medium)
            # If max_corr = 0.9 (strong), penalty = exp(-2.025) ~= 0.132 (HUGE)
            # If max_corr = 1.0 (redundant), penalty = exp(-2.5) ~= 0.082 (MASSIVE)
            correlation_penalty = np.exp(-2.5 * max_correlations**2)
            
            # Apply the penalty to the gains
            gains = gains * correlation_penalty
            print(f"[Context] Applied correlation penalty. Avg penalty: {(1.0 - correlation_penalty.mean()):.2f}")
        # --- END NEW ---
        
        
        # Selection Logic (No change)
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
        """(No changes to this method)"""
        # ... (omitted for brevity, no changes) ...
        candidates_indices = np.array(candidates_indices, dtype=np.int32)
        if len(candidates_indices) <= 400:
            return candidates_indices
        
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
        """(No changes to this method)"""
        # ... (omitted for brevity, no changes) ...
        q_count = game_state['question_count']
        if probs.sum() < 1e-10:
            return False, None, None
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else -1
        top_prob = probs[top_idx]
        second_prob = probs[second_idx] if second_idx != -1 else 0.0
        top_animal = self.animals[top_idx]
        margin = top_prob - second_prob
        
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 8:
                return False, None, None
        
        if top_prob >= 0.998 and margin >= 0.90:
            print(f"[Q{q_count}] CONFIDENT GUESS: {top_animal} (prob={top_prob:.4f}, margin={margin:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
        
        if q_count >= 25 and not game_state.get('continue_mode', False):
            print(f"[Q{q_count}] FORCED GUESS (Limit Reached): {top_animal} (prob={top_prob:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
            
        return False, None, None
    
    def get_features_for_data_collection(self, item_name, num_features=5):
        """
        MODIFIED: Uses 'is_imputed_mask' to find original NaNs.
        """
        try:
            matches = np.where(self.animals == item_name)[0]
            if len(matches) == 0:
                matches = np.where(np.char.lower(self.animals.astype(str)) == item_name.lower())[0]
            if len(matches) == 0:
                return self._get_random_allowed_features(num_features)
            item_idx = matches[0]
            
            # --- MODIFIED: Use the is_imputed_mask ---
            # Get the boolean mask for the item
            original_item_feats_mask = self.is_imputed_mask[item_idx]
            # Find the indices that WERE NaNs
            nan_indices = np.where(original_item_feats_mask)[0]
            # --- END MODIFIED ---
            
        except Exception:
            return self._get_random_allowed_features(num_features)
        
        useful_nan_indices = np.intersect1d(nan_indices, self.allowed_feature_indices).copy()
        
        if len(useful_nan_indices) < num_features:
            needed = num_features - len(useful_nan_indices)
            
            # This is still correct, as sparse_indices is based on original nan_frac
            extras = np.setdiff1d(self.sparse_indices, useful_nan_indices).copy()
            
            if len(extras) > 0:
                np.random.shuffle(extras)
                selected_indices = np.concatenate((useful_nan_indices, extras[:needed]))
            else:
                selected_indices = np.random.choice(
                    self.allowed_feature_indices, 
                    size=min(num_features, len(self.allowed_feature_indices)), 
                    replace=False
                )
        else:
            np.random.shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
            
        return self._format_features(selected_indices[:num_features])
    
    def _get_random_allowed_features(self, num_features):
        """(No changes)"""
        if len(self.allowed_feature_indices) == 0: return []
        num_to_select = min(num_features, len(self.allowed_feature_indices))
        selected = np.random.choice(self.allowed_feature_indices, size=num_to_select, replace=False)
        return self._format_features(selected)
    
    def _format_features(self, indices):
        """(No changes, but nan_percentage is now correct)"""
        results = []
        for idx in indices:
            py_idx = int(idx)
            if py_idx >= len(self.feature_cols) or py_idx >= len(self.col_nan_frac): continue
            fname = str(self.feature_cols[py_idx])
            results.append({
                "feature_name": fname,
                "question": str(self.questions_map.get(fname, f"Is it {fname}?")),
                # This now correctly reflects the ORIGINAL nan %
                "nan_percentage": float(self.col_nan_frac[py_idx])
            })
        return results
    
    def get_all_feature_gains(self, initial_prior: np.ndarray = None) -> list[dict]:
        """(No changes, but nan_percentage is now correct)"""
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
                # This now correctly reflects the ORIGINAL nan %
                'nan_percentage': float(self.col_nan_frac[idx])
            })
        results.sort(key=lambda x: x['initial_gain'], reverse=True)
        return results