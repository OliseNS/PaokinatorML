import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Akinator guessing engine (V9.1 - Aggressive Elimination).
    
    CHANGELOG V9.1:
    - STRICT ELIMINATION: Implements "Kill Zones". If user says 'No', 
      animals with 'Yes' are reduced to near-zero probability instantly.
    - SHARPER CURVES: Reduced sigma in likelihood tables to punish 
      minor deviations more severely.
    - IMPUTATION SAFETY: The strict elimination logic explicitly 
      ignores imputed values to prevent false negatives.
    - CORRELATION PENALTY: Retained from V9.0 to prevent redundant questions.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df # Will be replaced by dense version
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        # Store original sparse data for masking
        sparse_df = df.copy()
        
        # Granular answer values
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'mostly': 0.75, 'usually': 0.75, 'probably': 0.75,
            'sort of': 0.5, 'sometimes': 0.5,
            'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0,
        }
        
        # --- Step 1: Matrix Completion ---
        print("🧠 Engine received sparse data, starting matrix completion...")
        try:
            self.df = self._complete_matrix(df, feature_cols)
            print("✅ Matrix completion successful.")
        except Exception as e:
            print(f"❌ WARNING: Matrix completion failed ({e}). Using sparse data.")
            self.df = df
            
        # --- Step 2: Correlation Matrix ---
        print("   Calculating feature correlation matrix...")
        try:
            self.feature_correlation_matrix = self.df[self.feature_cols].corr().values.astype(np.float32)
            self.feature_correlation_matrix = np.nan_to_num(self.feature_correlation_matrix, nan=0.0)
        except Exception:
            n_feat = len(self.feature_cols)
            self.feature_correlation_matrix = np.eye(n_feat, dtype=np.float32)

        # --- Step 3: Precomputations ---
        self._precompute_likelihood_tables()
        self._build_arrays(sparse_df) 
        print(f"✓ Engine initialized: {len(self.animals)} items, {len(self.feature_cols)} features.")
    
    def _complete_matrix(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        if df.empty or not feature_cols:
            return df
        
        # Use KNN to fill gaps intelligently
        animal_names = df['animal_name'].values
        features_matrix = df[feature_cols].values.astype(np.float32)
        
        # Weights='distance' means closer neighbors have more influence
        imputer = KNNImputer(n_neighbors=5, weights='distance', missing_values=np.nan)
        
        completed_matrix = imputer.fit_transform(features_matrix)
        completed_matrix = np.clip(completed_matrix, 0.0, 1.0)

        completed_df = pd.DataFrame(completed_matrix, columns=feature_cols)
        completed_df['animal_name'] = animal_names
        return completed_df[['animal_name'] + feature_cols]

    def _precompute_likelihood_tables(self):
        """
        Creates the probability lookup table.
        V9.1 CHANGE: Sigmas are much tighter.
        """
        steps = 1001
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        
        f_grid_col = self.feature_grid[:, np.newaxis]
        a_vals_row = self.answer_values[np.newaxis, :]
        diffs = np.abs(f_grid_col - a_vals_row)
        
        # V9.1: Tighter sigmas (0.10 / 0.08) compared to standard (0.20 / 0.15)
        # This means probability drops off very quickly if values don't match exactly.
        sigmas = np.where(self.answer_values == 0.5, 0.12, 0.08)
        sigmas_row = sigmas[np.newaxis, :]
        
        likelihoods = np.exp(-0.5 * (diffs / sigmas_row) ** 2)
        self.likelihood_table = np.maximum(likelihoods, 1e-12).astype(np.float32)
    
    def _build_arrays(self, sparse_df: pd.DataFrame):
        self.animals = self.df['animal_name'].values
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        # Identify which values were originally NaN
        original_features = sparse_df[self.feature_cols].values.astype(np.float32)
        self.is_imputed_mask = np.isnan(original_features)
        
        # Metrics based on ORIGINAL data quality
        self.col_nan_frac = np.mean(self.is_imputed_mask, axis=0) 
        
        nan_masked_features = np.ma.masked_invalid(self.features)
        col_var = nan_masked_features.var(axis=0).data
        col_mean = nan_masked_features.mean(axis=0).data
        self.col_ambiguity = 1.0 - 2.0 * np.abs(col_mean - 0.5)
        
        # Allow features that are not 100% empty
        self.allowed_feature_mask = (self.col_nan_frac < 1.0) 
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)
        
        # Mark features with >50% missing data as sparse
        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask 
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        self.col_var = col_var
        
        if len(self.animals) > 0:
            n = len(self.animals)
            self.uniform_prior = np.ones(n, dtype=np.float32) / n
            
            # Pre-calculate ranks for feature selection optimization
            initial_gains = self._compute_gains(self.uniform_prior, self.allowed_feature_indices)
            sorted_indices = np.argsort(initial_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
            
            max_rank = len(self.feature_cols) + 1
            self.feature_ranks = np.full(len(self.feature_cols), max_rank, dtype=np.float32)
            self.feature_ranks[self.sorted_initial_feature_indices] = np.arange(len(self.sorted_initial_feature_indices)) + 1
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)
            self.feature_ranks = np.array([], dtype=np.float32)
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))
    
    def _compute_gains(self, prior: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
        """
        Calculates Information Gain (Entropy reduction).
        Safe against imputed data skewing results.
        """
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._calculate_entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains

        # Optimization: Only consider active candidates
        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active < len(prior) * 0.8 and n_active > 0:
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum()
            active_features = self.features[active_mask][:, feature_indices]
            active_nan_mask = self.is_imputed_mask[active_mask][:, feature_indices]
        else:
            active_prior = prior
            active_features = self.features[:, feature_indices]
            active_nan_mask = self.is_imputed_mask[:, feature_indices]
            active_mask = slice(None)
        
        if active_prior.size == 0: return gains
        
        # Quantize features for table lookup
        f_filled = np.nan_to_num(active_features, nan=0.5) 
        f_quant = np.clip(np.rint(f_filled * 1000), 0, 1000).astype(np.int32)
        flat_quant = f_quant.flatten()
        
        n_actual_rows = active_features.shape[0]
        likelihoods = self.likelihood_table[flat_quant, :].reshape(n_actual_rows, n_features, n_answers)
        
        # NEUTRALIZE IMPUTED VALUES:
        # If a value was imputed, we treat it as 1.0 likelihood (neutral)
        # so it contributes 0 information gain.
        nan_expand = np.repeat(active_nan_mask[:, :, np.newaxis], n_answers, axis=2)
        likelihoods[nan_expand] = 1.0
        
        # P(Answer) = sum(P(Item) * P(Answer|Item))
        p_answers = np.einsum('i,ijk->jk', active_prior, likelihoods)
        
        # P(Item|Answer)
        posteriors = active_prior[:, np.newaxis, np.newaxis] * likelihoods
        posteriors /= (p_answers[np.newaxis, :, :] + 1e-10)
        posteriors = np.clip(posteriors, 1e-10, 1.0)
        
        # H(Item|Answer)
        p_logs = np.log2(posteriors)
        entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)
        
        # Expected Entropy
        expected_entropy = np.sum(p_answers * entropies_per_answer, axis=1)
        
        gains = current_entropy - expected_entropy
        return gains
    
    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """
        V9.1 UPDATE LOGIC: AGGRESSIVE & STRICT
        """
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        f_col = self.features[:, feature_idx]
        
        # We must know which values were guessed (imputed)
        imputed_mask = self.is_imputed_mask[:, feature_idx] 
        
        # 1. Standard Gaussian Likelihood Update
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 1000), 0, 1000).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # 2. Calculate Severity and Apply Penalty
        certainty = np.clip(2.0 * np.abs(f_col - 0.5), 0.0, 1.0)
        distance = np.abs(f_col - answer_val)
        severity = distance * certainty
        
        # Imputed values get a "Free Pass" on severity penalties
        severity[imputed_mask] = 0.0 
        
        # V9.1: Increased Penalty Factor (8.0 -> 25.0)
        # This punishes "leanings" heavily. If you say "No", a "0.6" feature value is punished hard.
        penalty_factor = 25.0
        penalty_multiplier = np.exp(-penalty_factor * severity)
        likelihoods *= penalty_multiplier
        
        # 3. THE KILL ZONES (Strict Elimination)
        # We create a mask of items that are DEFINITELY wrong.
        definite_mismatch = np.zeros_like(imputed_mask, dtype=bool)
        
        # Logic: If user is confident, kill the opposites.
        # Thresholds: >0.7 is "Yes/Usually", <0.3 is "No/Rarely"
        
        if answer_val >= 0.9: # User said YES
            # Kill NO (0.0) and RARELY (0.25)
            definite_mismatch = (f_col <= 0.3)
            
        elif answer_val <= 0.1: # User said NO
            # Kill YES (1.0) and USUALLY (0.75)
            definite_mismatch = (f_col >= 0.7)
            
        elif answer_val == 0.75: # User said PROBABLY
            # Kill strict NO
            definite_mismatch = (f_col <= 0.1)
            
        elif answer_val == 0.25: # User said RARELY
            # Kill strict YES
            definite_mismatch = (f_col >= 0.9)
        
        # SAFETY: NEVER kill based on an imputed value.
        definite_mismatch[imputed_mask] = False
        
        # Execute Elimination (1e-12 is effectively 0 probability)
        likelihoods[definite_mismatch] = 1e-12
        
        scores = np.log(likelihoods + 1e-12)
        return current_scores + scores
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Selects the best question using Entropy + Correlation Penalty.
        """
        asked_mask = np.isin(self.feature_cols, asked_features)
        candidates_mask = self.allowed_feature_mask & ~asked_mask
        candidates_indices = np.where(candidates_mask)[0].astype(np.int32)

        if not candidates_indices.any():
            return None, None
            
        # --- Random Start for Variety ---
        if question_count == 0:
            top_initial = self.sorted_initial_feature_indices
            top_is_available_mask = np.isin(top_initial, candidates_indices)
            available_top = top_initial[top_is_available_mask]
            # Pick from top 15% to ensure quality but variety
            top_limit = max(1, len(self.sorted_initial_feature_indices) // 7)
            available_top_subset = available_top[available_top < top_limit]
            
            if available_top_subset.any():
                best_feat_idx = np.random.choice(available_top_subset)
                feature_name = self.feature_cols[best_feat_idx]
                return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        # --- Entropy Calculation ---
        candidates_to_eval = self._select_candidate_subset(candidates_indices)
        if len(candidates_to_eval) == 0: return None, None
        
        gains = self._compute_gains(prior, candidates_to_eval)
        
        # --- Quality Penalties ---
        # Penalize questions with too many missing values or too much ambiguity
        nan_fracs = self.col_nan_frac[candidates_to_eval]
        ambiguities = self.col_ambiguity[candidates_to_eval]
        quality_penalties = 1.0 - (0.8 * nan_fracs + 0.6 * ambiguities)
        quality_penalties = np.clip(quality_penalties, 0.1, 1.0)
        gains = gains * quality_penalties
        
        # --- Correlation Penalty (The "Africa" Fix) ---
        asked_indices = np.where(asked_mask)[0]
        if len(asked_indices) > 0:
            # Check correlation against ALL asked questions
            corr_slice = self.feature_correlation_matrix[candidates_to_eval, :][:, asked_indices]
            max_correlations = np.abs(corr_slice).max(axis=1)
            
            # Strong exponential penalty for high correlation
            # If corr > 0.8, penalty is massive.
            correlation_penalty = np.exp(-3.0 * max_correlations**2)
            gains = gains * correlation_penalty
        
        # --- Final Selection ---
        # Small amount of randomness in early game to prevent loops
        if question_count < 3 and len(gains) > 1:
            top_n = min(3, len(gains))
            top_indices_local = np.argpartition(gains, -top_n)[-top_n:]
            best_feat_idx = candidates_to_eval[np.random.choice(top_indices_local)]
        else:
            best_local_idx = np.argmax(gains)
            best_feat_idx = candidates_to_eval[best_local_idx]
        
        feature_name = self.feature_cols[best_feat_idx]
        return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
    
    def _select_candidate_subset(self, candidates_indices):
        """Optimization to avoid calculating gain for every single feature."""
        candidates_indices = np.array(candidates_indices, dtype=np.int32)
        if len(candidates_indices) <= 300:
            return candidates_indices
        
        # Heuristic: Rank + Variance
        ranks = self.feature_ranks[candidates_indices]
        rank_scores = np.where(ranks < self.feature_ranks.size + 1, 1.0 / ranks, 0.0)
        var_scores = self.col_var[candidates_indices]
        scores = rank_scores + var_scores
        
        sorted_local_indices = np.argsort(scores)[::-1]
        return candidates_indices[sorted_local_indices[:200]]
    
    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        Decides if the engine is confident enough to guess.
        Maintains HIGH STANDARDS for confidence.
        """
        q_count = game_state['question_count']
        if probs.sum() < 1e-10: return False, None, None
            
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else -1
        
        top_prob = probs[top_idx]
        second_prob = probs[second_idx] if second_idx != -1 else 0.0
        top_animal = self.animals[top_idx]
        margin = top_prob - second_prob
        
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 5:
                return False, None, None

        # V9.1 GUESS LOGIC: High Confidence
        # We require 99% probability AND a massive lead (0.95 margin).
        # Because elimination is strict, this state is reachable quickly.
        if top_prob >= 0.99 and margin >= 0.95:
            print(f"[Q{q_count}] 🎯 SNIPER GUESS: {top_animal} (P={top_prob:.4f}, M={margin:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
        
        # Emergency guess at limit
        if q_count >= 25 and not game_state.get('continue_mode', False):
            print(f"[Q{q_count}] ⚠️ TIMEOUT GUESS: {top_animal} (P={top_prob:.4f})")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'
            
        return False, None, None
    
    def get_features_for_data_collection(self, item_name, num_features=5):
        """Returns features to validate, prioritizing original NaNs."""
        try:
            matches = np.where(self.animals == item_name)[0]
            if len(matches) == 0:
                matches = np.where(np.char.lower(self.animals.astype(str)) == item_name.lower())[0]
            if len(matches) == 0:
                return self._get_random_allowed_features(num_features)
            item_idx = matches[0]
            
            # Find original NaNs using mask
            original_item_feats_mask = self.is_imputed_mask[item_idx]
            nan_indices = np.where(original_item_feats_mask)[0]
            
        except Exception:
            return self._get_random_allowed_features(num_features)
        
        useful_nan_indices = np.intersect1d(nan_indices, self.allowed_feature_indices).copy()
        
        if len(useful_nan_indices) < num_features:
            needed = num_features - len(useful_nan_indices)
            extras = np.setdiff1d(self.sparse_indices, useful_nan_indices).copy()
            if len(extras) > 0:
                np.random.shuffle(extras)
                selected_indices = np.concatenate((useful_nan_indices, extras[:needed]))
            else:
                selected_indices = np.random.choice(self.allowed_feature_indices, size=num_features, replace=False)
        else:
            np.random.shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
            
        return self._format_features(selected_indices[:num_features])
    
    def _get_random_allowed_features(self, num_features):
        if len(self.allowed_feature_indices) == 0: return []
        selected = np.random.choice(self.allowed_feature_indices, size=min(num_features, len(self.allowed_feature_indices)), replace=False)
        return self._format_features(selected)
    
    def _format_features(self, indices):
        results = []
        for idx in indices:
            py_idx = int(idx)
            if py_idx >= len(self.feature_cols): continue
            fname = str(self.feature_cols[py_idx])
            results.append({
                "feature_name": fname,
                "question": str(self.questions_map.get(fname, f"Is it {fname}?")),
                "nan_percentage": float(self.col_nan_frac[py_idx])
            })
        return results
    
    def get_all_feature_gains(self, initial_prior: np.ndarray = None) -> list[dict]:
        if initial_prior is None:
            initial_prior = self.uniform_prior if hasattr(self, 'uniform_prior') else np.ones(len(self.animals))/len(self.animals)
        
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