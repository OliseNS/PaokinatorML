import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Real-time Padded Akinator Engine (V13.0 - Smart & Patient).
    
    Updates:
    - Stricter guessing thresholds with confidence requirements
    - Mistake tolerance through answer history tracking
    - Stronger elimination logic
    - Verification mechanism for wrong guesses
    """
    
    def __init__(self, df, feature_cols, questions_map, row_padding=200, col_padding=100):
        self.questions_map = questions_map
        
        # Track active counts
        self.active_feature_names = np.array(feature_cols, dtype=object)
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        # O(1) Lookups
        self.feature_map = {name: i for i, name in enumerate(self.active_feature_names)}
        self.item_map = {} 
        
        # --- 1. Setup Capacity (Double Padding) ---
        self.cap_rows = self.n_items + row_padding
        self.cap_cols = self.n_features + col_padding
        
        print(f"🧠 Engine Alloc: {self.cap_rows} Rows (Items) x {self.cap_cols} Cols (Features)")

        # --- 2. Initialize Padded Arrays ---
        self.features = np.full((self.cap_rows, self.cap_cols), np.nan, dtype=np.float32)
        self.animals = np.full(self.cap_rows, None, dtype=object)
        self.feature_cols_array = np.full(self.cap_cols, None, dtype=object)
        
        # --- 3. Load Initial Data ---
        if self.n_features > 0:
            self.feature_cols_array[:self.n_features] = self.active_feature_names

        if not df.empty:
            names = df['animal_name'].values
            self.animals[:self.n_items] = names
            for idx, name in enumerate(names):
                self.item_map[name] = idx
                
            if len(df.columns) > 1:
                existing_data = df[feature_cols].values.astype(np.float32)
                self.features[:self.n_items, :self.n_features] = existing_data

        # --- 4. Initial Math Prep ---
        print("   Imputing initial data block...")
        self._impute_active_block()
        
        # Stats setup with improved fuzzy matching
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'mostly': 0.75, 'usually': 0.75, 
            'probably': 0.75, 'sort of': 0.5, 'sometimes': 0.5, 
            'somewhat': 0.5, 'not really': 0.25, 'rarely': 0.25, 
            'no': 0.0, 'n': 0.0
        }
        
        # Precomputations
        self._precompute_likelihood_tables()
        self._refresh_column_stats()
        self._recalc_correlation()
        self._compute_initial_priors()
        
        print(f"✓ Engine Ready. Active: {self.n_items} items, {self.n_features} features.")

    def _impute_active_block(self):
        """Smart Imputation with safety checks."""
        if self.n_items == 0 or self.n_features == 0: return
        
        active_view = self.features[:self.n_items, :self.n_features]
        
        if not np.isnan(active_view).any():
            return

        try:
            nan_mask = np.isnan(active_view)
            all_nan_cols = nan_mask.all(axis=0)
            
            if all_nan_cols.any():
                active_view[:, all_nan_cols] = 0.5
            
            if np.isnan(active_view).any():
                imputer = KNNImputer(n_neighbors=5, weights='distance', missing_values=np.nan)
                filled = imputer.fit_transform(active_view)
                
                if filled.shape == active_view.shape:
                    self.features[:self.n_items, :self.n_features] = np.clip(filled, 0.0, 1.0)
                else:
                    print(f"⚠️ Imputation Mismatch: Imputer returned {filled.shape}, expected {active_view.shape}. Falling back to 0.5 fill.")
                    mask = np.isnan(active_view)
                    self.features[:self.n_items, :self.n_features][mask] = 0.5
                    
        except Exception as e:
            print(f"⚠️ Imputation Error: {e}")
            mask = np.isnan(active_view)
            self.features[:self.n_items, :self.n_features][mask] = 0.5

    def _recalc_correlation(self):
        """Calculates feature correlation matrix for penalty logic."""
        if self.n_items < 2 or self.n_features == 0:
            self.feature_correlation_matrix = np.eye(self.n_features, dtype=np.float32)
            return
        try:
            active_feats = self.features[:self.n_items, :self.n_features]
            df_temp = pd.DataFrame(active_feats)
            self.feature_correlation_matrix = df_temp.corr().fillna(0).values.astype(np.float32)
        except:
            self.feature_correlation_matrix = np.eye(self.n_features, dtype=np.float32)

    def _refresh_column_stats(self):
        """Updates Variance and Mean for entropy calculations."""
        active_view = self.features[:self.n_items, :self.n_features]
        clean_view = np.nan_to_num(active_view, nan=0.5)
        
        self.col_var = np.var(clean_view, axis=0)
        self.col_mean = np.mean(clean_view, axis=0)
        self.col_ambiguity = 1.0 - 2.0 * np.abs(self.col_mean - 0.5)
        
        self.allowed_feature_mask = (self.col_var > 1e-6)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

    def _compute_initial_priors(self):
        if self.n_items > 0 and len(self.allowed_feature_indices) > 0:
            self.uniform_prior = np.ones(self.n_items, dtype=np.float32) / self.n_items
            initial_gains = self._compute_gains(self.uniform_prior, self.allowed_feature_indices)
            
            variance_boost = self.col_var[self.allowed_feature_indices]
            boosted_gains = initial_gains * (1.0 + variance_boost)
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        """Dynamically adds Items OR Features without full reload."""
        idx = self.item_map.get(item_name)
        if idx is None:
            if self.n_items >= self.cap_rows:
                print(f"⚠️ Row Capacity Reached ({self.cap_rows}). Cannot add '{item_name}'.")
                return
            
            idx = self.n_items
            self.animals[idx] = item_name
            self.item_map[item_name] = idx
            self.n_items += 1
            self.features[idx, :self.n_features] = 0.5 
            self.uniform_prior = np.ones(self.n_items, dtype=np.float32) / self.n_items
            print(f"   [Engine] +ITEM: '{item_name}' @ idx {idx}")

        f_idx = self.feature_map.get(feature_name)
        if f_idx is None:
            if self.n_features >= self.cap_cols:
                print(f"⚠️ Col Capacity Reached ({self.cap_cols}). Cannot add '{feature_name}'.")
                return
            
            f_idx = self.n_features
            self.feature_cols_array[f_idx] = feature_name
            self.feature_map[feature_name] = f_idx
            self.n_features += 1
            
            if question_text:
                self.questions_map[feature_name] = question_text
            
            self.features[:self.n_items, f_idx] = 0.5
            print(f"   [Engine] +FEATURE: '{feature_name}' @ col {f_idx}")

        self.features[idx, f_idx] = value
        self._refresh_column_stats()

    def recalculate_stats(self):
        """Called by service.py after a batch of updates to refresh priorities."""
        self._compute_initial_priors()

    def _precompute_likelihood_tables(self):
        """Precompute likelihood tables with tighter distributions for better discrimination."""
        steps = 1001
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        f_grid_col = self.feature_grid[:, np.newaxis]
        a_vals_row = self.answer_values[np.newaxis, :]
        diffs = np.abs(f_grid_col - a_vals_row)
        
        # Tighter sigmas for better discrimination
        sigmas = np.where(self.answer_values == 0.5, 0.15, 0.10)
        likelihoods = np.exp(-0.5 * (diffs / sigmas[np.newaxis, :]) ** 2)
        self.likelihood_table = np.maximum(likelihoods, 1e-12).astype(np.float32)

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))

    def _compute_gains(self, prior: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
        active_features = self.features[:self.n_items, :][:, feature_indices]
        
        n_rows = active_features.shape[0]
        n_feats = len(feature_indices)
        if n_rows == 0 or n_feats == 0: return np.zeros(n_feats, dtype=np.float32)

        f_filled = np.nan_to_num(active_features, nan=0.5)
        f_quant = np.clip(np.rint(f_filled * 1000), 0, 1000).astype(np.int32)
        flat_quant = f_quant.flatten()

        n_answers = 5
        likelihoods = self.likelihood_table[flat_quant, :].reshape(n_rows, n_feats, n_answers)
        
        p_answers = np.einsum('i,ijk->jk', prior, likelihoods)
        posteriors = prior[:, np.newaxis, np.newaxis] * likelihoods
        posteriors /= (p_answers[np.newaxis, :, :] + 1e-10)
        
        p_logs = np.log2(np.clip(posteriors, 1e-10, 1.0))
        entropies = -np.sum(posteriors * p_logs, axis=0)
        expected_entropy = np.sum(p_answers * entropies, axis=1)
        
        current_entropy = self._calculate_entropy(prior)
        return current_entropy - expected_entropy

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """Updated with MUCH stricter elimination logic."""
        if len(current_scores) < self.n_items:
            padding = np.zeros(self.n_items - len(current_scores), dtype=np.float32)
            current_scores = np.concatenate([current_scores, padding])
        elif len(current_scores) > self.n_items:
            current_scores = current_scores[:self.n_items]
        
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        f_col = self.features[:self.n_items, feature_idx]
        f_col_clean = np.nan_to_num(f_col, nan=0.5)
        
        f_quant = np.clip(np.rint(f_col_clean * 1000), 0, 1000).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # Enhanced penalty calculation
        certainty = np.clip(2.0 * np.abs(f_col_clean - 0.5), 0.0, 1.0)
        distance = np.abs(f_col_clean - answer_val)
        severity = distance * certainty
        
        # Stronger penalty for contradictions
        penalty_multiplier = np.exp(-25.0 * severity)  # Increased from 20.0
        likelihoods *= penalty_multiplier
        
        # MUCH stricter hard elimination thresholds
        hard_eliminate = np.zeros(self.n_items, dtype=bool)
        if answer_val == 1.0:  # "Yes"
            # Eliminate if feature value is low (< 0.4 instead of 0.3)
            hard_eliminate = (f_col_clean < 0.4)
        elif answer_val == 0.0:  # "No"
            # Eliminate if feature value is high (> 0.6 instead of 0.7)
            hard_eliminate = (f_col_clean > 0.6)
        elif answer_val == 0.75:  # "Mostly/Usually"
            # Eliminate if feature value is very low
            hard_eliminate = (f_col_clean < 0.3)
        elif answer_val == 0.25:  # "Not really/Rarely"
            # Eliminate if feature value is very high
            hard_eliminate = (f_col_clean > 0.7)
        
        # Apply elimination with severe penalty
        likelihoods[hard_eliminate] = 1e-15  # Even more severe than before
        scores = np.log(likelihoods + 1e-12)
        
        return current_scores + scores

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        active_cols = self.feature_cols_array[:self.n_features]
        
        asked_mask = np.isin(active_cols, asked_features)
        candidates_mask = self.allowed_feature_mask & ~asked_mask
        candidates_indices = np.where(candidates_mask)[0].astype(np.int32)

        if not candidates_indices.any(): return None, None
        
        # Q0 Optimization
        if question_count == 0 and len(self.sorted_initial_feature_indices) > 0:
            top_initial = self.sorted_initial_feature_indices
            valid_initial = top_initial[top_initial < self.n_features]
            
            top_is_available_mask = np.isin(valid_initial, candidates_indices)
            available_top = valid_initial[top_is_available_mask]
            
            top_cut = max(1, len(available_top) // 5)
            if len(available_top) > 0:
                best_feat_idx = np.random.choice(available_top[:top_cut])
                fname = str(active_cols[best_feat_idx])
                return fname, self.questions_map.get(fname, f"Is it {fname}?")

        candidates_to_eval = self._select_candidate_subset(candidates_indices)
        combined_gains = self._compute_gains(prior, candidates_to_eval)
        
        # Add Correlation Penalty
        asked_indices = np.where(asked_mask)[0]
        if len(asked_indices) > 0:
            try:
                limit = min(self.n_features, self.feature_correlation_matrix.shape[0])
                valid_cand_mask = candidates_to_eval < limit
                valid_candidates = candidates_to_eval[valid_cand_mask]
                
                valid_asked_mask = asked_indices < limit
                valid_asked = asked_indices[valid_asked_mask]
                
                if len(valid_candidates) > 0 and len(valid_asked) > 0:
                    corr_slice = self.feature_correlation_matrix[valid_candidates, :][:, valid_asked]
                    max_correlations = np.abs(corr_slice).max(axis=1)
                    combined_gains[valid_cand_mask] *= np.exp(-2.5 * max_correlations**2)
            except Exception: pass

        best_local_idx = np.argmax(combined_gains)
        best_feat_idx = candidates_to_eval[best_local_idx]
        
        fname = str(active_cols[best_feat_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def _select_candidate_subset(self, candidates_indices):
        candidates_indices = np.array(candidates_indices, dtype=np.int32)
        if len(candidates_indices) <= 300: return candidates_indices
        var_scores = self.col_var[candidates_indices]
        sorted_local = np.argsort(var_scores)[::-1]
        return candidates_indices[sorted_local[:300]]

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        MUCH more conservative guessing strategy with stronger requirements.
        
        Changes:
        - Higher confidence thresholds
        - Larger margin requirements
        - More questions required before guessing
        - Stricter multi-factor checks
        """
        if probs.sum() < 1e-10: return False, None, None
        
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_prob = probs[top_idx]
        
        top_animal = self.animals[top_idx]
        q_count = game_state['question_count']
        continue_mode = game_state.get('continue_mode', False)
        
        second_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0
        margin = top_prob - second_prob
        
        # Calculate relative dominance (how much better top is than second)
        relative_confidence = margin / (top_prob + 1e-10)
        
        # Don't guess too early in continue mode
        if continue_mode and game_state.get('questions_since_last_guess', 0) < 7:
            return False, None, None
        
        # Early guess only with EXTREME confidence (very rare)
        if q_count <= 10:
            if top_prob >= 0.998 and margin >= 0.95 and relative_confidence >= 0.95:
                return True, top_animal, 'final'
            return False, None, None
        
        # Mid-game guessing: require high confidence AND large margin
        if 10 < q_count <= 20:
            if top_prob >= 0.97 and margin >= 0.85 and relative_confidence >= 0.80:
                return True, top_animal, 'final'
            return False, None, None
        
        # Late mid-game: slightly relaxed but still strict
        if 20 < q_count <= 30:
            if top_prob >= 0.95 and margin >= 0.75 and relative_confidence >= 0.70:
                return True, top_animal, 'final'
            return False, None, None
        
        # Late game: more willing to guess but still cautious
        if 30 < q_count <= 40:
            if top_prob >= 0.90 and margin >= 0.60:
                return True, top_animal, 'final'
            return False, None, None
        
        # Very late game: must make a guess eventually
        if q_count > 40:
            if top_prob >= 0.70 and margin >= 0.40:
                return True, top_animal, 'final'
            # After 50 questions, guess the top candidate regardless
            if q_count >= 50:
                return True, top_animal, 'final'
            
        return False, None, None
    
    @property
    def feature_cols(self):
        return self.feature_cols_array[:self.n_features]

    def get_features_for_data_collection(self, item_name, num_features=5):
        if len(self.allowed_feature_indices) == 0: return []
        selected = np.random.choice(self.allowed_feature_indices, size=min(num_features, len(self.allowed_feature_indices)), replace=False)
        results = []
        active_cols = self.feature_cols_array[:self.n_features]
        for idx in selected:
            fname = str(active_cols[idx])
            results.append({
                "feature_name": fname,
                "question": self.questions_map.get(fname, f"Is it {fname}?")
            })
        return results
    
    def get_all_feature_gains(self) -> list[dict]:
        if self.n_items == 0: return []
        prior = np.ones(self.n_items, dtype=np.float32) / self.n_items
        gains = self._compute_gains(prior, self.allowed_feature_indices)
        results = []
        active_cols = self.feature_cols_array[:self.n_features]
        for i, idx in enumerate(self.allowed_feature_indices):
            results.append({
                "feature_name": str(active_cols[idx]),
                "initial_gain": float(gains[i])
            })
        return sorted(results, key=lambda x: x['initial_gain'], reverse=True)