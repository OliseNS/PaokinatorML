import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Real-time Padded Akinator Engine (V16.0 - Normalized & Tuned).
    """
    
    def __init__(self, df, feature_cols, questions_map, row_padding=200, col_padding=100):
        self.questions_map = questions_map
        
        # Track active counts
        self.active_feature_names = np.array(feature_cols, dtype=object)
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        # --- NORMALIZATION & DISPLAY MAP ---
        self.display_name_map = {} 
        
        # O(1) Lookups - NORMALIZED
        self.feature_map = {name: i for i, name in enumerate(self.active_feature_names)}
        self.item_map = {} 
        
        self.cap_rows = self.n_items + row_padding
        self.cap_cols = self.n_features + col_padding
        
        print(f"🧠 Engine Alloc: {self.cap_rows} Rows x {self.cap_cols} Cols")

        self.features = np.full((self.cap_rows, self.cap_cols), np.nan, dtype=np.float32)
        self.animals = np.full(self.cap_rows, None, dtype=object)
        self.feature_cols_array = np.full(self.cap_cols, None, dtype=object)
        
        if self.n_features > 0:
            self.feature_cols_array[:self.n_features] = self.active_feature_names

        if not df.empty:
            names = df['animal_name'].values
            
            # --- DEDUPLICATION LOGIC ---
            unique_indices = []
            seen_keys = set()
            
            for original_idx, name in enumerate(names):
                clean_key = self._normalize(name)
                if clean_key not in seen_keys:
                    seen_keys.add(clean_key)
                    unique_indices.append(original_idx)
                    
                    # Map normalized key to the new dense index
                    new_idx = len(unique_indices) - 1
                    self.item_map[clean_key] = new_idx
                    self.animals[new_idx] = name # Keep original casing for display
                    self.display_name_map[clean_key] = name
            
            # Update n_items to reflect deduplicated count
            self.n_items = len(unique_indices)
            
            if len(df.columns) > 1:
                # Load only the unique rows
                raw_features = df[feature_cols].values.astype(np.float32)
                self.features[:self.n_items, :self.n_features] = raw_features[unique_indices]

        self._impute_active_block()
        
        # Stats setup 
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 
            'probably': 0.75, 'mostly': 0.75, 'usually': 0.75, 
            'somewhat': 0.5, 'sort of': 0.5, 'sometimes': 0.5, 'idk': 0.5, 'unknown': 0.5,
            'probably not': 0.25, 'not really': 0.25, 'rarely': 0.25, 
            'no': 0.0, 'n': 0.0
        }
        
        self._refresh_column_stats()
        self._recalc_correlation()
        self._compute_initial_priors()
        
        print(f"✓ Engine Ready. Active: {self.n_items} items (deduplicated).")

    def _normalize(self, text):
        """Centralized normalization for the engine."""
        return str(text).strip().lower()

    def _impute_active_block(self):
        """Fills gaps in knowledge using KNN."""
        if self.n_items == 0 or self.n_features == 0: return
        
        active_view = self.features[:self.n_items, :self.n_features]
        
        nan_mask = np.isnan(active_view)
        all_nan_cols = nan_mask.all(axis=0)
        if all_nan_cols.any():
            active_view[:, all_nan_cols] = 0.5
            
        if np.isnan(active_view).any():
            try:
                imputer = KNNImputer(n_neighbors=7, weights='distance')
                filled = imputer.fit_transform(active_view)
                self.features[:self.n_items, :self.n_features] = np.clip(filled, 0.01, 0.99)
            except Exception as e:
                print(f"⚠️ Imputation fallback: {e}")
                mask = np.isnan(active_view)
                self.features[:self.n_items, :self.n_features][mask] = 0.5

    def _recalc_correlation(self):
        """Calculates correlation to prevent asking redundant questions."""
        if self.n_items < 5 or self.n_features == 0:
            self.feature_correlation_matrix = np.eye(self.n_features, dtype=np.float32)
            return
        try:
            active_feats = self.features[:self.n_items, :self.n_features]
            centered = active_feats - np.mean(active_feats, axis=0)
            cov = np.dot(centered.T, centered) / (self.n_items - 1)
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            self.feature_correlation_matrix = np.nan_to_num(corr, nan=0.0)
        except:
            self.feature_correlation_matrix = np.eye(self.n_features, dtype=np.float32)

    def _refresh_column_stats(self):
        """Refreshes variance to find 'informative' features."""
        active_view = self.features[:self.n_items, :self.n_features]
        clean_view = np.nan_to_num(active_view, nan=0.5)
        
        self.col_var = np.var(clean_view, axis=0)
        self.col_mean = np.mean(clean_view, axis=0)
        
        self.allowed_feature_mask = (self.col_var > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

    def _compute_initial_priors(self):
        """Calculates the best Q1-Q3 questions (Global Variance)."""
        if self.n_items > 0 and len(self.allowed_feature_indices) > 0:
            variances = self.col_var[self.allowed_feature_indices]
            sorted_indices = np.argsort(variances)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        """DEDUPLICATED Ingestion."""
        clean_item_key = self._normalize(item_name)
        idx = self.item_map.get(clean_item_key)
        
        # Add Item if New
        if idx is None:
            if self.n_items >= self.cap_rows: return
            idx = self.n_items
            self.animals[idx] = item_name  # Store Display Name
            self.item_map[clean_item_key] = idx 
            self.display_name_map[clean_item_key] = item_name
            self.n_items += 1
            if self.n_features > 0:
                self.features[idx, :self.n_features] = self.col_mean[:self.n_features]

        # Add Feature if New
        f_idx = self.feature_map.get(feature_name)
        if f_idx is None:
            if self.n_features >= self.cap_cols: return
            f_idx = self.n_features
            self.feature_cols_array[f_idx] = feature_name
            self.feature_map[feature_name] = f_idx
            self.n_features += 1
            if question_text:
                self.questions_map[feature_name] = question_text
            self.features[:self.n_items, f_idx] = 0.5

        self.features[idx, f_idx] = value
        self._refresh_column_stats()

    def recalculate_stats(self):
        self._refresh_column_stats()
        self._recalc_correlation()
        self._compute_initial_priors()

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        if len(current_scores) < self.n_items:
            padding_len = self.n_items - len(current_scores)
            valid_scores = current_scores[np.isfinite(current_scores)]
            base_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
            padding = np.full(padding_len, base_score, dtype=np.float32)
            current_scores = np.concatenate([current_scores, padding])
        
        current_scores = current_scores[:self.n_items]
        answer_val = self.fuzzy_map.get(str(answer_str).lower(), 0.5)
        
        f_col = self.features[:self.n_items, feature_idx]
        f_col_clean = np.nan_to_num(f_col, nan=0.5)
        
        sigma = 0.18
        distance = np.abs(f_col_clean - answer_val)
        gaussian_likelihood = np.exp(-0.5 * (distance / sigma) ** 2)
        
        p_noise = 0.02
        final_likelihood = (1.0 - p_noise) * gaussian_likelihood + p_noise
        
        if answer_val > 0.75:
             final_likelihood[f_col_clean > 0.8] *= 1.25
        elif answer_val < 0.25:
             final_likelihood[f_col_clean < 0.2] *= 1.25
             
        return current_scores + np.log(np.clip(final_likelihood, 1e-9, None))

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        active_cols = self.feature_cols_array[:self.n_features]
        asked_mask = np.isin(active_cols, asked_features)
        
        if prior is not None and len(prior) > 0:
            sorted_indices = np.argsort(prior)[::-1]
            cum_sum = np.cumsum(prior[sorted_indices])
            cutoff = np.argmax(cum_sum > 0.90)
            top_k = max(20, cutoff + 1)
            target_indices = sorted_indices[:top_k]
        else:
            target_indices = np.arange(self.n_items)

        target_features = self.features[target_indices, :]
        target_var = np.var(np.nan_to_num(target_features, nan=0.5), axis=0)
        candidate_indices = np.where((~asked_mask) & self.allowed_feature_mask)[0]
        
        if len(candidate_indices) == 0: return None, None

        scores = target_var.copy()
        asked_indices = np.where(asked_mask)[0]
        
        if len(asked_indices) > 0:
            limit = min(self.n_features, self.feature_correlation_matrix.shape[0])
            valid_candidates = candidate_indices[candidate_indices < limit]
            valid_asked = asked_indices[asked_indices < limit]
            if len(valid_candidates) > 0 and len(valid_asked) > 0:
                corr_sub = self.feature_correlation_matrix[valid_candidates][:, valid_asked]
                max_corr = np.max(np.abs(corr_sub), axis=1)
                scores[valid_candidates] *= (1.0 - max_corr**2)

        if question_count == 0:
            candidate_scores = scores[candidate_indices]
            sorted_local_indices = np.argsort(candidate_scores)[::-1]
            pool_size = min(len(sorted_local_indices), 25)
            if pool_size == 0: return None, None
            random_pool_index = np.random.choice(sorted_local_indices[:pool_size])
            best_candidate_idx = candidate_indices[random_pool_index]
        else:
            best_candidate_idx = candidate_indices[np.argmax(scores[candidate_indices])]
            
        fname = str(active_cols[best_candidate_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        Decides if we should guess.
        FIXED: Lower thresholds, stricter buffers for wrong guesses.
        """
        if probs.sum() < 1e-10: return False, None, None
        
        probs = probs / probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        q_count = game_state.get('question_count', 0)
        q_since_guess = game_state.get('questions_since_last_guess', 999) 
        
        # Ratio Calculation
        second_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0.001
        ratio = top_prob / (second_prob + 1e-9)
        
        # --- 1. HARD LIMIT ---
        if q_count >= 25:
            return True, top_animal, "max_questions_25"
            
        # --- 2. ANTI-INFINITE LOOP BUFFER (CRITICAL FIX) ---
        # If we have guessed recently, we FORCE at least 5 questions between guesses 
        # unless probability is nearly 100%.
        min_questions_between_guesses = 5
        if q_since_guess < min_questions_between_guesses:
             if top_prob < 0.999:
                 return False, None, None

        # --- 3. TUNED CONFIDENCE LOGIC ---
        # Faster guessing: 95% + 2x ratio or 85% + 4x ratio
        if top_prob > 0.95 and ratio > 2.0:
            return True, top_animal, "high_confidence"
            
        if top_prob > 0.85 and ratio > 4.0:
            return True, top_animal, "clear_leader"
            
        return False, None, None
    
    @property
    def feature_cols(self):
        return self.feature_cols_array[:self.n_features]

    def get_features_for_data_collection(self, item_name, num_features=5):
        if len(self.allowed_feature_indices) == 0: return []
        variances = self.col_var[self.allowed_feature_indices]
        prob_dist = variances / variances.sum()
        selected = np.random.choice(
            self.allowed_feature_indices, 
            size=min(num_features, len(self.allowed_feature_indices)), 
            replace=False, p=prob_dist
        )
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
        results = []
        active_cols = self.feature_cols_array[:self.n_features]
        for idx in self.allowed_feature_indices:
            results.append({
                "feature_name": str(active_cols[idx]),
                "initial_gain": float(self.col_var[idx])
            })
        return sorted(results, key=lambda x: x['initial_gain'], reverse=True)

    def build_feature_vector(self, user_answers: dict) -> np.ndarray:
        vector = np.full(self.n_features, 0.5, dtype=np.float32)
        for fname, val in user_answers.items():
            idx = self.feature_map.get(fname)
            if idx is not None and idx < self.n_features:
                vector[idx] = float(val)
        return vector

    def find_nearest_neighbors(self, target_vector: np.ndarray, exclude_name: str = None, n: int = 3) -> list[str]:
        if self.n_items == 0 or self.n_features == 0: return []
        
        exclude_clean = self._normalize(exclude_name) if exclude_name else None
        
        active_matrix = np.nan_to_num(self.features[:self.n_items, :self.n_features], nan=0.5)
        diff = active_matrix - target_vector
        distances = np.linalg.norm(diff, axis=1)
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices:
            if len(results) >= n: break
            candidate_name = self.animals[idx]
            if exclude_clean and self._normalize(candidate_name) == exclude_clean:
                continue
            results.append(candidate_name)
        return results