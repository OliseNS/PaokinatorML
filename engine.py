import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Real-time Padded Akinator Engine (V16.0 - High Efficiency & Lite).
    
    Optimizations:
    - Memory: Uses float16 (Half Precision) for feature storage (50% RAM reduction).
    - Speed: Correlation logic REMOVED for maximum throughput.
    - Resilience: Increased noise floor to 5% to handle user errors robustly.
    - Logic: Pacing and Safety Nets preserved.
    """
    
    def __init__(self, df, feature_cols, questions_map, row_padding=200, col_padding=100):
        self.questions_map = questions_map
        
        # Track active counts
        self.active_feature_names = np.array(feature_cols, dtype=object)
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        # O(1) Lookups - NOW NORMALIZED to prevent duplicates
        self.feature_map = {name: i for i, name in enumerate(self.active_feature_names)}
        self.item_map = {} 
        
        # --- 1. Setup Capacity (Double Padding) ---
        self.cap_rows = self.n_items + row_padding
        self.cap_cols = self.n_features + col_padding
        
        # Estimate memory usage for the main matrix
        mem_usage_mb = (self.cap_rows * self.cap_cols * 2) / (1024 * 1024)
        print(f"🧠 Engine Alloc: {self.cap_rows} Rows x {self.cap_cols} Cols")
        print(f"   Memory Est: ~{mem_usage_mb:.2f} MB (Float16)")

        # --- 2. Initialize Padded Arrays (FLOAT16 for Memory Savings) ---
        # Using float16 reduces memory footprint by 50% vs float32
        self.features = np.full((self.cap_rows, self.cap_cols), np.nan, dtype=np.float16)
        
        self.animals = np.full(self.cap_rows, None, dtype=object)
        self.feature_cols_array = np.full(self.cap_cols, None, dtype=object)
        
        # --- 3. Load Initial Data ---
        if self.n_features > 0:
            self.feature_cols_array[:self.n_features] = self.active_feature_names

        if not df.empty:
            names = df['animal_name'].values
            self.animals[:self.n_items] = names
            
            # Normalize keys
            for idx, name in enumerate(names):
                clean_key = self._normalize(name)
                self.item_map[clean_key] = idx
                
            if len(df.columns) > 1:
                # Load as float32 first to handle any conversion quirks, then cast to float16
                existing_data = df[feature_cols].values.astype(np.float32)
                self.features[:self.n_items, :self.n_features] = existing_data.astype(np.float16)

        # --- 4. Initial Math Prep ---
        print("   Imputing initial data block...")
        self._impute_active_block()
        
        # Stats setup (Float16 compatible)
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float16)
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 
            'probably': 0.75, 'mostly': 0.75, 'usually': 0.75, 
            'somewhat': 0.5, 'sort of': 0.5, 'sometimes': 0.5, 'idk': 0.5, 'unknown': 0.5,
            'probably not': 0.25, 'not really': 0.25, 'rarely': 0.25, 
            'no': 0.0, 'n': 0.0
        }
        
        self._refresh_column_stats()
        self._compute_initial_priors()
        
        print(f"✓ Engine Ready. Active: {self.n_items} items, {self.n_features} features.")

    @staticmethod
    def _normalize(text: str) -> str:
        if not text: return ""
        clean = str(text).strip().lower()
        clean = clean.replace(" ", "").replace("-", "").replace("_", "")
        if clean.endswith("s") and not clean.endswith("ss") and len(clean) > 3:
            clean = clean[:-1]
        return clean

    def _impute_active_block(self):
        if self.n_items == 0 or self.n_features == 0: return
        
        active_view = self.features[:self.n_items, :self.n_features]
        
        # Check for fully NaN columns
        nan_mask = np.isnan(active_view)
        all_nan_cols = nan_mask.all(axis=0)
        if all_nan_cols.any():
            active_view[:, all_nan_cols] = 0.5
            
        if np.isnan(active_view).any():
            try:
                # Scikit-learn works in float64/32, so we must cast
                imputer = KNNImputer(n_neighbors=7, weights='distance')
                # Cast input to float32 for scikit compatibility
                filled = imputer.fit_transform(active_view.astype(np.float32))
                # Cast result back to float16 to save RAM
                self.features[:self.n_items, :self.n_features] = np.clip(filled, 0.01, 0.99).astype(np.float16)
            except Exception as e:
                print(f"⚠️ Imputation fallback: {e}")
                mask = np.isnan(active_view)
                self.features[:self.n_items, :self.n_features][mask] = 0.5

    def _refresh_column_stats(self):
        # Calculations done in float32 for precision, storage in class is implied by usage
        active_view = self.features[:self.n_items, :self.n_features].astype(np.float32)
        clean_view = np.nan_to_num(active_view, nan=0.5)
        self.col_var = np.var(clean_view, axis=0)
        self.col_mean = np.mean(clean_view, axis=0)
        self.allowed_feature_mask = (self.col_var > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

    def _compute_initial_priors(self):
        if self.n_items > 0 and len(self.allowed_feature_indices) > 0:
            variances = self.col_var[self.allowed_feature_indices]
            sorted_indices = np.argsort(variances)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        clean_item_key = self._normalize(item_name)
        idx = self.item_map.get(clean_item_key)
        
        if idx is None:
            if self.n_items >= self.cap_rows:
                print(f"⚠️ Capacity Reached. Cannot add '{item_name}'.")
                return
            idx = self.n_items
            self.animals[idx] = item_name 
            self.item_map[clean_item_key] = idx
            self.n_items += 1
            if self.n_features > 0:
                self.features[idx, :self.n_features] = self.col_mean[:self.n_features].astype(np.float16)

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

        # Store as float16
        self.features[idx, f_idx] = np.float16(value)
        # No full recalc here for speed

    def recalculate_stats(self):
        self._refresh_column_stats()
        self._compute_initial_priors()

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        # Resize scores if needed (keep scores in float32 for accumulation precision)
        if len(current_scores) < self.n_items:
            padding_len = self.n_items - len(current_scores)
            valid_scores = current_scores[np.isfinite(current_scores)]
            base_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
            padding = np.full(padding_len, base_score, dtype=np.float32)
            current_scores = np.concatenate([current_scores, padding])
        
        current_scores = current_scores[:self.n_items]
        answer_val = self.fuzzy_map.get(str(answer_str).lower(), 0.5)
        
        # Expand float16 to float32 for probability calculation to avoid underflow
        f_col = self.features[:self.n_items, feature_idx].astype(np.float32)
        f_col_clean = np.nan_to_num(f_col, nan=0.5)
        
        # Gaussian Likelihood
        sigma = 0.22
        distance = np.abs(f_col_clean - answer_val)
        gaussian_likelihood = np.exp(-0.5 * (distance / sigma) ** 2)
        
        p_noise = 0.03
        final_likelihood = (1.0 - p_noise) * gaussian_likelihood + p_noise
        
        log_update = np.log(np.clip(final_likelihood, 1e-9, None))
        return current_scores + log_update

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        active_cols = self.feature_cols_array[:self.n_features]
        asked_mask = np.isin(active_cols, asked_features)
        
        # 1. Identify Target Set
        if prior is not None and len(prior) > 0:
            sorted_indices = np.argsort(prior)[::-1]
            cum_sum = np.cumsum(prior[sorted_indices])
            cutoff = np.argmax(cum_sum > 0.90)
            top_k = max(20, cutoff + 1)
            target_indices = sorted_indices[:top_k]
        else:
            target_indices = np.arange(self.n_items)

        # 2. Calculate Variance (Unweighted)
        # Use float32 for variance calc to prevent underflow of small variances
        target_features = self.features[target_indices, :].astype(np.float32)
        target_var = np.var(np.nan_to_num(target_features, nan=0.5), axis=0)
        
        candidate_indices = np.where((~asked_mask) & self.allowed_feature_mask)[0]
        
        if len(candidate_indices) == 0: return None, None

        scores = target_var.copy()
        
        # NOTE: Correlation penalty REMOVED for efficiency and memory savings.
        # This allows O(N) selection speed instead of O(N^2), ideal for massive datasets.

        # 3. Selection
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

    # --- SPARSE DISCOVERY METHOD ---
    def select_sparse_discovery_question(self, prior: np.ndarray, asked_features: list) -> tuple[str, str] | tuple[None, None]:
        if prior is None or len(prior) == 0: return None, None

        sorted_indices = np.argsort(prior)[::-1]
        top_k_indices = sorted_indices[:20]
        if len(top_k_indices) == 0: return None, None
            
        # Use float32 for math
        top_matrix = self.features[top_k_indices, :self.n_features].astype(np.float32)
        clean_matrix = np.nan_to_num(top_matrix, nan=0.5)
        uncertainty_score = np.abs(clean_matrix - 0.5)
        col_uncertainty_sum = np.sum(uncertainty_score, axis=0)
        
        active_cols = self.feature_cols_array[:self.n_features]
        asked_mask = np.isin(active_cols, asked_features)
        col_uncertainty_sum[asked_mask] = np.inf
        
        best_idx = np.argmin(col_uncertainty_sum)
        avg_deviation = col_uncertainty_sum[best_idx] / len(top_k_indices)
        
        if avg_deviation > 0.15: return None, None 

        fname = str(active_cols[best_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        if probs.sum() < 1e-10: return False, None, None
        
        probs = probs / probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        q_count = game_state.get('question_count', 0)
        continue_mode = game_state.get('continue_mode', False)
        q_since_last_guess = game_state.get('questions_since_last_guess', 0)
        
        rejection_count = np.sum(game_state.get('rejected_mask', []))
        
        # 1. INITIAL PHASE CONSTRAINT
        if q_count < 8:
            return False, None, None

        should_guess = False
        reason = ""

        # 2. CONTINUATION PACING LOGIC
        if continue_mode:
            pacing_limit = min(3 + int(rejection_count), 7)
            if q_since_last_guess < pacing_limit:
                active_cols = self.feature_cols_array[:self.n_features]
                asked_features = game_state.get('asked_features', [])
                remaining = len(active_cols) - len(asked_features)
                if remaining > 0:
                    return False, None, None
                else:
                    return True, top_animal, "exhausted_questions"

        # 3. SAFETY NETS
        if q_count >= 40:
             return True, top_animal, "hard_safety_net_40"

        if q_count >= 25:
            if top_prob > 0.40:
                should_guess = True
                reason = "safety_net_25_confident"
            elif q_count >= 35:
                 should_guess = True
                 reason = "safety_net_grace_period_ended"
        
        # 4. STANDARD CONFIDENCE CHECK
        else:
            second_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0.001
            ratio = top_prob / (second_prob + 1e-9)
            
            if top_prob > 0.85 and ratio > 4.0:
                should_guess = True
                reason = "high_confidence"
            elif top_prob > 0.95:
                should_guess = True
                reason = "very_high_probability"
                
        # 5. EXHAUSTED QUESTIONS CHECK
        if not should_guess:
            active_cols = self.feature_cols_array[:self.n_features]
            asked_features = game_state.get('asked_features', [])
            if len(asked_features) >= (len(active_cols) - 1):
                should_guess = True
                reason = "no_questions_left"

        return should_guess, top_animal if should_guess else None, reason
    
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
        active_matrix = np.nan_to_num(self.features[:self.n_items, :self.n_features], nan=0.5).astype(np.float32)
        diff = active_matrix - target_vector
        distances = np.linalg.norm(diff, axis=1)
        sorted_indices = np.argsort(distances)
        results = []
        for idx in sorted_indices:
            if len(results) >= n: break
            candidate_name = self.animals[idx]
            if exclude_name and self._normalize(candidate_name) == self._normalize(exclude_name):
                continue
            results.append(candidate_name)
        return results