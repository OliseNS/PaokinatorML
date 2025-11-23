import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import deque
from typing import Optional
import threading

class AkinatorEngine:
    """
    Akinator Engine V18.0 - SVD Optimized
    Uses Matrix Factorization (SVD) for O(N) scalability and noise reduction.
    """
    
    def __init__(self, df, feature_cols, questions_map, row_padding=200, col_padding=100):
        self.questions_map = questions_map
        self._rw_lock = _RWLock()
        
        self.active_feature_names = np.array(feature_cols, dtype=object)
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        self.feature_map = {name: i for i, name in enumerate(self.active_feature_names)}
        self.item_map = {}
        
        self.cap_rows = self.n_items + row_padding
        self.cap_cols = self.n_features + col_padding
        
        # Main storage (float16 for memory efficiency)
        self.features = np.full((self.cap_rows, self.cap_cols), np.nan, dtype=np.float16)
        self.animals = np.full(self.cap_rows, None, dtype=object)
        self.feature_cols_array = np.full(self.cap_cols, None, dtype=object)
        
        if self.n_features > 0:
            self.feature_cols_array[:self.n_features] = self.active_feature_names

        if not df.empty:
            names = df['animal_name'].values
            self.animals[:self.n_items] = names
            for idx, name in enumerate(names):
                self.item_map[self._normalize(name)] = idx
            if len(df.columns) > 1:
                self.features[:self.n_items, :self.n_features] = df[feature_cols].values.astype(np.float16)

        # State flags
        self._imputation_dirty = True
        self._imputed_cache = None
        self._stats_dirty = True
        
        # Incremental stats buffers
        self._col_sum = np.zeros(self.cap_cols, dtype=np.float64)
        self._col_sq_sum = np.zeros(self.cap_cols, dtype=np.float64)
        self._col_count = np.zeros(self.cap_cols, dtype=np.int32)
        
        # Batched updates
        self._pending_updates = deque(maxlen=1000)
        self._batch_threshold = 50
        
        # Pre-allocated buffers
        self._score_buffer = np.zeros(self.cap_rows, dtype=np.float32)
        self._var_buffer = np.zeros(self.cap_cols, dtype=np.float32)
        
        self._compute_initial_stats()
        
        # Fuzzy logic map
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float16)
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'probably': 0.75, 'mostly': 0.75, 'usually': 0.75,
            'somewhat': 0.5, 'sort of': 0.5, 'sometimes': 0.5, 'idk': 0.5, 'unknown': 0.5,
            'probably not': 0.25, 'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        print(f"✓ Engine Ready (SVD Mode): {self.n_items} items, {self.n_features} features")

    @staticmethod
    def _normalize(text: str) -> str:
        if not text: return ""
        clean = str(text).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        if clean.endswith("s") and not clean.endswith("ss") and len(clean) > 3:
            clean = clean[:-1]
        return clean

    def _get_imputed_features(self) -> np.ndarray:
        """
        Returns imputed feature matrix using Truncated SVD (Matrix Factorization).
        Significantly faster (O(N)) and more scalable than KNN (O(N^2)).
        """
        if not self._imputation_dirty and self._imputed_cache is not None:
            return self._imputed_cache
        
        with self._rw_lock.write():
            if not self._imputation_dirty and self._imputed_cache is not None:
                return self._imputed_cache
            
            active = self.features[:self.n_items, :self.n_features].copy()
            
            # Fast path: no NaNs
            nan_mask = np.isnan(active)
            if not nan_mask.any():
                self._imputed_cache = active.astype(np.float32)
                self._imputation_dirty = False
                return self._imputed_cache
            
            try:
                # 1. Initial Fill: Column Means
                # SVD requires a dense matrix to start. We fill holes with the average.
                col_means = np.nanmean(active, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.5)
                
                inds = np.where(nan_mask)
                active[inds] = np.take(col_means, inds[1])
                
                # 2. Matrix Factorization (SVD)
                # Captures latent concepts (e.g., "Aquatic", "Flying") instead of just row similarity
                n_components = min(40, self.n_features - 1, self.n_items - 1)
                
                if n_components > 1:
                    # randomized algorithm is faster for approximate results
                    svd = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=3)
                    reduced = svd.fit_transform(active)
                    reconstructed = svd.inverse_transform(reduced)
                    
                    # 3. Refine: Only update the originally missing values
                    # We trust explicit data more than the reconstruction
                    active[nan_mask] = reconstructed[nan_mask]
                
                active = np.clip(active, 0.01, 0.99)
                
            except Exception as e:
                print(f"[Engine] SVD Fallback: {e}")
                inds = np.where(np.isnan(active))
                active[inds] = 0.5

            self._imputed_cache = active.astype(np.float32)
            self._imputation_dirty = False
            return self._imputed_cache

    def _compute_initial_stats(self):
        if self.n_items == 0 or self.n_features == 0: return
        active = self.features[:self.n_items, :self.n_features].astype(np.float64)
        valid_mask = ~np.isnan(active)
        active_clean = np.where(valid_mask, active, 0.0)
        
        self._col_sum[:self.n_features] = np.sum(active_clean, axis=0)
        self._col_sq_sum[:self.n_features] = np.sum(active_clean ** 2, axis=0)
        self._col_count[:self.n_features] = np.sum(valid_mask, axis=0)
        self._update_derived_stats()

    def _update_derived_stats(self):
        n = self._col_count[:self.n_features].astype(np.float64)
        n = np.maximum(n, 1)
        
        mean = self._col_sum[:self.n_features] / n
        variance = (self._col_sq_sum[:self.n_features] / n) - (mean ** 2)
        variance = np.maximum(variance, 0)
        
        self.col_mean = mean.astype(np.float32)
        self.col_var = variance.astype(np.float32)
        self.allowed_feature_mask = (self.col_var > 1e-4)
        
        # Update cached feature order
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)
        if len(self.allowed_feature_indices) > 0:
            variances = self.col_var[self.allowed_feature_indices]
            sorted_idx = np.argsort(variances)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_idx]
        else:
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)
        
        self._stats_dirty = False

    def _incremental_stat_update(self, item_idx: int, feat_idx: int, old_val: float, new_val: float):
        old_valid, new_valid = not np.isnan(old_val), not np.isnan(new_val)
        
        if old_valid:
            self._col_sum[feat_idx] -= old_val
            self._col_sq_sum[feat_idx] -= old_val ** 2
            self._col_count[feat_idx] -= 1
        
        if new_valid:
            self._col_sum[feat_idx] += new_val
            self._col_sq_sum[feat_idx] += new_val ** 2
            self._col_count[feat_idx] += 1
        
        self._stats_dirty = True

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        clean_key = self._normalize(item_name)
        idx = self.item_map.get(clean_key)
        
        # Add new item
        if idx is None:
            if self.n_items >= self.cap_rows: return
            idx = self.n_items
            self.animals[idx] = item_name
            self.item_map[clean_key] = idx
            self.n_items += 1
            
            if self.n_features > 0:
                self.features[idx, :self.n_features] = self.col_mean[:self.n_features].astype(np.float16)
                # Quick stat update
                for f in range(self.n_features):
                    self._col_sum[f] += self.col_mean[f]
                    self._col_sq_sum[f] += self.col_mean[f] ** 2
                    self._col_count[f] += 1
            self._imputation_dirty = True

        # Add new feature
        f_idx = self.feature_map.get(feature_name)
        if f_idx is None:
            if self.n_features >= self.cap_cols: return
            f_idx = self.n_features
            self.feature_cols_array[f_idx] = feature_name
            self.feature_map[feature_name] = f_idx
            self.n_features += 1
            if question_text: self.questions_map[feature_name] = question_text
            self.features[:self.n_items, f_idx] = 0.5
            
            self._col_sum[f_idx] = 0.5 * self.n_items
            self._col_sq_sum[f_idx] = 0.25 * self.n_items
            self._col_count[f_idx] = self.n_items
            self._imputation_dirty = True

        # Update value
        old_val = float(self.features[idx, f_idx])
        self.features[idx, f_idx] = np.float16(value)
        self._incremental_stat_update(idx, f_idx, old_val, value)
        self._imputation_dirty = True
        
        self._pending_updates.append((idx, f_idx, value))
        if len(self._pending_updates) >= self._batch_threshold:
            self._flush_updates()

    def _flush_updates(self):
        if self._stats_dirty: self._update_derived_stats()
        self._pending_updates.clear()

    def recalculate_stats(self):
        self._flush_updates()

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        n = self.n_items
        if len(current_scores) < n:
            pad_len = n - len(current_scores)
            valid = current_scores[np.isfinite(current_scores)]
            base = np.mean(valid) if len(valid) > 0 else 0.0
            current_scores = np.concatenate([current_scores, np.full(pad_len, base, dtype=np.float32)])
        
        current_scores = current_scores[:n]
        answer_val = self.fuzzy_map.get(str(answer_str).lower(), 0.5)
        f_col = self._get_imputed_features()[:, feature_idx]
        
        # Bayesian Update (Vectorized)
        sigma_sq_2 = 2 * (0.22 ** 2)
        p_noise = 0.03
        
        np.abs(f_col - answer_val, out=self._score_buffer[:n])
        np.square(self._score_buffer[:n], out=self._score_buffer[:n])
        self._score_buffer[:n] /= -sigma_sq_2
        np.exp(self._score_buffer[:n], out=self._score_buffer[:n])
        
        self._score_buffer[:n] *= (1.0 - p_noise)
        self._score_buffer[:n] += p_noise
        np.log(np.clip(self._score_buffer[:n], 1e-9, None), out=self._score_buffer[:n])
        
        return current_scores + self._score_buffer[:n]

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple:
        """
        Optimized question selection using Entropy/Variance on SVD-imputed data.
        Strictly enforces exclusion of asked_features to prevent duplicates.
        """
        active_cols = self.feature_cols_array[:self.n_features]
        
        # STRICT Filter: O(1) lookup set
        asked_set = set(asked_features)
        
        # Identify target subset (top 90% mass or top 20 items)
        if prior is not None and len(prior) > 0:
            sorted_idx = np.argsort(prior)[::-1]
            cum_sum = np.cumsum(prior[sorted_idx])
            cutoff = np.searchsorted(cum_sum, 0.90)
            top_k = max(20, cutoff + 1)
            target_indices = sorted_idx[:top_k]
        else:
            target_indices = np.arange(self.n_items)
        
        # Calculate variance over the target subset
        target_features = self._get_imputed_features()[target_indices, :]
        np.var(target_features, axis=0, out=self._var_buffer[:self.n_features])
        
        # Create Mask: Must NOT be asked AND must have Variance
        # Using list comprehension is safe for string matching
        asked_mask = np.array([c in asked_set for c in active_cols], dtype=bool)
        candidate_mask = (~asked_mask) & self.allowed_feature_mask[:self.n_features]
        
        candidate_indices = np.where(candidate_mask)[0]
        
        if len(candidate_indices) == 0:
            return None, None
        
        scores = self._var_buffer[:self.n_features]
        
        # Selection Logic
        if question_count == 0:
            # Random start from high-variance features
            candidate_scores = scores[candidate_indices]
            top_n = min(25, len(candidate_indices))
            top_indices = np.argpartition(candidate_scores, -top_n)[-top_n:]
            best_idx = candidate_indices[np.random.choice(top_indices)]
        else:
            # Max variance (Max Entropy)
            best_idx = candidate_indices[np.argmax(scores[candidate_indices])]
        
        fname = str(active_cols[best_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def select_sparse_discovery_question(self, prior: np.ndarray, asked_features: list) -> tuple:
        if prior is None or len(prior) == 0: return None, None
        
        sorted_idx = np.argsort(prior)[::-1][:20]
        if len(sorted_idx) == 0: return None, None
        
        top_matrix = self._get_imputed_features()[sorted_idx, :]
        uncertainty = np.sum(np.abs(top_matrix - 0.5), axis=0)
        
        asked_set = set(asked_features)
        active_cols = self.feature_cols_array[:self.n_features]
        
        for i, c in enumerate(active_cols):
            if c in asked_set: uncertainty[i] = np.inf
        
        best_idx = np.argmin(uncertainty)
        avg_dev = uncertainty[best_idx] / len(sorted_idx)
        
        if avg_dev > 0.15: return None, None
        
        fname = str(active_cols[best_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple:
        if probs.sum() < 1e-10: return False, None, None
        
        probs = probs / probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        q_count = game_state.get('question_count', 0)
        continue_mode = game_state.get('continue_mode', False)
        q_since_last = game_state.get('questions_since_last_guess', 0)
        rejection_count = np.sum(game_state.get('rejected_mask', []))
        
        if q_count < 8: return False, None, None
        
        should_guess = False
        reason = ""
        
        if continue_mode:
            pacing_limit = min(3 + int(rejection_count), 7)
            if q_since_last < pacing_limit:
                remaining = self.n_features - len(game_state.get('asked_features', []))
                if remaining > 0: return False, None, None
                return True, top_animal, "exhausted_questions"
        
        if q_count >= 40: return True, top_animal, "hard_safety_net_40"
        
        if q_count >= 25:
            if top_prob > 0.40: should_guess, reason = True, "safety_net_25_confident"
            elif q_count >= 35: should_guess, reason = True, "safety_net_grace_period_ended"
        else:
            second_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0.001
            ratio = top_prob / (second_prob + 1e-9)
            if top_prob > 0.85 and ratio > 4.0: should_guess, reason = True, "high_confidence"
            elif top_prob > 0.95: should_guess, reason = True, "very_high_probability"
        
        if not should_guess:
            asked = game_state.get('asked_features', [])
            if len(asked) >= self.n_features - 1:
                should_guess, reason = True, "no_questions_left"
        
        return should_guess, top_animal if should_guess else None, reason

    @property
    def feature_cols(self):
        return self.feature_cols_array[:self.n_features]

    def get_features_for_data_collection(self, item_name, num_features=5):
        if len(self.allowed_feature_indices) == 0: return []
        variances = self.col_var[self.allowed_feature_indices]
        prob_dist = variances / (variances.sum() + 1e-10)
        selected = np.random.choice(
            self.allowed_feature_indices,
            size=min(num_features, len(self.allowed_feature_indices)),
            replace=False, p=prob_dist
        )
        return [{"feature_name": str(self.feature_cols_array[i]),
                 "question": self.questions_map.get(str(self.feature_cols_array[i]), f"Is it {self.feature_cols_array[i]}?")}
                for i in selected]

    def get_all_feature_gains(self) -> list:
        return sorted([{"feature_name": str(self.feature_cols_array[i]), "initial_gain": float(self.col_var[i])}
                       for i in self.allowed_feature_indices],
                      key=lambda x: x['initial_gain'], reverse=True)

    def build_feature_vector(self, user_answers: dict) -> np.ndarray:
        vector = np.full(self.n_features, 0.5, dtype=np.float32)
        for fname, val in user_answers.items():
            idx = self.feature_map.get(fname)
            if idx is not None and idx < self.n_features:
                vector[idx] = float(val)
        return vector

    def find_nearest_neighbors(self, target_vector: np.ndarray, exclude_name: str = None, n: int = 3) -> list:
        if self.n_items == 0: return []
        active_matrix = self._get_imputed_features()
        distances = np.linalg.norm(active_matrix - target_vector, axis=1)
        sorted_idx = np.argsort(distances)
        
        results = []
        exclude_clean = self._normalize(exclude_name) if exclude_name else None
        for idx in sorted_idx:
            if len(results) >= n: break
            name = self.animals[idx]
            if exclude_clean and self._normalize(name) == exclude_clean: continue
            results.append(name)
        return results


class _RWLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
    def read(self): return _RWLockReadContext(self)
    def write(self): return _RWLockWriteContext(self)

class _RWLockReadContext:
    def __init__(self, lock): self._lock = lock
    def __enter__(self):
        with self._lock._read_ready: self._lock._readers += 1
    def __exit__(self, *args):
        with self._lock._read_ready:
            self._lock._readers -= 1
            if self._lock._readers == 0: self._lock._read_ready.notify_all()

class _RWLockWriteContext:
    def __init__(self, lock): self._lock = lock
    def __enter__(self):
        self._lock._read_ready.acquire()
        while self._lock._readers > 0: self._lock._read_ready.wait()
    def __exit__(self, *args): self._lock._read_ready.release()