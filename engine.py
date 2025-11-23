import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Dict
import threading

class AkinatorEngine:
    """
    Akinator Engine V19.1 - Light (Pure NumPy)
    Restored Service Compatibility (Sparse Questions, Neighbor Search).
    """
    __slots__ = (
        'questions_map', 'lock', 'n_items', 'n_features', 
        'feature_map', 'item_map', 'cap_rows', 'cap_cols',
        'features', 'animals', 'feature_cols_array',
        '_stats_dirty', '_col_sum', '_col_sq_sum', '_col_count',
        'col_mean', 'col_var', 'allowed_feature_mask', 
        'allowed_feature_indices', 'sorted_initial_feature_indices',
        '_pending_updates', '_batch_threshold',
        '_score_buffer', 'answer_values', 'fuzzy_map'
    )

    def __init__(self, df, feature_cols, questions_map, row_padding=50, col_padding=20):
        self.questions_map = questions_map
        self.lock = threading.Lock()
        
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        self.cap_rows = self.n_items + row_padding
        self.cap_cols = self.n_features + col_padding
        
        self.feature_map = {name: i for i, name in enumerate(feature_cols)}
        self.item_map = {} 
        
        # Storage: float16 (2 bytes)
        self.features = np.full((self.cap_rows, self.cap_cols), np.nan, dtype=np.float16)
        self.animals = [None] * self.cap_rows 
        self.feature_cols_array = [None] * self.cap_cols
        
        if self.n_features > 0:
            for i, name in enumerate(feature_cols):
                self.feature_cols_array[i] = name

        if not df.empty:
            names = df['animal_name'].values
            for idx, name in enumerate(names):
                self.animals[idx] = name
                self.item_map[self._normalize(name)] = idx
            
            if len(df.columns) > 1:
                self.features[:self.n_items, :self.n_features] = df[feature_cols].values.astype(np.float16)

        # Stats Buffers (float32)
        self._stats_dirty = True
        self._col_sum = np.zeros(self.cap_cols, dtype=np.float32)
        self._col_sq_sum = np.zeros(self.cap_cols, dtype=np.float32)
        self._col_count = np.zeros(self.cap_cols, dtype=np.int32)
        
        # Derived Stats
        self.col_mean = np.zeros(self.cap_cols, dtype=np.float32)
        self.col_var = np.zeros(self.cap_cols, dtype=np.float32)
        self.allowed_feature_mask = np.zeros(self.cap_cols, dtype=bool)
        self.allowed_feature_indices = np.array([], dtype=np.int32)
        self.sorted_initial_feature_indices = np.array([], dtype=np.int32) # Required by main.py stats

        self._pending_updates = deque(maxlen=1000)
        self._batch_threshold = 50
        
        # Pre-allocated buffer
        self._score_buffer = np.zeros(self.cap_rows, dtype=np.float32)
        
        self._compute_initial_stats()
        
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float16)
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'probably': 0.75, 'mostly': 0.75, 'usually': 0.75,
            'somewhat': 0.5, 'sort of': 0.5, 'sometimes': 0.5, 'idk': 0.5, 'unknown': 0.5,
            'probably not': 0.25, 'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        print(f"✓ Engine Ready (Light Mode V2): {self.n_items} items, {self.n_features} features")

    @staticmethod
    def _normalize(text: str) -> str:
        if not text: return ""
        clean = str(text).strip().lower().translate(str.maketrans('', '', ' -_'))
        if clean.endswith("s") and not clean.endswith("ss") and len(clean) > 3:
            return clean[:-1]
        return clean

    def _compute_initial_stats(self):
        if self.n_items == 0 or self.n_features == 0: return
        active = self.features[:self.n_items, :self.n_features]
        valid_mask = ~np.isnan(active)
        safe_active = np.where(valid_mask, active, 0).astype(np.float32)
        
        self._col_sum[:self.n_features] = np.sum(safe_active, axis=0)
        self._col_sq_sum[:self.n_features] = np.sum(safe_active ** 2, axis=0)
        self._col_count[:self.n_features] = np.sum(valid_mask, axis=0)
        self._update_derived_stats()

    def _update_derived_stats(self):
        n = self._col_count[:self.n_features]
        mask = n > 0
        div_n = np.where(mask, n, 1.0).astype(np.float32)
        
        self.col_mean[:self.n_features] = 0.5
        np.divide(self._col_sum[:self.n_features], div_n, out=self.col_mean[:self.n_features], where=mask)
        
        mean_sq = (self._col_sq_sum[:self.n_features] / div_n)
        var = mean_sq - (self.col_mean[:self.n_features] ** 2)
        self.col_var[:self.n_features] = np.maximum(var, 0)
        
        self.allowed_feature_mask[:self.n_features] = (self.col_var[:self.n_features] > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask[:self.n_features])[0].astype(np.int32)
        
        # Create sorted indices for performance stats in main.py
        if len(self.allowed_feature_indices) > 0:
            variances = self.col_var[self.allowed_feature_indices]
            sorted_idx = np.argsort(variances)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_idx]
        else:
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

        self._stats_dirty = False

    def _incremental_stat_update(self, feat_idx: int, old_val: float, new_val: float):
        if not np.isnan(old_val):
            self._col_sum[feat_idx] -= old_val
            self._col_sq_sum[feat_idx] -= old_val ** 2
            self._col_count[feat_idx] -= 1
        
        if not np.isnan(new_val):
            self._col_sum[feat_idx] += new_val
            self._col_sq_sum[feat_idx] += new_val ** 2
            self._col_count[feat_idx] += 1
        self._stats_dirty = True

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        clean_key = self._normalize(item_name)
        idx = self.item_map.get(clean_key)
        
        if idx is None:
            if self.n_items >= self.cap_rows: return
            with self.lock:
                idx = self.n_items
                self.animals[idx] = item_name
                self.item_map[clean_key] = idx
                self.n_items += 1

        f_idx = self.feature_map.get(feature_name)
        if f_idx is None:
            if self.n_features >= self.cap_cols: return
            with self.lock:
                f_idx = self.n_features
                self.feature_cols_array[f_idx] = feature_name
                self.feature_map[feature_name] = f_idx
                self.n_features += 1
                if question_text: self.questions_map[feature_name] = question_text

        val_f16 = np.float16(value)
        old_val = self.features[idx, f_idx]
        
        if old_val != val_f16 and not (np.isnan(old_val) and np.isnan(val_f16)):
            self.features[idx, f_idx] = val_f16
            self._incremental_stat_update(f_idx, float(old_val), float(value))
            
            self._pending_updates.append(1)
            if len(self._pending_updates) >= self._batch_threshold:
                self._update_derived_stats()
                self._pending_updates.clear()

    # Service.py expects this alias
    def recalculate_stats(self):
        self._update_derived_stats()
        self._pending_updates.clear()

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        n = self.n_items
        if len(current_scores) < n:
            pad = np.full(n - len(current_scores), np.mean(current_scores) if len(current_scores) else 0.0)
            current_scores = np.concatenate([current_scores, pad])
        current_scores = current_scores[:n]
        
        answer_val = self.fuzzy_map.get(str(answer_str).lower(), 0.5)
        f_col = self.features[:n, feature_idx]
        col_mean = self.col_mean[feature_idx]
        
        diffs = self._score_buffer[:n]
        
        # Vectorized safe fill
        np.copyto(diffs, f_col) # Copy raw data (with NaNs) to buffer
        mask = np.isnan(diffs)
        diffs[mask] = col_mean  # Impute on fly
        
        diffs -= answer_val
        np.abs(diffs, out=diffs)
        
        diffs *= diffs
        diffs /= -0.0968 # 2 * 0.22^2
        np.exp(diffs, out=diffs)
        
        diffs *= 0.97
        diffs += 0.03
        np.log(diffs, out=diffs)
        
        return current_scores + diffs

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple:
        if self._stats_dirty: self._update_derived_stats()
        
        if prior is not None and len(prior) > 0:
            k = min(25, self.n_items)
            top_indices = np.argpartition(prior, -k)[-k:]
        else:
            top_indices = np.arange(self.n_items)

        sub_matrix = self.features[top_indices, :self.n_features]
        variances = np.nanvar(sub_matrix, axis=0)
        
        asked_set = set(asked_features)
        best_idx = -1
        best_var = -1.0
        
        for idx in self.allowed_feature_indices:
            fname = self.feature_cols_array[idx]
            if fname in asked_set: continue
            
            var = variances[idx]
            if var > best_var:
                best_var = var
                best_idx = idx
        
        if best_idx == -1: return None, None
        fname = self.feature_cols_array[best_idx]
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    # --- RESTORED METHOD ---
    def select_sparse_discovery_question(self, prior: np.ndarray, asked_features: list) -> tuple:
        """
        Selects a question to fill knowledge gaps (NaNs/0.5s) for top candidates.
        """
        if prior is None or len(prior) == 0: return None, None
        
        # 1. Get Top Candidates
        sorted_idx = np.argsort(prior)[::-1][:20]
        if len(sorted_idx) == 0: return None, None
        
        # 2. Calculate Uncertainty on Subset
        # In Light mode, uncertainty is high if value is NaN or close to 0.5
        top_matrix = self.features[sorted_idx, :self.n_features]
        
        # Fill NaNs with 0.5 so we can just check distance from 0.5
        filled_matrix = np.nan_to_num(top_matrix, nan=0.5)
        
        # High "uncertainty score" = Value is close to 0.5
        # Score = 1.0 - 2*|val - 0.5|. If val is 0.5, score is 1.0. If val is 1.0, score is 0.
        uncertainty_scores = 1.0 - (2.0 * np.abs(filled_matrix - 0.5))
        total_uncertainty = np.sum(uncertainty_scores, axis=0)
        
        asked_set = set(asked_features)
        
        # Find feature with highest uncertainty that hasn't been asked
        best_idx = -1
        max_uncertainty = -1.0
        
        for idx in self.allowed_feature_indices:
            fname = self.feature_cols_array[idx]
            if fname in asked_set: continue
            
            score = total_uncertainty[idx]
            if score > max_uncertainty:
                max_uncertainty = score
                best_idx = idx
                
        if best_idx == -1: return None, None
        
        # Check if it's actually worth asking (average uncertainty > threshold)
        avg_dev = max_uncertainty / len(sorted_idx)
        if avg_dev < 0.15: return None, None # Data is already pretty clean
        
        fname = self.feature_cols_array[best_idx]
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple:
        if probs.sum() < 1e-9: return False, None, None
        
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx] / probs.sum()
        q_count = game_state.get('question_count', 0)
        
        if q_count < 8: return False, None, None
        
        if top_prob > 0.95: 
            return True, self.animals[top_idx], "very_high_probability"
            
        sorted_idx = np.argsort(probs)[::-1]
        top_prob = probs[sorted_idx[0]] / probs.sum()
        second = probs[sorted_idx[1]] / probs.sum() if len(sorted_idx) > 1 else 0.001
        
        if q_count >= 25 and top_prob > 0.40: 
            return True, self.animals[sorted_idx[0]], "safety_net"
            
        ratio = top_prob / (second + 1e-9)
        if top_prob > 0.85 and ratio > 4.0:
            return True, self.animals[sorted_idx[0]], "high_confidence"
            
        return False, None, None

    # --- RESTORED HELPERS ---
    
    @property
    def feature_cols(self):
        return [x for x in self.feature_cols_array if x is not None]

    def get_features_for_data_collection(self, item_name, num_features=5):
        if len(self.allowed_feature_indices) == 0: return []
        variances = self.col_var[self.allowed_feature_indices]
        total = variances.sum()
        if total <= 0: return []
        p = variances / total
        chosen = np.random.choice(self.allowed_feature_indices, 
                                size=min(num_features, len(self.allowed_feature_indices)), 
                                replace=False, p=p)
        return [{"feature_name": self.feature_cols_array[i],
                 "question": self.questions_map.get(self.feature_cols_array[i], "?")}
                for i in chosen]

    def get_all_feature_gains(self) -> list:
        # Used by admin stats
        return sorted([{"feature_name": str(self.feature_cols_array[i]), "initial_gain": float(self.col_var[i])}
                       for i in self.allowed_feature_indices],
                      key=lambda x: x['initial_gain'], reverse=True)

    def build_feature_vector(self, user_answers: dict) -> np.ndarray:
        # Used by report generator
        vector = np.full(self.n_features, 0.5, dtype=np.float32)
        for fname, val in user_answers.items():
            idx = self.feature_map.get(fname)
            if idx is not None and idx < self.n_features:
                vector[idx] = float(val)
        return vector

    def find_nearest_neighbors(self, target_vector: np.ndarray, exclude_name: str = None, n: int = 3) -> list:
        # Used by report generator
        if self.n_items == 0: return []
        
        # Impute active data on the fly for distance calc
        active_matrix = np.nan_to_num(self.features[:self.n_items, :self.n_features], nan=0.5)
        
        # Euclidean distance
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