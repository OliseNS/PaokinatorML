import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Dict
import threading

class AkinatorEngine:
    """
    Akinator Engine V19.0 - Light (Pure NumPy)
    Removed SVD in favor of Probabilistic Mean Imputation and O(1) Ops.
    """
    __slots__ = (
        'questions_map', 'lock', 'n_items', 'n_features', 
        'feature_map', 'item_map', 'cap_rows', 'cap_cols',
        'features', 'animals', 'feature_cols_array',
        '_stats_dirty', '_col_sum', '_col_sq_sum', '_col_count',
        'col_mean', 'col_var', 'allowed_feature_mask', 'allowed_feature_indices',
        '_pending_updates', '_batch_threshold',
        '_score_buffer', '_subset_buffer', 'answer_values', 'fuzzy_map'
    )

    def __init__(self, df, feature_cols, questions_map, row_padding=50, col_padding=20):
        self.questions_map = questions_map
        self.lock = threading.Lock()
        
        # Dimensions
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        self.cap_rows = self.n_items + row_padding
        self.cap_cols = self.n_features + col_padding
        
        # Mappings
        self.feature_map = {name: i for i, name in enumerate(feature_cols)}
        self.item_map = {} # Populated below
        
        # storage: float16 is crucial for memory reduction (2 bytes per cell)
        self.features = np.full((self.cap_rows, self.cap_cols), np.nan, dtype=np.float16)
        self.animals = [None] * self.cap_rows # List is faster/lighter than object array for resizing
        self.feature_cols_array = [None] * self.cap_cols
        
        # Initialize Metadata
        if self.n_features > 0:
            for i, name in enumerate(feature_cols):
                self.feature_cols_array[i] = name

        # Initialize Data
        if not df.empty:
            names = df['animal_name'].values
            for idx, name in enumerate(names):
                self.animals[idx] = name
                self.item_map[self._normalize(name)] = idx
            
            if len(df.columns) > 1:
                # Direct load
                self.features[:self.n_items, :self.n_features] = df[feature_cols].values.astype(np.float16)

        # Incremental Stats Buffers (float32 is sufficient)
        self._stats_dirty = True
        self._col_sum = np.zeros(self.cap_cols, dtype=np.float32)
        self._col_sq_sum = np.zeros(self.cap_cols, dtype=np.float32)
        self._col_count = np.zeros(self.cap_cols, dtype=np.int32)
        
        # Cached Derived Stats
        self.col_mean = np.zeros(self.cap_cols, dtype=np.float32)
        self.col_var = np.zeros(self.cap_cols, dtype=np.float32)
        self.allowed_feature_mask = np.zeros(self.cap_cols, dtype=bool)
        self.allowed_feature_indices = np.array([], dtype=np.int32)

        # Update Queue
        self._pending_updates = deque(maxlen=1000)
        self._batch_threshold = 50
        
        # Pre-allocated buffers for vector ops
        self._score_buffer = np.zeros(self.cap_rows, dtype=np.float32)
        
        self._compute_initial_stats()
        
        # Fuzzy Logic Constants
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float16)
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'probably': 0.75, 'mostly': 0.75, 'usually': 0.75,
            'somewhat': 0.5, 'sort of': 0.5, 'sometimes': 0.5, 'idk': 0.5, 'unknown': 0.5,
            'probably not': 0.25, 'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        print(f"✓ Engine Ready (Light Mode): {self.n_items} items, {self.n_features} features")

    @staticmethod
    def _normalize(text: str) -> str:
        if not text: return ""
        clean = str(text).strip().lower().translate(str.maketrans('', '', ' -_'))
        if clean.endswith("s") and not clean.endswith("ss") and len(clean) > 3:
            return clean[:-1]
        return clean

    def _compute_initial_stats(self):
        if self.n_items == 0 or self.n_features == 0: return
        
        # Process chunks to save memory if needed, but here we do full slice
        # nanmean/nanvar are slow, so we use manual sum/count
        active = self.features[:self.n_items, :self.n_features]
        valid_mask = ~np.isnan(active)
        
        # Zero out NaNs for summation without allocation
        # We assume active is a view, so we copy to cast to float32 safely
        safe_active = np.where(valid_mask, active, 0).astype(np.float32)
        
        self._col_sum[:self.n_features] = np.sum(safe_active, axis=0)
        self._col_sq_sum[:self.n_features] = np.sum(safe_active ** 2, axis=0)
        self._col_count[:self.n_features] = np.sum(valid_mask, axis=0)
        self._update_derived_stats()

    def _update_derived_stats(self):
        """O(F) update of means and variances."""
        n = self._col_count[:self.n_features]
        mask = n > 0
        
        # Avoid divide by zero
        div_n = np.where(mask, n, 1.0).astype(np.float32)
        
        self.col_mean[:self.n_features] = 0.5 # Default prior
        np.divide(self._col_sum[:self.n_features], div_n, out=self.col_mean[:self.n_features], where=mask)
        
        # Var = E[X^2] - (E[X])^2
        mean_sq = (self._col_sq_sum[:self.n_features] / div_n)
        var = mean_sq - (self.col_mean[:self.n_features] ** 2)
        self.col_var[:self.n_features] = np.maximum(var, 0) # Clip negative noise
        
        self.allowed_feature_mask[:self.n_features] = (self.col_var[:self.n_features] > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask[:self.n_features])[0].astype(np.int32)
        self._stats_dirty = False

    def _incremental_stat_update(self, feat_idx: int, old_val: float, new_val: float):
        old_nan = np.isnan(old_val)
        new_nan = np.isnan(new_val)
        
        if not old_nan:
            self._col_sum[feat_idx] -= old_val
            self._col_sq_sum[feat_idx] -= old_val ** 2
            self._col_count[feat_idx] -= 1
        
        if not new_nan:
            self._col_sum[feat_idx] += new_val
            self._col_sq_sum[feat_idx] += new_val ** 2
            self._col_count[feat_idx] += 1
            
        self._stats_dirty = True

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        clean_key = self._normalize(item_name)
        idx = self.item_map.get(clean_key)
        
        # Add new item
        if idx is None:
            if self.n_items >= self.cap_rows: return # Reached capacity
            with self.lock:
                idx = self.n_items
                self.animals[idx] = item_name
                self.item_map[clean_key] = idx
                self.n_items += 1
                # Initialize with NaNs (Already done by initialization)

        # Add new feature
        f_idx = self.feature_map.get(feature_name)
        if f_idx is None:
            if self.n_features >= self.cap_cols: return
            with self.lock:
                f_idx = self.n_features
                self.feature_cols_array[f_idx] = feature_name
                self.feature_map[feature_name] = f_idx
                self.n_features += 1
                if question_text: 
                    self.questions_map[feature_name] = question_text
                # New feature has no data yet, counts are 0

        # Update value
        val_f16 = np.float16(value)
        old_val = self.features[idx, f_idx]
        
        if old_val != val_f16 and not (np.isnan(old_val) and np.isnan(val_f16)):
            self.features[idx, f_idx] = val_f16
            self._incremental_stat_update(f_idx, float(old_val), float(value))
            
            self._pending_updates.append(1)
            if len(self._pending_updates) >= self._batch_threshold:
                self._update_derived_stats()
                self._pending_updates.clear()

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """
        Vectorized Bayesian Update.
        Missing values (NaN) in data act as weak priors (using column mean).
        """
        n = self.n_items
        if len(current_scores) < n:
            # Resize scores on the fly
            pad = np.full(n - len(current_scores), np.mean(current_scores) if len(current_scores) else 0.0)
            current_scores = np.concatenate([current_scores, pad])
        
        current_scores = current_scores[:n]
        answer_val = self.fuzzy_map.get(str(answer_str).lower(), 0.5)
        
        # Fetch raw column
        f_col = self.features[:n, feature_idx]
        
        # IMPUTATION ON THE FLY:
        # Instead of SVD, use Column Mean for NaNs. 
        # This is O(N) and statistically neutral.
        col_mean = self.col_mean[feature_idx]
        
        # Prepare buffer
        diffs = self._score_buffer[:n]
        
        # Logic: If cell is NaN, assume it's the average value for that feature
        # This works because np.nan_to_num is extremely fast
        np.nan_to_num(f_col, copy=False, nan=col_mean) 
        # Note: copy=False might modify array if not careful, 
        # but f_col is a view. To be safe with float16->float32 ops:
        
        # Safe vectorized approach:
        # 1. Calculate abs diff
        diffs[:] = f_col
        mask = np.isnan(f_col)
        diffs[mask] = col_mean # Fill NaNs in buffer
        
        diffs -= answer_val
        np.abs(diffs, out=diffs)
        
        # 2. Gaussian Score
        # sigma_sq_2 = 2 * (0.22 ** 2) approx 0.0968
        diffs *= diffs
        diffs /= -0.0968
        np.exp(diffs, out=diffs)
        
        # 3. Noise & Log
        # p_noise = 0.03
        diffs *= 0.97
        diffs += 0.03
        np.log(diffs, out=diffs)
        
        return current_scores + diffs

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple:
        """
        Selects question based on Variance of the subset of top candidates.
        Uses np.nanvar to handle missing data without explicit imputation.
        """
        if self._stats_dirty: self._update_derived_stats()
        
        # 1. Identify Top Candidates (Subset)
        if prior is not None and len(prior) > 0:
            # Get top 25 items or top 90% mass
            # Faster approach: partition
            k = min(25, self.n_items)
            # argpartition is O(N), argsort is O(N log N)
            top_indices = np.argpartition(prior, -k)[-k:]
        else:
            top_indices = np.arange(self.n_items)

        # 2. Calculate Variance on Subset
        # This effectively asks: "Which feature splits these specific animals best?"
        sub_matrix = self.features[top_indices, :self.n_features]
        
        # nanvar is slightly slower than var, but robust.
        # axis=0 computes var for each column
        # O(K * F) where K is small (25). Very fast.
        variances = np.nanvar(sub_matrix, axis=0)
        
        # 3. Masking
        # asked_features is a list of strings, convert to set for O(1)
        asked_set = set(asked_features)
        
        best_idx = -1
        best_var = -1.0
        
        # Iterate only allowed features (skip empty ones)
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

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple:
        """Same logic, optimized reads."""
        if probs.sum() < 1e-9: return False, None, None
        
        # Quick check max probability without full sort
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx] / probs.sum()
        
        q_count = game_state.get('question_count', 0)
        
        if q_count < 8: return False, None, None
        
        # Fast path for high confidence
        if top_prob > 0.95: 
            return True, self.animals[top_idx], "very_high_probability"
            
        # Detailed check
        sorted_idx = np.argsort(probs)[::-1]
        top_prob = probs[sorted_idx[0]] / probs.sum() # Recalc normalized
        second = probs[sorted_idx[1]] / probs.sum() if len(sorted_idx) > 1 else 0.001
        
        if q_count >= 25 and top_prob > 0.40: 
            return True, self.animals[sorted_idx[0]], "safety_net"
            
        ratio = top_prob / (second + 1e-9)
        if top_prob > 0.85 and ratio > 4.0:
            return True, self.animals[sorted_idx[0]], "high_confidence"
            
        return False, None, None

    def get_features_for_data_collection(self, item_name, num_features=5):
        """Weighted random choice based on variance (Entropy)."""
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

    @property
    def feature_cols(self):
        return [x for x in self.feature_cols_array if x is not None]