import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Real-time Padded Akinator Engine (V14.2 - Ultra-Strict Confidence).
    
    Updates:
    - should_make_guess: Now requires 99.5% confidence and 20x probability ratio.
    - should_make_guess: Hard limit set to 25 questions.
    - select_question: Randomizes Q1 (Top 5), then Greedy (Argmax).
    """
    
    def __init__(self, df, feature_cols, questions_map, row_padding=200, col_padding=100):
        self.questions_map = questions_map
        
        # Track active counts
        self.active_feature_names = np.array(feature_cols, dtype=object)
        self.n_items = len(df)
        self.n_features = len(feature_cols)
        
        # O(1) Lookups - NOW NORMALIZED to prevent duplicates
        self.feature_map = {name: i for i, name in enumerate(self.active_feature_names)}
        self.item_map = {} # Populated below
        
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
            
            # CRITICAL FIX: Normalize keys for deduplication
            for idx, name in enumerate(names):
                clean_key = str(name).strip().lower()
                self.item_map[clean_key] = idx
                
            if len(df.columns) > 1:
                existing_data = df[feature_cols].values.astype(np.float32)
                self.features[:self.n_items, :self.n_features] = existing_data

        # --- 4. Initial Math Prep ---
        print("   Imputing initial data block...")
        self._impute_active_block()
        
        # Stats setup 
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        # Improved fuzzy map
        self.fuzzy_map = {
            'yes': 1.0, 
            'y': 1.0, 
            
            'probably': 0.75, 
            'mostly': 0.75, 
            'usually': 0.75, 
            
            'somewhat': 0.5, 
            'sort of': 0.5, 
            'sometimes': 0.5, 
            'idk': 0.5, 
            'unknown': 0.5,
            
            'probably not': 0.25,  
            'not really': 0.25, 
            'rarely': 0.25, 
            
            'no': 0.0, 
            'n': 0.0
        }
        # Precomputations
        self._refresh_column_stats()
        self._recalc_correlation()
        self._compute_initial_priors()
        
        print(f"✓ Engine Ready. Active: {self.n_items} items, {self.n_features} features.")

    def _impute_active_block(self):
        """
        Fills gaps in knowledge. If 'projector' is new and missing data, 
        this infers it from similar items (like 'TV'), making it guessable immediately.
        """
        if self.n_items == 0 or self.n_features == 0: return
        
        active_view = self.features[:self.n_items, :self.n_features]
        
        # 1. Fill complete empty columns with uncertainty (0.5)
        nan_mask = np.isnan(active_view)
        all_nan_cols = nan_mask.all(axis=0)
        if all_nan_cols.any():
            active_view[:, all_nan_cols] = 0.5
            
        # 2. KNN Imputation for partial data
        if np.isnan(active_view).any():
            try:
                # increased k-neighbors for better generalization
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
            # Using numpy for speed
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
        
        # Features with very low variance (everyone matches) are useless
        self.allowed_feature_mask = (self.col_var > 1e-4)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

    def _compute_initial_priors(self):
        """Calculates the best Q1-Q3 questions (Global Variance)."""
        if self.n_items > 0 and len(self.allowed_feature_indices) > 0:
            # Just rank by variance for initial speed
            variances = self.col_var[self.allowed_feature_indices]
            sorted_indices = np.argsort(variances)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def smart_ingest_update(self, item_name: str, feature_name: str, value: float, question_text: str = None):
        """
        DEDUPLICATED Ingestion.
        Checks for 'Projector' via 'projector' key.
        """
        # 1. Sanitize Key
        clean_item_key = str(item_name).strip().lower()
        
        idx = self.item_map.get(clean_item_key)
        
        # 2. Add Item if New
        if idx is None:
            if self.n_items >= self.cap_rows:
                print(f"⚠️ Capacity Reached. Cannot add '{item_name}'.")
                return
            
            idx = self.n_items
            self.animals[idx] = item_name # Store Display Name
            self.item_map[clean_item_key] = idx # Store Clean Key
            self.n_items += 1
            
            # Initialize with average of existing items (Not 0.5, but Mean)
            # This helps it blend in immediately rather than being an outlier
            if self.n_features > 0:
                self.features[idx, :self.n_features] = self.col_mean[:self.n_features]
                
            print(f"   [Engine] +ITEM: '{item_name}' @ idx {idx}")

        # 3. Add Feature if New
        f_idx = self.feature_map.get(feature_name)
        if f_idx is None:
            if self.n_features >= self.cap_cols:
                return
            
            f_idx = self.n_features
            self.feature_cols_array[f_idx] = feature_name
            self.feature_map[feature_name] = f_idx
            self.n_features += 1
            
            if question_text:
                self.questions_map[feature_name] = question_text
            
            # Initialize new feature to 0.5 (Uncertainty)
            self.features[:self.n_items, f_idx] = 0.5
            print(f"   [Engine] +FEATURE: '{feature_name}' @ col {f_idx}")

        # 4. Update Value
        self.features[idx, f_idx] = value
        
        # Quick stats refresh if significant updates happen
        # (Optional optimization: only run every N updates)
        self._refresh_column_stats()

    def recalculate_stats(self):
        self._refresh_column_stats()
        self._recalc_correlation()
        self._compute_initial_priors()

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """
        ROBUST SCORING (The "Climbing" Logic).
        """
        # Resize scores if engine grew
        if len(current_scores) < self.n_items:
            padding_len = self.n_items - len(current_scores)
            valid_scores = current_scores[np.isfinite(current_scores)]
            base_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
            padding = np.full(padding_len, base_score, dtype=np.float32)
            current_scores = np.concatenate([current_scores, padding])
        
        current_scores = current_scores[:self.n_items]
        
        answer_val = self.fuzzy_map.get(str(answer_str).lower(), 0.5)
        
        # 1. Get Data
        f_col = self.features[:self.n_items, feature_idx]
        f_col_clean = np.nan_to_num(f_col, nan=0.5)
        
        # 2. Gaussian Likelihood
        sigma = 0.18
        distance = np.abs(f_col_clean - answer_val)
        gaussian_likelihood = np.exp(-0.5 * (distance / sigma) ** 2)
        
        # 3. NOISE FLOOR
        p_noise = 0.02
        final_likelihood = (1.0 - p_noise) * gaussian_likelihood + p_noise
        
        # 4. Boost Confidence
        if answer_val > 0.75:
             bonus_mask = (f_col_clean > 0.8)
             final_likelihood[bonus_mask] *= 1.25
        elif answer_val < 0.25:
             bonus_mask = (f_col_clean < 0.2)
             final_likelihood[bonus_mask] *= 1.25
             
        # 5. Convert to Log-Prob update
        log_update = np.log(np.clip(final_likelihood, 1e-9, None))
        
        return current_scores + log_update

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        SMART NARROWING with STOCHASTIC START.
        
        Logic:
        - If Question 0: Pick RANDOMLY from the Top 25 features (High Variety).
        - If Question > 0: GREEDY AF (Strict Argmax) to converge rapidly.
        """
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

        # --- SELECTION LOGIC START ---
        
        if question_count == 0:
            # RANDOM START: Widen pool to Top 25 for "Extreme Randomness"
            
            # 1. Get scores for candidates
            candidate_scores = scores[candidate_indices]
            
            # 2. Sort descending (indices relative to candidate_indices)
            sorted_local_indices = np.argsort(candidate_scores)[::-1]
            
            # 3. Define pool (Top 25 instead of Top 5)
            # This forces variety. It won't just ask the #1 best question every time.
            pool_size = min(len(sorted_local_indices), 25)
            
            if pool_size == 0: 
                return None, None
                
            # 4. Pick Randomly
            random_pool_index = np.random.choice(sorted_local_indices[:pool_size])
            best_candidate_idx = candidate_indices[random_pool_index]
            
        else:
            # GREEDY AF: Strict Argmax for maximum efficiency after Q1
            best_candidate_idx = candidate_indices[np.argmax(scores[candidate_indices])]
            
        # --- SELECTION LOGIC END ---
        
        fname = str(active_cols[best_candidate_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")
    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        if probs.sum() < 1e-10: return False, None, None
        
        probs = probs / probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        q_count = game_state['question_count']
        second_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0.001
        ratio = top_prob / (second_prob + 1e-9)
        
        should_guess = False
        reason = ""
        
        # 1. Hard Limit (Guess at 25 if not found yet)
        if q_count >= 25:
            should_guess = True
            reason = "max_questions_25"
            
        # 2. Ultra-Strict Confidence Rule (Regardless of Question Count)
        # We demand 99.5% certainty and a massive lead (20x ratio).
        # This ensures the engine is essentially never "overconfident".
        elif top_prob > 0.995 and ratio > 20.0:
            should_guess = True
            reason = "strict_confidence"
                
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
            replace=False,
            p=prob_dist
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
        """
        Builds a feature vector from a dictionary of user answers.
        Unanswered features are set to 0.5 (uncertainty).
        """
        vector = np.full(self.n_features, 0.5, dtype=np.float32)
        for fname, val in user_answers.items():
            idx = self.feature_map.get(fname)
            if idx is not None and idx < self.n_features:
                vector[idx] = float(val)
        return vector

    def find_nearest_neighbors(self, target_vector: np.ndarray, exclude_name: str = None, n: int = 3) -> list[str]:
        """
        Calculates Euclidean distance between target vector and all active items.
        Returns top N most similar item names.
        """
        if self.n_items == 0 or self.n_features == 0:
            return []

        # 1. Get Active Features (Handle NaNs by converting to 0.5)
        # shape: (n_items, n_features)
        active_matrix = np.nan_to_num(
            self.features[:self.n_items, :self.n_features], 
            nan=0.5
        )
        
        # 2. Calculate Distances (Euclidean)
        # broadcast target_vector across the matrix
        # diff shape: (n_items, n_features)
        diff = active_matrix - target_vector
        
        # dist shape: (n_items,)
        distances = np.linalg.norm(diff, axis=1)
        
        # 3. Sort
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices:
            if len(results) >= n:
                break
                
            candidate_name = self.animals[idx]
            
            # Skip if it's the item itself (case-insensitive check)
            if exclude_name and candidate_name.lower().strip() == exclude_name.lower().strip():
                continue
                
            results.append(candidate_name)
            
        return results