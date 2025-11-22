import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class AkinatorEngine:
    """
    Real-time Padded Akinator Engine (V15.0 - Logic Refinement).
    
    Major Refinements:
    - Weighted Variance: Selects questions that split the TOP candidates, not just global variance.
    - Pacing Controls: Strict Q8 floor, Dynamic continuation pacing (3-7 qs), Q25/Q40 safety nets.
    - Robustness: Handles exhausted question pools by forcing guesses.
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
            
            # Normalize keys for deduplication
            for idx, name in enumerate(names):
                clean_key = self._normalize(name)
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
            'yes': 1.0, 'y': 1.0, 
            'probably': 0.75, 'mostly': 0.75, 'usually': 0.75, 
            'somewhat': 0.5, 'sort of': 0.5, 'sometimes': 0.5, 'idk': 0.5, 'unknown': 0.5,
            'probably not': 0.25, 'not really': 0.25, 'rarely': 0.25, 
            'no': 0.0, 'n': 0.0
        }
        # Precomputations
        self._refresh_column_stats()
        self._recalc_correlation()
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
        active_view = self.features[:self.n_items, :self.n_features]
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
        """
        Ingests a single data point update in real-time.
        Handles dynamic resizing if capacity is reached.
        """
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
                self.features[idx, :self.n_features] = self.col_mean[:self.n_features]

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
        
        # NOTE: We do NOT call full stats refresh here for speed. 
        # The service layer calls recalculate_stats() after a batch.

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
        
        # Gaussian Likelihood (Sigma 0.22)
        sigma = 0.22
        distance = np.abs(f_col_clean - answer_val)
        gaussian_likelihood = np.exp(-0.5 * (distance / sigma) ** 2)
        
        p_noise = 0.01
        final_likelihood = (1.0 - p_noise) * gaussian_likelihood + p_noise
        
        log_update = np.log(np.clip(final_likelihood, 1e-9, None))
        return current_scores + log_update

    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Selects the best question using Weighted Variance (Information Gain approximation) 
        focused on the top probability candidates.
        """
        active_cols = self.feature_cols_array[:self.n_features]
        asked_mask = np.isin(active_cols, asked_features)
        
        candidate_indices = np.where((~asked_mask) & self.allowed_feature_mask)[0]
        if len(candidate_indices) == 0: 
            return None, None

        # --- SMART NARROWING ---
        if prior is not None and len(prior) > 0:
            # 1. Identify Target Set (The "Likely" Animals)
            # We sort candidates by probability and take the cumulative 95% mass,
            # or at least top 20 items to avoid tunnel vision.
            sorted_indices = np.argsort(prior)[::-1]
            cum_sum = np.cumsum(prior[sorted_indices])
            
            # Dynamic cutoff: Focus on the items that matter
            cutoff = np.argmax(cum_sum > 0.95)
            top_k = max(20, cutoff + 1)
            # Cap top_k to prevent processing too much data in late game
            top_k = min(top_k, 500) 
            
            target_indices = sorted_indices[:top_k]
            
            # 2. Extract Sub-Matrix for Target Set
            # Shape: (Items_subset, Features)
            target_features = self.features[target_indices, :]
            target_probs = prior[target_indices]
            
            # Normalize probs within this subset to act as weights
            weights = target_probs / (target_probs.sum() + 1e-10)
            weights = weights.reshape(-1, 1) # reshape for broadcasting
            
            # 3. Clean Data
            clean_features = np.nan_to_num(target_features, nan=0.5)
            
            # 4. Calculate Weighted Variance for each feature
            # Var = E[x^2] - (E[x])^2
            # E[x] = sum(p * x)
            weighted_mean = np.sum(clean_features * weights, axis=0)
            weighted_sq_mean = np.sum((clean_features ** 2) * weights, axis=0)
            weighted_variance = weighted_sq_mean - (weighted_mean ** 2)
            
            scores = weighted_variance
        else:
            # Fallback for Q1 (Uniform prior)
            target_features = self.features[:self.n_items, :]
            scores = np.var(np.nan_to_num(target_features, nan=0.5), axis=0)

        # Apply constraints
        # Ensure we only look at unasked candidates
        final_scores = scores[candidate_indices]
        
        # --- Correlation Reduction (Anti-Redundancy) ---
        # If we have asked questions, penalize features highly correlated with them
        # This prevents asking "Is it a dog?" after "Is it a canine?"
        asked_indices = np.where(asked_mask)[0]
        if len(asked_indices) > 0 and len(candidate_indices) > 0:
            # Limit check to avoid massive matrix ops
            limit = min(self.n_features, self.feature_correlation_matrix.shape[0])
            
            valid_cand = candidate_indices[candidate_indices < limit]
            valid_asked = asked_indices[asked_indices < limit]
            
            if len(valid_cand) > 0 and len(valid_asked) > 0:
                # Get max correlation with ANY asked question
                corr_sub = self.feature_correlation_matrix[valid_cand][:, valid_asked]
                max_corr = np.max(np.abs(corr_sub), axis=1)
                
                # Apply penalty: Reduce score of highly correlated features
                # Map back to final_scores indices
                mask_in_final = np.isin(candidate_indices, valid_cand)
                final_scores[mask_in_final] *= (1.0 - max_corr * 0.8) # 0.8 factor allows some overlap if variance is huge

        if question_count == 0:
            # Randomize start slightly to avoid same Q1 every time
            sorted_local_indices = np.argsort(final_scores)[::-1]
            pool_size = min(len(sorted_local_indices), 15)
            if pool_size == 0: return None, None
            random_pool_index = np.random.choice(sorted_local_indices[:pool_size])
            best_candidate_idx = candidate_indices[random_pool_index]
        else:
            # Pick the single best feature that splits the remaining probability mass
            best_local_idx = np.argmax(final_scores)
            best_candidate_idx = candidate_indices[best_local_idx]
            
        fname = str(active_cols[best_candidate_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    # --- SPARSE DISCOVERY METHOD ---
    def select_sparse_discovery_question(self, prior: np.ndarray, asked_features: list) -> tuple[str, str] | tuple[None, None]:
        """
        Finds a question where the top candidates have 'Unknown' (0.5) data.
        """
        if prior is None or len(prior) == 0: return None, None

        sorted_indices = np.argsort(prior)[::-1]
        top_k_indices = sorted_indices[:20]
        if len(top_k_indices) == 0: return None, None
            
        top_matrix = self.features[top_k_indices, :self.n_features]
        clean_matrix = np.nan_to_num(top_matrix, nan=0.5)
        uncertainty_score = np.abs(clean_matrix - 0.5)
        col_uncertainty_sum = np.sum(uncertainty_score, axis=0)
        
        active_cols = self.feature_cols_array[:self.n_features]
        asked_mask = np.isin(active_cols, asked_features)
        col_uncertainty_sum[asked_mask] = np.inf
        
        best_idx = np.argmin(col_uncertainty_sum)
        avg_deviation = col_uncertainty_sum[best_idx] / len(top_k_indices)
        
        if avg_deviation > 0.15: return None, None # Candidates know too much

        fname = str(active_cols[best_idx])
        return fname, self.questions_map.get(fname, f"Is it {fname}?")

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        Determines if the engine should make a guess based on:
        1. Game constraints (min questions).
        2. Continuation pacing (min questions since last rejection).
        3. Safety nets (max questions).
        4. Confidence thresholds.
        """
        if probs.sum() < 1e-10: return False, None, None
        
        # Normalize probabilities
        probs = probs / probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        q_count = game_state.get('question_count', 0)
        continue_mode = game_state.get('continue_mode', False)
        q_since_last_guess = game_state.get('questions_since_last_guess', 0)
        
        # Calculate rejection count
        rejection_count = np.sum(game_state.get('rejected_mask', []))
        
        # --- 1. INITIAL PHASE CONSTRAINT ---
        # Strictly no guesses before question 8
        if q_count < 8:
            return False, None, None

        should_guess = False
        reason = ""

        # --- 2. CONTINUATION PACING LOGIC ---
        # If user continued, force a pacing delay based on number of rejections.
        # Formula: min(3 + rejections, 7)
        if continue_mode:
            pacing_limit = min(3 + int(rejection_count), 7)
            if q_since_last_guess < pacing_limit:
                # Check if we literally have no questions left, then we MUST guess
                active_cols = self.feature_cols_array[:self.n_features]
                asked_features = game_state.get('asked_features', [])
                remaining = len(active_cols) - len(asked_features)
                if remaining > 0:
                    return False, None, None
                else:
                    return True, top_animal, "exhausted_questions"

        # --- 3. SAFETY NETS ---
        
        # Hard Cap: Question 40 - Force guess regardless of confidence to stop infinite play
        if q_count >= 40:
             return True, top_animal, "hard_safety_net_40"

        # Soft Cap: Question 25
        if q_count >= 25:
            # Grace Period Logic:
            # If we are slightly unsure (prob < 0.40), allow a grace period until Q35.
            # But if we are reasonably sure (> 0.40), just guess now.
            if top_prob > 0.40:
                should_guess = True
                reason = "safety_net_25_confident"
            elif q_count >= 35:
                 should_guess = True
                 reason = "safety_net_grace_period_ended"
        
        # --- 4. STANDARD CONFIDENCE CHECK ---
        else:
            second_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0.001
            ratio = top_prob / (second_prob + 1e-9)
            
            # Require higher confidence to interrupt early
            if top_prob > 0.85 and ratio > 4.0:
                should_guess = True
                reason = "high_confidence"
            elif top_prob > 0.95:
                should_guess = True
                reason = "very_high_probability"
                
        # --- 5. EXHAUSTED QUESTIONS CHECK ---
        # If logic says False, but we have no questions left to ask, we MUST guess.
        if not should_guess:
            active_cols = self.feature_cols_array[:self.n_features]
            asked_features = game_state.get('asked_features', [])
            # Simple check: if asked count is near feature count
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
        active_matrix = np.nan_to_num(self.features[:self.n_items, :self.n_features], nan=0.5)
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