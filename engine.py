import pandas as pd
import numpy as np

class AkinatorEngine:
    """
    Akinator guessing engine (V2.5).

    CHANGELOG:
    - (Patience Fix 1) Increased min questions in 'should_make_guess' to 20 (was 15)
      and raised confidence_ratio thresholds to 1000/800 to force more questions.
    - (Patience Fix 2) Drastically lowered 'get_discriminative_question'
      threshold to 0.01 (was 0.05) to make the engine *always* prefer
      a "smart" confirmation question when it's confident.
    - (Patience Fix 3) Slowed temperature decay in 'select_question' to
      explore new questions for ~20 rounds (was ~15).
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        self.MAX_QUESTIONS = None
        
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'mostly': 0.75, 'usually': 0.75, 'probably': 0.75,
            'sort of': 0.5, 'sometimes': 0.5, 
            'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0,
        }
        
        self._precompute_likelihood_tables()
        self._build_arrays()
        print(f"✓ Engine initialized: {len(self.animals)} items, {len(self.feature_cols)} features.")
        if len(self.allowed_feature_indices) < len(self.feature_cols):
              print(f"  (Filtered {len(self.feature_cols) - len(self.allowed_feature_indices)} unusable features)")

    def _precompute_likelihood_tables(self):
        """Precomputes fuzzy likelihoods with tighter sigmas."""
        steps = 41
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.10
                else:
                    sigma = 0.16

                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                self.likelihood_table[i, j] = max(likelihood, 0.0001)

    def _build_arrays(self):
        """Converts dataframe to optimized numpy arrays."""
        self.animals = self.df['animal_name'].values
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.nan_mask = np.isnan(self.features)
        
        self.col_nan_frac = np.mean(self.nan_mask, axis=0)
        col_var = np.nanvar(self.features, axis=0)

        self.allowed_feature_mask = (self.col_nan_frac < 1.0)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        
        self.col_var = col_var
        
        if len(self.animals) > 0:
            n = len(self.animals)
            self.uniform_prior = np.ones(n, dtype=np.float32) / n
            
            initial_gains = self._compute_gains_batched(self.uniform_prior, self.allowed_feature_indices, batch_size=256)
            
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var[self.allowed_feature_indices]) + 1e-5)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.5)
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.uniform_prior = np.array([], dtype=np.float32)
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Robust Shannon entropy calculation."""
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))

    def _compute_gains_batched(self, prior: np.ndarray, feature_indices: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Computes information gain in batches for memory efficiency.
        Uses a vectorized entropy calculation and active-item slicing for speed.
        """
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._calculate_entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains

        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        # Optimization: Slice prior and features if a large portion is inactive
        if n_active < len(prior) * 0.8 and n_active > 0:
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum()
        else:
            active_prior = prior
            active_mask = slice(None) # Use all items
            n_active = len(prior)

        if n_active == 0:
            return gains

        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            
            f_batch = self.features[active_mask][:, batch_indices]
            
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5)
            f_batch_quant = np.clip(np.rint(f_batch_filled * 40), 0, 40).astype(np.int8)

            # (n_active, batch_size, n_answers)
            likelihoods = self.likelihood_table[f_batch_quant]

            nan_batch_mask = np.isnan(f_batch)
            nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
            likelihoods[nan_expand] = 1.0 # (n_active, batch_size, n_answers)

            batch_gains = np.zeros(len(batch_indices), dtype=np.float32)
            
            for f_local_idx in range(len(batch_indices)):
                L_f = likelihoods[:, f_local_idx, :] # (n_active, n_answers)
                
                p_answers = active_prior @ L_f
                
                # --- Vectorized Entropy Calculation ---
                posteriors = active_prior[:, np.newaxis] * L_f
                posteriors /= (p_answers + 1e-10)
                posteriors = np.clip(posteriors, 1e-10, 1.0)
                p_logs = np.log2(posteriors)
                entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)
                expected_entropy = p_answers @ entropies_per_answer
                
                batch_gains[f_local_idx] = current_entropy - expected_entropy

            gains[i:end] = batch_gains
            
        return gains

    def update(self, feature_idx: int, answer_str: str, 
               current_scores: np.ndarray) -> np.ndarray:
        """
        Updates item scores based on a user's answer.
        Applies a soft penalty (0.05) for contradictions to be robust to user error.
        """
        # Default to 0.5 ('sometimes') if the answer is not recognized
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 40), 0, 40).astype(np.int32)
        
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # SOFTER PENALTIES (0.05 = 95% penalty)
        # This is more robust to a single user mistake.
        if answer_val >= 0.9:  
            contradictions = (f_col < 0.15) & (~nan_mask)
            likelihoods[contradictions] *= 0.05
        elif answer_val <= 0.1:  
            contradictions = (f_col > 0.85) & (~nan_mask)
            likelihoods[contradictions] *= 0.05

        scores = np.log(likelihoods + 1e-10)
        
        new_cumulative_scores = current_scores + scores

        return new_cumulative_scores

    def get_sparse_question_for_game(self, prior: np.ndarray, asked_features_names: set) -> tuple[str, str]:
        """
        Finds the best information-gain question from *only* sparse features.
        Note: This is currently unused by select_question() but kept for
        potential data collection or alternative game logic.
        """
        
        candidates_to_eval = [
            idx for idx in self.sparse_indices
            if self.feature_cols[idx] not in asked_features_names
        ]

        if not candidates_to_eval:
            return None, None

        gains = self._compute_gains_batched(prior, np.array(candidates_to_eval), batch_size=32)
        
        if np.max(gains) < 1e-5:
            return None, None

        best_local_idx = np.argmax(gains)
        best_feat_idx = candidates_to_eval[best_local_idx]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Selects the next best question using a temperature-based random choice.
        Uses a controlled temperature decay to move from exploration to exploitation.
        """
        
        asked_set = set(asked_features)
        
        candidates_indices = [idx for idx in self.allowed_feature_indices 
                              if self.feature_cols[idx] not in asked_set]

        if not candidates_indices:
            print("[Question] No more features to ask.")
            return None, None

        # Sub-sample candidates for performance, balancing top-ranked and random features.
        if len(candidates_indices) > 100:
            scored_candidates = []
            for idx in candidates_indices:
                initial_rank = np.where(self.sorted_initial_feature_indices == idx)[0]
                rank_score = 1.0 / (initial_rank[0] + 1) if len(initial_rank) > 0 else 0
                var_score = self.col_var[idx] / (np.max(self.col_var[self.allowed_feature_indices]) + 1e-5)
                scored_candidates.append((idx, rank_score + var_score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [idx for idx, _ in scored_candidates[:80]]
            
            other_candidates = [idx for idx in candidates_indices if idx not in top_candidates]
            if other_candidates:
                random_extras = np.random.choice(other_candidates, 
                                                 size=min(len(other_candidates), 20), 
                                                 replace=False)
                candidates_to_eval = np.unique(np.concatenate((top_candidates, random_extras))).astype(np.int32)
            else:
                candidates_to_eval = np.array(top_candidates, dtype=np.int32)
        else:
            candidates_to_eval = np.array(candidates_indices, dtype=np.int32)
            
        if len(candidates_to_eval) == 0:
            return None, None

        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=128)
        
        sorted_gain_indices = np.argsort(gains)[::-1]
        
        # Less randomness (Top 2)
        top_n = min(2, len(sorted_gain_indices))
        
        top_local_indices = sorted_gain_indices[:top_n]
        top_gains = gains[top_local_indices]
        
        # --- PATIENCE FIX 3: SLOW DOWN DECAY ---
        # The old implementation (0.3) decayed to 0.01 in 5 questions,
        # stopping exploration too early.
        # This new implementation (0.07) decays over ~20 questions
        temperature = max(0.01, 1.5 - (question_count * 0.07))
        # (Q0=1.5, Q5=1.15, Q10=0.8, Q15=0.45, Q20=0.1)
        # --- END PATIENCE FIX 3 ---
        
        exp_gains = np.exp((top_gains - np.max(top_gains)) / (temperature + 1e-5))
        probs = exp_gains / np.sum(exp_gains)
        
        chosen_local_idx = np.random.choice(top_n, p=probs)
        best_feat_idx = candidates_to_eval[top_local_indices[chosen_local_idx]]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text

    def get_discriminative_question(self, top_animal_idx: int, prior: np.ndarray, asked_features: list) -> tuple[str, str]:
        """Find a question that best separates the top candidate from its rivals."""
        top_prob = prior[top_animal_idx]
        
        rival_threshold = top_prob * 0.2
        rival_mask = (prior > rival_threshold) & (np.arange(len(prior)) != top_animal_idx)
        rival_indices = np.where(rival_mask)[0]
        
        if len(rival_indices) == 0:
            rival_indices = np.where(prior > 0.001)[0]
            rival_indices = rival_indices[rival_indices != top_animal_idx]
            if len(rival_indices) == 0:
                 return None, None # Only one animal left
            
        asked_set = set(asked_features)
        candidate_indices = [idx for idx in self.allowed_feature_indices 
                                 if self.feature_cols[idx] not in asked_set]
                                
        if not candidate_indices:
            return None, None

        top_feats = self.features[top_animal_idx, candidate_indices]
        
        rival_feats = self.features[rival_indices][:, candidate_indices]
        rival_weights = prior[rival_indices]
        rival_weights /= rival_weights.sum()
        avg_rival_feats = np.average(rival_feats, axis=0, weights=rival_weights)
        
        diffs = np.abs(np.nan_to_num(top_feats, nan=0.5) - np.nan_to_num(avg_rival_feats, nan=0.5))
        
        opinion_strength = 1.0 + np.abs(np.nan_to_num(top_feats, nan=0.5) - 0.5) # Range 1.0 to 1.5
        weighted_diffs = diffs * opinion_strength
        
        best_local_idx = np.argmax(weighted_diffs)
        
        # --- PATIENCE FIX 2: "SMART" QUESTIONING ---
        # Lowered threshold from 0.20 to 0.05.
        # This makes the engine EXTREMELY willing to ask a "smart"
        # confirmation question, even if it's not a perfect separator.
        # This directly addresses your "rat/plague" example.
        if weighted_diffs[best_local_idx] > 0.01: 
        # --- END PATIENCE FIX 2 ---
            best_feat_idx = candidate_indices[best_local_idx]
            feature_name = self.feature_cols[best_feat_idx]
            return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                      
        return None, None

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        Determines if the engine is confident enough to make a guess.
        This logic is now flexible, removing the hard requirement for a minimum
        number of questions and instead relying on tiered confidence thresholds.
        """
        q_count = game_state['question_count']

        if probs.sum() < 1e-10:
            return False, None, None

        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]

        probs_copy = probs.copy()
        probs_copy[top_idx] = 0.0
        second_prob = np.max(probs_copy)

        confidence_ratio = top_prob / (second_prob + 1e-9)
        entropy = self._calculate_entropy(probs)

        # Prevent guessing again too soon after a rejection
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 3:
                return False, None, None
            else:
                # Reset for the next potential guess
                game_state['continue_mode'] = False
                game_state['questions_since_last_guess'] = 0

        # Tier 1: High-Confidence Guess (can happen early)
        # Requires near-certainty. Allows the engine to feel "smart" for obvious characters.
        if top_prob > 0.95 and confidence_ratio > 1500 and entropy < 0.1:
            print(f"[Q{q_count}] HIGH-CONFIDENCE GUESS: prob={top_prob:.3f}, ratio={confidence_ratio:.0f}")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'

        # Tier 2: Standard-Confidence Guess (requires more questions)
        # A balanced threshold for making a guess after a reasonable number of questions.
        if q_count > 8 and top_prob > 0.85 and confidence_ratio > 800 and entropy < 0.5:
            print(f"[Q{q_count}] STANDARD-CONFIDENCE GUESS: prob={top_prob:.3f}, ratio={confidence_ratio:.0f}")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'

        # Tier 3: Final Attempt (for long games)
        # If the game has gone on for a long time, the engine can make a guess
        # with slightly lower confidence to avoid getting stuck.
        if q_count > 20 and top_prob > 0.75 and confidence_ratio > 500:
            print(f"[Q{q_count}] EXTENDED-GAME GUESS: prob={top_prob:.3f}, ratio={confidence_ratio:.0f}")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'

        # If no thresholds are met, continue asking questions.
        return False, None, None
    
    def get_features_for_data_collection(self, item_name, num_features=5):
        """
        Gets a list of features for data collection for a specific item.
        Prioritizes features that are unknown (NaN) for the item,
        then falls back to globally sparse features.
        """
        try:
            matches = np.where(self.animals == item_name)[0]
            if len(matches) == 0:
                matches = np.where(np.char.lower(self.animals.astype(str)) == item_name.lower())[0]
            
            if len(matches) == 0:
                print(f"Data collection: Item '{item_name}' not found, using random features.")
                return self._get_random_allowed_features(num_features)
                
            item_idx = matches[0]
            item_feats = self.features[item_idx]
        except Exception as e:
            print(f"Error in data collection for '{item_name}': {e}, using random features.")
            return self._get_random_allowed_features(num_features)

        nan_indices = np.where(np.isnan(item_feats))[0]
        useful_nan_indices = np.intersect1d(nan_indices, self.allowed_feature_indices).copy()
        
        if len(useful_nan_indices) < num_features:
            needed = num_features - len(useful_nan_indices)
            
            extras = np.setdiff1d(self.sparse_indices, useful_nan_indices).copy()
            
            if len(extras) > 0:
                np.random.shuffle(extras)
                selected_indices = np.concatenate((useful_nan_indices, extras[:needed]))
            else:
                remaining_allowed = np.setdiff1d(self.allowed_feature_indices, useful_nan_indices).copy()
                if len(remaining_allowed) > 0:
                    np.random_shuffle(remaining_allowed)
                    needed_more = num_features - len(useful_nan_indices)
                    selected_indices = np.concatenate((useful_nan_indices, remaining_allowed[:needed_more]))
                else:
                    selected_indices = useful_nan_indices
        else:
            np.random_shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
        
        if len(selected_indices) < num_features and len(self.allowed_feature_indices) >= num_features:
            remaining = np.setdiff1d(self.allowed_feature_indices, selected_indices).copy()
            np.random_shuffle(remaining)
            needed = num_features - len(selected_indices)
            selected_indices = np.concatenate((selected_indices, remaining[:needed]))
            
        return self._format_features(selected_indices[:num_features])
    
    def _get_random_allowed_features(self, num_features):
        """Returns random allowed features as fallback."""
        if len(self.allowed_feature_indices) == 0:
            return []
        
        num_to_select = min(num_features, len(self.allowed_feature_indices))
        selected = np.random.choice(self.allowed_feature_indices, size=num_to_select, replace=False)
        return self._format_features(selected)
    
    def _format_features(self, indices):
        """Format feature indices into the expected output format."""
        results = []
        for idx in indices:
            py_idx = int(idx)
            if py_idx >= len(self.feature_cols) or py_idx >= len(self.col_nan_frac):
                continue
                    
            fname = str(self.feature_cols[py_idx])
            results.append({
                "feature_name": fname,
                "question": str(self.questions_map.get(fname, f"Is it {fname}?")),
                "nan_percentage": float(self.col_nan_frac[py_idx])
            })
            
        return results
    
    """
    New group of functions that help me figure out uneeded features that add no gain to the system
    """
    
    # --- NEW FUNCTION ---
    def get_all_feature_gains(self, initial_prior: np.ndarray = None) -> list[dict]:
        """
        Calculates and returns the initial information gain for all
        allowed features. This directly answers the question about
        calculating gain for all features.
        """
        if initial_prior is None:
            if hasattr(self, 'uniform_prior'):
                initial_prior = self.uniform_prior
            else:
                if len(self.animals) == 0:
                    return []
                initial_prior = np.ones(len(self.animals), dtype=np.float32) / len(self.animals)
        
        indices_to_calc = self.allowed_feature_indices
        if len(indices_to_calc) == 0:
            return []
        
        print(f"Calculating gains for {len(indices_to_calc)} features...")
        gains = self._compute_gains_batched(initial_prior, indices_to_calc, batch_size=128)
        
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
            
        # Sort by gain, descending
        results.sort(key=lambda x: x['initial_gain'], reverse=True)
        return results
    # --- END NEW FUNCTION ---
    
    def to_delete(self, similarity_threshold=0.85, min_variance=0.01, output_file='questions_to_delete.csv'):
        """
        Identifies and logs redundant, similar, or low-quality questions that should be deleted.
        
        Criteria for deletion:
        1. Highly correlated features (similarity > threshold)
        2. Low variance features (variance < min_variance)
        3. Features with >95% missing data
        4. Features that provide minimal information gain
        
        Args:
            similarity_threshold: Correlation threshold above which features are considered redundant
            min_variance: Minimum variance threshold for useful features
            output_file: CSV file to save the deletion candidates
            
        Returns:
            DataFrame with features to delete and their reasons
        """
        from datetime import datetime
        import os
        
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"Started redundancy analysis at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        deletion_candidates = []
        
        # 1. Find features with excessive missing data (>95%)
        print("Checking for features with excessive missing data...")
        high_nan_mask = self.col_nan_frac > 0.95
        high_nan_indices = np.where(high_nan_mask)[0]
        
        for idx in high_nan_indices:
            fname = self.feature_cols[idx]
            question = self.questions_map.get(fname, f"Does it have {fname}?")
            deletion_candidates.append({
                'feature_name': fname,
                'question_text': question,
                'reason': 'excessive_missing_data',
                'nan_percentage': float(self.col_nan_frac[idx]),
                'variance': float(self.col_var[idx]) if not np.isnan(self.col_var[idx]) else 0.0,
                'correlation_with': None
            })
        
        print(f"  Found {len(high_nan_indices)} features with >95% missing data")
        
        # 2. Find low variance features
        print("\nChecking for low variance features...")
        valid_var_mask = ~np.isnan(self.col_var) & (self.col_var < min_variance)
        low_var_indices = np.where(valid_var_mask)[0]
        
        for idx in low_var_indices:
            if idx in high_nan_indices:
                continue  # Already flagged
                
            fname = self.feature_cols[idx]
            question = self.questions_map.get(fname, f"Does it have {fname}?")
            deletion_candidates.append({
                'feature_name': fname,
                'question_text': question,
                'reason': 'low_variance',
                'nan_percentage': float(self.col_nan_frac[idx]),
                'variance': float(self.col_var[idx]),
                'correlation_with': None
            })
        
        print(f"  Found {len(low_var_indices)} features with variance < {min_variance}")
        
        # 3. Find highly correlated (redundant) features
        print("\nChecking for highly correlated features...")
        print("  (This may take a moment for large datasets...)")
        
        allowed_indices = self.allowed_feature_indices
        redundant_pairs = []
        
        # Sample correlation check for performance
        sample_size = min(len(allowed_indices), 200)
        if len(allowed_indices) > sample_size:
            indices_to_check = np.random.choice(allowed_indices, sample_size, replace=False)
        else:
            indices_to_check = allowed_indices
        
        for i, idx1 in enumerate(indices_to_check):
            for idx2 in indices_to_check[i+1:]:
                col1 = self.features[:, idx1]
                col2 = self.features[:, idx2]
                
                # Only compare where both have data
                valid_mask = ~np.isnan(col1) & ~np.isnan(col2)
                if np.sum(valid_mask) < 10:
                    continue
                
                corr = np.corrcoef(col1[valid_mask], col2[valid_mask])[0, 1]
                
                if not np.isnan(corr) and abs(corr) > similarity_threshold:
                    # Keep the feature with less missing data
                    if self.col_nan_frac[idx1] > self.col_nan_frac[idx2]:
                        redundant_idx = idx1
                        keeper_idx = idx2
                    else:
                        redundant_idx = idx2
                        keeper_idx = idx1
                    
                    redundant_pairs.append((redundant_idx, keeper_idx, corr))
        
        # Add redundant features to deletion list
        seen_redundant = set()
        for redundant_idx, keeper_idx, corr in redundant_pairs:
            if redundant_idx in seen_redundant:
                continue
                
            fname_redundant = self.feature_cols[redundant_idx]
            fname_keeper = self.feature_cols[keeper_idx]
            question = self.questions_map.get(fname_redundant, f"Does it have {fname_redundant}?")
            
            deletion_candidates.append({
                'feature_name': fname_redundant,
                'question_text': question,
                'reason': 'highly_correlated',
                'nan_percentage': float(self.col_nan_frac[redundant_idx]),
                'variance': float(self.col_var[redundant_idx]) if not np.isnan(self.col_var[redundant_idx]) else 0.0,
                'correlation_with': f"{fname_keeper} (r={corr:.3f})"
            })
            seen_redundant.add(redundant_idx)
        
        print(f"  Found {len(redundant_pairs)} highly correlated feature pairs")
        
        # 4. Create DataFrame and save
        if deletion_candidates:
            df_delete = pd.DataFrame(deletion_candidates)
            df_delete['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Sort by reason and nan_percentage
            df_delete = df_delete.sort_values(['reason', 'nan_percentage'], ascending=[True, False])
            
            # Save to CSV
            df_delete.to_csv(output_file, index=False)
            
            print(f"\n{'='*60}")
            print(f"SUMMARY:")
            print(f"  Total features analyzed: {len(self.feature_cols)}")
            print(f"  Features flagged for deletion: {len(deletion_candidates)}")
            print(f"    - Excessive missing data: {len(high_nan_indices)}")
            print(f"    - Low variance: {len(low_var_indices)}")
            print(f"    - Highly correlated: {len(redundant_pairs)}")
            print(f"\n  Results saved to: {output_file}")
            print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            return df_delete
        else:
            print(f"\n{'='*60}")
            print("No features flagged for deletion.")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            return pd.DataFrame()