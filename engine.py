import pandas as pd
import numpy as np

class AkinatorEngine:
    """
    Akinator guessing engine (V2.3).

    Key logic principles:
    1.  Aggressive Temperature Decay:
        - 'select_question' temperature starts at 1.5 (high exploration)
          and decays rapidly to 0.01 (high exploitation) by Question 5.
    2.  Stricter Guessing Thresholds:
        - All guessing thresholds ('should_make_guess') are high
          to ensure extreme confidence before making a guess.
        - Minimum question count before any guess is 12.
    3.  Error Robustness:
        - 'update' function uses a soft contradiction penalty (0.05)
          to be more forgiving of single user mistakes.
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
            uniform_prior = np.ones(n, dtype=np.float32) / n
            
            initial_gains = self._compute_gains_batched(uniform_prior, self.allowed_feature_indices, batch_size=256)
            
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var[self.allowed_feature_indices]) + 1e-5)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.5)
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
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
        Uses an aggressive temperature decay (1.5 -> 0.01 by Q5) to move
        from exploration to exploitation quickly.
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
        
        # Q0=1.5, Q1=1.2, Q2=0.9, Q3=0.6, Q4=0.3, Q5=0.01 (min)
        temperature = max(0.01, 15 - (question_count * 5))
        
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
        
        if weighted_diffs[best_local_idx] > 0.35: 
            best_feat_idx = candidate_indices[best_local_idx]
            feature_name = self.feature_cols[best_feat_idx]
            return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                      
        return None, None

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        Determines if the engine is confident enough to make a guess.
        Uses "super-patient" logic: requires 12 questions minimum and has
        very high, tiered confidence thresholds that relax over time.
        """
        q_count = game_state['question_count']
        
        if q_count < 12: 
            return False, None, None
            
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
        
        cumulative_scores = game_state.get('cumulative_scores')
        is_consistent = False
        if cumulative_scores is not None and q_count > 0:
            top_animal_score = cumulative_scores[top_idx]
            plausible_mask = (probs > 0.001) & (~game_state['rejected_mask'])
            
            if np.sum(plausible_mask) > 1:
                avg_score = np.mean(cumulative_scores[plausible_mask])
                score_std = np.std(cumulative_scores[plausible_mask])
                
                # Stays at 1.0 std dev
                consistency_threshold = avg_score + (score_std * 1.0) 
            else:
                consistency_threshold = -np.inf
            
            if top_animal_score >= consistency_threshold:
                is_consistent = True
            else:
                print(f"[GUESS] REJECTED: {top_animal} prob={top_prob:.2f}, "
                      f"score={top_animal_score:.2f} below strict_threshold={consistency_threshold:.2f}")
        elif q_count > 0:
            is_consistent = True
        
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 3:
                return False, None, None
            else:
                game_state['continue_mode'] = False
                game_state['questions_since_last_guess'] = 0

        
        # Stricter thresholds
        # Q12-Q15: "Perfect"
        if q_count < 25:
            if top_prob > 0.995 and confidence_ratio > 900.0 and entropy < 0.005 and is_consistent:
                print(f"[Q{q_count}] PERFECT: prob={top_prob:.3f}, ratio={confidence_ratio:.0f}")
                game_state['has_made_initial_guess'] = True
                return True, top_animal, 'final'
            
        
        # Q22+ "Confident"
        elif q_count >= 22:
            if top_prob > 0.99 and confidence_ratio > 900.0 and entropy < 0.003 and is_consistent:
                print(f"[Q{q_count}] CONFIDENT: prob={top_prob:.3f}, ratio={confidence_ratio:.0f}")
                game_state['has_made_initial_guess'] = True
                return True, top_animal, 'final'

        # No forced guess, engine will continue until it gets it or until the user gets tired.
        
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