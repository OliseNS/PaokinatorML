import pandas as pd
import numpy as np

class AkinatorEngine:
    """
    REFACTORED: Fuzzy Logic Expert System Engine.
    
    This engine implements the new logic:
    1.  Driven by Information Gain on fuzzy-valued features.
    2.  'update' is stateless: it accepts scores, calculates a new score,
        and returns the result. It does NOT store probabilities.
    3.  'should_make_guess' is also stateless: it receives the game_state
        (for scores/counts) and the on-the-fly probabilities (for confidence)
        from the service.
    4.  Confidence thresholds are high to prevent "eager" guesses.
    5.  Allows sparse features to be asked during the game.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        # Convert feature_cols to a numpy array for faster lookups
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        self.MAX_QUESTIONS = 35 # Increased for a more patient game
        
        # The 5 fuzzy answer choices
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        # Map from client-side answers to fuzzy values
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'mostly': 0.75, 'usually': 0.75, 'probably': 0.75,
            'sort of': 0.5, 'sometimes': 0.5, 'maybe': 0.5, 'idk': 0.5,
            'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0,
        }

        self.sparse_questions_asked = set()
        
        self._precompute_likelihood_tables()
        self._build_arrays()
        print(f"✓ Engine initialized: {len(self.animals)} items, {len(self.feature_cols)} features.")
        if len(self.allowed_feature_indices) < len(self.feature_cols):
             print(f"   (Filtered {len(self.feature_cols) - len(self.allowed_feature_indices)} unusable features)")

    def _precompute_likelihood_tables(self):
        """
        Precomputes the fuzzy likelihoods for all possible
        feature_value vs. answer_value combinations.
        
        Uses tight sigmas for high confidence.
        """
        steps = 41 # 0.0, 0.025, ..., 1.0
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                
                # 'Yes'/'No' answers are high-confidence (tight curve)
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.12  # Very strict
                # Mid-answers are lower-confidence (wider curve)
                else:
                    sigma = 0.18  # Strict

                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                
                # Floor to prevent true zero, but keep it very low
                self.likelihood_table[i, j] = max(likelihood, 0.0001)

    def _build_arrays(self):
        """Converts dataframe to optimized numpy arrays."""
        self.animals = self.df['animal_name'].values
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.nan_mask = np.isnan(self.features)
        
        # Pre-calculate column stats
        self.col_nan_frac = np.mean(self.nan_mask, axis=0)
        col_var = np.nanvar(self.features, axis=0)
        
        # --- MODIFICATION ---
        # Per your request, we only filter out features that are *completely*
        # unusable (e.g., all NaN or zero variance).
        # Sparse features (e.g., 90% NaN) are NOW ALLOWED.
        self.allowed_feature_mask = (self.col_nan_frac < 0.98) & (col_var > 1e-6)
        # --- END MODIFICATION ---
        
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

        # Identify sparse features (>=50% NaN) from the *allowed* list
        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        
        # Store variance for smarter selection
        self.col_var = col_var
        
        # Initial feature ranking for Q0 optimization
        if len(self.animals) > 0:
            n = len(self.animals)
            # Create a uniform prior (for calculating initial info gain)
            uniform_prior = np.ones(n, dtype=np.float32) / n
            
            # Calculate info gain for all allowed features
            initial_gains = self._compute_gains_batched(uniform_prior, self.allowed_feature_indices, batch_size=256)
            
            # Boost gain by variance (questions that split 0.5/0.5 are good)
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var[self.allowed_feature_indices]) + 1e-5)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.5)
            
            # Sort and store
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
        Memory-optimized Information Gain calculation.
        This calculates the expected reduction in entropy for each feature.
        """
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._calculate_entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains # No entropy to reduce, or no features to ask

        # --- Optimization: Subset calculation ---
        # If the prior is highly concentrated (e.g., 99% on 10 items),
        # only calculate entropy over that active subset.
        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active < len(prior) * 0.1 and n_active > 0:
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum() # Re-normalize
        else:
            active_prior = prior
            active_mask = slice(None) # Use all items
            n_active = len(prior)

        if n_active == 0:
            return gains
        # --- End Optimization ---

        # Process features in batches to save memory
        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            
            # 1. Get feature data for the active items
            f_batch = self.features[active_mask][:, batch_indices]
            
            # 2. Quantize feature values to match our likelihood table grid
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5) # Treat NaN as 'Sometimes'
            f_batch_quant = np.clip(np.rint(f_batch_filled * 40), 0, 40).astype(np.int8)

            # 3. Get all 5 likelihoods for each item/feature
            # Shape: (n_active, k_batch, n_answers)
            likelihoods = self.likelihood_table[f_batch_quant]  

            # 4. Handle NaNs: If the original data was NaN, likelihood is 1.0
            # (a NaN feature value doesn't change our belief)
            nan_batch_mask = np.isnan(f_batch)
            nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
            likelihoods[nan_expand] = 1.0

            # 5. Calculate expected entropy for each feature in the batch
            batch_gains = np.zeros(len(batch_indices), dtype=np.float32)
            
            for f_local_idx in range(len(batch_indices)):
                L_f = likelihoods[:, f_local_idx, :] # Likelihoods for one feature
                
                # P(Answer|Feature) = sum_items [ P(Item) * P(Answer|Item, Feature) ]
                p_answers = active_prior @ L_f # Shape: (n_answers,)
                
                expected_entropy = 0.0
                for a_idx in range(n_answers):
                    p_a = p_answers[a_idx]
                    if p_a > 1e-7:
                        # Bayes' Rule: P(Item|Answer) = P(Item) * P(Answer|Item) / P(Answer)
                        post = active_prior * L_f[:, a_idx]
                        post /= p_a
                        expected_entropy += p_a * self._calculate_entropy(post)
                
                # Info Gain = H(Prior) - H(Prior|Feature)
                batch_gains[f_local_idx] = current_entropy - expected_entropy

            gains[i:end] = batch_gains
            
        return gains

    def update(self, feature_idx: int, answer_str: str, 
               current_scores: np.ndarray) -> np.ndarray:
        """
        NEW STATELESS UPDATE:
        Calculates the score adjustment for a single answer and adds
        it to the cumulative scores.
        """
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        f_col = self.features[:, feature_idx] # All items for this feature
        nan_mask = self.nan_mask[:, feature_idx]
        
        # 1. Quantize feature values to match likelihood table
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 40), 0, 40).astype(np.int32)
        
        # 2. Find the index for the user's answer (e.g., 'yes' -> 1.0 -> index 0)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        
        # 3. Get the precomputed likelihood for every item
        # This is P(Answer | Item)
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # 4. Apply contradiction penalty
        # If user said 'Yes' (1.0) but DB says 'No' (e.g., < 0.15), penalize heavily.
        if answer_val >= 0.9:  
            contradictions = (f_col < 0.15) & (~nan_mask)
            likelihoods[contradictions] *= 0.03  # 97% penalty
        # If user said 'No' (0.0) but DB says 'Yes' (e.g., > 0.85), penalize heavily.
        elif answer_val <= 0.1:  
            contradictions = (f_col > 0.85) & (~nan_mask)
            likelihoods[contradictions] *= 0.03  # 97% penalty

        # 5. Convert likelihoods to log-space scores
        # We use log-likelihoods for cumulative scoring to prevent
        # scores from vanishing to zero.
        # This turns multiplication (Bayes) into addition.
        scores = np.log(likelihoods + 1e-10) 
        
        # 6. Add to the cumulative scores
        new_cumulative_scores = current_scores + scores

        return new_cumulative_scores

    def get_sparse_question_for_game(self, prior: np.ndarray, asked_features_names: set) -> tuple[str, str]:
        """Find the most useful sparse question to ask right now."""
        
        # Find sparse features that haven't been asked
        candidates_to_eval = [
            idx for idx in self.sparse_indices
            if self.feature_cols[idx] not in asked_features_names
        ]

        if not candidates_to_eval:
            return None, None

        # Find the info gain for these sparse candidates
        gains = self._compute_gains_batched(prior, np.array(candidates_to_eval), batch_size=32)
        
        if np.max(gains) < 1e-5:
            return None, None # No useful sparse questions

        best_local_idx = np.argmax(gains)
        best_feat_idx = candidates_to_eval[best_local_idx]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Smarter question selection using Information Gain.
        """
        
        # --- Sparse Question Strategy ---
        # At Q6 and Q12, try to ask a high-gain sparse question.
        if question_count == 6 or question_count == 12:
            asked_set_for_sparse = set(asked_features) | self.sparse_questions_asked
            sparse_feat, sparse_q = self.get_sparse_question_for_game(prior, asked_set_for_sparse)
            if sparse_feat:
                self.sparse_questions_asked.add(sparse_feat) # Mark as asked
                print(f"[Question] Asking sparse Q{question_count}: {sparse_feat}")
                return sparse_feat, sparse_q
        
        # --- Standard Question Strategy ---
        
        # 1. Get all features we *can* ask
        asked_set = set(asked_features)
        candidates_indices = [idx for idx in self.allowed_feature_indices  
                              if self.feature_cols[idx] not in asked_set]

        if not candidates_indices:
            print("[Question] No more features to ask.")
            return None, None # Out of questions

        # 2. Filter down candidates to a manageable number for gain calculation
        if len(candidates_indices) > 100:
            # Score candidates based on initial rank (from Q0) and variance
            scored_candidates = []
            for idx in candidates_indices:
                initial_rank = np.where(self.sorted_initial_feature_indices == idx)[0]
                rank_score = 1.0 / (initial_rank[0] + 1) if len(initial_rank) > 0 else 0
                var_score = self.col_var[idx] / (np.max(self.col_var[self.allowed_feature_indices]) + 1e-5)
                scored_candidates.append((idx, rank_score + var_score))
            
            # Take the 80 best-scoring candidates
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [idx for idx, _ in scored_candidates[:80]]
            
            # Add 20 random candidates to prevent getting stuck
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

        # 3. Calculate info gain for the filtered list
        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=64)
        
        # 4. Probabilistic Selection (to avoid asking the same questions)
        # Get top 3 candidates
        sorted_gain_indices = np.argsort(gains)[::-1]
        top_n = min(3, len(sorted_gain_indices))
        top_local_indices = sorted_gain_indices[:top_n]
        top_gains = gains[top_local_indices]
        
        # Use a "temperature" to scale probabilities
        # Early game (low Q#): high temp, more random
        # Late game (high Q#): low temp, pick the best
        temperature = max(0.3, 1.5 - (question_count * 0.1))
        
        # Softmax on the gains to get probabilities
        exp_gains = np.exp((top_gains - np.max(top_gains)) / temperature)
        probs = exp_gains / np.sum(exp_gains)
        
        # Choose one of the top 3 based on these probabilities
        chosen_local_idx = np.random.choice(top_n, p=probs)
        best_feat_idx = candidates_to_eval[top_local_indices[chosen_local_idx]]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text

    def get_discriminative_question(self, top_animal_idx: int, prior: np.ndarray, asked_features: list) -> tuple[str, str]:
        """Find a question that best separates the top candidate from its rivals."""
        top_prob = prior[top_animal_idx]
        
        # Rivals are items with probability > 20% of the top item
        rival_threshold = top_prob * 0.2
        rival_mask = (prior > rival_threshold) & (np.arange(len(prior)) != top_animal_idx)
        rival_indices = np.where(rival_mask)[0]
        
        if len(rival_indices) == 0:
            return None, None # No rivals to discriminate from
            
        asked_set = set(asked_features)
        candidate_indices = [idx for idx in self.allowed_feature_indices 
                                 if self.feature_cols[idx] not in asked_set]
                                
        if not candidate_indices:
            return None, None

        # Get feature values for the top animal
        top_feats = self.features[top_animal_idx, candidate_indices]
        
        # Get weighted-average feature values for all rivals
        rival_feats = self.features[rival_indices][:, candidate_indices]
        rival_weights = prior[rival_indices]
        rival_weights /= rival_weights.sum() # Normalize rival weights
        avg_rival_feats = np.average(rival_feats, axis=0, weights=rival_weights)
        
        # Find the feature with the biggest *difference*
        diffs = np.abs(np.nan_to_num(top_feats, nan=0.5) - np.nan_to_num(avg_rival_feats, nan=0.5))
        
        best_local_idx = np.argmax(diffs)
        
        # Only return this question if the difference is significant
        if diffs[best_local_idx] > 0.35: # e.g., 'Yes' (1.0) vs 'Sort of' (0.5)
            best_feat_idx = candidate_indices[best_local_idx]
            feature_name = self.feature_cols[best_feat_idx]
            return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                    
        return None, None # No good discriminating question found

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        NEW STATELESS GUESS LOGIC:
        Requires high probability AND high consistency.
        
        Receives game_state (for counts/scores) and probs (for confidence).
        """
        q_count = game_state['question_count']
        
        # Probs are already masked by the service, but we re-check sum
        if probs.sum() < 1e-10:
            return False, None, None
        
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        probs_copy = probs.copy()
        probs_copy[top_idx] = 0.0
        second_prob = np.max(probs_copy)
        
        # Ratio of top candidate to the runner-up
        confidence_ratio = top_prob / (second_prob + 1e-9)
        
        # Measure total uncertainty
        entropy = self._calculate_entropy(probs)
        
        # --- NEW CONSISTENCY CHECK ---
        # This is the "anti-fooling" logic.
        # The top item must not only have high probability, but also
        # a high cumulative score compared to other plausible items.
        cumulative_scores = game_state.get('cumulative_scores')
        is_consistent = False
        if cumulative_scores is not None and q_count > 0:
            top_animal_score = cumulative_scores[top_idx]
            
            # Check against other "plausible" items (prob > 0.1%)
            plausible_mask = (probs > 0.001) & (~game_state['rejected_mask'])
            
            if np.sum(plausible_mask) > 1:
                avg_score = np.mean(cumulative_scores[plausible_mask])
                score_std = np.std(cumulative_scores[plausible_mask])
                # Top item must be at least 0.5 std dev *above* the average
                consistency_threshold = avg_score + (score_std * 0.5) 
            else:
                consistency_threshold = -np.inf # Only one item left
            
            if top_animal_score >= consistency_threshold:
                is_consistent = True
            else:
                print(f"[GUESS] REJECTED: {top_animal} prob ({top_prob:.2f}) is high, "
                      f"but score ({top_animal_score:.2f}) is not above "
                      f"threshold ({consistency_threshold:.2f})")
        
        elif q_count == 0:
            is_consistent = True # No data to check, allow (won't pass prob check anyway)
        # --- END CONSISTENCY CHECK ---

        # Continue mode logic
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 3:
                return False, None, None # Ask 3 more questions
            else:
                # Reset continue mode for the *next* guess
                game_state['continue_mode'] = False
                game_state['questions_since_last_guess'] = 0

        # --- HIGH CONFIDENCE THRESHOLDS ---
        
        # Zone 1: Early Game (q < 15). Almost impossible to guess.
        if q_count < 15:
            if top_prob > 0.99 and confidence_ratio > 300.0 and entropy < 0.1 and is_consistent:
                print(f"[GUESS] Early Slam Dunk: prob={top_prob:.4f}, ratio={confidence_ratio:.1f}, entropy={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Zone 2: Mid-Game (q < 25). Extremely high confidence needed.
        if q_count < 25:
            # Requires 95% prob, 80x lead, low entropy, and consistency
            if top_prob > 0.95 and confidence_ratio > 80.0 and entropy < 0.3 and is_consistent:
                print(f"[GUESS] Mid-Game: prob={top_prob:.4f}, ratio={confidence_ratio:.1f}, entropy={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Zone 3: Late-Game (q < MAX_QUESTIONS). High confidence needed.
        if q_count < self.MAX_QUESTIONS: 
            # Requires 90% prob, 40x lead, low-ish entropy, and consistency
            if top_prob > 0.90 and confidence_ratio > 40.0 and entropy < 0.5 and is_consistent:
                print(f"[GUESS] Late-Game: prob={top_prob:.4f}, ratio={confidence_ratio:.1f}, entropy={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Zone 4: Timeout (MAX_QUESTIONS reached).
        # We are forced to guess. We'll take the most probable *consistent* item.
        if is_consistent:
             print(f"[GUESS] Forced Timeout (Consistent): prob={top_prob:.4f}, entropy={entropy:.3f}")
             return True, top_animal, 'final'
        else:
             # Top item is inconsistent. Find the *most consistent* item.
             plausible_mask = (probs > 0.001) & (~game_state['rejected_mask'])
             if np.any(plausible_mask):
                plausible_scores = cumulative_scores[plausible_mask]
                plausible_animals = self.animals[plausible_mask]
                best_consistent_idx = np.argmax(plausible_scores)
                best_consistent_animal = plausible_animals[best_consistent_idx]
                print(f"[GUESS] Forced Timeout (Inconsistent): Choosing {best_consistent_animal} instead of {top_animal}")
                return True, best_consistent_animal, 'final'
             else:
                 # Fallback, just guess the top item
                 print(f"[GUESS] Forced Timeout (Fallback): prob={top_prob:.4f}, entropy={entropy:.3f}")
                 return True, top_animal, 'final'

    def get_features_for_data_collection(self, item_name, num_features=5):
        """
        Gets 5 features for data collection.
        (This function is unchanged from your original file).
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

        # 1. Find NaN features for this item (highest priority)
        nan_indices = np.where(np.isnan(item_feats))[0]
        useful_nan_indices = np.intersect1d(nan_indices, self.allowed_feature_indices).copy()
        
        # 2. Add globally sparse features if needed
        if len(useful_nan_indices) < num_features:
            needed = num_features - len(useful_nan_indices)
            extras = np.setdiff1d(self.sparse_indices, useful_nan_indices).copy()
            
            if len(extras) > 0:
                np.random.shuffle(extras)
                selected_indices = np.concatenate((useful_nan_indices, extras[:needed]))
            else:
                # 3. Fallback to any allowed features
                remaining_allowed = np.setdiff1d(self.allowed_feature_indices, useful_nan_indices).copy()
                if len(remaining_allowed) > 0:
                    np.random.shuffle(remaining_allowed)
                    needed_more = num_features - len(useful_nan_indices)
                    selected_indices = np.concatenate((useful_nan_indices, remaining_allowed[:needed_more]))
                else:
                    selected_indices = useful_nan_indices
        else:
            np.random.shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
        
        # Always ensure we return exactly num_features if possible
        if len(selected_indices) < num_features and len(self.allowed_feature_indices) >= num_features:
            remaining = np.setdiff1d(self.allowed_feature_indices, selected_indices).copy()
            np.random.shuffle(remaining)
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