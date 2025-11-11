import pandas as pd
import numpy as np

class AkinatorEngine:
    """
    --- UPDATED ---
    Akinator engine refactored for patience and accuracy.
    - Softer likelihood curves to tolerate fuzzy answers.
    - Patient guessing logic to prevent "rushing".
    - More forgiving update logic.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        
        # --- Configuration ---
        # --- FIX: Increased max questions to allow for more patient guessing ---
        self.MAX_QUESTIONS = 25
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        # --- Mappings ---
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'mostly': 0.75, 'usually': 0.75, 'probably': 0.75,
            'sort of': 0.5, 'sometimes': 0.5, 'maybe': 0.5, 'idk': 0.5,
            'not really': 0.25, 'rarely': 0.25,
            'no': 0.0, 'n': 0.0,
        }

        # --- Internal State ---
        self.answer_history = []
        self.sparse_questions_asked = set()

        # --- Initialization ---
        self._precompute_likelihood_tables()
        self._build_arrays()

    def _precompute_likelihood_tables(self):
        """
        --- FIX: Softer likelihood curves. ---
        This is the most important change. The old curves were too "sharp" (sigma=0.12).
        If a user said "Mostly" (0.75) but the DB had "Yes" (1.0), the
        correct animal's probability was almost zeroed out.
        These wider curves (sigma=0.2, sigma=0.3) are more tolerant
        of small differences between user answers and database values.
        """
        steps = 41
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                
                # --- FIX: Widened sigmas for more tolerance ---
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.2  # Was 0.12 (too aggressive).
                else:
                    sigma = 0.3  # Was 0.25. Softer for intermediate answers.

                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                
                # Floor to prevent true zero probabilities (allows recovery later)
                self.likelihood_table[i, j] = max(likelihood, 0.0001)

    def _build_arrays(self):
        """Converts dataframe to optimized numpy arrays."""
        self.animals = self.df['animal_name'].values
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.nan_mask = np.isnan(self.features)
        
        # Pre-calculate column stats
        self.col_nan_frac = np.mean(self.nan_mask, axis=0)
        col_var = np.nanvar(self.features, axis=0)
        col_mean = np.nanmean(self.features, axis=0)
        
        # Filter usable features
        self.allowed_feature_mask = (self.col_nan_frac < 0.95) & (col_var > 1e-5)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

        # Identify sparse features (>=50% NaN)
        sparse_mask = (self.col_nan_frac >= 0.50) & self.allowed_feature_mask
        self.sparse_indices = np.where(sparse_mask)[0].astype(np.int32)
        
        # Store variance and mean for smarter selection
        self.col_var = col_var
        self.col_mean = col_mean
        
        # Initial feature ranking
        if len(self.animals) > 0:
            n = len(self.animals)
            uniform_prior = np.ones(n, dtype=np.float32) / n
            initial_gains = self._compute_gains_batched(uniform_prior, self.allowed_feature_indices, batch_size=256)
            
            variance_boost = col_var[self.allowed_feature_indices] / (np.max(col_var) + 1e-5)
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.5)
            
            sorted_indices = np.argsort(boosted_gains)[::-1]
            self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
            self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def _entropy(self, probs):
        """Robust Shannon entropy calculation."""
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))

    def _compute_gains_batched(self, prior, feature_indices, batch_size=64):
        """
        Memory-optimized information gain.
        (Logic unchanged, but will perform better with stabler priors)
        """
        n_animals = len(prior)
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains

        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active < n_animals * 0.1 and n_active > 0:
            use_subset = True
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum()
        else:
            use_subset = False
            active_prior = prior
            active_mask = slice(None)
            n_active = n_animals

        if n_active == 0:
            return gains
            
        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            k_batch = len(batch_indices)
            
            f_batch = self.features[active_mask][:, batch_indices]
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5)
            f_batch_quant = np.clip(np.rint(f_batch_filled * 40), 0, 40).astype(np.int8)

            likelihoods = self.likelihood_table[f_batch_quant] 

            nan_batch_mask = np.isnan(f_batch)
            nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
            likelihoods[nan_expand] = 1.0

            batch_gains = np.zeros(k_batch, dtype=np.float32)
            
            for f_local_idx in range(k_batch):
                L_f = likelihoods[:, f_local_idx, :]
                p_answers = active_prior @ L_f
                
                expected_entropy = 0.0
                for a_idx in range(n_answers):
                    p_a = p_answers[a_idx]
                    if p_a > 1e-7:
                        post = active_prior * L_f[:, a_idx]
                        post /= p_a
                        expected_entropy += p_a * self._entropy(post)
                
                batch_gains[f_local_idx] = current_entropy - expected_entropy

            gains[i:end] = batch_gains
            
        return gains

    def update(self, prior, feature_idx, answer_str):
        """
        --- FIX: Softer Bayesian update penalties ---
        """
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 40), 0, 40).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        
        likelihoods = self.likelihood_table[f_quant, a_idx]
        
        # --- FIX: Softer contradiction penalties ---
        # The old penalty (0.05) was too harsh and killed candidates too early.
        # 0.2 is much more forgiving.
        if answer_val >= 0.9:  # User said YES
            contradictions = (f_col < 0.15) & (~nan_mask)
            likelihoods[contradictions] *= 0.2  # Was 0.05
        elif answer_val <= 0.1:  # User said NO
            contradictions = (f_col > 0.85) & (~nan_mask)
            likelihoods[contradictions] *= 0.2  # Was 0.05

        posterior = prior * likelihoods
        
        total_p = np.sum(posterior)
        if total_p < 1e-9:
            # Fallback: dampen the prior to show some update, but don't collapse
            return prior * 0.8 + (1.0/len(prior)) * 0.2
            
        return posterior / total_p

    def get_sparse_question_for_game(self, prior, asked_features_names):
        """Find the most useful sparse question. (Unchanged)"""
        sparse_indices = self.sparse_indices
        
        asked_set = set(asked_features_names)
        candidates_to_eval = [
            idx for idx in sparse_indices
            if self.feature_cols[idx] not in asked_set
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
        
    def select_question(self, prior, asked_features, question_count):
        """
        Smarter question selection. (Logic unchanged, but will be called
        more appropriately by the updated service layer).
        """
        
        if question_count == 6:
            sparse_feat, sparse_q = self.get_sparse_question_for_game(prior, asked_features)
            if sparse_feat:
                self.sparse_questions_asked.add(sparse_feat)
                return sparse_feat, sparse_q
        
        asked_set = set(asked_features)
        asked_set.update(self.sparse_questions_asked)
        
        candidates_indices = [idx for idx in self.allowed_feature_indices 
                             if self.feature_cols[idx] not in asked_set]

        if not candidates_indices:
            return None, None

        # Smarter pre-filtering
        if len(candidates_indices) > 100:
            scored_candidates = []
            for idx in candidates_indices:
                initial_rank = np.where(self.sorted_initial_feature_indices == idx)[0]
                rank_score = 1.0 / (initial_rank[0] + 1) if len(initial_rank) > 0 else 0
                var_score = self.col_var[idx] / (np.max(self.col_var) + 1e-5)
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

        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=64)
        
        sorted_gain_indices = np.argsort(gains)[::-1]
        top_n = min(3, len(sorted_gain_indices))
        
        top_local_indices = sorted_gain_indices[:top_n]
        top_gains = gains[top_local_indices]
        
        temperature = max(0.3, 1.5 - (question_count * 0.1))
        
        exp_gains = np.exp((top_gains - np.max(top_gains)) / temperature)
        probs = exp_gains / np.sum(exp_gains)
        
        chosen_local_idx = np.random.choice(top_n, p=probs)
        best_feat_idx = candidates_to_eval[top_local_indices[chosen_local_idx]]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text

    def get_discriminative_question(self, top_animal_idx, prior, asked_features):
        """Find a question that separates the top candidate from rivals. (Unchanged)"""
        top_prob = prior[top_animal_idx]
        
        rival_threshold = top_prob * 0.2
        rival_mask = (prior > rival_threshold) & (np.arange(len(prior)) != top_animal_idx)
        rival_indices = np.where(rival_mask)[0]
        
        if len(rival_indices) == 0:
            return None, None
            
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
        
        best_local_idx = np.argmax(diffs)
        
        if diffs[best_local_idx] > 0.35:
            best_feat_idx = candidate_indices[best_local_idx]
            feature_name = self.feature_cols[best_feat_idx]
            return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
             
        return None, None

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs_safe * np.log2(probs_safe))

    def should_make_guess(self, game_state: dict) -> tuple[bool, str | None, str | None]:
        """
        --- FIX: PATIENT Guessing Logic ---
        This is the second most important change. It stops the engine from
        "rushing". We will not guess early unless we are absolutely certain.
        It's better to ask 15 smart questions and be right than 8
        rushed questions and be wrong.
        """
        q_count = game_state['question_count']
        probs = game_state['probabilities'].copy()
        mask = game_state['rejected_mask']
        probs[mask] = 0.0 

        if probs.sum() < 1e-10:
            return False, None, None
        
        probs = probs / probs.sum()
        
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        top_animal = self.animals[top_idx]
        
        probs_copy = probs.copy()
        probs_copy[top_idx] = 0.0
        second_prob = np.max(probs_copy)
        
        # Ratio of top candidate to the runner-up
        confidence_ratio = top_prob / (second_prob + 1e-9)
        
        # entropy = self._calculate_entropy(probs)
        # significant_candidates = np.sum(probs > 0.05)
        
        # Continue mode logic
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 3: # Was 3, this is fine
                return False, None, None
            else:
                game_state['continue_mode'] = False
                game_state['questions_since_last_guess'] = 0

        # --- NEW PATIENT THRESHOLDS ---
        
        if q_count < 8:
            if top_prob > 1 and confidence_ratio > 120.0:
                print(f"[GUESS] Slam Dunk: prob={top_prob:.3f}")
                return True, top_animal, 'final'
            return False, None, None # Force "no"
        
        if q_count < 15:
            if top_prob > 0.98 and confidence_ratio > 30.0:
                print(f"[GUESS] Mid-Game High-Conf: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}")
                return True, top_animal, 'final'
            return False, None, None # Force "no"
        
        if q_count < self.MAX_QUESTIONS:
            if top_prob > 0.96 and confidence_ratio > 10.0:
                print(f"[GUESS] Late-Game Standard: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}")
                return True, top_animal, 'final'
            return False, None, None # Force "no"
        
        # Zone 4: Timeout (MAX_QUESTIONS reached).
        # We've run out of questions. Make the best guess we have.
        print(f"[GUESS] Forced (Timeout): prob={top_prob:.3f}")
        return True, top_animal, 'final'

    def get_features_for_data_collection(self, item_name, num_features=5):
        """
        ALWAYS returns features for data collection.
        (Unchanged)
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