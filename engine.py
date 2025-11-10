import pandas as pd
import numpy as np

class AkinatorEngine:
    """
    Memory-optimized Akinator engine with batched calculations and improved selection logic.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        
        # --- Configuration ---
        self.MAX_QUESTIONS = 25
        # Standardized answer values for the engine's internal logic
        self.answer_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
        
        # Confidence schedule for guessing (prob threshold, ratio threshold)
        self.confidence_schedule = {
             range(0, 8): (0.96, 25.0),   # Very strict early
             range(8, 15): (0.92, 18.0),  # Strict mid-game
             range(15, 20): (0.88, 12.0), # Moderate late-game
             range(20, 26): (0.80, 8.0),  # looser end-game
        }

        # --- Mappings ---
        # Ensures compatibility with: Yes, Mostly, Sort of, Not really, No
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
        Precomputes likelihood lookup tables for fast Bayesian updates.
        Uses sharper curves for definite answers to quickly eliminate mismatches.
        """
        # 41 bins allows mapping 0.0-1.0 with 0.025 precision
        steps = 41
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                
                # Base likelihood uses Gaussian-like falloff
                # Sharper falloff if the answer is "Yes"(1.0) or "No"(0.0)
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.2  # Narrow curve for definite answers
                else:
                    sigma = 0.35 # Wider curve for fuzzy answers

                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                
                # Clamp small values to avoid zero-division issues later
                self.likelihood_table[i, j] = max(likelihood, 0.001)

    def _build_arrays(self):
        """Converts dataframe to optimized numpy arrays."""
        self.animals = self.df['animal_name'].values
        
        # Main feature matrix (keeps NaNs)
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.nan_mask = np.isnan(self.features)
        
        # Pre-calculate column stats for feature selection
        self.col_nan_frac = np.mean(self.nan_mask, axis=0)
        col_var = np.nanvar(self.features, axis=0)
        
        # Filter usable features (ignore ones that are >95% empty or have 0 variance)
        self.allowed_feature_mask = (self.col_nan_frac < 0.95) & (col_var > 1e-5)
        self.allowed_feature_indices = np.where(self.allowed_feature_mask)[0].astype(np.int32)

        # Identify sparse features for data collection phase
        self.sparse_indices = np.where(self.col_nan_frac >= 0.50)[0].astype(np.int32)
        
        # --- Initial Feature Ranking ---
        # Pre-calculate the best starting questions assuming uniform prior
        if len(self.animals) > 0:
             n = len(self.animals)
             uniform_prior = np.ones(n, dtype=np.float32) / n
             # Use a larger batch for initial precomputation since it only runs once
             initial_gains = self._compute_gains_batched(uniform_prior, self.allowed_feature_indices, batch_size=256)
             
             # Sort features by initial information gain
             sorted_indices = np.argsort(initial_gains)[::-1]
             self.sorted_initial_feature_indices = self.allowed_feature_indices[sorted_indices]
        else:
             self.sorted_initial_feature_indices = np.array([], dtype=np.int32)

    def _entropy(self, probs):
        """Robust Shannon entropy calculation."""
        # Clip to avoid log(0)
        p = np.clip(probs, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))

    def _compute_gains_batched(self, prior, feature_indices, batch_size=64):
        """
        Memory-optimized information gain calculation.
        Processes features in small batches to prevent huge memory spikes.
        """
        n_animals = len(prior)
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._entropy(prior)
        if current_entropy < 1e-5:
            return gains

        # Identify active animals to speed up calc (skip animals with ~0 probability)
        active_mask = prior > 1e-7
        n_active = np.sum(active_mask)
        
        # If too few animals are active, just use standard calculation to avoid overhead
        if n_active < n_animals * 0.1 and n_active > 0:
             use_subset = True
             active_prior = prior[active_mask]
             # Renormalize active prior
             active_prior = active_prior / active_prior.sum()
        else:
             use_subset = False
             active_prior = prior
             active_mask = slice(None) # Select all

        # Main batch loop
        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            k_batch = len(batch_indices)
            
            # 1. Get features for this batch AND active animals only
            f_batch = self.features[active_mask][:, batch_indices]
            
            # 2. Fill NaNs temporarily for lookup (we'll mask them later)
            # Using 0.5 as neutral filler for NaNs
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5)
            
            # 3. Quantize to look up in likelihood table (bins 0-40)
            f_batch_quant = np.clip(np.rint(f_batch_filled * 40), 0, 40).astype(np.int8)

            # 4. Compute Likelihoods: shape (n_active, k_batch, n_answers)
            # This is the memory-intensive part, now bounded by batch_size
            likelihoods = self.likelihood_table[f_batch_quant] 

            # 5. Apply NaN neutrality
            # If a feature value is NaN, it shouldn't affect discriminability much.
            # We set likelihood to 1.0 (neutral update) for all answers if feature is NaN.
            nan_batch_mask = np.isnan(f_batch)
            # Expand mask to match likelihoods shape for broadcasting
            nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
            likelihoods[nan_expand] = 1.0

            # 6. Compute Expected Entropy for each feature in batch
            batch_gains = np.zeros(k_batch, dtype=np.float32)
            
            for f_local_idx in range(k_batch):
                # For one feature, across all active animals, for all answers:
                # L_f shape: (n_active, n_answers)
                L_f = likelihoods[:, f_local_idx, :]
                
                # P(answer | feature) = sum(P(animal) * P(answer | animal, feature))
                # Shape: (n_answers,)
                p_answers = active_prior @ L_f
                
                expected_entropy = 0.0
                for a_idx in range(n_answers):
                    p_a = p_answers[a_idx]
                    if p_a > 1e-7:
                        # Posterior if this answer were given
                        post = active_prior * L_f[:, a_idx]
                        post /= p_a # Normalize
                        expected_entropy += p_a * self._entropy(post)
                
                batch_gains[f_local_idx] = current_entropy - expected_entropy

            gains[i:end] = batch_gains
            
        return gains

    def update(self, prior, feature_idx, answer_str):
        """
        Bayesian update of prior probabilities based on an answer.
        """
        # 1. Parse answer
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        # 2. Get feature column
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        # 3. Calculate Likelihoods
        # Quantize feature values to indices [0, 40]
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 40), 0, 40).astype(np.int32)
        
        # Find which answer column in table matches user answer
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        
        # Look up base likelihoods
        likelihoods = self.likelihood_table[f_quant, a_idx]
        
        # 4. Handle NaNs and Contradictions
        # If feature data is missing (NaN), this question gives us NO info about this animal.
        # Likelihood stays 1.0 (neutral).
        likelihoods[nan_mask] = 1.0

        # Strong Contradiction Dampening:
        # If user says DEFINITELY YES (1.0) but we have DEFINITELY NO (0.0ish),
        # apply an extra penalty.
        if answer_val >= 0.9: # User said YES
             contradictions = (f_col < 0.2) & (~nan_mask)
             likelihoods[contradictions] *= 0.1
        elif answer_val <= 0.1: # User said NO
             contradictions = (f_col > 0.8) & (~nan_mask)
             likelihoods[contradictions] *= 0.1

        # 5. Apply Bayes Rule
        posterior = prior * likelihoods
        
        # 6. Normalize
        total_p = np.sum(posterior)
        if total_p < 1e-9:
             # Avoid hard crash if all candidates ruled out; revert to prior slightly dampened
             return prior * 0.9 + (1.0/len(prior)) * 0.1
             
        return posterior / total_p

    def select_question(self, prior, asked_features, question_count):
        """
        Selects the next best question using information gain and smart randomization.
        """
        # 1. Filter available features
        asked_set = set(asked_features)
        # Only consider features that aren't asked AND are marked as 'allowed'
        candidates_indices = [idx for idx in self.allowed_feature_indices 
                              if self.feature_cols[idx] not in asked_set]

        if not candidates_indices:
            return None, None

        # Optimization: If we have too many candidates, just look at the 
        # ones that were historically good (initial ranking) plus a few randoms
        # to ensure we don't get stuck in loops if the initial ranking is bad for this specific animal.
        if len(candidates_indices) > 100:
             # Take top 80 from pre-calculated good features that are still available
             top_candidates = [idx for idx in self.sorted_initial_feature_indices 
                               if idx in candidates_indices][:80]
             # Add 20 random other available features for diversity
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

        # 2. Compute Information Gain (Batched!)
        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=64)
        
        # 3. Smart Selection (Softmax)
        # Don't always just take the max gain. Add a little randomness based on 
        # how good the questions are relative to each other.
        
        # Sort by gain descending
        sorted_gain_indices = np.argsort(gains)[::-1]
        top_n = min(5, len(sorted_gain_indices))
        
        top_local_indices = sorted_gain_indices[:top_n]
        top_gains = gains[top_local_indices]
        
        # Dynamic Temperature:
        # Early game (q<5): Higher temp (more exploration/randomness)
        # Late game (q>15): Lower temp (greedy selection of best question)
        temperature = max(0.5, 3.0 - (question_count * 0.15))
        
        # Softmax calculation
        exp_gains = np.exp((top_gains - np.max(top_gains)) / temperature) # sub max for numerical stability
        probs = exp_gains / np.sum(exp_gains)
        
        # Choose one
        chosen_local_idx = np.random.choice(top_n, p=probs)
        best_feat_idx = candidates_to_eval[top_local_indices[chosen_local_idx]]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text

    def get_discriminative_question(self, top_animal_idx, prior, asked_features):
        """
        Finds a question that specifically separates the #1 candidate from its closest rivals.
        Used when we are fairly confident but need that final confirmation.
        """
        top_prob = prior[top_animal_idx]
        
        # Find rivals: animals with at least 15% of the top animal's probability
        rival_threshold = top_prob * 0.15
        rival_mask = (prior > rival_threshold) & (np.arange(len(prior)) != top_animal_idx)
        rival_indices = np.where(rival_mask)[0]
        
        if len(rival_indices) == 0:
            return None, None
            
        # Candidates to ask
        asked_set = set(asked_features)
        candidate_indices = [idx for idx in self.allowed_feature_indices 
                             if self.feature_cols[idx] not in asked_set]
                             
        if not candidate_indices:
             return None, None

        # We want a feature where Top Animal is very different from Rivals.
        # 1. Get top animal feature values
        top_feats = self.features[top_animal_idx, candidate_indices]
        
        # 2. Get weighted average of rival feature values
        rival_feats = self.features[rival_indices][:, candidate_indices]
        rival_weights = prior[rival_indices]
        rival_weights /= rival_weights.sum() # Normalize
        avg_rival_feats = np.average(rival_feats, axis=0, weights=rival_weights)
        
        # 3. Calculate difference (ignoring NaNs)
        # We use nan_to_num(0.5) so NaNs don't look "different" from 0.5 neutral values
        diffs = np.abs(np.nan_to_num(top_feats, nan=0.5) - np.nan_to_num(avg_rival_feats, nan=0.5))
        
        # 4. Find best separator
        best_local_idx = np.argmax(diffs)
        
        # Only use if the separation is significant enough (> 0.4 difference)
        if diffs[best_local_idx] > 0.4:
             best_feat_idx = candidate_indices[best_local_idx]
             feature_name = self.feature_cols[best_feat_idx]
             return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
             
        return None, None

    def get_features_for_data_collection(self, item_name, num_features=5):
        """
        Selects features to ask user about for improving the database.
        Prioritizes:
        1. Features that are 'active' (allowed) but missing (NaN) for this item.
        2. Features that are globally sparse (need more data overall).
        """
        try:
            item_idx = np.where(self.animals == item_name)[0][0]
            item_feats = self.features[item_idx]
        except IndexError:
            # Item might be new or not in current engine loaded state
            return []

        # Find features that are NaN for this item AND are in our allowed list
        nan_indices = np.where(np.isnan(item_feats))[0]
        useful_nan_indices = np.intersect1d(nan_indices, self.allowed_feature_indices)
        
        # If we need more, look at globally sparse features
        if len(useful_nan_indices) < num_features:
             needed = num_features - len(useful_nan_indices)
             # Take some globally sparse ones that we haven't already selected
             extras = np.setdiff1d(self.sparse_indices, useful_nan_indices)
             # Shuffle extras to get variety across different users
             np.random.shuffle(extras)
             selected_indices = np.concatenate((useful_nan_indices, extras[:needed]))
        else:
             # We have enough NaNs specific to this item. 
             # Shuffle them so we don't always ask the same ones first.
             np.random.shuffle(useful_nan_indices)
             selected_indices = useful_nan_indices[:num_features]
             
        # Format for return
        results = []
        for idx in selected_indices[:num_features]:
             fname = self.feature_cols[idx]
             results.append({
                  "feature_name": fname,
                  "question": self.questions_map.get(fname, f"Is it {fname}?"),
                  "nan_percentage": float(self.col_nan_frac[idx])
             })
             
        return results