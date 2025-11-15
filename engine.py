import pandas as pd
import numpy as np
from datetime import datetime
import os

class AkinatorEngine:
    """
    Akinator guessing engine (V5.0 - High Precision Mode).

    CHANGES:
    1. STRICT CONFIDENCE: Will NOT guess unless probability >= 94% (except at Q25 limit).
    2. RANDOM START: Q1 is random from top features to vary gameplay.
    3. DISCRIMINATION: Aggressive tie-breaking logic when two items are close.
    """
    
    def __init__(self, df, feature_cols, questions_map):
        self.df = df
        self.feature_cols = np.array(feature_cols)
        self.questions_map = questions_map
        
        self.MAX_QUESTIONS = None 
        
        # Granular answer values for precision
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
        """
        Likelihood calculation with adaptive sigma to punish mismatches.
        """
        steps = 101
        self.feature_grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        n_answers = len(self.answer_values)
        
        self.likelihood_table = np.zeros((steps, n_answers), dtype=np.float32)
        
        for i, f_val in enumerate(self.feature_grid):
            for j, a_val in enumerate(self.answer_values):
                diff = abs(f_val - a_val)
                
                # Tighter sigma for definite answers (0.0/1.0) to drive probabilities apart faster
                if a_val == 1.0 or a_val == 0.0:
                    sigma = 0.06
                elif a_val == 0.75 or a_val == 0.25:
                    sigma = 0.12
                else:
                    sigma = 0.18

                likelihood = np.exp(-0.5 * (diff / sigma) ** 2)
                self.likelihood_table[i, j] = max(likelihood, 0.001)

    def _build_arrays(self):
        """Converts dataframe to optimized numpy arrays with quality metrics."""
        self.animals = self.df['animal_name'].values
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.nan_mask = np.isnan(self.features)
        
        self.col_nan_frac = np.mean(self.nan_mask, axis=0)
        
        nan_masked_features = np.ma.masked_invalid(self.features)
        col_var = nan_masked_features.var(axis=0).data
        
        col_mean = nan_masked_features.mean(axis=0).data
        self.col_ambiguity = 1.0 - 2.0 * np.abs(col_mean - 0.5)

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
            
            penalty = 1.0 - (
                0.4 * self.col_nan_frac[self.allowed_feature_indices] + 
                0.4 * self.col_ambiguity[self.allowed_feature_indices]
            )
            penalty = np.clip(penalty, 0.3, 1.0)
            
            boosted_gains = initial_gains * (1.0 + variance_boost * 0.6) * penalty
            
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
        """Computes information gain in batches for memory efficiency."""
        n_features = len(feature_indices)
        n_answers = len(self.answer_values)
        gains = np.zeros(n_features, dtype=np.float32)
        
        current_entropy = self._calculate_entropy(prior)
        if current_entropy < 1e-5 or n_features == 0:
            return gains

        active_mask = prior > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active < len(prior) * 0.8 and n_active > 0:
            active_prior = prior[active_mask]
            active_prior = active_prior / active_prior.sum()
        else:
            active_prior = prior
            active_mask = slice(None)
            n_active = len(prior)

        if n_active == 0:
            return gains

        for i in range(0, n_features, batch_size):
            end = min(i + batch_size, n_features)
            batch_indices = feature_indices[i:end]
            
            f_batch = self.features[active_mask][:, batch_indices]
            f_batch_filled = np.nan_to_num(f_batch, nan=0.5)
            f_batch_quant = np.clip(np.rint(f_batch_filled * 100), 0, 100).astype(np.int16)

            likelihoods = self.likelihood_table[f_batch_quant]

            nan_batch_mask = np.isnan(f_batch)
            nan_expand = np.repeat(nan_batch_mask[:, :, np.newaxis], n_answers, axis=2)
            likelihoods[nan_expand] = 1.0

            batch_gains = np.zeros(len(batch_indices), dtype=np.float32)
            
            for f_local_idx in range(len(batch_indices)):
                L_f = likelihoods[:, f_local_idx, :]
                p_answers = active_prior @ L_f
                
                posteriors = active_prior[:, np.newaxis] * L_f
                posteriors /= (p_answers + 1e-10)
                posteriors = np.clip(posteriors, 1e-10, 1.0)
                
                p_logs = np.log2(posteriors)
                entropies_per_answer = -np.sum(posteriors * p_logs, axis=0)
                expected_entropy = p_answers @ entropies_per_answer
                
                batch_gains[f_local_idx] = current_entropy - expected_entropy

            gains[i:end] = batch_gains
            
        return gains

    def update(self, feature_idx: int, answer_str: str, current_scores: np.ndarray) -> np.ndarray:
        """Updates probability scores based on answer with contradiction handling."""
        answer_val = self.fuzzy_map.get(answer_str.lower(), 0.5)
        
        f_col = self.features[:, feature_idx]
        nan_mask = self.nan_mask[:, feature_idx]
        
        f_quant = np.clip(np.rint(np.nan_to_num(f_col, nan=0.5) * 100), 0, 100).astype(np.int32)
        a_idx = np.abs(self.answer_values - answer_val).argmin()
        
        likelihoods = self.likelihood_table[f_quant, a_idx].copy()
        
        # Contradiction penalty
        distance = np.abs(f_col - answer_val)
        certainty = np.clip(1.0 - 2.0 * np.abs(f_col - 0.5), 0.0, 1.0)
        severity = distance * certainty
        severity[nan_mask] = 0.0
        
        penalty_factor = 12.0
        penalty_multiplier = np.exp(-penalty_factor * severity)
        
        likelihoods *= penalty_multiplier

        scores = np.log(likelihoods + 1e-10)
        new_cumulative_scores = current_scores + scores

        return new_cumulative_scores
        
    def select_question(self, prior: np.ndarray, asked_features: list, question_count: int) -> tuple[str, str]:
        """
        Smart question selection.
        - Q1: Random from top features (Exploration/Variety)
        - Q2-Q14: Greedy Info Gain
        - Q15+: Discriminative (Tie-breaking)
        """
        asked_set = set(asked_features)
        candidates_indices = [idx for idx in self.allowed_feature_indices 
                              if self.feature_cols[idx] not in asked_set]

        if not candidates_indices:
            print("[Question] No more features to ask.")
            return None, None

        # Q1: Random exploration from top features (Kept as per request)
        if question_count == 0:
            top_20_pct = max(1, len(self.sorted_initial_feature_indices) // 5)
            available_top = [idx for idx in self.sorted_initial_feature_indices[:top_20_pct] 
                           if idx in candidates_indices]
            
            if available_top:
                best_feat_idx = np.random.choice(available_top)
                feature_name = self.feature_cols[best_feat_idx]
                question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                print(f"[Q1] EXPLORATION: Random question from top features")
                return feature_name, question_text

        # Check if we should use discriminative questions (Q15+)
        if question_count >= 14:
            top_idx = np.argmax(prior)
            top_prob = prior[top_idx]
            
            # If we have a leader (>40%) but haven't hit 94%, we need to specifically attack rivals
            if top_prob > 0.40 and top_prob < 0.94:
                discriminative_q = self.get_discriminative_question(top_idx, prior, asked_features)
                if discriminative_q[0] is not None:
                    print(f"[Q{question_count + 1}] DISCRIMINATIVE: Separating top candidate (prob={top_prob:.3f})")
                    return discriminative_q

        # Standard greedy information gain
        candidates_to_eval = self._select_candidate_subset(candidates_indices)
            
        if len(candidates_to_eval) == 0:
            return None, None

        gains = self._compute_gains_batched(prior, candidates_to_eval, batch_size=128)
        
        # Apply quality penalties to prefer clean data
        penalty_gains = np.ones_like(gains)
        for i, feat_idx in enumerate(candidates_to_eval):
            nan_frac = self.col_nan_frac[feat_idx]
            ambiguity = self.col_ambiguity[feat_idx]
            penalty = 1.0 - (0.6 * nan_frac + 0.5 * ambiguity)
            penalty = np.clip(penalty, 0.15, 1.0)
            penalty_gains[i] = gains[i] * penalty

        gains = penalty_gains
        
        # Select best gain
        best_local_idx = np.argmax(gains)
        best_feat_idx = candidates_to_eval[best_local_idx]
        
        feature_name = self.feature_cols[best_feat_idx]
        question_text = self.questions_map.get(feature_name, f"Does it have {feature_name}?")
        
        return feature_name, question_text

    def _select_candidate_subset(self, candidates_indices):
        """Helper to sample feature candidates efficiently."""
        if len(candidates_indices) <= 120:
            return np.array(candidates_indices, dtype=np.int32)
        
        scored_candidates = []
        for idx in candidates_indices:
            initial_rank = np.where(self.sorted_initial_feature_indices == idx)[0]
            rank_score = 1.0 / (initial_rank[0] + 1) if len(initial_rank) > 0 else 0
            var_score = self.col_var[idx] / (np.max(self.col_var[self.allowed_feature_indices]) + 1e-5)
            scored_candidates.append((idx, rank_score + var_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [idx for idx, _ in scored_candidates[:90]]
        
        other_candidates = [idx for idx in candidates_indices if idx not in top_candidates]
        if other_candidates:
            random_extras = np.random.choice(other_candidates, size=min(len(other_candidates), 30), replace=False)
            candidates_to_eval = np.unique(np.concatenate((top_candidates, random_extras))).astype(np.int32)
        else:
            candidates_to_eval = np.array(top_candidates, dtype=np.int32)
            
        return candidates_to_eval

    def get_discriminative_question(self, top_animal_idx: int, prior: np.ndarray, asked_features: list) -> tuple[str, str]:
        """Finds questions that best separate top candidate from rivals."""
        top_prob = prior[top_animal_idx]
        
        rival_threshold = top_prob * 0.20
        rival_mask = (prior > rival_threshold) & (np.arange(len(prior)) != top_animal_idx)
        rival_indices = np.where(rival_mask)[0]
        
        if len(rival_indices) == 0:
            rival_indices = np.where(prior > 0.001)[0]
            rival_indices = rival_indices[rival_indices != top_animal_idx]
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
        opinion_strength = 1.0 + 2.0 * np.abs(np.nan_to_num(top_feats, nan=0.5) - 0.5)
        weighted_diffs = diffs * opinion_strength
        
        best_local_idx = np.argmax(weighted_diffs)
        
        if weighted_diffs[best_local_idx] > 0.15:
            best_feat_idx = candidate_indices[best_local_idx]
            feature_name = self.feature_cols[best_feat_idx]
            return feature_name, self.questions_map.get(feature_name, f"Does it have {feature_name}?")
                      
        return None, None

    def should_make_guess(self, game_state: dict, probs: np.ndarray) -> tuple[bool, str | None, str | None]:
        """
        STRICT GUESSING LOGIC:
        Only guesses if probability >= 94%, OR if we hit the safety question limit (Q25).
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

        # Ratio of Top / Second best (e.g., 0.95 / 0.01 = 95)
        confidence_ratio = top_prob / (second_prob + 1e-9)
        
        # Prevent guessing too soon after a user rejection
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < 4:
                if q_count < 25: # Still honor forced guess
                    return False, None, None

        # --- TIER 1: SAFETY NET (Q25) ---
        # Must guess here to prevent infinite games, even if prob < 94%
        if q_count >= 25:
            print(f"[Q{q_count}] FORCED GUESS (Safety Net): prob={top_prob:.4f}")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'

        # --- TIER 2: ULTRA CONFIDENT (Early Game) ---
        if top_prob > 0.98 and confidence_ratio > 500 and q_count >= 5:
            print(f"[Q{q_count}] INSTANT CERTAINTY: prob={top_prob:.4f}")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'

        # --- TIER 3: HIGH PRECISION (Standard Win) ---
        # STRICT 94% THRESHOLD
        if top_prob >= 0.94 and confidence_ratio > 100 and q_count >= 10:
            print(f"[Q{q_count}] PRECISION GUESS: prob={top_prob:.4f}")
            game_state['has_made_initial_guess'] = True
            return True, top_animal, 'final'

        # No guess if below 94% and not at Q25
        return False, None, None
    
    def get_sparse_question_for_game(self, prior: np.ndarray, asked_features_names: set) -> tuple[str, str]:
        """Finds the best information-gain question from *only* sparse features."""
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
        
    def get_features_for_data_collection(self, item_name, num_features=5):
        """Gets a list of features for data collection for a specific item."""
        try:
            matches = np.where(self.animals == item_name)[0]
            if len(matches) == 0:
                matches = np.where(np.char.lower(self.animals.astype(str)) == item_name.lower())[0]
            
            if len(matches) == 0:
                return self._get_random_allowed_features(num_features)
            item_idx = matches[0]
            item_feats = self.features[item_idx]
        except Exception:
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
                    np.random.shuffle(remaining_allowed)
                    needed_more = num_features - len(useful_nan_indices)
                    selected_indices = np.concatenate((useful_nan_indices, remaining_allowed[:needed_more]))
                else:
                    selected_indices = useful_nan_indices
        else:
            np.random.shuffle(useful_nan_indices)
            selected_indices = useful_nan_indices[:num_features]
        
        if len(selected_indices) < num_features and len(self.allowed_feature_indices) >= num_features:
            remaining = np.setdiff1d(self.allowed_feature_indices, selected_indices).copy()
            np.random.shuffle(remaining)
            needed = num_features - len(selected_indices)
            selected_indices = np.concatenate((selected_indices, remaining[:needed]))
            
        return self._format_features(selected_indices[:num_features])
    
    def _get_random_allowed_features(self, num_features):
        if len(self.allowed_feature_indices) == 0:
            return []
        num_to_select = min(num_features, len(self.allowed_feature_indices))
        selected = np.random.choice(self.allowed_feature_indices, size=num_to_select, replace=False)
        return self._format_features(selected)
    
    def _format_features(self, indices):
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
    
    def get_all_feature_gains(self, initial_prior: np.ndarray = None) -> list[dict]:
        if initial_prior is None:
            if hasattr(self, 'uniform_prior'):
                initial_prior = self.uniform_prior
            else:
                if len(self.animals) == 0: return []
                initial_prior = np.ones(len(self.animals), dtype=np.float32) / len(self.animals)
        
        indices_to_calc = self.allowed_feature_indices
        if len(indices_to_calc) == 0: return []
        
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
        results.sort(key=lambda x: x['initial_gain'], reverse=True)
        return results
 
    def to_delete(self, similarity_threshold=0.85, min_variance=0.01, output_file='questions_to_delete.csv'):
            """Identifies and logs redundant, similar, or low-quality questions that should be deleted."""
            start_time = datetime.now()
            print(f"\n{'='*60}\nStarted redundancy analysis at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
            
            deletion_candidates = []
            
            # 1. Find features with excessive missing data (>95%)
            print("Checking for features with excessive missing data...")
            high_nan_mask = self.col_nan_frac >= 0.40
            high_nan_indices = np.where(high_nan_mask)[0]
            
            for idx in high_nan_indices:
                fname = self.feature_cols[idx]
                deletion_candidates.append({
                    'feature_name': fname,
                    'question_text': self.questions_map.get(fname, f"Does it have {fname}?"),
                    'reason': 'excessive_missing_data',
                    'nan_percentage': float(self.col_nan_frac[idx]),
                    'variance': float(self.col_var[idx]) if not np.isnan(self.col_var[idx]) else 0.0,
                    'correlation_with': None
                })
            
            # 2. Find low variance features
            print("Checking for low variance features...")
            valid_var_mask = ~np.isnan(self.col_var) & (self.col_var < min_variance)
            low_var_indices = np.where(valid_var_mask)[0]
            
            for idx in low_var_indices:
                if idx in high_nan_indices: continue
                fname = self.feature_cols[idx]
                deletion_candidates.append({
                    'feature_name': fname,
                    'question_text': self.questions_map.get(fname, f"Does it have {fname}?"),
                    'reason': 'low_variance',
                    'nan_percentage': float(self.col_nan_frac[idx]),
                    'variance': float(self.col_var[idx]),
                    'correlation_with': None
                })
            
            # 3. Find highly correlated (redundant) features
            print("Checking for highly correlated features...")
            allowed_indices = self.allowed_feature_indices
            redundant_pairs = []
            sample_size = min(len(allowed_indices), 200)
            indices_to_check = np.random.choice(allowed_indices, sample_size, replace=False) if len(allowed_indices) > sample_size else allowed_indices
            
            for i, idx1 in enumerate(indices_to_check):
                for idx2 in indices_to_check[i+1:]:
                    col1 = self.features[:, idx1]
                    col2 = self.features[:, idx2]
                    valid_mask = ~np.isnan(col1) & ~np.isnan(col2)
                    
                    # Skip if not enough overlapping data points
                    if np.sum(valid_mask) < 10: continue
                    
                    c1_masked = col1[valid_mask]
                    c2_masked = col2[valid_mask]
                    
                    # FIX: Check for zero variance in the masked subset to avoid RuntimeWarning
                    if np.std(c1_masked) == 0 or np.std(c2_masked) == 0:
                        continue

                    corr = np.corrcoef(c1_masked, c2_masked)[0, 1]
                    
                    if not np.isnan(corr) and abs(corr) > similarity_threshold:
                        if self.col_nan_frac[idx1] > self.col_nan_frac[idx2]:
                            redundant_pairs.append((idx1, idx2, corr))
                        else:
                            redundant_pairs.append((idx2, idx1, corr))
            
            seen_redundant = set()
            for redundant_idx, keeper_idx, corr in redundant_pairs:
                if redundant_idx in seen_redundant: continue
                fname_redundant = self.feature_cols[redundant_idx]
                fname_keeper = self.feature_cols[keeper_idx]
                
                deletion_candidates.append({
                    'feature_name': fname_redundant,
                    'question_text': self.questions_map.get(fname_redundant, f"Does it have {fname_redundant}?"),
                    'reason': 'highly_correlated',
                    'nan_percentage': float(self.col_nan_frac[redundant_idx]),
                    'variance': float(self.col_var[redundant_idx]) if not np.isnan(self.col_var[redundant_idx]) else 0.0,
                    'correlation_with': f"{fname_keeper} (r={corr:.3f})"
                })
                seen_redundant.add(redundant_idx)
            
            if deletion_candidates:
                df_delete = pd.DataFrame(deletion_candidates)
                df_delete['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
                df_delete = df_delete.sort_values(['reason', 'nan_percentage'], ascending=[True, False])
                df_delete.to_csv(output_file, index=False)
                return df_delete
            return pd.DataFrame()