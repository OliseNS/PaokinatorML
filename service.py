import threading
import numpy as np

import db
from engine import AkinatorEngine
from state_manager import StateManager


class AkinatorService:
    """
    Manages multiple AkinatorEngine instances (one per domain) and game logic.
    (Updated get_next_question for smarter strategy)
    """
    
    def __init__(self):
        print("Initializing AkinatorService...")
        self.engines_lock = threading.Lock()
        self.engines: dict[str, AkinatorEngine] = {}
        self.state_manager = StateManager()
        
        # Performance caches
        self._prediction_cache = {}
        self._cache_lock = threading.Lock()
        
        # Load all engines on startup
        self._load_all_engines()
        print(f"✅ Service initialized with {len(self.engines)} domains.")

    def _create_engine(self, domain_name: str) -> AkinatorEngine:
        """Loads data from DB and builds a new engine instance."""
        print(f"   Loading data for domain '{domain_name}'...")
        try:
            df, feature_cols, questions_map = db.load_data_from_supabase(domain_name)
            engine = AkinatorEngine(df, feature_cols, questions_map)
            print(f"   ✓ Engine for '{domain_name}' created with {len(engine.animals)} items.")
            return engine
        except Exception as e:
            print(f"   ✗ ERROR: Failed to create engine for '{domain_name}': {e}")
            return None

    def _load_all_engines(self):
        """Loads all available domains and builds engines."""
        print("Loading all engines...")
        domain_names = db.get_all_domains()
        if not domain_names:
            print("❌ CRITICAL: No domains found in database.")
            return

        new_engines: dict[str, AkinatorEngine] = {}
        for domain in domain_names:
            engine = self._create_engine(domain)
            if engine:
                new_engines[domain] = engine
        
        # Atomic swap
        with self.engines_lock:
            self.engines = new_engines
        
        # Clear prediction cache
        with self._cache_lock:
            self._prediction_cache.clear()
            
        print(f"✅ All engines loaded. Serving {len(self.engines)} domains.")

    def _background_reload(self):
        """Runs in a thread to reload all engines."""
        print("🚀 Starting background engine reload...")
        try:
            self._load_all_engines()
            print(f"✅ Engine hot-swap complete.")
        except Exception as e:
            print(f"❌ ERROR: Background reload failed: {e}")

    def _get_engine_and_migrate_state(self, game_state: dict) -> tuple[AkinatorEngine, dict]:
        """
        Gets the correct engine and migrates state if necessary.
        MUST be called within self.engines_lock.
        """
        domain_name = game_state.get('domain_name')
        if not domain_name:
            raise ValueError("Game state is missing 'domain_name'")
        
        engine = self.engines.get(domain_name)
        if not engine:
            print(f"Warning: No engine for '{domain_name}'. Using fallback.")
            if not self.engines:
                raise ValueError("No engines loaded.")
            domain_name, engine = next(iter(self.engines.items()))
            game_state['domain_name'] = domain_name
            
        # Migrate state if needed
        game_state = self.state_manager.migrate_state(
            game_state, 
            len(engine.animals),
            engine.animals
        )

        # --- FIX: Ensure consistency state exists in game_state ---
        # This moves state from the engine to the session
        if 'cumulative_scores' not in game_state or len(game_state['cumulative_scores']) != len(engine.animals):
            # If mismatch (e.g., new engine), reset it.
            game_state['cumulative_scores'] = np.zeros(len(engine.animals), dtype=np.float32)
        
        if 'answer_history' not in game_state:
            game_state['answer_history'] = []
        # --- END FIX ---

        return engine, game_state

    # --- Public API Methods ---

    def get_available_domains(self) -> list[str]:
        """Returns list of loaded domain names."""
        with self.engines_lock:
            return list(self.engines.keys())

    def create_initial_state(self, domain_name: str) -> dict:
        """Creates a new game session state."""
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found.")
            
            # Get base state from manager
            initial_state = self.state_manager.create_initial_state(
                domain_name, 
                len(engine.animals)
            )

            # --- FIX: Add consistency state to the new game_state ---
            if 'cumulative_scores' not in initial_state:
                initial_state['cumulative_scores'] = np.zeros(len(engine.animals), dtype=np.float32)
            if 'answer_history' not in initial_state:
                 initial_state['answer_history'] = []
            # --- END FIX ---
            
            return initial_state

    def get_top_predictions(self, game_state: dict, n: int = 5) -> list[dict]:
        """Gets top N predictions with caching."""
        state_key = self.state_manager.get_state_cache_key(game_state, n)
        
        with self._cache_lock:
            if state_key in self._prediction_cache:
                return self._prediction_cache[state_key]
        
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        probs = game_state['probabilities'].copy()
        mask = game_state['rejected_mask']
        probs[mask] = 0.0
        
        sorted_indices = np.argsort(probs)[::-1]
        
        top_n = min(n, len(sorted_indices))
        results = []
        for i in range(top_n):
            idx = sorted_indices[i]
            prob = probs[idx]
            if prob < 0.001:
                break
            results.append({
                'animal': engine.animals[idx],
                'probability': float(prob)
            })
        
        # Cache result
        with self._cache_lock:
            self._prediction_cache[state_key] = results
            # Limit cache size
            if len(self._prediction_cache) > 1000:
                oldest_keys = list(self._prediction_cache.keys())[:100]
                for key in oldest_keys:
                    self._prediction_cache.pop(key, None)
        
        return results

    def should_make_guess(self, game_state: dict) -> tuple[bool, str | None, str | None]:
        """
        Delegates all guessing logic to the (now much more patient) engine.
        """
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
        
        # Delegate directly to the engine
        return engine.should_make_guess(game_state)

    def activate_continue_mode(self, game_state: dict) -> dict:
        """Sets game to continue asking questions after wrong guess."""
        game_state['continue_mode'] = True
        game_state['questions_since_last_guess'] = 0
        game_state['middle_guess_made'] = False
        return game_state
    def get_next_question(self, game_state: dict) -> tuple[str | None, str | None, dict]:
        """
        BALANCED STRATEGY: Explore early, disambiguate when confident
        
        - Early game (entropy high): Pure exploration via info gain
        - Late game (top candidate emerges): Discriminate between similar items
        - Prevents premature convergence while ensuring precise identification
        """
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)

        prior = game_state['probabilities'].copy()
        prior[game_state['rejected_mask']] = 0.0
        
        prior_sum = prior.sum()
        if prior_sum < 1e-10:
            prior = np.ones_like(prior)
            prior[game_state['rejected_mask']] = 0.0
            prior_sum = prior.sum()
            if prior_sum < 1e-10:
                return None, None, game_state
                
        prior = prior / (prior_sum + 1e-10)
        
        asked = game_state['asked_features']
        q_count = game_state['question_count']
        
        # First question optimization
        if q_count == 0 and hasattr(engine, 'sorted_initial_feature_indices'):
            feature, q = engine.select_question(prior, asked, q_count)
            return feature, q, game_state
        
        # Calculate metrics for strategy decision
        top_idx = np.argmax(prior)
        top_prob = prior[top_idx]
        
        rival_threshold = top_prob * 0.3  # Items within 30% of top probability
        num_rivals = np.sum((prior > rival_threshold) & (np.arange(len(prior)) != top_idx))
        
        entropy = engine._calculate_entropy(prior)
        
        
        should_disambiguate = (
            top_prob > 0.20 and 
            2 <= num_rivals <= 10 and 
            entropy < 3.0 and 
            q_count > 8
        )
        
        if should_disambiguate:
            print(f"[Question] DISAMBIGUATE: top_prob={top_prob:.3f}, rivals={num_rivals}, entropy={entropy:.2f}")
            
            # Try to find a discriminative question
            feature, q = engine.get_discriminative_question(top_idx, prior, asked)
            
            if feature and q:
                print(f"  → Found discriminative question: {q[:60]}...")
                return feature, q, game_state
            else:
                print(f"  → No discriminative question found, falling back to exploration")
        
        # --- EXPLORATION STRATEGY (default) ---
        # Used when:
        # - Early game (high entropy, no clear leader)
        # - No suitable rivals to discriminate between
        # - Discriminative question search failed
        
        print(f"[Question] EXPLORE: Finding best split across {np.sum(prior > 0.001):.0f} items (entropy={entropy:.2f})")
        feature, q = engine.select_question(prior, asked, q_count)
        
        return feature, q, game_state
    
    def process_answer(self, game_state: dict, feature: str, answer: str) -> dict:
        """Updates game state based on an answer."""
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        if feature not in engine.feature_cols:
            print(f"Warning: Feature '{feature}' not in engine. Skipping.")
            return game_state
            
        feature_idx = engine.feature_cols.index(feature)
        
        prior = game_state['probabilities'].copy()
        prior[game_state['rejected_mask']] = 0.0
        
        prior_sum = prior.sum()
        if prior_sum < 1e-10:
             return game_state
             
        prior = prior / (prior_sum + 1e-10)
        
        # --- FIX: Pass state to update() and get new state back ---
        current_scores = game_state['cumulative_scores']
        q_count = game_state['question_count'] # This is the length of answer history

        posterior, new_scores = engine.update(
            prior,
            feature_idx,
            answer,
            current_scores,
            q_count
        )
        
        game_state['probabilities'] = posterior
        game_state['cumulative_scores'] = new_scores
        # --- END FIX ---
        
        # Also add to answer_history for the report
        game_state['answer_history'].append({'feature': feature, 'answer': answer})
        
        return game_state

    def reject_guess(self, game_state: dict, animal_name: str) -> dict:
        """Marks an animal as rejected."""
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        try:
            idx_list = np.where(engine.animals == animal_name)[0]
            if len(idx_list) > 0:
                idx = idx_list[0]
                game_state['rejected_mask'][idx] = True
                game_state['probabilities'][idx] = 0.0 
                
                total = game_state['probabilities'].sum()
                if total > 1e-10:
                    game_state['probabilities'] = game_state['probabilities'] / total
            else:
                 print(f"Warning: Cannot reject '{animal_name}', not found.")
                
        except Exception as e:
            print(f"Error rejecting '{animal_name}': {e}")
            
        return game_state

    def get_features_for_data_collection(self, domain_name: str, 
                                    item_name: str) -> list[dict]:
        """Gets 5 features for data collection."""
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found.")
        
        return engine.get_features_for_data_collection(item_name, num_features=5)

    def record_suggestion(self, animal_name: str, 
                         answered_features: dict, domain_name: str) -> str:
        """Persists a suggestion for an existing animal."""
        return db.persist_suggestion(animal_name, answered_features, domain_name)

    def learn_new_animal(self, animal_name: str, 
                        answered_features: dict, domain_name: str) -> str:
        """Persists a new animal to the database."""
        return db.persist_new_animal(animal_name, answered_features, domain_name)

    def start_engine_reload(self):
        """Starts background reload of all engines."""
        threading.Thread(target=self._background_reload, daemon=True).start()

    def get_game_report(self, domain_name: str, item_name: str, 
                        user_answers: dict, is_new: bool) -> dict:
        """
        Generates a report comparing user answers to consensus answers.
        """
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found.")
        
        questions_report = []
        item_idx = -1
        
        # Handle new items
        if is_new:
            report_name = f"{item_name} (New Item)"
        
        # Handle existing items
        else:
            report_name = item_name
            try:
                item_idx = np.where(engine.animals == item_name)[0][0]
            except (IndexError, TypeError):
                # Fallback: Item not found, treat as new
                report_name = f"{item_name} (Item not found)"
                is_new = True

        # Build the question list
        for feature_name, user_value in user_answers.items():
            question_text = engine.questions_map.get(
                feature_name, 
                f"Is it/does it have {feature_name.replace('_', ' ')}?"
            )
            
            consensus_value = None
            if not is_new and item_idx != -1:
                try:
                    # Find the feature's column index
                    feature_idx = engine.feature_cols.index(feature_name)
                    # Get the consensus value from the engine's feature matrix
                    consensus_raw = engine.features[item_idx, feature_idx]
                    # Convert np.nan to None
                    if not np.isnan(consensus_raw):
                        consensus_value = float(consensus_raw)
                except (ValueError, IndexError):
                    # Feature not in engine (e.g., new feature), skip consensus
                    pass
            
            questions_report.append({
                "question": question_text,
                "feature": feature_name,
                "user_answer": float(user_value),
                "consensus_answer": consensus_value
            })

        return {
            "item_name": report_name,
            "is_new_item": is_new,
            "questions": questions_report
        }