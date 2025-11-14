import threading
import numpy as np

import db
from engine import AkinatorEngine
from state_manager import StateManager


class AkinatorService:
    """
    REFACTORED (V2.1): Manages game logic and state.
    
    CHANGELOG:
    - Lowered thresholds in 'get_next_question' to trigger "smart"
      confirmation logic earlier (prob > 0.40, entropy < 2.5).
      This is the primary fix for the "dumb" feeling.
    - Added 'get_feature_gains' to call the new engine function.
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
        engine = None # To hold one engine instance
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

        # --- Regarding your question about calculating gains ---
        # The 'to_delete' method below (which you had commented out)
        # is what analyzes features. You are correct to keep it
        # commented out in production.
        # I have added a *new*, safe endpoint (/admin/feature_gains)
        # to let you see the gains without writing a file.
        
        if engine: # If at least one engine loaded
            print(f"Running 'to_delete' analysis (will save to questions_to_delete.csv)...")
            # Run the analysis
            df_to_delete = engine.to_delete(
                similarity_threshold=0.85,  # Correlation threshold
                min_variance=0.01,          # Minimum useful variance
                output_file='questions_to_delete.csv'
            )
            """Pushing to prod and do not want this method to run uncomment to run it
            This method in the engine.py can be run on the development environment and it will save a csv file on low variance or bad quality data
            ***NEVER PUSH TO PRODUCTION WITH THIS UNCOMMENTED***
            print(df_to_delete)
            """

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
            
        # Migrate state if needed (e.g., new animals added)
        game_state = self.state_manager.migrate_state(
            game_state, 
            len(engine.animals),
            engine.animals
        )

        # Ensure core state arrays exist (from state_manager)
        if 'cumulative_scores' not in game_state or len(game_state['cumulative_scores']) != len(engine.animals):
            game_state['cumulative_scores'] = np.zeros(len(engine.animals), dtype=np.float32)
        
        if 'answer_history' not in game_state:
            game_state['answer_history'] = []

        return engine, game_state

    # --- NEW HELPER ---
    def _get_probs_and_state(self, game_state: dict) -> tuple[np.ndarray, dict, AkinatorEngine]:
        """
        Calculates probabilities on-the-fly from cumulative_scores using softmax.
        This is the core of the new fuzzy logic system.
        """
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        scores = game_state['cumulative_scores'].copy()
        mask = game_state['rejected_mask']
        
        # Apply mask: rejected animals have no chance
        scores[mask] = -np.inf
        
        # Softmax calculation (stable version)
        if len(scores) > 0:
            e_x = np.exp(scores - np.max(scores))
            probs = e_x / (e_x.sum() + 1e-10)
        else:
            probs = np.array([], dtype=np.float32)

        probs[mask] = 0.0  # Ensure rejected are 0
        
        return probs, game_state, engine
    # --- END NEW HELPER ---

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
            
            # Get base state from manager (now includes 'cumulative_scores')
            initial_state = self.state_manager.create_initial_state(
                domain_name, 
                len(engine.animals)
            )
            
            return initial_state

    def get_top_predictions(self, game_state: dict, n: int = 5) -> list[dict]:
        """Gets top N predictions with caching."""
        # Cache key is now based on 'cumulative_scores' (see state_manager)
        state_key = self.state_manager.get_state_cache_key(game_state, n)
        
        with self._cache_lock:
            if state_key in self._prediction_cache:
                return self._prediction_cache[state_key]
        
        # --- REFACTORED ---
        # Get on-the-fly probabilities
        probs, game_state, engine = self._get_probs_and_state(game_state)
        # --- END REFACTOR ---
            
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
        Delegates guessing logic to the engine, passing in the
        on-the-fly probabilities.
        """
        # --- REFACTORED ---
        probs, game_state, engine = self._get_probs_and_state(game_state)
        
        # Pass both game_state (for scores) and probs (for confidence)
        return engine.should_make_guess(game_state, probs)
        # --- END REFACTOR ---

    def activate_continue_mode(self, game_state: dict) -> dict:
        """Sets game to continue asking questions after wrong guess."""
        game_state['continue_mode'] = True
        game_state['questions_since_last_guess'] = 0
        game_state['middle_guess_made'] = False # This can likely be removed if not used
        return game_state

    def get_next_question(self, game_state: dict) -> tuple[str | None, str | None, dict]:
        """
        ### MODIFIED (This is the "SMART" fix) ###
        Gets the next question. If confidence is high, it switches
        to a "confirmation" strategy to ask a discriminative question.
        Otherwise, it uses the "exploration" (information gain) strategy.
        """
        # --- REFACTORED ---
        # Get on-the-fly probs to use as the 'prior' for info gain
        prior, game_state, engine = self._get_probs_and_state(game_state)
        # --- END REFACTOR ---

        asked = game_state['asked_features']
        q_count = game_state['question_count']
        
        # First question optimization
        if q_count == 0 and hasattr(engine, 'sorted_initial_feature_indices'):
            # Pass uniform prior (which 'prior' is at q_count=0)
            feature, q = engine.select_question(prior, asked, q_count)
            return feature, q, game_state
        
        # --- "SMART" FIX: "CONFIRMATION" STRATEGY ---
        top_prob = np.max(prior)
        entropy = engine._calculate_entropy(prior)
        
        # If confidence is high (e.g., >40% prob, low-ish entropy),
        # try to find a "confirmation" question.
        # --- THRESHOLDS LOWERED ---
        if top_prob > 0.40 and entropy < 2.5 and q_count > 5:
        # --- END THRESHOLD CHANGE ---
            top_idx = np.argmax(prior)
            feature, q = engine.get_discriminative_question(top_idx, prior, asked)
            
            if feature:
                print(f"[Question] CONFIRM: Asking discriminative question for {engine.animals[top_idx]} (prob={top_prob:.2f})")
                return feature, q, game_state
            # If no good confirmation question found, fall through to exploration
        
        # --- EXPLORATION STRATEGY (default) ---
        print(f"[Question] EXPLORE: Finding best split across {np.sum(prior > 0.001):.0f} items (entropy={entropy:.2f})")
        feature, q = engine.select_question(prior, asked, q_count)
        
        return feature, q, game_state
    
    def process_answer(self, game_state: dict, feature: str, answer: str) -> dict:
        """
        Updates game state based on an answer using the new engine.
        """
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        if feature not in engine.feature_cols:
            print(f"Warning: Feature '{feature}' not in engine. Skipping.")
            return game_state
            
        # Find the index for the feature name
        feature_idx = np.where(engine.feature_cols == feature)[0]
        if len(feature_idx) == 0:
             print(f"Warning: Feature '{feature}' not in engine column list. Skipping.")
             return game_state
        feature_idx = int(feature_idx[0])

        # --- REFACTORED ---
        # Get current scores from state
        current_scores = game_state['cumulative_scores']

        # Call the new stateless update function
        new_scores = engine.update(
            feature_idx,
            answer,
            current_scores
        )
        
        # Store the new scores back into the game state
        game_state['cumulative_scores'] = new_scores
        # 'probabilities' are no longer stored or updated here.
        # --- END REFACTOR ---
        
        # Also add to answer_history for the report
        if 'answer_history' not in game_state:
             game_state['answer_history'] = []
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
                
                # --- REFACTORED ---
                # The only action needed is to update the mask.
                # The softmax helper (`_get_probs_and_state`) will see this
                # mask and assign -inf score, resulting in 0 probability.
                game_state['rejected_mask'][idx] = True
                # All 'probabilities' manipulation is removed.
                # --- END REFACTOR ---
            else:
                 print(f"Warning: Cannot reject '{animal_name}', not found.")
                
        except Exception as e:
            print(f"Error rejecting '{animal_name}': {e}")
            
        return game_state

    def get_features_for_data_collection(self, domain_name: str, 
                                    item_name: str) -> list[dict]:
        """Gets 5 features for data collection (unchanged)."""
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found.")
        
        return engine.get_features_for_data_collection(item_name, num_features=5)

    # --- NEW FUNCTION ---
    def get_feature_gains(self, domain_name: str) -> list[dict]:
        """
        Calls the engine to get all feature gains.
        """
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found.")
        
        return engine.get_all_feature_gains()
    # --- END NEW FUNCTION ---

    def record_suggestion(self, animal_name: str, 
                         answered_features: dict, domain_name: str) -> str:
        """Persists a suggestion for an existing animal (unchanged)."""
        return db.persist_suggestion(animal_name, answered_features, domain_name)

    def learn_new_animal(self, animal_name: str, 
                        answered_features: dict, domain_name: str) -> str:
        """Persists a new animal to the database (unchanged)."""
        return db.persist_new_animal(animal_name, answered_features, domain_name)

    def start_engine_reload(self):
        """Starts background reload of all engines (unchanged)."""
        threading.Thread(target=self._background_reload, daemon=True).start()

    def get_game_report(self, domain_name: str, item_name: str, 
                        user_answers: dict, is_new: bool) -> dict:
        """
        Generates a report comparing user answers to consensus answers (unchanged).
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
                    feature_idx_list = np.where(engine.feature_cols == feature_name)[0]
                    if len(feature_idx_list) > 0:
                        feature_idx = int(feature_idx_list[0])
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