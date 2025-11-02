import json
import threading
import numpy as np

# Import our project files
import db
from engine import AkinatorEngine

class AkinatorService:
    """
    Manages multiple AkinatorEngine instances (one per domain) and game logic.
    
    This class handles the "hot-swap" of all engines when new
    data is learned, and seamlessly migrates active game sessions
    to the new engine's array shapes.
    """
    
    def __init__(self):
        print("Initializing AkinatorService...")
        self.engines_lock = threading.Lock()
        self.engines: dict[str, AkinatorEngine] = {}
        
        # Performance caches
        self._prediction_cache = {}  # Cache for top predictions
        self._cache_lock = threading.Lock()
        
        # Load all engines blocking on the first startup
        self._load_all_engines()
        print(f"✅ Service initialized with {len(self.engines)} domains.")

    def _create_engine(self, domain_name: str) -> AkinatorEngine:
        """Loads data from DB for a specific domain and builds a new engine instance."""
        print(f"   Loading data for domain '{domain_name}'...")
        try:
            df, feature_cols, questions_map = db.load_data_from_supabase(domain_name)
            engine = AkinatorEngine(df, feature_cols, questions_map)
            print(f"   ✓ Engine for '{domain_name}' created with {len(engine.animals)} items.")
            return engine
        except Exception as e:
            print(f"   ✗ ERROR: Failed to create engine for '{domain_name}': {e}")
            return None # Return None on failure

    def _load_all_engines(self):
        """
        Loads all available domains from the database and builds
        an engine for each one.
        """
        print("Loading all engines...")
        domain_names = db.get_all_domains()
        if not domain_names:
            print("❌ CRITICAL: No domains found in database.")
            return

        new_engines: dict[str, AkinatorEngine] = {}
        for domain in domain_names:
            engine = self._create_engine(domain)
            if engine: # Only add if creation was successful
                new_engines[domain] = engine
        
        # Atomic swap
        with self.engines_lock:
            self.engines = new_engines
        
        # Clear prediction cache when engines change
        with self._cache_lock:
            self._prediction_cache.clear()
            
        print(f"✅ All engines loaded. Serving {len(self.engines)} domains.")

    def _background_reload(self):
        """
        Runs in a separate thread to build new engines for all
        domains and hot-swap them atomically.
        """
        print("🚀 Starting background engine reload for all domains...")
        try:
            self._load_all_engines() # This now reloads everything
            print(f"✅ All engines hot-swap complete.")
        except Exception as e:
            print(f"❌ ERROR: Background reload failed: {e}")

    def _get_engine_and_migrate_state(self, game_state: dict) -> tuple[AkinatorEngine, dict]:
        """
        Gets the correct engine for the game state's domain
        and migrates the state if necessary.
        
        This MUST be called within the 'self.engines_lock'.
        """
        domain_name = game_state.get('domain_name')
        if not domain_name:
            raise ValueError("Game state is missing 'domain_name'")
        
        engine = self.engines.get(domain_name)
        if not engine:
            print(f"Warning: No engine found for domain '{domain_name}'. Using default.")
            # Fallback to the first available engine if domain is missing
            # This is a safety measure, but ideally should not be hit.
            if not self.engines:
                raise ValueError("No engines are loaded at all.")
            domain_name, engine = next(iter(self.engines.items()))
            game_state['domain_name'] = domain_name # Correct the state
            
        # Now that we have the correct engine, migrate the state
        game_state = self._migrate_state(game_state, engine)
        return engine, game_state

    def _migrate_state(self, game_state: dict, engine: AkinatorEngine) -> dict:
        """
        Ensures the game state arrays match the current engine's dimensions.
        This is the core of the seamless "hot-swap".
        """
        current_n = len(engine.animals)
        state_n = game_state.get('animal_count', 0)
        
        # If counts match, state is valid.
        if state_n == current_n:
            return game_state

        print(f"🔄 Migrating session state for domain '{game_state.get('domain_name')}' from {state_n} to {current_n} items.")
        
        # --- Migrate Probabilities ---
        old_probs = game_state['probabilities']
        new_probs = np.ones(current_n, dtype=np.float32)
        
        copy_len = min(state_n, current_n)
        if copy_len > 0:
            new_probs[:copy_len] = old_probs[:copy_len]
        
        if current_n > state_n:
            fill_prob = np.mean(old_probs) if state_n > 0 else (1.0 / current_n)
            new_probs[state_n:] = max(fill_prob, 1e-9) # Ensure non-zero
        
        # Re-normalize
        new_probs_sum = new_probs.sum()
        game_state['probabilities'] = new_probs / (new_probs_sum + 1e-10)
        
        # --- Migrate Rejected Mask ---
        old_mask = game_state['rejected_mask']
        new_mask = np.zeros(current_n, dtype=bool)
        if copy_len > 0:
            new_mask[:copy_len] = old_mask[:copy_len]
        game_state['rejected_mask'] = new_mask
        
        # --- Update animal count ---
        game_state['animal_count'] = current_n
        
        return game_state

    def _get_state_key(self, game_state: dict, n: int) -> str:
        """Create a cache key for the game state."""
        # Create a hashable key from the state
        probs_hash = hash(tuple(game_state['probabilities'].tolist()))
        mask_hash = hash(tuple(game_state['rejected_mask'].tolist()))
        asked_hash = hash(tuple(sorted(game_state['asked_features'])))
        domain_hash = hash(game_state.get('domain_name', ''))
        return f"{domain_hash}_{probs_hash}_{mask_hash}_{asked_hash}_{n}"

    # --- Public API Methods (Called by FastAPI) ---

    def get_available_domains(self) -> list[str]:
        """Returns a list of loaded domain names."""
        with self.engines_lock:
            return list(self.engines.keys())

    def create_initial_state(self, domain_name: str) -> dict:
        """Creates a new game session state for a specific domain."""
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found or not loaded.")
            
            n_animals = len(engine.animals)
            
            # Initial prior: uniform distribution
            probabilities = np.ones(n_animals, dtype=np.float32) / (n_animals + 1e-10)
            
            state = {
                'domain_name': domain_name, # CRITICAL: Store the domain
                'probabilities': probabilities,
                'rejected_mask': np.zeros(n_animals, dtype=bool),
                'asked_features': [],
                'answered_features': {},
                'question_count': 0,
                'middle_guess_made': False,
                'animal_count': n_animals,  # CRITICAL: Stamp the state with animal count
                'continue_mode': False, # <-- NEW
                'questions_since_last_guess': 0, # <-- NEW
                'last_guess_type': None # <-- NEW
            }
            return state

    def get_top_predictions(self, game_state: dict, n: int = 5) -> list[dict]:
        """Gets top N predictions from a game state with caching."""
        state_key = self._get_state_key(game_state, n)
        
        with self._cache_lock:
            if state_key in self._prediction_cache:
                return self._prediction_cache[state_key]
        
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        # Now we have a valid state and engine, proceed without the lock
        probs = game_state['probabilities'].copy() # Copy to avoid mutation
        mask = game_state['rejected_mask']
        
        probs[mask] = 0.0
        
        # Get sorted indices
        sorted_indices = np.argsort(probs)[::-1]
        
        top_n = min(n, len(sorted_indices))
        results = []
        for i in range(top_n):
            idx = sorted_indices[i]
            prob = probs[idx]
            if prob < 0.001:
                break
            results.append({
                'animal': engine.animals[idx], # 'animal' is generic item name
                'probability': float(prob)
            })
        
        with self._cache_lock:
            self._prediction_cache[state_key] = results
            if len(self._prediction_cache) > 1000:
                oldest_keys = list(self._prediction_cache.keys())[:100]
                for key in oldest_keys:
                    self._prediction_cache.pop(key, None)
        
        return results

    def should_make_guess(self, game_state: dict) -> tuple[bool, str | None, str | None]:
        """
        Determines if the engine should make a guess.
        This method mutates the game_state for 'middle_guess_made' and 'continue_mode'.
        """
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        q_count = game_state['question_count']
        probs = game_state['probabilities'].copy()
        mask = game_state['rejected_mask']
        probs[mask] = 0.0 

        if probs.sum() < 1e-10:
            return False, None, None
        
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        top_animal = engine.animals[top_idx]
        
        probs[top_idx] = 0.0
        second_prob = np.max(probs)
        
        confidence_ratio = top_prob / (second_prob + 1e-9)
        
        # --- NEW CONTINUE LOGIC ---
        QUESTIONS_TO_WAIT = 3 # Ask 3 more questions before re-guessing
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < QUESTIONS_TO_WAIT:
                return False, None, None # Force more questions
            else:
                # Time to guess again. Reset flags and fall through to final guess.
                game_state['continue_mode'] = False
                game_state['questions_since_last_guess'] = 0
        # --- END NEW LOGIC ---

        if (
            q_count in [5, 10, 15] and 
            top_prob > 0.3 and
            not game_state['middle_guess_made']
        ):
            game_state['middle_guess_made'] = True # Mutates state
            return True, top_animal, 'middle'
            
        if (top_prob > 0.7 and confidence_ratio > 5.0 and q_count > 5):
            return True, top_animal, 'final'
        if (top_prob > 0.5 and confidence_ratio > 10.0 and q_count > 8):
            return True, top_animal, 'final'
        if q_count >= 20:
            return True, top_animal, 'final'
            
        return False, None, None

    # --- NEW METHOD ---
    def activate_continue_mode(self, game_state: dict) -> dict:
        """Sets the game state to continue asking questions."""
        game_state['continue_mode'] = True
        game_state['questions_since_last_guess'] = 0
        game_state['middle_guess_made'] = False # Allow another middle guess
        return game_state
    # --- END NEW METHOD ---

    def get_next_question(self, game_state: dict) -> tuple[str | None, str | None, dict]:
        """Gets the next best question, returns (feature, question, modified_state)."""
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
        
        if q_count == 0 and hasattr(engine, 'sorted_initial_feature_indices'):
            feature, q = engine.select_question(prior, asked, q_count)
            return feature, q, game_state
        
        top_idx = np.argmax(prior)
        top_prob = prior[top_idx]
        if top_prob > 0.2:
            feature, q = engine.get_discriminative_question(top_idx, prior, asked)
            if feature:
                return feature, q, game_state

        feature, q = engine.select_question(prior, asked, q_count)
        return feature, q, game_state

    def process_answer(self, game_state: dict, feature: str, answer: str) -> dict:
        """Updates game state based on an answer."""
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        if feature not in engine.feature_cols:
            print(f"Warning: Feature '{feature}' not in engine for domain '{game_state.get('domain_name')}'. Skipping update.")
            return game_state
            
        feature_idx = engine.feature_cols.index(feature)
        
        prior = game_state['probabilities'].copy()
        prior[game_state['rejected_mask']] = 0.0
        
        prior_sum = prior.sum()
        if prior_sum < 1e-10:
             return game_state
             
        prior = prior / (prior_sum + 1e-10)
        
        posterior = engine.update(prior, feature_idx, answer)
        game_state['probabilities'] = posterior
        
        return game_state

    def reject_guess(self, game_state: dict, animal_name: str) -> dict:
        """Marks an animal as rejected in the game state."""
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        try:
            # Find index of animal in the *current* engine
            idx_list = np.where(engine.animals == animal_name)[0]
            if len(idx_list) > 0:
                idx = idx_list[0]
                game_state['rejected_mask'][idx] = True
                game_state['probabilities'][idx] = 0.0 
                
                total = game_state['probabilities'].sum()
                if total > 1e-10:
                    game_state['probabilities'] = game_state['probabilities'] / total
            else:
                 print(f"Warning: Could not reject '{animal_name}', not found in engine for domain '{game_state.get('domain_name')}'.")
                
        except Exception as e:
            print(f"Error rejecting '{animal_name}': {e}")
            
        return game_state

    def get_data_collection_features(self, domain_name: str, item_name: str) -> list[dict]:
        """
        Gets 5 features for data collection for a specific item.
        Prioritizes nulls for that item, then pads with sparse features.
        """
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine:
                raise ValueError(f"Domain '{domain_name}' not found.")
        
        # Call the engine method (it handles item-not-found logic)
        # We don't need the lock for this part as it's read-only
        return engine.get_features_for_data_collection(item_name, num_features=5)

    def record_suggestion(self, animal_name: str, answered_features: dict, domain_name: str) -> str:
        """
        Persists a "suggestion" (a completed game for an EXISTING
        animal) to the database for a specific domain.
        """
        result = db.persist_suggestion(animal_name, answered_features, domain_name)
        return result

    def learn_new_animal(self, animal_name: str, answered_features: dict, domain_name: str) -> str:
        """
        Persists a NEW animal to the database for a specific domain.
        """
        result = db.persist_new_animal(animal_name, answered_features, domain_name)
        return result

    def start_engine_reload(self):
        """
        Starts a background engine reload for ALL domains.
        """
        threading.Thread(target=self._background_reload, daemon=True).start()