import threading
import numpy as np

import db
from engine import AkinatorEngine
from state_manager import StateManager


class AkinatorService:
    """
    Manages multiple AkinatorEngine instances (one per domain) and game logic.
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
            
            return self.state_manager.create_initial_state(
                domain_name, 
                len(engine.animals)
            )

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
        *** MASSIVELY STRICTER *** guessing logic.
        
        Now requires:
        - Much higher confidence (0.85-0.99+)
        - Much higher separation ratios (15x-50x+)
        - Lower entropy thresholds
        - Minimum question counts
        - Active candidate analysis
        
        Returns: (should_guess, animal_name, guess_type)
        """
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        q_count = game_state['question_count']
        probs = game_state['probabilities'].copy()
        mask = game_state['rejected_mask']
        probs[mask] = 0.0 

        if probs.sum() < 1e-10:
            return False, None, None
        
        # Normalize
        probs = probs / probs.sum()
        
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        top_animal = engine.animals[top_idx]
        
        # Calculate separation from second place
        probs_copy = probs.copy()
        probs_copy[top_idx] = 0.0
        second_prob = np.max(probs_copy)
        confidence_ratio = top_prob / (second_prob + 1e-9)
        
        # Calculate entropy
        entropy = self._calculate_entropy(probs)
        
        # Count significant candidates (prob > 0.05)
        significant_candidates = np.sum(probs > 0.05)
        
        # Count viable candidates (prob > 0.01)
        viable_candidates = np.sum(probs > 0.01)
        
        # --- Continue Mode Logic ---
        QUESTIONS_TO_WAIT = 4  # Increased from 3
        if game_state.get('continue_mode', False):
            if game_state.get('questions_since_last_guess', 0) < QUESTIONS_TO_WAIT:
                return False, None, None
            else:
                # Reset and allow final guess with stricter threshold
                game_state['continue_mode'] = False
                game_state['questions_since_last_guess'] = 0

        # --- REMOVED MIDDLE GUESSES ---
        # Middle guesses are disruptive and don't help accuracy
        # Removed entirely
        
        # --- (NEW) STRICTER FINAL GUESS LOGIC ---
        
        # Early game (< 10 questions): NEVER guess unless absolutely certain
        if q_count < 10:
            if (top_prob > 0.99 and  # TIGHTENED
                confidence_ratio > 60.0 and  # TIGHTENED
                entropy < 0.10 and  # TIGHTENED
                significant_candidates <= 1):
                print(f"[GUESS] Early game confidence: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}, ent={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Mid game (10-15 questions): Very strict
        if q_count < 15:
            if (top_prob > 0.97 and  # TIGHTENED
                confidence_ratio > 40.0 and  # TIGHTENED
                entropy < 0.20 and  # TIGHTENED
                significant_candidates <= 2):
                print(f"[GUESS] Mid game confidence: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}, ent={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Late mid game (15-20 questions): Strict
        if q_count < 20:
            if (top_prob > 0.92 and  # TIGHTENED
                confidence_ratio > 25.0 and  # TIGHTENED
                entropy < 0.35 and  # TIGHTENED
                significant_candidates <= 3):
                print(f"[GUESS] Late mid confidence: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}, ent={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Late game (20-25 questions): Moderately strict (USER'S TARGET)
        if q_count < 25:
            if (top_prob > 0.88 and  # TIGHTENED
                confidence_ratio > 20.0 and  # TIGHTENED
                entropy < 0.5):  # TIGHTENED
                print(f"[GUESS] Late game confidence: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}, ent={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Very late game (25-30 questions): Still require decent confidence
        if q_count < 30:
            if (top_prob > 0.80 and  # TIGHTENED
                confidence_ratio > 12.0 and  # TIGHTENED
                entropy < 0.8):  # TIGHTENED
                print(f"[GUESS] Very late confidence: prob={top_prob:.3f}, ratio={confidence_ratio:.1f}, ent={entropy:.3f}")
                return True, top_animal, 'final'
            return False, None, None
        
        # Forced guess after 30 questions (but still with minimum standards)
        if q_count >= 30:
            if top_prob > 0.60 and confidence_ratio > 5.0:  # TIGHTENED
                print(f"[GUESS] Forced (30+): prob={top_prob:.3f}, ratio={confidence_ratio:.1f}")
                return True, top_animal, 'final'
            # If even forced guess fails, return top candidate anyway
            print(f"[GUESS] Ultimate forced (30+): prob={top_prob:.3f}")
            return True, top_animal, 'final'
            
        return False, None, None

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs_safe * np.log(probs_safe))

    def activate_continue_mode(self, game_state: dict) -> dict:
        """Sets game to continue asking questions after wrong guess."""
        game_state['continue_mode'] = True
        game_state['questions_since_last_guess'] = 0
        game_state['middle_guess_made'] = False
        return game_state

    def get_next_question(self, game_state: dict) -> tuple[str | None, str | None, dict]:
        """Gets the next question. Returns (feature, question, modified_state)."""
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
        
        # Try discriminative question
        top_idx = np.argmax(prior)
        top_prob = prior[top_idx]
        if top_prob > 0.2:
            feature, q = engine.get_discriminative_question(top_idx, prior, asked)
            if feature:
                return feature, q, game_state

        # Default question selection
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
        
        posterior = engine.update(prior, feature_idx, answer)
        game_state['probabilities'] = posterior
        
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

    def get_data_collection_features(self, domain_name: str, 
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