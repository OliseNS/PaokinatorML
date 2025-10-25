import json
import threading
import torch
import numpy as np

# Import our project files
import db
from engine import AkinatorEngine

class AkinatorService:
    """
    Manages the AkinatorEngine instance and game logic.
    
    This class handles the "hot-swap" of the engine when new
    data is learned, and seamlessly migrates active game sessions
    to the new engine's tensor shapes.
    """
    
    def __init__(self, questions_path: str):
        print("Initializing AkinatorService...")
        self.engine_lock = threading.Lock()
        self.questions_map = self._load_questions(questions_path)
        
        # Performance caches
        self._prediction_cache = {}  # Cache for top predictions
        self._cache_lock = threading.Lock()
        
        # Load the engine blocking on the first startup
        self.engine = self._create_engine()
        print(f"✅ Service initialized with {len(self.engine.animals)} animals.")

    def _load_questions(self, questions_path: str) -> dict:
        """Loads the questions.json file."""
        try:
            with open(questions_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ WARNING: {questions_path} not found. Using empty questions map.")
            return {}
        except json.JSONDecodeError:
            print(f"❌ ERROR: Failed to parse {questions_path}. Using empty map.")
            return {}

    def _create_engine(self) -> AkinatorEngine:
        """Loads data from DB and builds a new engine instance."""
        print("Loading data from Supabase to build engine...")
        df, feature_cols = db.load_data_from_supabase()
        return AkinatorEngine(df, feature_cols, self.questions_map)

    def _background_reload(self):
        """
        Runs in a separate thread to build a new engine
        and hot-swap it atomically.
        """
        print("🚀 Starting background engine reload...")
        try:
            new_engine = self._create_engine()
            
            # Atomic swap
            with self.engine_lock:
                self.engine = new_engine
            
            # Clear prediction cache when engine changes
            with self._cache_lock:
                self._prediction_cache.clear()
            
            print(f"✅ Engine hot-swap complete. Now serving {len(new_engine.animals)} animals.")
        except Exception as e:
            print(f"❌ ERROR: Background reload failed: {e}")

    def _migrate_state(self, game_state: dict, engine: AkinatorEngine) -> dict:
        """
        Ensures the game state tensors match the current engine's dimensions.
        This is the core of the seamless "hot-swap".
        """
        current_n = len(engine.animals)
        state_n = game_state.get('animal_count', 0)
        
        # If counts match, state is valid.
        if state_n == current_n:
            return game_state

        print(f"🔄 Migrating session state from {state_n} to {current_n} animals.")
        
        # --- Migrate Probabilities ---
        old_probs = game_state['probabilities']
        new_probs = torch.ones(current_n, dtype=torch.float32)
        
        # Copy old probabilities
        new_probs[:state_n] = old_probs
        
        # Fill new animals with a low-ish, normalized probability
        # We use mean as a heuristic, but could be any small value
        fill_prob = torch.mean(old_probs).item() if state_n > 0 else (1.0 / current_n)
        new_probs[state_n:] = fill_prob
        
        # Re-normalize
        game_state['probabilities'] = new_probs / new_probs.sum()
        
        # --- Migrate Rejected Mask ---
        old_mask = game_state['rejected_mask']
        new_mask = torch.zeros(current_n, dtype=torch.bool)
        new_mask[:state_n] = old_mask
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
        return f"{probs_hash}_{mask_hash}_{asked_hash}_{n}"

    # --- Public API Methods (Called by FastAPI) ---

    def create_initial_state(self) -> dict:
        """Creates a new game session state."""
        with self.engine_lock:
            engine = self.engine
            n_animals = len(engine.animals)
            
            # Initial prior: uniform distribution
            probabilities = torch.ones(n_animals, dtype=torch.float32) / n_animals
            
            state = {
                'probabilities': probabilities,
                'rejected_mask': torch.zeros(n_animals, dtype=torch.bool),
                'asked_features': [],
                'answered_features': {},
                'question_count': 0,
                'middle_guess_made': False,
                'animal_count': n_animals  # CRITICAL: Stamp the state with animal count
            }
            return state

    def get_top_predictions(self, game_state: dict, n: int = 5) -> list[dict]:
        """Gets top N predictions from a game state with caching."""
        # Create cache key based on state
        state_key = self._get_state_key(game_state, n)
        
        # Check cache first
        with self._cache_lock:
            if state_key in self._prediction_cache:
                return self._prediction_cache[state_key]
        
        # This function is read-only on the engine, but we acquire
        # the lock to ensure we get a consistent engine and migrate
        # the state *before* trying to read from it.
        with self.engine_lock:
            engine = self.engine
            game_state = self._migrate_state(game_state, engine)
            
            probs = game_state['probabilities'].clone() # Clone to avoid mutation
            mask = game_state['rejected_mask']
            
            # Apply rejected mask
            probs[mask] = 0.0
            
            sorted_probs, indices = torch.sort(probs, descending=True)
            
            top_n = min(n, len(indices))
            results = []
            for i in range(top_n):
                idx = indices[i].item()
                prob = sorted_probs[i].item()
                if prob < 0.001:
                    break
                results.append({
                    'animal': engine.animals[idx],
                    'probability': prob
                })
            
            # Cache the results
            with self._cache_lock:
                self._prediction_cache[state_key] = results
                # Limit cache size to prevent memory issues
                if len(self._prediction_cache) > 1000:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self._prediction_cache.keys())[:100]
                    for key in oldest_keys:
                        del self._prediction_cache[key]
            
            return results

    def should_make_guess(self, game_state: dict) -> tuple[bool, str | None, str | None]:
        """Determines if the engine should make a guess."""
        with self.engine_lock:
            engine = self.engine
            game_state = self._migrate_state(game_state, engine)
            
            q_count = game_state['question_count']
            probs = game_state['probabilities'].clone() # Clone to avoid mutation
            mask = game_state['rejected_mask']
            probs[mask] = 0.0 # Apply mask
            
            # --- START MODIFICATION ---
            
            # Find top TWO probabilities
            top_prob, top_idx = torch.max(probs, dim=0)
            top_animal = engine.animals[top_idx.item()]
            top_prob_val = top_prob.item()
            
            # Temporarily zero out the top one to find the second
            probs[top_idx] = 0.0
            second_prob_val = torch.max(probs, dim=0)[0].item()
            
            # Calculate confidence ratio
            confidence_ratio = top_prob_val / (second_prob_val + 1e-9) # Add epsilon for safety
            
            # --- LOGIC RE-ORDERING ---
            # 1. Check for Middle guess ("sneaky guess") condition FIRST
            if (
                q_count in [5, 10, 15] and 
                top_prob_val > 0.3 and  # Use the value we already calculated
                not game_state['middle_guess_made']
            ):
                game_state['middle_guess_made'] = True # Mutates state
                return True, top_animal, 'middle'
                
            # 2. Check for Final guess conditions SECOND
            # We now check *both* absolute prob and relative ratio
            if (top_prob_val > 0.7 and confidence_ratio > 5.0 and q_count > 5):
                # e.g., P(A) = 0.7, P(B) = 0.13 -> ratio = 5.3 (Guess)
                return True, top_animal, 'final'
            if (top_prob_val > 0.5 and confidence_ratio > 10.0 and q_count > 8):
                # e.g., P(A) = 0.5, P(B) = 0.04 -> ratio = 12.5 (Guess)
                return True, top_animal, 'final'
            if q_count >= 20: # Max questions
                return True, top_animal, 'final'
            
            # --- END MODIFICATION ---
                
            return False, None, None

    def get_next_question(self, game_state: dict) -> tuple[str | None, str | None, dict]:
        """Gets the next best question, returns (feature, question, modified_state)."""
        with self.engine_lock:
            engine = self.engine
            game_state = self._migrate_state(game_state, engine)

            prior = game_state['probabilities'].clone()
            prior[game_state['rejected_mask']] = 0.0
            prior = prior / (prior.sum() + 1e-10)
            
            asked = game_state['asked_features']
            q_count = game_state['question_count']
            
            # Optimization: Use precomputed uniform prior for Q0 (INSTANT)
            if q_count == 0 and hasattr(engine, '_uniform_prior') and engine._uniform_prior is not None:
                # For Q0, use the precomputed uniform prior directly - NO CALCULATION
                feature, q = engine.select_question(engine._uniform_prior, asked, q_count)
                return feature, q, game_state
            
            # 1. Try to find a discriminative question
            top_prob, top_idx = torch.max(prior, dim=0)
            if top_prob > 0.2:
                feature, q = engine.get_discriminative_question(top_idx, prior, asked)
                if feature:
                    return feature, q, game_state

            # 2. If not, find the max info-gain question
            feature, q = engine.select_question(prior, asked, q_count)
            return feature, q, game_state

    def process_answer(self, game_state: dict, feature: str, answer: str) -> dict:
        """Updates game state based on an answer."""
        with self.engine_lock:
            engine = self.engine
            game_state = self._migrate_state(game_state, engine)
            
            if feature not in engine.feature_cols:
                print(f"Warning: Feature '{feature}' not in engine. Skipping update.")
                return game_state
                
            feature_idx = engine.feature_cols.index(feature)
            
            prior = game_state['probabilities'].clone()
            prior[game_state['rejected_mask']] = 0.0
            prior = prior / (prior.sum() + 1e-10) # Re-normalize
            
            posterior = engine.update(prior, feature_idx, answer)
            game_state['probabilities'] = posterior
            
            return game_state

    def reject_guess(self, game_state: dict, animal_name: str) -> dict:
        """Marks an animal as rejected in the game state."""
        with self.engine_lock:
            engine = self.engine
            game_state = self._migrate_state(game_state, engine)
            
            try:
                # Find index of animal in the *current* engine
                idx = np.where(engine.animals == animal_name)[0][0]
                game_state['rejected_mask'][idx] = True
                game_state['probabilities'][idx] = 0.0 
                
                # Re-normalize probabilities
                total = game_state['probabilities'].sum()
                if total > 1e-10:
                    game_state['probabilities'] = game_state['probabilities'] / total
                
            except IndexError:
                print(f"Warning: Could not reject '{animal_name}', not found in engine.")
                
            return game_state

    def learn_animal(self, animal_name: str, answered_features: dict) -> str:
        """
        Persists a new animal to the database and triggers a
        non-blocking background reload of the engine.
        """
        animal_data = answered_features.copy()
        animal_data['animal_name'] = animal_name
        
        # Persist to DB
        result = db.persist_learned_animal(animal_data)
        
        return result

    def start_engine_reload(self):
        """
        Starts a background engine reload in a separate thread.
        This is the public method called by the admin endpoint.
        """
        import threading
        threading.Thread(target=self._background_reload, daemon=True).start()