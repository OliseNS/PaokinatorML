import threading
import numpy as np

import db
from engine import AkinatorEngine
from state_manager import StateManager


class AkinatorService:
    """
    Service Layer.
    Manages game flow and engine interaction.
    """
    
    def __init__(self):
        print("Initializing AkinatorService...")
        self.engines_lock = threading.Lock()
        self.engines: dict[str, AkinatorEngine] = {}
        self.state_manager = StateManager()
        self._prediction_cache = {}
        self._cache_lock = threading.Lock()
        self._load_all_engines()

    def _create_engine(self, domain_name: str) -> AkinatorEngine:
        try:
            df, feature_cols, questions_map = db.load_data_from_supabase(domain_name)
            engine = AkinatorEngine(df, feature_cols, questions_map)
            return engine
        except Exception as e:
            print(f"   ✗ ERROR: Failed to create engine for '{domain_name}': {e}")
            return None

    def _load_all_engines(self):
        print("Loading all engines...")
        domain_names = db.get_all_domains()
        new_engines: dict[str, AkinatorEngine] = {}
        for domain in domain_names:
            engine = self._create_engine(domain)
            if engine:
                new_engines[domain] = engine
        
        with self.engines_lock:
            self.engines = new_engines
        with self._cache_lock:
            self._prediction_cache.clear()
        print(f"✅ All engines loaded. Serving {len(self.engines)} domains.")

    def _background_reload(self):
        print("🚀 Starting background engine reload...")
        try:
            self._load_all_engines()
            print(f"✅ Engine hot-swap complete.")
        except Exception as e:
            print(f"❌ ERROR: Background reload failed: {e}")

    def _get_engine_and_migrate_state(self, game_state: dict) -> tuple[AkinatorEngine, dict]:
        domain_name = game_state.get('domain_name')
        engine = self.engines.get(domain_name)
        if not engine:
            if not self.engines: raise ValueError("No engines loaded.")
            domain_name, engine = next(iter(self.engines.items()))
            game_state['domain_name'] = domain_name
            
        game_state = self.state_manager.migrate_state(game_state, len(engine.animals), engine.animals)
        if 'cumulative_scores' not in game_state or len(game_state['cumulative_scores']) != len(engine.animals):
            game_state['cumulative_scores'] = np.zeros(len(engine.animals), dtype=np.float32)
        if 'answer_history' not in game_state:
            game_state['answer_history'] = []
        return engine, game_state

    def _get_probs_and_state(self, game_state: dict) -> tuple[np.ndarray, dict, AkinatorEngine]:
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
        scores = game_state['cumulative_scores'].copy()
        mask = game_state['rejected_mask']
        scores[mask] = -np.inf
        
        if len(scores) > 0:
            e_x = np.exp(scores - np.max(scores))
            probs = e_x / (e_x.sum() + 1e-10)
        else:
            probs = np.array([], dtype=np.float32)
        probs[mask] = 0.0
        return probs, game_state, engine

    def get_available_domains(self) -> list[str]:
        with self.engines_lock:
            return list(self.engines.keys())

    def create_initial_state(self, domain_name: str) -> dict:
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError(f"Domain '{domain_name}' not found.")
            return self.state_manager.create_initial_state(domain_name, len(engine.animals))

    def get_top_predictions(self, game_state: dict, n: int = 5) -> list[dict]:
        state_key = self.state_manager.get_state_cache_key(game_state, n)
        with self._cache_lock:
            if state_key in self._prediction_cache: return self._prediction_cache[state_key]
        
        probs, game_state, engine = self._get_probs_and_state(game_state)
        sorted_indices = np.argsort(probs)[::-1]
        results = []
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = probs[idx]
            if prob < 0.001: break
            results.append({'animal': engine.animals[idx], 'probability': float(prob)})
        
        with self._cache_lock:
            self._prediction_cache[state_key] = results
        return results

    def should_make_guess(self, game_state: dict) -> tuple[bool, str | None, str | None]:
        probs, game_state, engine = self._get_probs_and_state(game_state)
        return engine.should_make_guess(game_state, probs)

    def activate_continue_mode(self, game_state: dict) -> dict:
        game_state['continue_mode'] = True
        game_state['questions_since_last_guess'] = 0
        return game_state

    def get_next_question(self, game_state: dict) -> tuple[str | None, str | None, dict]:
        """
        Determines the next best question to ask.
        """
        prior, game_state, engine = self._get_probs_and_state(game_state)
        asked = game_state['asked_features']
        q_count = game_state['question_count']
        
        # Q0 optimization: Uses pre-computed indices.
        if q_count == 0 and hasattr(engine, 'sorted_initial_feature_indices'):
            feature, q = engine.select_question(prior, asked, q_count)
            return feature, q, game_state
        
        entropy = engine._calculate_entropy(prior)
        print(f"[Question] PURE ENTROPY: Finding best split across {np.sum(prior > 0.001):.0f} items (entropy={entropy:.2f})")
        
        feature, q = engine.select_question(prior, asked, q_count)
        
        return feature, q, game_state
    
    def process_answer(self, game_state: dict, feature: str, answer: str) -> dict:
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        if feature not in engine.feature_cols: return game_state
        feature_idx_list = np.where(engine.feature_cols == feature)[0]
        if len(feature_idx_list) == 0: return game_state
        feature_idx = int(feature_idx_list[0])

        current_scores = game_state['cumulative_scores']
        new_scores = engine.update(feature_idx, answer, current_scores)
        game_state['cumulative_scores'] = new_scores
        
        if 'answer_history' not in game_state: game_state['answer_history'] = []
        game_state['answer_history'].append({'feature': feature, 'answer': answer})
        
        return game_state

    def reject_guess(self, game_state: dict, animal_name: str) -> dict:
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
        
        idx_list = np.where(engine.animals == animal_name)[0]
        if len(idx_list) > 0:
            game_state['rejected_mask'][idx_list[0]] = True
        return game_state

    def get_features_for_data_collection(self, domain_name: str, item_name: str) -> list[dict]:
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError(f"Domain '{domain_name}' not found.")
        return engine.get_features_for_data_collection(item_name, num_features=5)

    def get_feature_gains(self, domain_name: str) -> list[dict]:
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError(f"Domain '{domain_name}' not found.")
        return engine.get_all_feature_gains()

    def record_suggestion(self, animal_name: str, answered_features: dict, domain_name: str) -> str:
        return db.persist_suggestion(animal_name, answered_features, domain_name)

    def learn_new_animal(self, animal_name: str, answered_features: dict, domain_name: str) -> str:
        return db.persist_new_animal(animal_name, answered_features, domain_name)

    def start_engine_reload(self):
        threading.Thread(target=self._background_reload, daemon=True).start()

    def get_game_report(self, domain_name: str, item_name: str, user_answers: dict, is_new: bool) -> dict:
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError(f"Domain '{domain_name}' not found.")
        
        questions_report = []
        item_idx = -1
        
        if is_new:
            report_name = f"{item_name} (New Item)"
        else:
            report_name = item_name
            try:
                item_idx = np.where(engine.animals == item_name)[0][0]
            except (IndexError, TypeError):
                report_name = f"{item_name} (Item not found)"
                is_new = True

        for feature_name, user_value in user_answers.items():
            question_text = engine.questions_map.get(feature_name, f"Is it {feature_name}?")
            consensus_value = None
            if not is_new and item_idx != -1:
                try:
                    f_idx = np.where(engine.feature_cols == feature_name)[0]
                    if len(f_idx) > 0:
                        val = engine.features[item_idx, int(f_idx[0])]
                        if not np.isnan(val): consensus_value = float(val)
                except: pass
            
            questions_report.append({
                "question": question_text,
                "feature": feature_name,
                "user_answer": float(user_value),
                "consensus_answer": consensus_value
            })

        return {"item_name": report_name, "is_new_item": is_new, "questions": questions_report}