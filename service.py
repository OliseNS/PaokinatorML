import threading
import time
import numpy as np
from datetime import datetime, timezone, timedelta

import db
from engine import AkinatorEngine
from state_manager import StateManager


class AkinatorService:
    """
    Service Layer with Precise Real-time Sync.
    """
    
    def __init__(self):
        print("Initializing AkinatorService...")
        self.engines_lock = threading.Lock()
        self.engines: dict[str, AkinatorEngine] = {}
        self.state_manager = StateManager()
        self._prediction_cache = {}
        self._cache_lock = threading.Lock()
        
        self.is_running = True
        self._load_all_engines()
        
        # Reset sync time to NOW after load to start polling fresh changes
        self.last_sync_time = datetime.now(timezone.utc)
        
        # Start Poller
        self.sync_thread = threading.Thread(target=self._background_sync_loop, daemon=True)
        self.sync_thread.start()

    def _create_engine(self, domain_name: str) -> AkinatorEngine:
        try:
            # Load full dataset (active + suggested)
            df, feature_cols, questions_map = db.load_data_from_supabase(domain_name)
            # Create with padding
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

    def _background_sync_loop(self):
        """
        Smart Poll: Fetches diffs -> Hot Patch.
        """
        print(f"🔄 Real-time sync loop started.")
        
        while self.is_running:
            try:
                time.sleep(3.0) # Poll every 3s
                
                # Prepare next timestamp anchor
                next_sync_time = datetime.now(timezone.utc)
                
                with self.engines_lock:
                    domains = list(self.engines.keys())
                
                total_updates = 0
                
                for domain in domains:
                    # Fetch updates since last_sync_time
                    updates = db.get_recent_updates(domain, self.last_sync_time)
                    
                    if updates:
                        with self.engines_lock:
                            engine = self.engines.get(domain)
                            if not engine: continue
                            
                            # 1. Ingest Data
                            for up in updates:
                                engine.smart_ingest_update(
                                    item_name=up['item_name'],
                                    feature_name=up['feature_name'],
                                    value=up['value'],
                                    question_text=up.get('question_text')
                                )
                            
                            # 2. Update Strategy
                            # We do this once per batch per domain
                            engine.recalculate_stats()
                            
                        total_updates += len(updates)
                        print(f"⚡ [Realtime] Synced {len(updates)} updates for '{domain}'")
                
                if total_updates > 0:
                    with self._cache_lock:
                        self._prediction_cache.clear()
                    self.last_sync_time = next_sync_time
                else:
                    self.last_sync_time = next_sync_time
                
            except Exception as e:
                print(f"⚠️ Sync Loop Error: {e}")
                time.sleep(5.0)

    def _get_engine_and_migrate_state(self, game_state: dict) -> tuple[AkinatorEngine, dict]:
        domain_name = game_state.get('domain_name')
        engine = self.engines.get(domain_name)
        if not engine:
            if not self.engines: raise ValueError("No engines loaded.")
            domain_name, engine = next(iter(self.engines.items()))
            game_state['domain_name'] = domain_name
            
        # Dynamic Session Migration
        game_state = self.state_manager.migrate_state(game_state, engine.n_items)
        return engine, game_state

    def _get_probs_and_state(self, game_state: dict) -> tuple[np.ndarray, dict, AkinatorEngine]:
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
        
        scores = game_state['cumulative_scores'].copy()
        mask = game_state['rejected_mask']
        
        if len(scores) != engine.n_items:
            game_state = self.state_manager.migrate_state(game_state, engine.n_items)
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

    # --- Proxy Methods ---
    
    def get_available_domains(self) -> list[str]:
        with self.engines_lock:
            return list(self.engines.keys())

    def create_initial_state(self, domain_name: str) -> dict:
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError(f"Domain '{domain_name}' not found.")
            return self.state_manager.create_initial_state(domain_name, engine.n_items)

    def get_top_predictions(self, game_state: dict, n: int = 5) -> list[dict]:
        state_key = self.state_manager.get_state_cache_key(game_state, n)
        with self._cache_lock:
            if state_key in self._prediction_cache: return self._prediction_cache[state_key]
        
        probs, game_state, engine = self._get_probs_and_state(game_state)
        if len(probs) == 0: return []
        
        sorted_indices = np.argsort(probs)[::-1]
        results = []
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = probs[idx]
            if prob < 0.001: break
            if idx < engine.n_items:
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
        prior, game_state, engine = self._get_probs_and_state(game_state)
        asked = game_state['asked_features']
        q_count = game_state['question_count']
        
        # --- DATA COLLECTION INJECTION LOGIC ---
        sparse_asked_count = game_state.get('sparse_questions_asked', 0)
        
        top_prob = np.max(prior) if len(prior) > 0 else 0
        
        # Trigger Conditions:
        # 1. "Mid-Game Pocket": Between Q5 and Q20.
        # 2. Limit: Max 2 per session.
        # 3. Safety: Not in continue mode, not highly confident yet (>0.85).
        # 4. Chance: 25% probability per turn (stochastic).
        should_try_sparse = (
            5 <= q_count <= 20 and
            sparse_asked_count < 2 and
            not game_state.get('continue_mode', False) and
            top_prob < 0.85 and
            np.random.random() < 0.25 
        )

        if should_try_sparse:
            # print(f"[Service] Attempting sparse discovery...")
            feature, q = engine.select_sparse_discovery_question(prior, asked)
            
            if feature:
                # print(f"[Service] 🟢 Inserted sparse question: {feature}")
                game_state['sparse_questions_asked'] = sparse_asked_count + 1
                return feature, q, game_state

        # --- END INJECTION LOGIC ---
        
        feature, q = engine.select_question(prior, asked, q_count)
        return feature, q, game_state
    
    def process_answer(self, game_state: dict, feature: str, answer: str) -> dict:
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
            
        try:
            f_idx = engine.feature_map.get(feature)
            if f_idx is None: return game_state
        except Exception:
            return game_state 

        current_scores = game_state['cumulative_scores']
        
        new_scores = engine.update(f_idx, answer, current_scores)
        
        game_state['cumulative_scores'] = new_scores
        if len(new_scores) > len(game_state['rejected_mask']):
            padding = len(new_scores) - len(game_state['rejected_mask'])
            game_state['rejected_mask'] = np.concatenate([game_state['rejected_mask'], np.zeros(padding, dtype=bool)])
            
        if 'answer_history' not in game_state: game_state['answer_history'] = []
        game_state['answer_history'].append({'feature': feature, 'answer': answer})
        
        return game_state

    def reject_guess(self, game_state: dict, animal_name: str) -> dict:
        with self.engines_lock:
            engine, game_state = self._get_engine_and_migrate_state(game_state)
        
        target_clean = engine._normalize(animal_name)
        for i in range(engine.n_items):
            current_name = engine.animals[i]
            if engine._normalize(current_name) == target_clean:
                if i < len(game_state['rejected_mask']):
                    game_state['rejected_mask'][i] = True
                break
                
        return game_state

    def get_features_for_data_collection(self, domain_name, item_name):
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError("Domain not found")
        return engine.get_features_for_data_collection(item_name)

    def get_feature_gains(self, domain_name):
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError("Domain not found")
        return engine.get_all_feature_gains()
    
    def record_suggestion(self, *args): return db.persist_suggestion(*args)
    def learn_new_animal(self, *args): return db.persist_new_animal(*args)
    def start_engine_reload(self): self._load_all_engines()

    def get_game_report(self, domain_name, item_name, user_answers, is_new, session_id=None, ai_won=False, sparse_count=0):
        with self.engines_lock:
            engine = self.engines.get(domain_name)
            if not engine: raise ValueError("Domain not found")
            
            clean_item_name = engine._normalize(item_name)
            existing_idx = engine.item_map.get(clean_item_name)
            
            if existing_idx is not None:
                is_new = False
                item_name = engine.animals[existing_idx]
                target_vector = np.nan_to_num(engine.features[existing_idx, :engine.n_features], nan=0.5)
            else:
                target_vector = engine.build_feature_vector(user_answers)

            similar_items = engine.find_nearest_neighbors(target_vector, exclude_name=item_name, n=3)
            questions_report = []
            item_idx = engine.item_map.get(clean_item_name, -1)

            for feature_name, user_value in user_answers.items():
                q_text = engine.questions_map.get(feature_name, f"Is it {feature_name}?")
                consensus_value = None
                
                if item_idx != -1:
                    try:
                        f_idx = engine.feature_map.get(feature_name)
                        if f_idx is not None:
                            val = engine.features[item_idx, f_idx]
                            if not np.isnan(val): consensus_value = float(val)
                    except: pass
                
                questions_report.append({
                    "question": q_text, 
                    "feature": feature_name,
                    "user_answer": float(user_value), 
                    "consensus_answer": consensus_value
                })

        report_data = {
            "item_name": item_name, 
            "is_new_item": is_new, 
            "questions": questions_report,
            "similar_items": similar_items,
            "sparse_questions_asked": sparse_count # <--- Added field
        }
        
        if session_id:
            threading.Thread(target=db.save_game_report, args=(
                session_id, domain_name, item_name, ai_won, report_data
            )).start()

        return report_data