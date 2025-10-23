import pandas as pd
import numpy as np
import json
import os
import torch
from threading import Lock, Thread

# Import from our new refactored files
from engine import AkinatorEngine
from db import load_data_from_supabase, persist_learned_animal
import config


class AkinatorService:
    """
    Game manager for Akinator sessions.
    This class connects the pure 'engine' to the 'db' layer
    and manages game logic.
    """
    
    def __init__(self, questions_path=config.QUESTIONS_PATH):
        # This lock protects the 'self.engine' and 'self.animal_to_idx'
        # references, allowing them to be "hot-swapped" safely.
        self.engine_lock = Lock()
        
        # These will be populated by reload_engine()
        self.engine: AkinatorEngine = None
        self.animal_to_idx: dict = {}
        self.feature_cols: list = [] # Stored for learn_animal
        
        # Initial load
        self.reload_engine()
        
        print("✅ AkinatorService initialized")
    
    def _load_engine_data(self) -> tuple[AkinatorEngine, dict, list]:
        """
        Helper function to load all data from scratch and build a new engine.
        This runs in the background during a hot-reload.
        """
        # 1. Load data from Supabase
        df, feature_cols = load_data_from_supabase()

        # 2. Load questions map
        questions_map = {}
        path_used = None
        tried = []

        if config.QUESTIONS_PATH:
            tried.append(os.path.abspath(config.QUESTIONS_PATH))
            if os.path.exists(config.QUESTIONS_PATH):
                path_used = config.QUESTIONS_PATH

        if path_used is None:
            base_dir = os.path.dirname(__file__)
            alt = os.path.join(base_dir, config.QUESTIONS_PATH)
            tried.append(os.path.abspath(alt))
            if os.path.exists(alt):
                path_used = alt

        if path_used:
            try:
                with open(path_used, 'r', encoding='utf-8') as f:
                    questions_map = json.load(f)
            except Exception as e:
                print(f"Warning: failed loading questions file '{path_used}': {e}")
        else:
            print(f"Warning: questions file not found. Tried: {tried}")
        
        # 3. Build the engine and lookup map
        engine = AkinatorEngine(df, feature_cols, questions_map)
        animal_to_idx = {
            name.lower(): i for i, name in enumerate(df['animal_name'].values)
        }
        
        return engine, animal_to_idx, feature_cols

    def reload_engine(self):
        """
        Builds a new engine and atomically swaps it into the service.
        This provides zero-downtime updates when new animals are learned.
        """
        print("🧠 Engine hot-reload starting...")
        try:
            new_engine, new_animal_map, new_feature_cols = self._load_engine_data()
            
            # This is the "atomic swap"
            # Any request happening right now will finish with the OLD
            # engine, and the next request will get the NEW one.
            with self.engine_lock:
                self.engine = new_engine
                self.animal_to_idx = new_animal_map
                self.feature_cols = new_feature_cols # Store for learn_animal
            
            print(f"✅ Engine hot-reloaded. Total animals: {len(self.engine.animals)}")
        except Exception as e:
            print(f"❌ CRITICAL: Engine hot-reload FAILED: {e}")
            # The service will continue to run with the old engine (if one exists)
    
    def create_initial_state(self):
        """Creates a new game state. Must be called *after* engine is loaded."""
        with self.engine_lock:
            N = len(self.engine.animals)
        
        return {
            'probabilities': torch.ones(N, dtype=torch.float32),
            'rejected_mask': torch.zeros(N, dtype=torch.bool),
            'answered_features': {},
            'asked_features': [],
            'rejected_animals': [],
            'question_count': 0,
            'middle_guess_made': False,
            'discriminative_mode': False,
            'discrimination_target': None
        }
    
    def get_next_question(self, state):
        with self.engine_lock:
            # Ensure probabilities tensor is the right size, in case of hot-reload
            if len(state['probabilities']) != len(self.engine.animals):
                # This is a simple fix: just reset the state.
                # A more complex fix would be to remap probabilities.
                print("Engine size changed, resetting game state.")
                state.update(self.create_initial_state())

            probs = state['probabilities'] / (state['probabilities'].sum() + 1e-10)
            
            if state.get('discriminative_mode'):
                target_idx = state.get('discrimination_target')
                if target_idx is not None and target_idx < len(self.engine.animals):
                    feature, question = self.engine.get_discriminative_question(
                        target_idx, probs, state['asked_features'])
                    if feature:
                        return feature, question
                state['discriminative_mode'] = False
            
            result = self.engine.select_question(
                probs, 
                state['asked_features'], 
                state['question_count']
            )
            return result
        
    def process_answer(self, state, feature, answer):
        try:
            # Must read feature_cols from self, not engine
            feature_idx = self.feature_cols.index(feature)
        except ValueError:
            return state
        
        with self.engine_lock:
            # Re-check tensor size
            if len(state['probabilities']) != len(self.engine.animals):
                print("Engine size changed, resetting game state.")
                state.update(self.create_initial_state())
                return state # Ask user to answer again

            probs = state['probabilities'] / (state['probabilities'].sum() + 1e-10)
            state['probabilities'] = self.engine.update(probs, feature_idx, answer)
        
        state['probabilities'][state['rejected_mask']] = 0.0
        return state
    
    def should_make_guess(self, state):
        """Determine if we should make a guess."""
        with self.engine_lock:
            # Re-check tensor size
            if len(state['probabilities']) != len(self.engine.animals):
                print("Engine size changed, resetting game state.")
                state.update(self.create_initial_state())
                return False, None, None

            probs = state['probabilities'] / (state['probabilities'].sum() + 1e-10)
            num_questions = len(state['answered_features'])
            
            rejected_set = set(state.get('rejected_animals', []))
            sorted_indices = torch.argsort(probs, descending=True)
            
            top_idx, second_idx = None, None
            top_prob, second_prob = 0.0, 0.0
            
            for idx in sorted_indices:
                idx_item = idx.item()
                if idx_item >= len(self.engine.animals):
                    continue # Index out of bounds (mid-reload)
                
                name = self.engine.animals[idx_item]
                if name in rejected_set:
                    continue
                if top_idx is None:
                    top_idx = idx_item
                    top_prob = probs[idx].item()
                elif second_idx is None:
                    second_prob = probs[idx].item()
                    break
            
            if top_idx is None:
                return False, None, None
            
            animal_name = self.engine.animals[top_idx]
            top_spec = self.engine.specificity[top_idx].item()
            entropy = self.engine.entropy(probs).item()
            ratio = top_prob / max(second_prob, 1e-12)
        
        # (Guessing logic remains outside the lock)
        if not state['middle_guess_made'] and num_questions >= 4:
            required_ratio = 6.0 if top_spec < 0.25 else 3.0
            if (top_prob > 0.93 and ratio > required_ratio) or top_prob > 0.98:
                state['middle_guess_made'] = True
                state['discriminative_mode'] = True
                state['discrimination_target'] = top_idx
                return True, animal_name, "middle"
        
        if (top_prob > 0.96 and top_spec > 0.08) or entropy < 0.06 or num_questions >= 25:
            return True, animal_name, "final"
        
        return False, None, None
    
    def reject_guess(self, state, animal_name):
        # This function modifies the session 'state' and 'animal_to_idx',
        # so it needs a lock for the read operation.
        with self.engine_lock:
            idx = self.animal_to_idx.get(animal_name.lower())
        
        if idx is not None:
            if idx < len(state['rejected_mask']):
                state['rejected_mask'][idx] = True
                state['probabilities'][idx] = 0.0
            
            if animal_name not in state['rejected_animals']:
                state['rejected_animals'].append(animal_name)
        
        return state
    
    def get_top_predictions(self, state, n=5):
        """Get top N predictions adjusted by specificity."""
        with self.engine_lock:
            # Re-check tensor size
            if len(state['probabilities']) != len(self.engine.animals):
                print("Engine size changed, resetting game state.")
                state.update(self.create_initial_state())
            
            probs = state['probabilities'] / (state['probabilities'].sum() + 1e-10)
            rejected = set(state.get('rejected_animals', []))
            
            answered_count = max(1, len(state.get('answered_features', {})))
            spec_influence = min(1.0, answered_count / 6.0)
            
            results = []
            k_scan = min(max(n * 3, 50), len(probs))
            
            if k_scan == 0:
                return [] # Should not happen if state is valid

            top_vals, top_idx = torch.topk(probs, k_scan)

            for i in range(len(top_idx)):
                idx_item = top_idx[i].item()
                if idx_item >= len(self.engine.animals):
                    continue # Index out of bounds (mid-reload)

                name = self.engine.animals[idx_item]
                if name in rejected:
                    continue

                prob = top_vals[i].item()
                spec = self.engine.specificity[idx_item].item()
                adjusted = prob * (1.0 + 0.8 * spec * spec_influence)

                results.append((name, prob, adjusted))
                if len(results) >= max(n, 10):
                    break
        
        results.sort(key=lambda x: x[2], reverse=True)
        return [(name, prob) for name, prob, _ in results[:n]]
    
    def learn_animal(self, name, answered_features):
        """
        Learns an animal by saving to DB and triggering a background reload.
        This no longer blocks the main thread.
        """
        name = name.strip().capitalize()
        
        # We must read self.feature_cols inside the lock
        with self.engine_lock:
            cols = self.feature_cols
            
        animal_data = {'animal_name': name}
        for f in cols:
            animal_data[f] = answered_features.get(f, np.nan)

        # 1. Persist to Supabase
        status = persist_learned_animal(animal_data)
        
        # 2. Conditionally trigger a background reload in a new thread
        #    so the HTTP request can return immediately.
        if status == "inserted":
            print(f"✅ New animal {name} saved. Triggering background engine reload.")
            reload_thread = Thread(target=self.reload_engine, daemon=True)
            reload_thread.start()
        
        elif status == "suggestion_inserted":
            print(f"✅ Suggestion for {name} saved. Engine not modified.")
            
        else: # status == "error"
            print(f"Skipping engine reload for {name} due to DB error.")