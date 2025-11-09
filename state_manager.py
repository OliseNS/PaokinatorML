"""
State management utilities for game sessions.
Extracted from service.py for better separation of concerns.
"""
import numpy as np
from typing import Dict, Tuple


class StateManager:
    """Manages game state migrations and validations."""
    
    @staticmethod
    def create_initial_state(domain_name: str, n_animals: int) -> dict:
        """Creates a new game session state."""
        probabilities = np.ones(n_animals, dtype=np.float32) / (n_animals + 1e-10)
        
        return {
            'domain_name': domain_name,
            'probabilities': probabilities,
            'rejected_mask': np.zeros(n_animals, dtype=bool),
            'asked_features': [],
            'answered_features': {},
            'question_count': 0,
            # 'middle_guess_made': False, # <-- REMOVED
            'animal_count': n_animals,
            'continue_mode': False,
            'questions_since_last_guess': 0,
            'last_guess_type': None,
            'state_type': 'initial'  # --- SMART FIX ---
        }
    
    @staticmethod
    def migrate_state(game_state: dict, new_animal_count: int, 
                     current_animals: np.ndarray) -> dict:
        """
        Migrates state to match new engine dimensions.
        Returns a new state dict (doesn't mutate input).
        """
        current_n = new_animal_count
        state_n = game_state.get('animal_count', 0)
        
        if state_n == current_n:
            return game_state
        
        print(f"🔄 Migrating state from {state_n} to {current_n} items.")
        
        # Create new state copy
        new_state = game_state.copy()
        
        # Migrate probabilities
        old_probs = game_state['probabilities']
        new_probs = np.ones(current_n, dtype=np.float32)
        
        copy_len = min(state_n, current_n)
        if copy_len > 0:
            new_probs[:copy_len] = old_probs[:copy_len]
        
        if current_n > state_n:
            fill_prob = np.mean(old_probs) if state_n > 0 else (1.0 / current_n)
            new_probs[state_n:] = max(fill_prob, 1e-9)
        
        new_probs_sum = new_probs.sum()
        new_state['probabilities'] = new_probs / (new_probs_sum + 1e-10)
        
        # Migrate rejected mask
        old_mask = game_state['rejected_mask']
        new_mask = np.zeros(current_n, dtype=bool)
        if copy_len > 0:
            new_mask[:copy_len] = old_mask[:copy_len]
        new_state['rejected_mask'] = new_mask
        
        new_state['animal_count'] = current_n
        
        return new_state
    
    @staticmethod
    def increment_question_count(game_state: dict) -> dict:
        """Increments question count. Returns modified state."""
        game_state['question_count'] += 1
        if game_state.get('continue_mode', False):
            game_state['questions_since_last_guess'] = \
                game_state.get('questions_since_last_guess', 0) + 1
        return game_state
    
    @staticmethod
    def get_state_cache_key(game_state: dict, n: int) -> str:
        """Create a cache key for the game state."""
        probs_hash = hash(tuple(game_state['probabilities'].tolist()))
        mask_hash = hash(tuple(game_state['rejected_mask'].tolist()))
        asked_hash = hash(tuple(sorted(game_state['asked_features'])))
        domain_hash = hash(game_state.get('domain_name', ''))
        return f"{domain_hash}_{probs_hash}_{mask_hash}_{asked_hash}_{n}"