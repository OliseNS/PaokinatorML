"""
REFACTORED: State management utilities for game sessions.
Now initializes and migrates 'cumulative_scores' instead of 'probabilities'.
"""
import numpy as np
from typing import Dict, Tuple


class StateManager:
    """Manages game state migrations and validations."""
    
    @staticmethod
    def create_initial_state(domain_name: str, n_animals: int) -> dict:
        """
        Creates a new game session state.
        REFACTORED: Initializes 'cumulative_scores' to zero.
        """
        return {
            'domain_name': domain_name,
            # 'probabilities' is REMOVED
            'cumulative_scores': np.zeros(n_animals, dtype=np.float32), # ADDED
            'rejected_mask': np.zeros(n_animals, dtype=bool),
            'asked_features': [],
            'answered_features': {},
            'question_count': 0,
            'animal_count': n_animals,
            'continue_mode': False,
            'questions_since_last_guess': 0,
            'last_guess_type': None,
            'state_type': 'initial',
            'answer_history': [] # Added for robustness
        }
    
    @staticmethod
    def migrate_state(game_state: dict, new_animal_count: int, 
                     current_animals: np.ndarray) -> dict:
        """
        Migrates state to match new engine dimensions.
        REFACTORED: Migrates 'cumulative_scores'.
        """
        current_n = new_animal_count
        state_n = game_state.get('animal_count', 0)
        
        if state_n == current_n:
            # --- Robustness Check ---
            # Ensure the new state key exists even if counts match
            if 'cumulative_scores' not in game_state:
                game_state['cumulative_scores'] = np.zeros(current_n, dtype=np.float32)
            # --- End Check ---
            return game_state
        
        print(f"🔄 Migrating state from {state_n} to {current_n} items.")
        
        # Create new state copy
        new_state = game_state.copy()
        
        # --- REFACTORED: Migrate cumulative_scores ---
        if 'cumulative_scores' in game_state:
            old_scores = game_state['cumulative_scores']
        else:
            # Migrating from an old state that had probabilities
            old_scores = np.zeros(state_n, dtype=np.float32)

        new_scores = np.zeros(current_n, dtype=np.float32)
        
        copy_len = min(state_n, current_n)
        if copy_len > 0:
            new_scores[:copy_len] = old_scores[:copy_len]
        
        # New animals (current_n > state_n) will just have the default 0 score
        
        new_state['cumulative_scores'] = new_scores
        # 'probabilities' key is removed from migration
        if 'probabilities' in new_state:
            del new_state['probabilities']
        # --- END REFACTOR ---

        # Migrate rejected mask (unchanged logic)
        old_mask = game_state.get('rejected_mask', np.zeros(state_n, dtype=bool))
        new_mask = np.zeros(current_n, dtype=bool)
        if copy_len > 0:
            new_mask[:copy_len] = old_mask[:copy_len]
        new_state['rejected_mask'] = new_mask
        
        new_state['animal_count'] = current_n
        
        return new_state
    
    @staticmethod
    def increment_question_count(game_state: dict) -> dict:
        """Increments question count. Returns modified state (unchanged)."""
        game_state['question_count'] += 1
        if game_state.get('continue_mode', False):
            game_state['questions_since_last_guess'] = \
                game_state.get('questions_since_last_guess', 0) + 1
        return game_state
    
    @staticmethod
    def get_state_cache_key(game_state: dict, n: int) -> str:
        """
        Create a cache key for the game state.
        REFACTORED: Hashes 'cumulative_scores'.
        """
        # --- REFACTORED ---
        scores_hash = hash(tuple(game_state['cumulative_scores'].tolist()))
        # --- END REFACTOR ---
        mask_hash = hash(tuple(game_state['rejected_mask'].tolist()))
        asked_hash = hash(tuple(sorted(game_state['asked_features'])))
        domain_hash = hash(game_state.get('domain_name', ''))
        
        return f"{domain_hash}_{scores_hash}_{mask_hash}_{asked_hash}_{n}"