import numpy as np

class StateManager:
    """
    Manages game state resizing.
    Updated to handle 'active count' (n_items) instead of fixed static array length.
    """
    
    @staticmethod
    def create_initial_state(domain_name: str, n_animals: int) -> dict:
        return {
            'domain_name': domain_name,
            'cumulative_scores': np.zeros(n_animals, dtype=np.float32),
            'rejected_mask': np.zeros(n_animals, dtype=bool),
            'asked_features': [],
            'answered_features': {},
            'question_count': 0,
            'animal_count': n_animals,
            'continue_mode': False,
            'questions_since_last_guess': 0,
            'last_guess_type': None,
            'state_type': 'initial',
            'answer_history': [],
            'sparse_questions_asked': 0  # Tracks injected data collection questions
        }
    
    @staticmethod
    def migrate_state(game_state: dict, new_animal_count: int) -> dict:
        """
        Resize state arrays if the engine grew (new items added in real-time).
        """
        current_scores = game_state.get('cumulative_scores', np.array([]))
        state_n = len(current_scores)
        
        # Case 1: Size matches (No change needed)
        if state_n == new_animal_count:
            if 'rejected_mask' not in game_state:
                game_state['rejected_mask'] = np.zeros(state_n, dtype=bool)
            return game_state
        
        # Case 2: Engine grew (New items added)
        if new_animal_count > state_n:
            padding_len = new_animal_count - state_n
            
            # Don't use 0.0. Use the worst score in the current active set.
            if state_n > 0:
                valid_scores = current_scores[np.isfinite(current_scores)]
                if len(valid_scores) > 0:
                    base_score = np.min(valid_scores) - 1.0
                else:
                    base_score = -20.0
            else:
                base_score = 0.0

            score_pad = np.full(padding_len, base_score, dtype=np.float32)
            mask_pad = np.zeros(padding_len, dtype=bool)
            
            game_state['cumulative_scores'] = np.concatenate([current_scores, score_pad])
            
            old_mask = game_state.get('rejected_mask', np.zeros(state_n, dtype=bool))
            game_state['rejected_mask'] = np.concatenate([old_mask, mask_pad])
            
            game_state['animal_count'] = new_animal_count
            
        # Case 3: Engine shrank
        elif new_animal_count < state_n:
            game_state['cumulative_scores'] = current_scores[:new_animal_count]
            game_state['rejected_mask'] = game_state['rejected_mask'][:new_animal_count]
            game_state['animal_count'] = new_animal_count
            
        if 'probabilities' in game_state: del game_state['probabilities']
        
        return game_state
    
    @staticmethod
    def increment_question_count(game_state: dict) -> dict:
        game_state['question_count'] += 1
        if game_state.get('continue_mode', False):
            game_state['questions_since_last_guess'] = \
                game_state.get('questions_since_last_guess', 0) + 1
        return game_state
    
    @staticmethod
    def get_state_cache_key(game_state: dict, n: int) -> str:
        # Convert numpy to tuple for hashing
        scores_hash = hash(tuple(game_state['cumulative_scores'].tolist()))
        mask_hash = hash(tuple(game_state['rejected_mask'].tolist()))
        asked_hash = hash(tuple(sorted(game_state['asked_features'])))
        domain_hash = hash(game_state.get('domain_name', ''))
        return f"{domain_hash}_{scores_hash}_{mask_hash}_{asked_hash}_{n}"