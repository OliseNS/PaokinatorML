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
            'answer_history': []
        }
    
    @staticmethod
    def migrate_state(game_state: dict, new_animal_count: int) -> dict:
        """
        Resize state arrays if the engine grew (new items added in real-time).
        """
        # Get current size from the arrays themselves
        current_scores = game_state.get('cumulative_scores', np.array([]))
        state_n = len(current_scores)
        
        # If size matches, return (ensure mask exists)
        if state_n == new_animal_count:
            if 'rejected_mask' not in game_state:
                game_state['rejected_mask'] = np.zeros(state_n, dtype=bool)
            return game_state
        
        # If engine is larger than session state -> Pad
        if new_animal_count > state_n:
            padding_len = new_animal_count - state_n
            
            # Pad Scores with 0.0 (Neutral probability in log space, relative to others doesn't matter 
            # as long as they are normalized later. But strictly, if we added a new item, 
            # its score should ideally be comparable to the start score. 
            # 0.0 in log space is log(1), which is high. 
            # The engine initializes new item scores to 0.0 implicitly in 'update' if we don't do it here.
            # Let's use 0.0 (log(1)) because they haven't been ruled out.)
            score_pad = np.zeros(padding_len, dtype=np.float32)
            mask_pad = np.zeros(padding_len, dtype=bool)
            
            game_state['cumulative_scores'] = np.concatenate([current_scores, score_pad])
            
            old_mask = game_state.get('rejected_mask', np.zeros(state_n, dtype=bool))
            game_state['rejected_mask'] = np.concatenate([old_mask, mask_pad])
            
            game_state['animal_count'] = new_animal_count
            
        # If engine is smaller (rare, maybe deletion?), slice it
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
        scores_hash = hash(tuple(game_state['cumulative_scores'].tolist()))
        mask_hash = hash(tuple(game_state['rejected_mask'].tolist()))
        asked_hash = hash(tuple(sorted(game_state['asked_features'])))
        domain_hash = hash(game_state.get('domain_name', ''))
        return f"{domain_hash}_{scores_hash}_{mask_hash}_{asked_hash}_{n}"