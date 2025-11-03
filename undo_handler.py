"""
Handles undo logic for game sessions.
Extracted from main.py for better testability and maintainability.
"""
import db
from typing import Dict, Optional, Tuple


class UndoHandler:
    """Manages undo operations for game sessions."""
    
    def __init__(self, service):
        """
        Args:
            service: AkinatorService instance for accessing engines
        """
        self.service = service
    
    def can_undo(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if undo is possible.
        
        Returns:
            (can_undo: bool, error_message: Optional[str])
        """
        history_len = db.get_session_history_length(session_id)
        
        # Need at least 3 states: [S1_Q2, S1, S0_Q1]
        if history_len < 3:
            return False, "Cannot undo at the first question"
        
        return True, None
    
    def perform_undo(self, session_id: str) -> Tuple[Optional[dict], Optional[str]]:
        """
        Performs the undo operation.
        
        Returns:
            (reverted_state: Optional[dict], error: Optional[str])
        """
        try:
            history_len = db.get_session_history_length(session_id)
            print(f"[UNDO] Session {session_id[:8]}... has {history_len} states in history")
            
            # Need at least 3 states: [S1_Q2, S1, S0_Q1]
            # If we have less, we can't undo
            if history_len < 3:
                # Return current state with error
                current_state = db.get_current_session_state(session_id)
                print(f"[UNDO] Can't undo - returning current state with error")
                if current_state:
                    print(f"[UNDO] Current state: asked={len(current_state.get('asked_features', []))}, count={current_state.get('question_count', 0)}")
                return current_state, "Cannot undo at the first question"
            
            # Pop answer state
            popped_answer = db.pop_session_state(session_id)
            if not popped_answer:
                print(f"[UNDO] ERROR: Failed to pop answer state")
                return None, "Session expired or not found"
            
            # Pop question state
            popped_question = db.pop_session_state(session_id)
            if not popped_question:
                print(f"[UNDO] ERROR: Failed to pop question state, restoring answer")
                # Restore answer state
                db.push_session_state(session_id, popped_answer)
                return None, "Session history corrupted"
            
            # Get the now-current state
            reverted_state = db.get_current_session_state(session_id)
            if not reverted_state:
                print(f"[UNDO] ERROR: No current state after pops")
                return None, "Session state corrupted after undo"
            
            print(f"[UNDO] Reverted to state: asked={len(reverted_state.get('asked_features', []))}, count={reverted_state.get('question_count', 0)}")
            
            # Loop to skip past any guess states
            reverted_state = self._skip_past_guesses(session_id, reverted_state)
            
            print(f"[UNDO] After skip guesses: asked={len(reverted_state.get('asked_features', []))}, count={reverted_state.get('question_count', 0)}")
            
            return reverted_state, None
        except Exception as e:
            print(f"[UNDO] Exception in perform_undo: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Undo failed: {str(e)}"
    
    def _skip_past_guesses(self, session_id: str, state: dict) -> dict:
        """
        Continues undoing if we're on a guess state.
        This ensures we land on a real question.
        """
        max_iterations = 20  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            asked_features = state.get('asked_features', [])
            if not asked_features:
                break
            
            last_feature = asked_features[-1]
            
            # If not a guess, we're done
            if last_feature not in ['final_guess', 'sneaky_guess']:
                break
            
            print(f"Undo found guess '{last_feature}', undoing again...")
            
            # Check if we can undo further
            history_len = db.get_session_history_length(session_id)
            if history_len < 3:
                print("Undo loop stopped at history base")
                break
            
            # Pop answer to guess
            popped_answer = db.pop_session_state(session_id)
            if not popped_answer:
                print("Failed to pop answer during guess skip")
                break
            
            # Pop guess question
            popped_question = db.pop_session_state(session_id)
            if not popped_question:
                db.push_session_state(session_id, popped_answer)
                print("Failed to pop question during guess skip")
                break
            
            # Get new current state
            state = db.get_current_session_state(session_id)
            if not state:
                print("Lost state during guess skip")
                break
            
            iterations += 1
        
        return state
    
    def build_response_from_state(self, state: dict, session_id: str,
                                   error: Optional[str] = None) -> dict:
        """
        Builds the API response from a game state.
        
        This reconstructs what question/guess should be shown.
        """
        if not state:
            return {'error': 'Session not found'}
        
        domain_name = state.get('domain_name')
        
        # Access engine with proper locking
        with self.service.engines_lock:
            engine = self.service.engines.get(domain_name)
        
        if not engine:
            return {'error': f"Domain {domain_name} not found"}
        
        asked_features = state.get('asked_features', [])
        
        # Handle case where we've undone to S0 (no questions asked yet)
        if not asked_features:
            return self._build_first_question_response(state, engine, error, session_id)
        
        last_feature = asked_features[-1]
        
        # If we have an error (like "can't undo"), we still need to show the current question
        # The error case means we're at S0_Q1 and can't go back further
        
        # Handle different feature types
        if last_feature == 'sneaky_guess':
            return self._build_sneaky_guess_response(state, engine, error)
        elif last_feature == 'final_guess':
            return self._build_final_guess_response(state, error)
        else:
            return self._build_regular_question_response(
                state, engine, last_feature, error
            )
    
    def _build_first_question_response(self, state: dict, 
                                       engine, error: Optional[str],
                                       session_id: str) -> dict:
        """Builds response for the first question (S0 -> S0_Q1)."""
        feature, question, updated_state = self.service.get_next_question(state)
        
        if feature:
            updated_state['asked_features'].append(feature)
        
        db.push_session_state(session_id, updated_state)
        
        top_pred = self.service.get_top_predictions(updated_state, n=1)
        
        response = {
            'question': question,
            'feature': feature,
            'question_number': 1,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False,
            'undo_status': 'failed_at_start' if error else 'ok'
        }
        
        if error:
            response['error'] = error
        
        return response
    
    def _build_sneaky_guess_response(self, state: dict, 
                                     engine, error: Optional[str]) -> dict:
        """Builds response for a sneaky guess."""
        top_pred = self.service.get_top_predictions(state, n=1)
        animal_name = top_pred[0]['animal'] if top_pred else "your animal"
        
        article = self._get_article(animal_name)
        
        # question_count is the number of ANSWERED questions
        # The current question being shown is question_count + 1
        response = {
            'question': f"Is it {article} {animal_name}?",
            'feature': 'sneaky_guess',
            'animal_name': animal_name,
            'is_sneaky_guess': True,
            'question_number': state.get('question_count', 0) + 1,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'undo_status': 'ok'
        }
        
        if error:
            response['error'] = error
            response['undo_status'] = 'failed_at_start'
        
        return response
    
    def _build_final_guess_response(self, state: dict, 
                                    error: Optional[str]) -> dict:
        """Builds response for a final guess."""
        top_preds = self.service.get_top_predictions(state, n=5)
        
        response = {
            'should_guess': True,
            'guess': top_preds[0]['animal'] if top_preds else "your animal",
            'guess_type': 'final',
            'top_predictions': top_preds,
            'undo_status': 'ok'
        }
        
        if error:
            response['error'] = error
            response['undo_status'] = 'failed_at_start'
        
        return response
    
    def _build_regular_question_response(self, state: dict, engine,
                                         feature: str, error: Optional[str]) -> dict:
        """Builds response for a regular question."""
        question_text = engine.questions_map.get(
            feature, 
            f"Does it have {feature.replace('_', ' ')}?"
        )
        
        top_pred = self.service.get_top_predictions(state, n=1)
        
        # question_count is the number of ANSWERED questions
        # The current question being shown is question_count + 1
        response = {
            'question': question_text,
            'feature': feature,
            'question_number': state.get('question_count', 0) + 1,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False,
            'undo_status': 'ok'
        }
        
        if error:
            response['error'] = error
            response['undo_status'] = 'failed_at_start'
        
        return response
    
    @staticmethod
    def _get_article(word: str) -> str:
        """Determines if 'a' or 'an' should precede the word."""
        if not word:
            return "a"
        try:
            first_char = word.strip()[0].lower()
            if first_char in 'aeiou':
                return "an"
        except IndexError:
            return "a"
        return "a"