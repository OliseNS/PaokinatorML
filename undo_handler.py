"""
Handles undo logic for game sessions.
Extracted from main.py for better testability and maintainability.
"""
import db
from typing import Dict, Optional, Tuple
import traceback


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
        
        # Need at least 2 states: [S_Answer, S_Question] or [S_Question, S_Initial]
        if history_len < 2:
            return False, "Cannot undo at the first question"
        
        return True, None

    # --- START OF REPLACEMENT ---
    def perform_undo(self, session_id: str) -> Tuple[Optional[dict], Optional[str]]:
        """
        Performs the undo operation intelligently based on state_type.
        
        Returns:
            (reverted_state: Optional[dict], error: Optional[str])
        """
        try:
            current_state = db.get_current_session_state(session_id)
            if not current_state:
                print(f"[UNDO] ERROR: No current state found for {session_id[:8]}")
                return None, "Session expired or not found"

            state_type = current_state.get('state_type')
            history_len = db.get_session_history_length(session_id)
            
            print(f"[UNDO] Session {session_id[:8]}: history_len={history_len}, state_type='{state_type}'")

            # Cannot undo from initial state
            if state_type == 'initial' or history_len < 2:
                print(f"[UNDO] Can't undo - at initial state or history too short")
                return current_state, "Cannot undo at the first question"
            
            reverted_state = None
            
            if state_type == 'answer':
                # We are on an 'answer' state. (e.g., [S_Answer, S_Question, ...])
                # Pop 1 state to get to the 'question' state.
                db.pop_session_state(session_id)
                reverted_state = db.get_current_session_state(session_id)
                
            elif state_type == 'question':
                # We are on a 'question' state. (e.g., [S_Question, S_Answer, ...])
                
                # --- SMART FIX: This check is robust and prevents the 404 ---
                # If history_len is 2, we are at [S0_Q1, S_Initial]. We can't pop 2.
                if history_len < 3: 
                     print(f"[UNDO] Can't undo - at first question (history_len={history_len})")
                     return current_state, "Cannot undo at the first question"
                
                db.pop_session_state(session_id) # Pop 'question' state
                db.pop_session_state(session_id) # Pop 'answer' state
                reverted_state = db.get_current_session_state(session_id)
                
            else:
                # Fallback for unknown state or old states without state_type
                print(f"[UNDO] WARN: Unknown state_type '{state_type}'. Popping 2.")
                if history_len < 3:
                    return current_state, "Cannot undo at the first question"
                db.pop_session_state(session_id)
                db.pop_session_state(session_id)
                reverted_state = db.get_current_session_state(session_id)

            if not reverted_state:
                print(f"[UNDO] ERROR: No state after popping")
                # This can happen if the stack is corrupted or empty
                return None, "Session state corrupted after undo"

            print(f"[UNDO] Reverted to state: asked={len(reverted_state.get('asked_features', []))}, count={reverted_state.get('question_count', 0)}, type={reverted_state.get('state_type')}")
            
            # Loop to skip past any final guess states we landed on
            reverted_state = self._skip_past_guesses(session_id, reverted_state)
            
            print(f"[UNDO] Final reverted state: asked={len(reverted_state.get('asked_features', []))}, count={reverted_state.get('question_count', 0)}, type={reverted_state.get('state_type')}")
            
            return reverted_state, None
        
        except Exception as e:
            print(f"[UNDO] Exception in perform_undo: {e}")
            traceback.print_exc()
            return None, f"Undo failed: {str(e)}"

    def _skip_past_guesses(self, session_id: str, state: dict) -> dict:
        """
        Continues undoing if we're on a state that should be skipped.
        This ensures we land on a real question or the initial state.
        
        *** SMART FIX: Now only skips 'final_guess' and ensures it
            always lands on a 'question' or 'initial' state. ***
        """
        max_iterations = 20  # Safety limit
        iterations = 0
        current_state = state
        
        while iterations < max_iterations:
            if not current_state:
                print("[UNDO] Skip loop lost state, bailing.")
                return state # Return original state
                
            state_type = current_state.get('state_type')
            
            # --- Valid Stopping Points ---
            
            # 1. Valid stopping point: initial state
            if state_type == 'initial':
                print("[UNDO] Skip loop stopped at initial state")
                break
                
            # 2. Valid stopping point: a 'question' that is NOT a final guess
            if state_type == 'question':
                asked_features = current_state.get('asked_features', [])
                if not asked_features:
                    # This is S0_Q1 (question 1), a valid state
                    print("[UNDO] Skip loop stopped at first question")
                    break 
                    
                last_feature = asked_features[-1]
                
                # --- SMART FIX #1 ---
                # Only 'final_guess' is skipped.
                if last_feature != 'final_guess':
                    print(f"[UNDO] Skip loop stopped on non-final-guess: '{last_feature}'")
                    break
                
                # If we are here, state_type is 'question' AND last_feature is 'final_guess'
                # This state must be skipped.
                print(f"[UNDO] Landed on guess '{last_feature}', undoing again...")

            # --- State Must Be Skipped ---
            # --- SMART FIX #2 ---
            # If we are here, the current_state must be skipped.
            # It's either:
            # 1. state_type == 'answer' (which we always skip)
            # 2. state_type == 'question' AND last_feature == 'final_guess'
            
            # Check if we can undo
            history_len = db.get_session_history_length(session_id)
            if history_len < 2:
                print("[UNDO] Skip loop stopped at history base")
                break
            
            # Pop the state we're on
            popped_state = db.pop_session_state(session_id)
            if not popped_state:
                print("[UNDO] Failed to pop state during skip")
                break # Return the last valid state we had
            
            # If we just popped a 'final_guess' question, we must ALSO pop its preceding answer
            if state_type == 'question': # (and we know from above it was a final_guess)
                if db.get_session_history_length(session_id) < 1:
                    print("[UNDO] History empty after popping guess, stopping")
                    current_state = db.get_current_session_state(session_id)
                    break
                
                answer_pop = db.pop_session_state(session_id) # Pop the answer
                if not answer_pop:
                    print("[UNDO] Failed to pop answer during guess skip")
                    db.push_session_state(session_id, popped_state) # Restore
                    current_state = db.get_current_session_state(session_id)
                    break

            # Get the new current state to evaluate in the next loop
            current_state = db.get_current_session_state(session_id)
            if not current_state:
                 print("[UNDO] Lost state during guess skip loop")
                 return state # Return original state
            
            iterations += 1
        
        return current_state
    # --- END OF REPLACEMENT ---
    
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
        state_type = state.get('state_type')
        
        # --- SMART FIX ---
        # Handle case where we've undone to S0 (initial state)
        if state_type == 'initial':
            return self._build_first_question_response(state, engine, error, session_id)
        
        # Handle empty features just in case
        if not asked_features:
             return self._build_first_question_response(state, engine, error, session_id)

        last_feature = asked_features[-1]
        
        # If we have an error (like "can't undo"), we still need to show the current question
        # The error case means we're at S0_Q1 and can't go back further
        
        # --- UPDATED: Removed Sneaky Guess ---
        if last_feature == 'final_guess':
            return self._build_final_guess_response(state, error)
        else:
            return self._build_regular_question_response(
                state, engine, last_feature, error
            )
        # --- END OF UPDATE ---
    
    def _build_first_question_response(self, state: dict, 
                                       engine, error: Optional[str],
                                       session_id: str) -> dict:
        """Builds response for the first question (S0 -> S0_Q1)."""
        
        # We are at 'initial' state (S0). We need to get the first question
        # and push its state (S0_Q1) so the stack is correct.
        
        feature, question, updated_state = self.service.get_next_question(state)
        
        if feature:
            updated_state['asked_features'].append(feature)
        
        # --- SMART FIX ---
        updated_state['state_type'] = 'question'
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
    
    # --- UPDATED: Removed _build_sneaky_guess_response ---
    # The _build_sneaky_guess_response method was here and is now removed.
    # --- END OF UPDATE ---
    
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