import uvicorn
import uuid
import time
import gzip
import json
import random
import traceback # For better error logging
from fastapi import FastAPI, HTTPException, Request, Header, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

# Import from our project files (Assumed to exist for the server to run)
import config
from service import AkinatorService
import db


class StartPayload(BaseModel):
    domain_name: str = "animals" # Default to "animals"

class AnswerPayload(BaseModel):
    feature: str
    answer: str
    animal_name: Optional[str] = None # For sneaky_guess

class RejectPayload(BaseModel):
    animal_name: str

class WinPayload(BaseModel):
    animal_name: str

class LearnPayload(BaseModel):
    animal_name: str

# --- NEW: Model for suggesting a feature ---
class SuggestFeaturePayload(BaseModel):
    domain_name: str
    feature_name: str
    question_text: str
    item_name: str     # Changed from animal_name to be generic
    fuzzy_value: float # The answer for that item (e.g., 1.0 for 'yes')

# --- Global Service Variable ---
service: AkinatorService | None = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("FastAPI server starting up...")
    try:
        service = AkinatorService()
        
        # Warmup: Precompute initial state and first question for default domain
        print("🔥 Warming up engine for default 'animals' domain...")
        if "animals" in service.get_available_domains():
            initial_state = service.create_initial_state(domain_name="animals")
            service.get_next_question(initial_state)
            service.get_top_predictions(initial_state, n=5)
        else:
            print("⚠ 'animals' domain not found, skipping warmup.")
        
        print(f"✅ FastAPI startup complete. {len(service.get_available_domains())} domains loaded.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize AkinatorService. Server cannot start.")
        print(f"Error: {e}")
        service = None
    
    yield
    
    print("FastAPI server shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Akinator API",
    description="Multi-domain Akinator-style game server.",
    version="1.3.2", # Incremented version for Undo fix
    lifespan=lifespan
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000 # Corrected process time calculation
    response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
    return response

# --- Helper function to check service ---
def get_service() -> AkinatorService:
    if service is None:
        print("ERROR: Service not available. It may have failed to load on startup.")
        raise HTTPException(status_code=503, detail="Service not available")
    return service

# --- API Endpoints ---

@app.get("/domains", summary="Get available domains")
async def get_available_domains():
    """
    Returns a list of all currently loaded game domains (e.g., "animals").
    """
    srv = get_service()
    return {"domains": srv.get_available_domains()}

@app.post("/start", summary="Start a new game session")
async def start_game(payload: StartPayload):
    """
    Initializes a new game session for a specific domain.
    """
    srv = get_service()
    session_id = str(uuid.uuid4())
    try:
        game_state = srv.create_initial_state(payload.domain_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # MODIFIED: Use push_session_state
    db.push_session_state(session_id, game_state)
    return {"session_id": session_id, "domain_name": payload.domain_name}


@app.get("/question/{session_id}", summary="Get the next question")
async def get_question(session_id: str):
    """
    Fetches the next question for the given session.
    This endpoint will also return a guess if the engine is confident.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    def get_article(word: str) -> str:
        """Determines if 'a' or 'an' should precede the word."""
        if not word:
            return "a"
        try:
            # Check if the first non-space character is a vowel (case-insensitive)
            first_char = word.strip()[0].lower()
            if first_char in 'aeiou':
                return "an"
        except IndexError:
            return "a" # Default for empty string
        return "a"
    
    try:
        # should_make_guess *mutates* state (middle_guess_made, continue_mode)
        should_guess, guess_animal, guess_type = srv.should_make_guess(game_state)
        
        # --- NEW: Store guess type in session ---
        game_state['last_guess_type'] = guess_type
        # --- END NEW ---

        if should_guess:
            top_predictions = srv.get_top_predictions(game_state, n=5)
            
            # --- NEW FIX: Add guess type to asked_features before saving ---
            if guess_type == 'final':
                game_state['asked_features'].append('final_guess')
            else: # 'middle'
                game_state['asked_features'].append('sneaky_guess')
            # --- END FIX ---

            # MODIFIED: Use push_session_state
            db.push_session_state(session_id, game_state) # Save state changes
            
            if guess_type == 'final':
                return {
                    'should_guess': True,
                    'guess': guess_animal,
                    'guess_type': guess_type,
                    'top_predictions': top_predictions
                }
            
            # --- This must be 'middle' guess (sneaky_guess) ---
            q_num = game_state.get('question_count', 0)
            top_pred = srv.get_top_predictions(game_state, n=1)
            
            article = get_article(guess_animal)
            
            return {
                'should_guess': False, # It's a question, not a final guess
                'question': f"Is it {article} {guess_animal}?",
                'feature': 'sneaky_guess',
                'animal_name': guess_animal,
                'is_sneaky_guess': True,
                'question_number': q_num,
                'top_prediction': top_pred[0] if top_pred else None
            }
        
        # --- Not guessing, so get next question ---
        feature, question, game_state = srv.get_next_question(game_state)

        # --- NEW FIX: Add the new question to the state *before* pushing ---
        if feature and feature not in game_state['asked_features']:
            # This state now represents the question being asked
            game_state['asked_features'].append(feature)
        # --- END FIX ---

        # MODIFIED: Use push_session_state
        db.push_session_state(session_id, game_state) # Save state (which now includes the new question)

        if not feature or not question:
            top_preds = srv.get_top_predictions(game_state, n=3)
            if not top_preds:
                 return {
                    'question': "I'm stumped! You win!",
                    'feature': "game_over_lost",
                    'top_predictions': [],
                    'should_guess': False,
                    'is_sneaky_guess': False
                }

            # Out of questions, force a final guess
            return {
                'should_guess': True,
                'guess': top_preds[0]['animal'],
                'guess_type': 'final',
                'top_predictions': top_preds
            }
        
        q_num = game_state.get('question_count', 0) + 1 # Get from state, default to 0
        top_pred = srv.get_top_predictions(game_state, n=1)
        
        return {
            'question': question,
            'feature': feature,
            'question_number': q_num,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False
        }
    except Exception as e:
        print(f"Error in /question: {e}")
        print(traceback.format_exc()) # More detailed error
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/answer/{session_id}", summary="Submit an answer")
async def submit_answer(session_id: str, payload: AnswerPayload):
    """
    Processes a user's answer to a question.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    domain_name = game_state['domain_name'] # Get domain from state

    client_answer = payload.answer.lower().strip()
    feature = payload.feature
    
    if feature == 'sneaky_guess':
        if client_answer in ['yes', 'y', 'usually']:
            return {
                'status': 'guess_correct',
                'animal': payload.animal_name,
                'top_predictions': srv.get_top_predictions(game_state, n=5)
            }

        game_state = srv.reject_guess(game_state, payload.animal_name)
        # Redundant check removed, 'sneaky_guess' is already in asked_features
        # from the /question endpoint.

        # MODIFIED: Use push_session_state
        db.push_session_state(session_id, game_state)
        return {
            'status': 'ok',
            'top_predictions': srv.get_top_predictions(game_state, n=5)
        }
    
    # This check is now correctly false, as /question added the feature
    # if feature not in game_state['asked_features']:
    #     game_state['asked_features'].append(feature)

    answer_map = {
        'yes': 'yes', 'y': 'yes',
        'no': 'no', 'n': 'no',
        'usually': 'usually',
        'sometimes': 'sometimes',
        'rarely': 'rarely'
    }
    brain_answer = answer_map.get(client_answer)

    status = 'skipped'
    if client_answer != 'skip' and brain_answer is not None:
        # Use the engine from the service to get the fuzzy map
        engine_fuzzy_map = srv.engines.get(domain_name).fuzzy_map if srv.engines.get(domain_name) else {}
        fuzzy_value = engine_fuzzy_map.get(brain_answer)
        if fuzzy_value is not None:
            game_state['answered_features'][feature] = fuzzy_value
            game_state = srv.process_answer(game_state, feature, brain_answer)
            game_state['question_count'] += 1
            
            # --- NEW: Increment continue counter ---
            if game_state.get('continue_mode', False):
                game_state['questions_since_last_guess'] = game_state.get('questions_since_last_guess', 0) + 1
            # --- END NEW ---
                
            status = 'ok'

    # MODIFIED: Use push_session_state
    db.push_session_state(session_id, game_state)
    
    return {
        'status': status,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }

# --- MODIFIED ENDPOINT ---
@app.post("/undo/{session_id}", summary="Undo the last action")
async def undo_last_action(session_id: str):
    """
    Reverts the game state to the previous question by popping
    the current state from the history.
    """
    srv = get_service() # Just to check service
    
    # Check history length.
    # We need at least 3 states to undo: [S1_Q2, S1, S0_Q1]
    # Popping S1_Q2 and S1 leaves S0_Q1 (the state for Q1)
    history_len = db.get_session_history_length(session_id)
    if history_len < 3:
        # Can't undo, just return the current (initial) state
        current_state = db.get_current_session_state(session_id)
        if not current_state:
            raise HTTPException(status_code=404, detail="Session expired or not found")
        
        # Convert to JSON-safe and add error status
        json_safe_state = db.convert_state_to_json_safe(current_state)
        if not json_safe_state:
             raise HTTPException(status_code=404, detail="Session expired or not found")
        
        json_safe_state['undo_status'] = 'failed_at_start'
        json_safe_state['error'] = 'Cannot undo at the first question.'
        
        # Re-build the current question view
        # This is likely state S0_Q1, which is fine
        if not json_safe_state.get('asked_features'):
             # This is state S0, before Q1. Re-run get_question logic.
             print("Undo called on S0, re-running get_question")
             # We must NOT push the state again, so we can't call get_question directly.
             # Re-create S0_Q1 logic here.
             feature, question, game_state = srv.get_next_question(current_state)
             if feature:
                 game_state['asked_features'].append(feature)
             db.push_session_state(session_id, game_state) # Push S0_Q1
             
             top_pred = srv.get_top_predictions(game_state, n=1)
             return {
                 'question': question,
                 'feature': feature,
                 'question_number': 1, # It's the first question
                 'top_prediction': top_pred[0] if top_pred else None,
                 'should_guess': False,
                 'is_sneaky_guess': False,
                 'undo_status': 'failed_at_start'
             }

        # It's S0_Q1 (or later), reconstruct the *current* question
        last_feature_asked = json_safe_state.get('asked_features', [])[-1]
        domain_name = json_safe_state.get('domain_name')
        engine = srv.engines.get(domain_name)
        if not engine:
            raise HTTPException(status_code=404, detail=f"Domain {domain_name} not found")
            
        question_text = engine.questions_map.get(last_feature_asked, f"Does it have {last_feature_asked.replace('_', ' ')}?")
        top_pred = srv.get_top_predictions(current_state, n=1)
        
        response = {
            'question': question_text,
            'feature': last_feature_asked,
            # FIX 1: Add 1 to the question count
            'question_number': json_safe_state.get('question_count', 0) + 1,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False,
            'undo_status': 'failed_at_start' # Add flag
        }
        return response

    # Pop the current "answer" state (e.g., S1)
    popped_a_state = db.pop_session_state(session_id)
    if not popped_a_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")

    # Pop the current "question" state (e.g., S0_Q1)
    popped_q_state = db.pop_session_state(session_id)
    if not popped_q_state:
        # This is bad, we're in a weird state. Push the a_state back on.
        db.push_session_state(session_id, popped_a_state)
        raise HTTPException(status_code=500, detail="Session history corrupted")
    
    reverted_state = db.get_current_session_state(session_id)
    if not reverted_state:
        raise HTTPException(status_code=404, detail="Session state corrupted after undo")

    # --- FIX 2: Loop to undo past guesses ---
    last_feature_list = reverted_state.get('asked_features', [])
    last_feature = last_feature_list[-1] if last_feature_list else None
    
    while last_feature in ['final_guess', 'sneaky_guess']:
        print(f"Undo found guess '{last_feature}', undoing again...")
        history_len = db.get_session_history_length(session_id)
        if history_len < 3:
            # We've undone as far as we can. This is the first question (which was a guess).
            print("Undo loop stopped at history base")
            break # Exit loop, will handle S0 or S0_Q_Guess state below

        # Pop the answer to the guess (e.g., S_Reject)
        popped_a_state = db.pop_session_state(session_id)
        if not popped_a_state:
            raise HTTPException(status_code=500, detail="Session history corrupted during undo loop")
            
        # Pop the guess-question (e.g., S1_Guess)
        popped_q_state = db.pop_session_state(session_id)
        if not popped_q_state:
            db.push_session_state(session_id, popped_a_state) # Restore
            raise HTTPException(status_code=500, detail="Session history corrupted during undo loop")

        reverted_state = db.get_current_session_state(session_id)
        if not reverted_state:
             raise HTTPException(status_code=500, detail="Session history corrupted during undo loop")
             
        last_feature_list = reverted_state.get('asked_features', [])
        last_feature = last_feature_list[-1] if last_feature_list else None
        
        if last_feature is None:
            # We've landed on S0
            print("Undo loop stopped at S0")
            break
    # --- END FIX 2 ---

    # --- THIS IS THE FIX ---
    # We cannot return 'reverted_state' as it contains numpy arrays.
    # We must build a JSON-safe response, just like /question does.
    
    # Handle S0 state (we've undone all the way to the start)
    if not reverted_state.get('asked_features', []):
        # This means we reverted to S0. We need to show Q1.
        print("Undo reverted to S0, re-running get_question")
        # We must NOT push the state again, so we can't call get_question directly.
        # Re-create S0_Q1 logic here.
        feature, question, game_state = srv.get_next_question(reverted_state)
        if feature:
            game_state['asked_features'].append(feature)
        db.push_session_state(session_id, game_state) # Push S0_Q1
        
        top_pred = srv.get_top_predictions(game_state, n=1)
        return {
            'question': question,
            'feature': feature,
            'question_number': 1, # It's the first question
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False,
            'undo_status': 'ok'
        }

    # The 'reverted_state' (e.g., S0_Q1) has the question info we want to display.
    
    domain_name = reverted_state.get('domain_name')
    engine = srv.engines.get(domain_name)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Domain {domain_name} not found")

    asked_features_list = reverted_state.get('asked_features', [])
    last_feature_asked = asked_features_list[-1]
    response = {}

    # Handle sneaky guess reconstruction
    if last_feature_asked == 'sneaky_guess':
        top_pred = srv.get_top_predictions(reverted_state, n=1)
        animal_name = top_pred[0]['animal'] if top_pred else "your animal"
        
        def get_article(word: str) -> str:
            if not word: return "a"
            try:
                first_char = word.strip()[0].lower()
                if first_char in 'aeiou': return "an"
            except IndexError: return "a"
            return "a"
        
        article = get_article(animal_name)
        
        response = {
            'question': f"Is it {article} {animal_name}?",
            'feature': 'sneaky_guess',
            'animal_name': animal_name,
            'is_sneaky_guess': True,
            # FIX 1: Add 1 to the question count
            'question_number': reverted_state.get('question_count', 0) + 1,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'undo_status': 'ok'
        }
    
    # Handle final guess reconstruction
    elif last_feature_asked == 'final_guess': # Check against the feature we added
        top_preds = srv.get_top_predictions(reverted_state, n=5)
        response = {
            'should_guess': True,
            'guess': top_preds[0]['animal'] if top_preds else "your animal",
            'guess_type': 'final',
            'top_predictions': top_preds,
            'undo_status': 'ok'
        }
        
    # Handle regular question reconstruction
    else:
        question_text = engine.questions_map.get(last_feature_asked, f"Does it have {last_feature_asked.replace('_', ' ')}?")
        top_pred = srv.get_top_predictions(reverted_state, n=1)
        response = {
            'question': question_text,
            'feature': last_feature_asked,
            # FIX 1: Add 1 to the question count
            'question_number': reverted_state.get('question_count', 0) + 1,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False,
            'undo_status': 'ok'
        }
    
    return response
# --- END MODIFIED ENDPOINT ---


@app.post("/continue/{session_id}", summary="Continue game after wrong guess")
async def continue_game(session_id: str):
    """
    Sets the game state to 'continue mode' to ask more questions
    after a final guess was rejected.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.activate_continue_mode(game_state)
    # MODIFIED: Use push_session_state
    db.push_session_state(session_id, game_state)
    
    return {"status": "continuing"}


@app.post("/reject/{session_id}", summary="Reject a guess")
async def reject_animal(session_id: str, payload: RejectPayload):
    """
    Tells the engine its guess was wrong.
    This now returns 'ask_to_continue' for final guesses.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.reject_guess(game_state, payload.animal_name)
    
    # We assume this endpoint is only for final guesses, as 'middle'
    # is handled by /answer. The user's request confirms this.
    
    # MODIFIED: Use push_session_state
    db.push_session_state(session_id, game_state)
    
    return {
        'status': 'ask_to_continue', # <-- CHANGED
        'animal': payload.animal_name,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }


@app.post("/win/{session_id}", summary="Confirm a final guess")
async def confirm_win(session_id: str, payload: WinPayload):
    """
    Confirms the engine's final guess was correct.
    This saves the session data as a suggestion.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    answered_features = game_state['answered_features']
    domain_name = game_state['domain_name'] # Get domain
    
    srv.record_suggestion(payload.animal_name, answered_features, domain_name) # Pass domain

    db.delete_session(session_id)
    
    return {
        'status': 'win_confirmed',
        'animal': payload.animal_name
    }


@app.post("/learn/{session_id}", summary="Learn a new animal")
async def learn_animal(session_id: str, payload: LearnPayload):
    """
    Finishes the game and teaches the engine a new animal.
    This is called when the user declines to continue and
    wants to add their animal.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    answered_features = game_state['answered_features']
    domain_name = game_state['domain_name'] # Get domain
    
    result = srv.learn_new_animal(payload.animal_name, answered_features, domain_name) # Pass domain

    db.delete_session(session_id)
    
    return {
        'message': f"Thank you! I've learned about {payload.animal_name}.",
        'features_learned': len(answered_features),
        'db_status': result
    }

@app.get("/items_for_questions/{domain_name}", summary="Get random items for feature suggestion")
async def get_items_for_questions(domain_name: str):
    """
    Returns a list of 5 random items (e.g., animals) from a
    given domain. Used by the 'add_questions' page in the client.
    """
    srv = get_service()
    
    with srv.engines_lock:
        engine = srv.engines.get(domain_name)
        
        if not engine:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found.")
        
        # engine.animals is the list of item names
        all_items = list(engine.animals)
        
        if len(all_items) == 0:
            # Handle empty domain, though unlikely
            return {"items": []}
        
        # Select 5 random items, or all items if fewer than 5
        num_to_select = min(5, len(all_items))
        selected_items = random.sample(all_items, num_to_select)
        
        return {"items": selected_items}


@app.post("/suggest_feature", summary="Suggest a new feature")
async def suggest_feature(payload: SuggestFeaturePayload):
    """
    Allows a user to suggest a new feature (question) and
    provide the first answer for a specific item.
    This does not require a session ID.
    """
    srv = get_service() # Just to check if service is up
    
    result = db.suggest_new_feature(
        domain_name=payload.domain_name,
        feature_name=payload.feature_name,
        question_text=payload.question_text,
        item_name=payload.item_name,
        fuzzy_value=payload.fuzzy_value
    )
    
    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result


@app.get("/features_for_data_collection/{domain_name}", summary="Get features for data collection")
async def get_features_for_data_collection(domain_name: str, item_name: str):
    """
    Gets 5 features for a specific item to collect data from the user.
    
    Prioritizes features that are 'null' (NaN) for that item,
    then pads the list with globally 'sparse' features
    to ensure 5 questions are always returned for data collection.
    """
    srv = get_service()
    try:
        features = srv.get_data_collection_features(domain_name, item_name)
        
        # This could happen if the engine has no allowed features at all
        if not features:
             engine = srv.engines.get(domain_name)
             if engine and len(engine.animals) > 0:
                print(f"Warning: No data collection features found for {item_name} in {domain_name}.")
        
        return {"features": features, "item_name": item_name, "domain_name": domain_name}
    
    except ValueError as e:
        # Raised by service if domain not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # General error
        print(f"Error in /features_for_data_collection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/stats", summary="Get server statistics")
async def get_stats():
    """
    Returns statistics about active sessions and all loaded domains.
    """
    srv = get_service()
    active_sessions = db.get_active_session_count()
    
    with srv.engines_lock:
        domain_stats = {}
        for domain, engine in srv.engines.items():
            domain_stats[domain] = {
                'total_items': len(engine.animals),
                'total_features': len(engine.feature_cols),
                'precomputed_features': len(engine.sorted_initial_feature_indices)
            }
    
    with srv._cache_lock:
        cache_size = len(srv._prediction_cache)
    
    return {
        'active_sessions': active_sessions,
        'prediction_cache_size': cache_size,
        'domains': domain_stats
    }


@app.get("/predictions/{session_id}", summary="Get top 10 predictions")
async def get_predictions(session_id: str):
    """
    Debug endpoint to see the current top 10 predictions for a session.
    """
    srv = get_service()
    # MODIFIED: Use get_current_session_state
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    top_predictions = srv.get_top_predictions(game_state, n=10)
    return {'top_predictions': top_predictions}


@app.get("/performance", summary="Get performance metrics")
async def get_performance():
    """
    Returns performance metrics for the default 'animals' domain.
    """
    srv = get_service()
    
    optimizations = {}
    with srv.engines_lock:
        engine = srv.engines.get("animals") # Get default domain
        if engine:
            optimizations = {
                'domain': 'animals',
                'uniform_prior_cached': hasattr(engine, 'sorted_initial_feature_indices'),
                'precomputed_features': len(engine.sorted_initial_feature_indices),
            }
        else:
            optimizations = {'domain': 'animals', 'status': 'not_loaded'}
            
    with srv._cache_lock:
        optimizations['prediction_cache_size'] = len(srv._prediction_cache)
    
    return {
        'optimizations': optimizations,
        'performance_tips': [
            "Q0 questions are precomputed for instant response",
            "Top predictions are cached for faster subsequent requests",
            "GZip compression is enabled for smaller responses",
            "Engines are warmed up on startup for optimal performance"
        ]
    }


@app.post("/admin/reload", summary="Trigger engine reload", include_in_schema=False)
async def trigger_reload(reload_token: str = Header(..., alias="X-Reload-Token")):
    """
    Secure endpoint to trigger a background reload of ALL engines.
    """
    if not config.RELOAD_SECRET_TOKEN:
        print("ERROR: Reload endpoint called but RELOAD_SECRET_TOKEN is not set.")
        raise HTTPException(status_code=503, detail="Reload endpoint is not configured")
        
    if reload_token != config.RELOAD_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing reload token")
    
    srv = get_service()
        
    try:
        srv.start_engine_reload()
        return {
            "status": "ok",
            "message": "Engine reload process for ALL domains started in background."
        }
    except Exception as e:
        print(f"ERROR: /admin/reload failed to start: {e}")
        raise HTTPException(status_code=500, detail="Failed to start reload process")

if __name__ == "__main__":
    print("Starting server with uvicorn...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=config.PORT, 
        reload=False
    )

