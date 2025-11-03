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
    version="1.2.0", # Incremented version
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
    
    db.set_session(session_id, game_state)
    return {"session_id": session_id, "domain_name": payload.domain_name}


@app.get("/question/{session_id}", summary="Get the next question")
async def get_question(session_id: str):
    """
    Fetches the next question for the given session.
    This endpoint will also return a guess if the engine is confident.
    """
    srv = get_service()
    game_state = db.get_session(session_id)
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
            db.set_session(session_id, game_state) # Save state changes
            
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
        db.set_session(session_id, game_state) # Save state

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
    game_state = db.get_session(session_id)
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
        if 'sneaky_guess' not in game_state['asked_features']:
            game_state['asked_features'].append('sneaky_guess')

        db.set_session(session_id, game_state)
        return {
            'status': 'ok',
            'top_predictions': srv.get_top_predictions(game_state, n=5)
        }
    
    if feature not in game_state['asked_features']:
        game_state['asked_features'].append(feature)

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

    db.set_session(session_id, game_state)
    
    return {
        'status': status,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }

# --- NEW ENDPOINT ---
@app.post("/continue/{session_id}", summary="Continue game after wrong guess")
async def continue_game(session_id: str):
    """
    Sets the game state to 'continue mode' to ask more questions
    after a final guess was rejected.
    """
    srv = get_service()
    game_state = db.get_session(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.activate_continue_mode(game_state)
    db.set_session(session_id, game_state)
    
    return {"status": "continuing"}
# --- END NEW ENDPOINT ---

@app.post("/reject/{session_id}", summary="Reject a guess")
async def reject_animal(session_id: str, payload: RejectPayload):
    """
    Tells the engine its guess was wrong.
    This now returns 'ask_to_continue' for final guesses.
    """
    srv = get_service()
    game_state = db.get_session(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.reject_guess(game_state, payload.animal_name)
    
    # We assume this endpoint is only for final guesses, as 'middle'
    # is handled by /answer. The user's request confirms this.
    
    db.set_session(session_id, game_state)
    
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
    game_state = db.get_session(session_id)
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
    game_state = db.get_session(session_id)
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
    game_state = db.get_session(session_id)
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