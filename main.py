import uvicorn
import uuid
import time
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

# Import from our project files
import config
from service import AkinatorService
import db  # We still need db for session functions

# --- Pydantic Models for Request Bodies ---

class AnswerPayload(BaseModel):
    feature: str
    answer: str
    animal_name: Optional[str] = None # For sneaky_guess

class RejectPayload(BaseModel):
    animal_name: str

class LearnPayload(BaseModel):
    animal_name: str

# --- Global Service Variable ---
# This will be populated during the 'lifespan' startup event
service: AkinatorService | None = None

# --- NEW: Lifespan Event Handler ---
# This replaces the deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global service
    print("FastAPI server starting up...")
    try:
        service = AkinatorService(config.QUESTIONS_PATH)
        print("✅ FastAPI startup complete, AkinatorService is loaded.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize AkinatorService. Server cannot start.")
        print(f"Error: {e}")
        service = None
    
    yield  # This is when the application is running
    
    # Code to run on shutdown
    print("FastAPI server shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Akinator API",
    description="Akinator-style game server with FastAPI and PyTorch.",
    version="1.0.0",
    lifespan=lifespan  # Set the lifespan handler
)

# --- Middleware for logging ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
    return response

# --- Helper function to check service ---
def get_service() -> AkinatorService:
    if service is None:
        print("ERROR: Service not available. It may have failed to load on startup.")
        raise HTTPException(status_code=503, detail="Service not available")
    return service

# --- API Endpoints ---

@app.post("/start", summary="Start a new game session")
async def start_game():
    """
    Initializes a new game session and returns a unique session ID.
    """
    srv = get_service()
    session_id = str(uuid.uuid4())
    game_state = srv.create_initial_state()
    
    db.set_session(session_id, game_state)
    return {"session_id": session_id}


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
    
    # Check if we should guess
    should_guess, guess_animal, guess_type = srv.should_make_guess(game_state)
    
    if should_guess:
        top_predictions = srv.get_top_predictions(game_state, n=5)
        # Save state changes (like 'middle_guess_made')
        db.set_session(session_id, game_state)
        
        if guess_type == 'final':
            return {
                'should_guess': True,
                'guess': guess_animal,
                'guess_type': guess_type,
                'top_predictions': top_predictions
            }
        
        # Middle guess (sneaky guess)
        q_num = game_state['question_count'] 
        top_pred = srv.get_top_predictions(game_state, n=1)
        return {
            'should_guess': False,
            'question': f"Is it a/an {guess_animal}?",
            'feature': 'sneaky_guess',
            'animal_name': guess_animal,
            'is_sneaky_guess': True,
            'question_number': q_num,
            'top_prediction': top_pred[0] if top_pred else None
        }
    
    # If not guessing, get next question
    feature, question, game_state = srv.get_next_question(game_state)
    db.set_session(session_id, game_state) 

    if not feature or not question:
        return {
            'question': None,
            'feature': None,
            'top_predictions': srv.get_top_predictions(game_state, n=3)
        }
    
    q_num = game_state['question_count'] + 1  # Add 1 to start from Q1
    top_pred = srv.get_top_predictions(game_state, n=1)
    
    return {
        'question': question,
        'feature': feature,
        'question_number': q_num,
        'top_prediction': top_pred[0] if top_pred else None,
        'should_guess': False,
        'is_sneaky_guess': False
    }


@app.post("/answer/{session_id}", summary="Submit an answer")
async def submit_answer(session_id: str, payload: AnswerPayload):
    """
    Processes a user's answer to a question.
    """
    srv = get_service()
    game_state = db.get_session(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")

    client_answer = payload.answer.lower().strip()
    feature = payload.feature
    
    # Handle sneaky guess
    if feature == 'sneaky_guess':
        if client_answer in ['yes', 'y', 'usually']:
            db.delete_session(session_id)
            return {
                'status': 'guess_correct',
                'animal': payload.animal_name,
                'top_predictions': srv.get_top_predictions(game_state, n=5)
            }

        # Rejected sneaky guess
        game_state = srv.reject_guess(game_state, payload.animal_name)
        if 'sneaky_guess' not in game_state['asked_features']:
            game_state['asked_features'].append('sneaky_guess')

        db.set_session(session_id, game_state)
        return {
            'status': 'ok',
            'top_predictions': srv.get_top_predictions(game_state, n=5)
        }
    
    # Handle normal question
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
        fuzzy_value = srv.engine.fuzzy_map.get(brain_answer)
        if fuzzy_value is not None:
            game_state['answered_features'][feature] = fuzzy_value
            game_state = srv.process_answer(game_state, feature, brain_answer)
            game_state['question_count'] += 1
            status = 'ok'

    db.set_session(session_id, game_state)
    
    return {
        'status': status,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }


@app.post("/reject/{session_id}", summary="Reject a guess")
async def reject_animal(session_id: str, payload: RejectPayload):
    """
    Tells the engine its guess was wrong.
    """
    srv = get_service()
    game_state = db.get_session(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.reject_guess(game_state, payload.animal_name)
    db.set_session(session_id, game_state)
    
    return {
        'status': 'rejected',
        'animal': payload.animal_name,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }


@app.post("/learn/{session_id}", summary="Learn a new animal")
async def learn_animal(session_id: str, payload: LearnPayload):
    """
    Finishes the game and teaches the engine a new animal.
    This endpoint NO LONGER triggers an engine reload.
    """
    srv = get_service()
    game_state = db.get_session(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    answered_features = game_state['answered_features']
    
    # This function now ONLY saves to DB.
    result = srv.learn_animal(payload.animal_name, answered_features)
    
    db.delete_session(session_id)
    
    return {
        'message': f"Thank you! I've learned about {payload.animal_name}.",
        'features_learned': len(answered_features),
        'db_status': result
    }


@app.get("/stats", summary="Get server statistics")
async def get_stats():
    """
    Returns statistics about active sessions and engine state.
    """
    srv = get_service()
    active_sessions = db.get_active_session_count()
    
    # Thread-safe read of total animals
    with srv.engine_lock:
        total_animals = len(srv.engine.animals)
    
    return {
        'active_sessions': active_sessions,
        'total_animals': total_animals
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

# --- NEW: Admin Endpoint ---

@app.post("/admin/reload", summary="Trigger engine reload", include_in_schema=False)
async def trigger_reload(reload_token: str = Header(..., alias="X-Reload-Token")):
    """
    Secure endpoint to trigger a background engine reload.
    This is intended to be called by a cron job.
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
            "message": "Engine reload process started in background."
        }
    except Exception as e:
        print(f"ERROR: /admin/reload failed to start: {e}")
        raise HTTPException(status_code=500, detail="Failed to start reload process")

# --- Run Server ---

if __name__ == "__main__":
    print("Starting server with uvicorn...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=config.PORT, 
        reload=True # Enable auto-reload for development
    )