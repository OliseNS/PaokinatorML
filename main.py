import uvicorn
import uuid
import time
import traceback
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager

import config
from service import AkinatorService
from undo_handler import UndoHandler
import db


# --- Request Models ---
class StartPayload(BaseModel):
    domain_name: str = "animals"

class AnswerPayload(BaseModel):
    feature: str
    answer: str
    animal_name: Optional[str] = None

class RejectPayload(BaseModel):
    animal_name: str

class SuggestFeaturePayload(BaseModel):
    domain_name: str
    feature_name: str
    question_text: str
    item_name: str
    fuzzy_value: float

# --- NEW: Report Models ---
class ReportQuestion(BaseModel):
    question: str
    feature: str
    user_answer: float
    consensus_answer: Optional[float] = None

class GameReport(BaseModel):
    item_name: str
    is_new_item: bool
    questions: List[ReportQuestion]


# --- Global Variables ---
service: AkinatorService = None
undo_handler: UndoHandler = None


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, undo_handler
    print("FastAPI server starting up...")
    try:
        service = AkinatorService()
        undo_handler = UndoHandler(service)
        
        # Warmup
        print("🔥 Warming up engine for default 'animals' domain...")
        if "animals" in service.get_available_domains():
            initial_state = service.create_initial_state(domain_name="animals")
            service.get_next_question(initial_state)
            service.get_top_predictions(initial_state, n=5)
        else:
            print("⚠ 'animals' domain not found, skipping warmup.")
        
        print(f"✅ FastAPI startup complete. {len(service.get_available_domains())} domains loaded.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize. Server cannot start.")
        print(f"Error: {e}")
        service = None
    
    yield
    
    print("FastAPI server shutting down...")


# --- App Initialization ---
app = FastAPI(
    title="Akinator API",
    description="Multi-domain Akinator-style game server.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
    return response


def get_service() -> AkinatorService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    return service


def get_article(word: str) -> str:
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


# --- Endpoints ---

@app.get("/domains")
async def get_available_domains():
    srv = get_service()
    return {"domains": srv.get_available_domains()}


@app.post("/start")
async def start_game(payload: StartPayload):
    srv = get_service()
    session_id = str(uuid.uuid4())
    try:
        game_state = srv.create_initial_state(payload.domain_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Initial state is set to 'initial' by state_manager
    db.push_session_state(session_id, game_state)
    return {"session_id": session_id, "domain_name": payload.domain_name}


@app.get("/question/{session_id}")
async def get_question(session_id: str):
    srv = get_service()
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    try:
        should_guess, guess_animal, guess_type = srv.should_make_guess(game_state)
        game_state['last_guess_type'] = guess_type

        if should_guess:
            top_predictions = srv.get_top_predictions(game_state, n=5)
            
            game_state['asked_features'].append('final_guess')
            game_state['state_type'] = 'question'
            db.push_session_state(session_id, game_state)
            
            return {
                'should_guess': True,
                'guess': guess_animal,
                'guess_type': 'final',
                'top_predictions': top_predictions
            }
        
        # Get regular question
        feature, question, game_state = srv.get_next_question(game_state)

        if feature and feature not in game_state['asked_features']:
            game_state['asked_features'].append(feature)

        game_state['state_type'] = 'question'
        db.push_session_state(session_id, game_state)

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

            return {
                'should_guess': True,
                'guess': top_preds[0]['animal'],
                'guess_type': 'final',
                'top_predictions': top_preds
            }
        
        q_num = game_state.get('question_count', 0) + 1
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
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/answer/{session_id}")
async def submit_answer(session_id: str, payload: AnswerPayload):
    srv = get_service()
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    domain_name = game_state['domain_name']
    client_answer = payload.answer.lower().strip()
    feature = payload.feature
    
    answer_map = {
            # Standard values
            'yes': 'yes', 
            'y': 'yes',
            'no': 'no', 
            'n': 'no',

            # NEW frontend values -> Engine values
            'mostly': 'usually',       # Maps "Mostly" to engine's 0.75
            'probably': 'usually',     # Keep for backward compatibility if needed
            'usually': 'usually',      # Keep for engine compatibility

            'sort of': 'sometimes',    # Maps "Sort of" to engine's 0.5
            'sometimes': 'sometimes',  # Keep for engine compatibility
            'maybe': 'sometimes',      # Keep for robustness

            'not really': 'rarely',    # Maps "Not really" to engine's 0.25
            'rarely': 'rarely'         # Keep for engine compatibility
        }
        
    brain_answer = answer_map.get(client_answer)

    status = 'skipped'
    if client_answer != 'skip' and brain_answer is not None:
        engine = srv.engines.get(domain_name)
        if engine:
            engine_fuzzy_map = engine.fuzzy_map
            fuzzy_value = engine_fuzzy_map.get(brain_answer)
            if fuzzy_value is not None:
                game_state['answered_features'][feature] = fuzzy_value
                game_state = srv.process_answer(game_state, feature, brain_answer)
                game_state['question_count'] += 1
                
                if game_state.get('continue_mode', False):
                    game_state['questions_since_last_guess'] = \
                        game_state.get('questions_since_last_guess', 0) + 1
                    
                status = 'ok'

    game_state['state_type'] = 'answer'
    db.push_session_state(session_id, game_state)
    
    return {
        'status': status,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }


@app.post("/undo/{session_id}")
async def undo_last_action(session_id: str):
    """
    FIXED: Now properly handles question numbering and guess states.
    """
    try:
        srv = get_service()
        
        # Perform the undo
        reverted_state, error = undo_handler.perform_undo(session_id)
        
        if not reverted_state:
            raise HTTPException(
                status_code=404, 
                detail=error or "Session expired or not found"
            )
        
        # Build response from the reverted state (pass session_id)
        response = undo_handler.build_response_from_state(reverted_state, session_id, error)
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /undo: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/continue/{session_id}")
async def continue_game(session_id: str):
    srv = get_service()
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.activate_continue_mode(game_state)
    db.push_session_state(session_id, game_state)
    
    return {"status": "continuing"}


@app.post("/reject/{session_id}")
async def reject_animal(session_id: str, payload: RejectPayload):
    srv = get_service()
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    game_state = srv.reject_guess(game_state, payload.animal_name)
    db.push_session_state(session_id, game_state)
    
    return {
        'status': 'ask_to_continue',
        'animal': payload.animal_name,
        'top_predictions': srv.get_top_predictions(game_state, n=5)
    }


# --- REMOVED /win and /learn ENDPOINTS ---
# The new /report endpoint now handles this functionality.


@app.get("/items_for_questions/{domain_name}")
async def get_items_for_questions(domain_name: str):
    import random
    srv = get_service()
    
    with srv.engines_lock:
        engine = srv.engines.get(domain_name)
        if not engine:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found.")
        
        all_items = list(engine.animals)
        if len(all_items) == 0:
            return {"items": []}
        
        num_to_select = min(5, len(all_items))
        selected_items = random.sample(all_items, num_to_select)
        
        return {"items": selected_items}


@app.post("/suggest_feature")
async def suggest_feature(payload: SuggestFeaturePayload):
    srv = get_service()
    
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


@app.get("/features_for_data_collection/{domain_name}")
async def get_features_for_data_collection(domain_name: str, item_name: str):
    srv = get_service()
    try:
        features = srv.get_features_for_data_collection(domain_name, item_name)
        return {"features": features, "item_name": item_name, "domain_name": domain_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error in /features_for_data_collection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/stats")
async def get_stats():
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


@app.get("/predictions/{session_id}")
async def get_predictions(session_id: str):
    srv = get_service()
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    top_predictions = srv.get_top_predictions(game_state, n=10)
    return {'top_predictions': top_predictions}


@app.get("/performance")
async def get_performance():
    srv = get_service()
    
    optimizations = {}
    with srv.engines_lock:
        engine = srv.engines.get("animals")
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

@app.get("/report/{session_id}", response_model=GameReport)
async def get_and_finalize_game_report(
    session_id: str, 
    item_name: str, 
    is_new: bool = False
):
    """
    Retrieves a final report AND finalizes the game.
    Replaces calls to /win and /learn.
    """
    srv = get_service()
    game_state = db.get_current_session_state(session_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Session expired or not found")

    try:
        user_answers = game_state.get('answered_features', {})
        domain_name = game_state.get('domain_name', 'animals')
        
        # --- FIX START: Check if already finalized to prevent duplicate votes ---
        if game_state.get('game_finalized', False):
             return srv.get_game_report(
                domain_name=domain_name,
                item_name=item_name,
                user_answers=user_answers,
                is_new=is_new
            )
        report_data = srv.get_game_report(
            domain_name=domain_name,
            item_name=item_name,
            user_answers=user_answers,
            is_new=is_new
        )
        
        # 2. Finalize the game (write-op)
        if is_new:
            srv.learn_new_animal(item_name, user_answers, domain_name)
        else:
            srv.record_suggestion(item_name, user_answers, domain_name)
        
        # 3. Clean up: Don't delete immediately, just mark as finalized.
        # Redis TTL will clean it up after 30 minutes.
        # db.delete_session(session_id)  <-- REMOVED THIS LINE
        game_state['game_finalized'] = True
        db.push_session_state(session_id, game_state)
        
        # 4. Return the report
        return report_data
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error in /report: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/admin/reload", include_in_schema=False)
async def trigger_reload(reload_token: str = Header(..., alias="X-Reload-Token")):
    if not config.RELOAD_SECRET_TOKEN:
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
        print(f"ERROR: /admin/reload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start reload process")


# --- NEW ENDPOINT ---
@app.get("/admin/feature_gains/{domain_name}", include_in_schema=False)
async def get_feature_gains(
    domain_name: str,
    reload_token: str = Header(..., alias="X-Reload-Token")
):
    """
    NEW: This endpoint directly answers your question about
    calculating the gain for all features. It returns a JSON
    list of all features and their initial information gain.
    """
    if not config.RELOAD_SECRET_TOKEN:
        raise HTTPException(status_code=503, detail="Admin endpoints are not configured")
        
    if reload_token != config.RELOAD_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing reload token")
    
    srv = get_service()
    
    try:
        gains_data = srv.get_feature_gains(domain_name)
        return {
            "domain_name": domain_name,
            "feature_count": len(gains_data),
            "features_by_gain": gains_data
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"ERROR: /admin/feature_gains failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate gains")
# --- END NEW ENDPOINT ---


if __name__ == "__main__":
    print("Starting server with uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=False)