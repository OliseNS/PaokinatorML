import os
import uuid
import time
from flask import Flask, request, jsonify

# Import from our new refactored files
import config
from service import AkinatorService
from db import get_session, set_session, delete_session, get_active_session_count

def create_app():
    """Application Factory"""
    app = Flask(__name__)

    # --- Initialize Service ---
    # This now happens inside the factory.
    # If this fails, the app will fail to start (which is good!)
    try:
        service = AkinatorService(config.QUESTIONS_PATH)
        print("Starting Akinator Server...")
        with service.engine_lock:
            print(f"Loaded {len(service.engine.animals)} animals")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize AkinatorService. Server cannot start.")
        print(f"Error: {e}")
        # Re-raise the exception to stop the app from starting in a broken state
        raise
    
    # --- Register Routes ---
    
    @app.route('/start', methods=['POST'])
    def start_game():
        """Start a new game session."""
        # No 'if service is None' check needed!
        session_id = str(uuid.uuid4())
        game_state = service.create_initial_state()
        
        set_session(session_id, game_state)
        return jsonify({'session_id': session_id})


    @app.route('/question/<session_id>', methods=['GET'])
    def get_question(session_id):
        start_t = time.time()
        game_state = get_session(session_id)
        if not game_state:
            return jsonify({'error': 'Session expired or not found'}), 404
        
        should_guess, guess_animal, guess_type = service.should_make_guess(game_state)
        
        if should_guess:
            top_predictions = service.get_top_predictions(game_state, n=5)
            # Save state changes (like 'middle_guess_made')
            set_session(session_id, game_state)
            
            if guess_type == 'final':
                return jsonify({
                    'should_guess': True,
                    'guess': guess_animal,
                    'guess_type': guess_type,
                    'top_predictions': top_predictions
                })
            
            # Middle guess
            q_num = game_state['question_count'] 
            top_pred = service.get_top_predictions(game_state, n=1)
            return jsonify({
                'should_guess': False,
                'question': f"Is it a/an {guess_animal}?",
                'feature': 'sneaky_guess',
                'animal_name': guess_animal,
                'is_sneaky_guess': True,
                'question_number': q_num,
                'top_prediction': top_pred[0] if top_pred else None
            })
        
        result = service.get_next_question(game_state)
        set_session(session_id, game_state) 
        dur = (time.time() - start_t) * 1000.0
        print(f"[TIMING] get_question {session_id} took {dur:.1f} ms")

        if not result or result[0] is None:
            return jsonify({
                'question': None,
                'feature': None,
                'top_predictions': service.get_top_predictions(game_state, n=3)
            })
        
        feature, question = result
        q_num = game_state['question_count'] + 1  # Add 1 to start from Q1
        top_pred = service.get_top_predictions(game_state, n=1)
        
        return jsonify({
            'question': question,
            'feature': feature,
            'question_number': q_num,
            'top_prediction': top_pred[0] if top_pred else None,
            'should_guess': False,
            'is_sneaky_guess': False
        })

    @app.route('/answer/<session_id>', methods=['POST'])
    def submit_answer(session_id):
        """Process user's answer to a question."""
        data = request.json
        feature = data.get('feature')
        client_answer = data.get('answer', 'skip').lower().strip()
        
        if not feature:
            return jsonify({'error': 'Missing feature'}), 400

        start_t = time.time()
        game_state = get_session(session_id)
        if not game_state:
            return jsonify({'error': 'Session expired or not found'}), 404
        
        # Handle sneaky guess
        if feature == 'sneaky_guess':
            animal_name = data.get('animal_name')
            if client_answer in ['yes', 'y', 'usually']:
                delete_session(session_id)
                dur = (time.time() - start_t) * 1000.0
                print(f"[TIMING] submit_answer (sneaky correct) {session_id} took {dur:.1f} ms")
                return jsonify({
                    'status': 'guess_correct',
                    'animal': animal_name,
                    'top_predictions': service.get_top_predictions(game_state, n=5)
                })

            # Rejected sneaky guess: mark animal as rejected but don't count it as a question
            service.reject_guess(game_state, animal_name)
            # Also add sneaky_guess to asked_features if not present
            if 'sneaky_guess' not in game_state['asked_features']:
                game_state['asked_features'].append('sneaky_guess')

            set_session(session_id, game_state)
            dur = (time.time() - start_t) * 1000.0
            print(f"[TIMING] submit_answer (sneaky rejected) {session_id} took {dur:.1f} ms")
            return jsonify({
                'status': 'ok',
                'top_predictions': service.get_top_predictions(game_state, n=5)
            })
        
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

        if client_answer != 'skip' and brain_answer is not None:
            fuzzy_value = service.engine.fuzzy_map.get(brain_answer)
            if fuzzy_value is not None:
                game_state['answered_features'][feature] = fuzzy_value
                service.process_answer(game_state, feature, brain_answer)
                game_state['question_count'] += 1

        set_session(session_id, game_state)
        dur = (time.time() - start_t) * 1000.0
        print(f"[TIMING] submit_answer {session_id} took {dur:.1f} ms")

        status = 'skipped' if client_answer == 'skip' else 'ok'
        return jsonify({
            'status': status,
            'top_predictions': service.get_top_predictions(game_state, n=5)
        })

    @app.route('/reject/<session_id>', methods=['POST'])
    def reject_animal(session_id):
        """Reject a guessed animal."""
        data = request.json
        animal_name = data.get('animal_name')
        
        if not animal_name:
            return jsonify({'error': 'Missing animal_name'}), 400
        
        game_state = get_session(session_id)
        if not game_state:
            return jsonify({'error': 'Session expired or not found'}), 404
        
        service.reject_guess(game_state, animal_name)
        set_session(session_id, game_state)
        
        return jsonify({
            'status': 'rejected',
            'animal': animal_name,
            'top_predictions': service.get_top_predictions(game_state, n=5)
        })

    @app.route('/learn/<session_id>', methods=['POST'])
    def learn_animal(session_id):
        """Learn a new animal from the session."""
        data = request.json
        animal_name = data.get('animal_name')
        
        if not animal_name:
            return jsonify({'error': 'Missing animal_name'}), 400

        game_state = get_session(session_id)
        if not game_state:
            return jsonify({'error': 'Session expired or not found'}), 404
        
        answered_features = game_state['answered_features']
        
        # This function now saves to DB and triggers a
        # non-blocking background reload of the engine.
        service.learn_animal(animal_name, answered_features)
        
        delete_session(session_id)
        
        return jsonify({
            'message': f"Thank you! I've learned about {animal_name}.",
            'features_learned': len(answered_features)
        })

    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get server statistics."""
        active_sessions = get_active_session_count()
        
        # Thread-safe read of total animals
        with service.engine_lock:
            total_animals = len(service.engine.animals)
        
        return jsonify({
            'active_sessions': active_sessions,
            'total_animals': total_animals
        })

    @app.route('/predictions/<session_id>', methods=['GET'])
    def get_predictions(session_id):
        """Get current top predictions for debugging."""
        game_state = get_session(session_id)
        if not game_state:
            return jsonify({'error': 'Session expired or not found'}), 404
        
        top_predictions = service.get_top_predictions(game_state, n=10)
        return jsonify({'top_predictions': top_predictions})

    # Return the app instance from the factory
    return app

# --- Entry Point ---

# This line creates the app instance, making it discoverable by Gunicorn
app = create_app()

if __name__ == '__main__':
    # This block only runs when you execute `python host.py`
    # It will use Flask's built-in dev server
    app.run(
        host='0.0.0.0', 
        port=config.PORT, 
        debug=False # Set to True for development auto-reload
    )
