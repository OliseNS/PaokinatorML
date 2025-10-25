import os
import redis
import pandas as pd
import numpy as np
import torch
import uuid
import msgpack
import msgpack_numpy as m
from supabase import create_client, Client

# Import centralized configuration
import config

# Configure msgpack-numpy to be the default handler
m.patch()

# --- Initialize Clients ---
try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    print("SUCCESS: Connected to Supabase")
except Exception as e:
    print(f"ERROR: Could not connect to Supabase: {e}")
    supabase = None

try:
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=False  # We will use msgpack, so we need bytes
    )
    redis_client.ping()
    print("SUCCESS: Connected to Redis")
except Exception as e:
    print(f"ERROR: Could not connect to Redis: {e}")
    redis_client = None


# --- Session Management (Optimized) ---

def set_session(session_id: str, state: dict):
    """
    Saves a session state to Redis.
    *** OPTIMIZED: Uses msgpack + raw numpy bytes + compression. ***
    """
    if not redis_client:
        print("Error: Redis client not initialized.")
        return
    try:
        # --- Make a shallow copy to avoid modifying the original state ---
        state_to_save = state.copy()

        # --- Convert Tensors to Numpy Arrays for msgpack ---
        # This is much faster than torch.save or pickle
        if 'probabilities' in state_to_save and isinstance(state_to_save['probabilities'], torch.Tensor):
            state_to_save['probabilities'] = state_to_save['probabilities'].numpy()
        if 'rejected_mask' in state_to_save and isinstance(state_to_save['rejected_mask'], torch.Tensor):
            state_to_save['rejected_mask'] = state_to_save['rejected_mask'].numpy()
        
        # msgpack is much faster than pickle
        packed_state = msgpack.packb(state_to_save, use_bin_type=True)
        
        # Compress for even better performance on large states
        if len(packed_state) > 1024:  # Only compress if larger than 1KB
            import gzip
            packed_state = gzip.compress(packed_state)
            key = f"session_gz:{session_id}"
        else:
            key = f"session:{session_id}"
        
        redis_client.setex(
            key, 
            config.SESSION_TTL_SECONDS, 
            packed_state
        )
    except Exception as e:
        print(f"Error setting session {session_id} in Redis: {e}")

def get_session(session_id: str) -> dict | None:
    """
    Retrieves a session state from Redis.
    *** OPTIMIZED: Uses msgpack and converts numpy back to torch. ***
    """
    if not redis_client:
        print("Error: Redis client not initialized.")
        return None
    try:
        # Try compressed version first, then fallback to uncompressed
        for key_prefix in ["session_gz:", "session:"]:
            key = f"{key_prefix}{session_id}"
            packed_state = redis_client.get(key)
            
            if packed_state:
                # Reset TTL on access to keep active sessions alive
                redis_client.expire(key, config.SESSION_TTL_SECONDS)
                
                # Decompress if needed
                if key_prefix == "session_gz:":
                    import gzip
                    packed_state = gzip.decompress(packed_state)
                
                # Unpack with msgpack
                state = msgpack.unpackb(packed_state, raw=False)

                # --- Convert Numpy Arrays back to Tensors ---
                # We use .copy() to ensure the tensor is writable,
                # which is crucial for our state migration logic.
                if 'probabilities' in state:
                    state['probabilities'] = torch.from_numpy(state['probabilities'].copy())
                if 'rejected_mask' in state:
                    state['rejected_mask'] = torch.from_numpy(state['rejected_mask'].copy())

                return state
        
        return None  # Session expired or not found
    except Exception as e:
        print(f"Error getting session {session_id} from Redis: {e}")
        return None

def delete_session(session_id: str):
    """Deletes a session from Redis."""
    if not redis_client:
        return
    try:
        redis_client.delete(f"session:{session_id}")
    except Exception as e:
        print(f"Error deleting session {session_id} from Redis: {e}")

def get_active_session_count() -> int:
    """Counts active sessions in Redis."""
    if not redis_client:
        return 0
    try:
        # Avoid using 'KEYS' in production on large dbs, 
        # but for a "session:" prefix, it's generally acceptable.
        return len(list(redis_client.scan_iter("session:*")))
    except Exception as e:
        print(f"Error counting sessions in Redis: {e}")
        return 0

# --- Data Persistence (Supabase) ---

def load_data_from_supabase() -> tuple[pd.DataFrame, list]:
    """Loads the entire animal database from Supabase."""
    if not supabase:
        raise Exception("Supabase client not initialized. Cannot load data.")
        
    print("Loading data from Supabase...")
    try:
        data = supabase.table("animals").select("*").execute()
        df = pd.DataFrame(data.data)
        
        if df.empty or 'animal_name' not in df.columns:
            print("WARNING: No data found in Supabase. Using dummy data.")
            df = pd.DataFrame({
                'animal_name': ['Dog', 'Cat'],
                'has_fur': [1.0, 1.0],
                'lives_in_water': [0.0, 0.0]
            })

        # Standardize: Remove Supabase internal cols
        feature_cols = [
            c for c in df.columns if c not in ['id', 'created_at', 'animal_name']
        ]
        
        # Ensure all feature columns are numeric, fill N/A
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Loaded {len(df)} animals and {len(feature_cols)} features.")
        return df, feature_cols

    except Exception as e:
        print(f"CRITICAL: Failed to load data from Supabase: {e}")
        raise

def persist_learned_animal(animal_data: dict) -> str:
    """
    Learns an animal by persisting it to Supabase.
    
    - If the animal is NEW, it inserts it into the main 'public.animals' table.
    - If the animal EXISTS, it inserts the new data as a suggestion
      into the 'public.animalsuggest' table.
      
    Returns:
        str: "inserted", "suggestion_inserted", or "error"
    """
    if not supabase:
        print("Error: Supabase client not initialized. Cannot persist data.")
        return "error"
        
    try:
        name = animal_data['animal_name']
        
        # --- Payload Cleaning ---
        # Convert np.nan to None for JSON/Supabase compatibility
        payload = {}
        for key, value in animal_data.items():
            payload[key] = None if pd.isna(value) else value
        # --- End Cleaning ---
        
        # Check if animal already exists (case-insensitive)
        existing = supabase.table("animals").select("id").ilike("animal_name", name).execute()
        
        if existing.data:
            # --- ANIMAL EXISTS ---
            # Save to 'animalsuggest' table with a new UUID
            payload['id'] = str(uuid.uuid4())
            supabase.table("animalsuggest").insert(payload).execute()
            print(f"Inserted suggestion for '{name}' into public.animalsuggest.")
            return "suggestion_inserted"
            
        else:
            # --- ANIMAL IS NEW ---
            payload['id'] = str(uuid.uuid4())
            
            # Insert into the main 'animals' table
            supabase.table("animals").insert(payload).execute()
            print(f"Inserted NEW animal '{name}' into public.animals.")
            return "inserted"
            
    except Exception as e:
        print(f"Error persisting learned animal {animal_data.get('animal_name')} to Supabase: {e}")
        return "error"