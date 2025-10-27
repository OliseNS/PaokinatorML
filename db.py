import redis
import pandas as pd
import torch
import msgpack
import msgpack_numpy as m
from supabase import create_client, Client
import config
import traceback

m.patch()
LIMIT = 10000

# --- Initialize Clients ---
try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    print("✓ Connected to Supabase")
except Exception as e:
    print(f"✗ Supabase connection failed: {e}")
    supabase = None

try:
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=False
    )
    redis_client.ping()
    print("✓ Connected to Redis")
except Exception as e:
    print(f"✗ Redis connection failed: {e}")
    redis_client = None


# --- Session Management ---
def _convert_tensors_to_numpy(state: dict) -> dict:
    """Convert torch tensors to numpy arrays for serialization."""
    state_copy = state.copy()
    for key in ['probabilities', 'rejected_mask']:
        if key in state_copy and isinstance(state_copy[key], torch.Tensor):
            state_copy[key] = state_copy[key].numpy()
    return state_copy


def _convert_numpy_to_tensors(state: dict) -> dict:
    """Convert numpy arrays back to torch tensors."""
    for key in ['probabilities', 'rejected_mask']:
        if key in state:
            state[key] = torch.from_numpy(state[key].copy())
    return state


def set_session(session_id: str, state: dict):
    """Save session state to Redis using msgpack."""
    if not redis_client:
        return print("✗ Redis client not initialized")
    
    try:
        packed = msgpack.packb(_convert_tensors_to_numpy(state), use_bin_type=True)
        redis_client.setex(f"session:{session_id}", config.SESSION_TTL_SECONDS, packed)
    except Exception as e:
        print(f"✗ Error setting session {session_id}: {e}")


def get_session(session_id: str) -> dict | None:
    """Retrieve session state from Redis."""
    if not redis_client:
        return None
    
    try:
        key = f"session:{session_id}"
        packed = redis_client.get(key)
        
        if not packed:
            return None
        
        redis_client.expire(key, config.SESSION_TTL_SECONDS)
        state = msgpack.unpackb(packed, raw=False)
        return _convert_numpy_to_tensors(state)
    except Exception as e:
        print(f"✗ Error getting session {session_id}: {e}")
        return None


def delete_session(session_id: str):
    """Delete session from Redis."""
    if redis_client:
        try:
            redis_client.delete(f"session:{session_id}")
        except Exception as e:
            print(f"✗ Error deleting session {session_id}: {e}")


def get_active_session_count() -> int:
    """Count active sessions in Redis."""
    if not redis_client:
        return 0
    try:
        return len(list(redis_client.scan_iter("session:*")))
    except Exception as e:
        print(f"✗ Error counting sessions: {e}")
        return 0

# --- NEW FUNCTION ---
def get_all_domains() -> list[str]:
    """Retrieves all domain names from the domains table."""
    if not supabase:
        print("✗ Supabase client not initialized, returning default domain")
        return ["animals"] # Default fallback
    try:
        result = supabase.table("domains").select("domain_name").execute()
        if not result.data:
            print("⚠ No domains found in DB, returning default 'animals'")
            return ["animals"]
        return [d['domain_name'] for d in result.data]
    except Exception as e:
        print(f"✗ Error fetching domains: {e}")
        return ["animals"] # Default fallback

# --- Data Loading ---
def load_data_from_supabase(domain_name: str = "animals") -> tuple[pd.DataFrame, list, dict]:
    """Load and pivot domain data from Supabase with dynamic batch sizing."""
    if not supabase:
        raise Exception("Supabase client not initialized")
    
    print(f"Loading domain '{domain_name}'...")
    
    try:
        # Get domain ID
        domain = supabase.table("domains").select("id,domain_name").ilike("domain_name", domain_name).limit(1).execute()
        if not domain.data:
            raise Exception(f"Domain '{domain_name}' not found")
        
        domain_id = domain.data[0]['id']
        print(f"✓ Domain ID: {domain_id}")
        
        # --- MODIFIED SECTION: Load only 'active' features ---
        domain_features_response = supabase.table("domain_features") \
            .select("features(id, feature_name, question_text)") \
            .eq("domain_id", domain_id) \
            .eq("features.status", "active") \
            .limit(LIMIT) \
            .execute()

        if not domain_features_response.data:
            # This is NOT a critical error, a domain might have no features yet
            print(f"⚠ No active features found for domain '{domain_name}'. Engine will be empty.")
            # Return empty structure
            return pd.DataFrame(columns=['animal_name']), [], {}

        active_features_data = [row['features'] for row in domain_features_response.data if row.get('features')]

        if not active_features_data:
            print(f"⚠ No active features found (after processing) for domain '{domain_name}'. Engine will be empty.")
            return pd.DataFrame(columns=['animal_name']), [], {}

        feature_ids = [f['id'] for f in active_features_data]
        feature_count = len(feature_ids)
        print(f"✓ Active Features: {feature_count}")

        feature_id_to_name = {f['id']: f['feature_name'] for f in active_features_data if f.get('id') and f.get('feature_name')}
        questions_map = {f['feature_name']: f['question_text'] for f in active_features_data if f.get('feature_name') and f.get('question_text')}
        print(f"✓ Question texts: {len(questions_map)}")
        
        # Get all items
        items = supabase.table("items") \
            .select("id,item_name") \
            .eq("domain_id", domain_id) \
            .limit(50000) \
            .execute()
        
        if not items.data:
            print(f"⚠ No items found for domain '{domain_name}'. Engine will be empty.")
            return pd.DataFrame(columns=['animal_name']), feature_id_to_name.values(), questions_map
        
        item_id_to_name = {i['id']: i['item_name'] for i in items.data}
        item_ids = list(item_id_to_name.keys())
        print(f"✓ Items: {len(item_ids)}")
        
        if feature_count == 0:
             print(f"⚠ No features, returning items list only.")
             df = pd.DataFrame({'animal_name': list(item_id_to_name.values())})
             return df, [], questions_map
             
        max_rows_per_query = 9500
        batch_size = max(1, max_rows_per_query // feature_count)
        rows_per_batch = batch_size * feature_count
        
        print(f"✓ Dynamic batch size: {batch_size} items ({rows_per_batch} rows/query)")
        
        all_feature_data = []
        total_batches = (len(item_ids) + batch_size - 1) // batch_size
        
        try:
            print("   Attempting to read from 'live_item_features' view...")
            for i in range(0, len(item_ids), batch_size):
                batch_ids = item_ids[i:i + batch_size]
                print(f"   Batch {i//batch_size + 1}/{total_batches}...", end='\r')
                
                batch = supabase.table("live_item_features") \
                    .select("item_id,feature_id,fuzzy_value") \
                    .in_("item_id", batch_ids) \
                    .in_("feature_id", feature_ids) \
                    .limit(rows_per_batch) \
                    .execute()
                
                if batch.data:
                    all_feature_data.extend(batch.data)
            
            print(f"\n✓ Retrieved {len(all_feature_data)} feature values (view)")
            
        except Exception as view_error:
            print(f"\n! View error: {view_error}")
            print("   Falling back to 'item_features' table...")
            all_feature_data = []
            
            for i in range(0, len(item_ids), batch_size):
                batch_ids = item_ids[i:i + batch_size]
                print(f"   Batch {i//batch_size + 1}/{total_batches}...", end='\r')
                
                batch = supabase.table("item_features") \
                    .select("item_id,feature_id,value_sum,vote_count") \
                    .in_("item_id", batch_ids) \
                    .in_("feature_id", feature_ids) \
                    .limit(rows_per_batch) \
                    .execute()
                
                for row in batch.data:
                    if row['vote_count'] > 0:
                        all_feature_data.append({
                            'item_id': row['item_id'],
                            'feature_id': row['feature_id'],
                            'fuzzy_value': row['value_sum'] / row['vote_count']
                        })
            
            print(f"\n✓ Retrieved {len(all_feature_data)} feature values (table)")
        
        long_data = []
        items_with_features = set()
        
        for row in all_feature_data:
            item_name = item_id_to_name.get(row['item_id'])
            feature_name = feature_id_to_name.get(row['feature_id'])
            
            if item_name and feature_name:
                long_data.append({
                    'animal_name': item_name,
                    'feature_name': feature_name,
                    'value': row['fuzzy_value']
                })
                items_with_features.add(row['item_id'])
        
        # Add all items, even those without features, to the dataframe
        all_item_names = set(item_id_to_name.values())
        items_in_long_data = set(d['animal_name'] for d in long_data)
        missing_items = all_item_names - items_in_long_data
        
        if missing_items:
            print(f"   Adding {len(missing_items)} items that have no feature data.")
            for item_name in missing_items:
                long_data.append({
                    'animal_name': item_name,
                    'feature_name': None, # This will create NaNs
                    'value': None
                })
        
        if not long_data:
            print("⚠ No data found, returning empty dataframe")
            return pd.DataFrame(columns=['animal_name'] + feature_cols), feature_cols, questions_map
        
        print("   Pivoting data...")
        df_long = pd.DataFrame(long_data)
        
        feature_cols = [f for f in feature_id_to_name.values()]

        if df_long.empty or 'feature_name' not in df_long.columns or df_long['feature_name'].isnull().all():
             print("   No feature data to pivot. Returning items list.")
             df_wide = pd.DataFrame({'animal_name': list(all_item_names)})
             # Add empty feature columns
             for col in feature_cols:
                 df_wide[col] = pd.NA
             return df_wide, feature_cols, questions_map

        dup_check = df_long.groupby(['animal_name', 'feature_name']).size()
        has_dups = (dup_check > 1).any()
        
        if has_dups:
            print(f"   Averaging {(dup_check > 1).sum()} duplicates")
            df_wide = df_long.pivot_table(
                index='animal_name',
                columns='feature_name',
                values='value',
                aggfunc='mean'
            )
        else:
            df_wide = df_long.pivot(
                index='animal_name',
                columns='feature_name',
                values='value'
            )
        
        # Ensure all items are in the index
        df_wide = df_wide.reindex(all_item_names)
        df_wide = df_wide.reset_index().rename(columns={'index': 'animal_name'})
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df_wide.columns:
                df_wide[col] = pd.NA
        
        df_wide.columns.name = None
        
        for col in feature_cols:
            df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
        
        print(f"✓ SUCCESS: {len(df_wide)} items × {len(feature_cols)} features")
        
        return df_wide, feature_cols, questions_map
    
    except Exception as e:
        print(f"✗ CRITICAL ERROR: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise


# --- Data Persistence ---
def _get_domain_id(domain_name: str) -> str | None:
    """Helper to get domain ID."""
    if not supabase: return None
    result = supabase.table("domains").select("id").ilike("domain_name", domain_name).limit(1).execute()
    return result.data[0]['id'] if result.data else None


def _get_item_id(domain_id: str, item_name: str) -> str | None:
    """Helper to get item ID."""
    if not supabase: return None
    result = supabase.table("items").select("id").eq("domain_id", domain_id).ilike("item_name", item_name).limit(1).execute()
    return result.data[0]['id'] if result.data else None

def _get_or_create_feature_id(feature_name: str, question_text: str = None) -> str | None:
    """Helper to get or create a feature ID."""
    if not supabase: return None
    
    # Try to find existing
    result = supabase.table("features").select("id").eq("feature_name", feature_name).limit(1).execute()
    if result.data:
        return result.data[0]['id']
    
    # Not found, create it
    if not question_text:
        question_text = f"Does it have/is it {feature_name.replace('_', ' ')}?"
        
    insert_data = {
        "feature_name": feature_name,
        "question_text": question_text,
        "status": "suggested"
    }
    result = supabase.table("features").insert(insert_data).execute()
    return result.data[0]['id'] if result.data else None

def _link_feature_to_domain(domain_id: str, feature_id: str):
    """Helper to link a feature to a domain if not already linked."""
    if not supabase: return
    try:
        supabase.table("domain_features").insert({
            "domain_id": domain_id,
            "feature_id": feature_id
        }).execute()
    except Exception as e:
        # Ignore duplicate key errors, which are expected
        if "duplicate key value" not in str(e):
            print(f"Error linking feature to domain: {e}")

def _get_feature_map() -> dict:
    """Helper to get feature name to ID mapping."""
    if not supabase: return {}
    features = supabase.table("features").select("id,feature_name").execute()
    return {f['feature_name']: f['id'] for f in features.data}


def persist_new_animal(animal_name: str, answered_features: dict, domain_name: str = "animals") -> str:
    """Insert new animal with its features."""
    if not supabase:
        return "error"
    
    try:
        domain_id = _get_domain_id(domain_name)
        if not domain_id:
            return "error_domain_not_found"
        
        if _get_item_id(domain_id, animal_name):
            print(f"'{animal_name}' exists, saving as suggestion")
            return persist_suggestion(animal_name, answered_features, domain_name)
        
        item_res = supabase.table("items").insert({
            "item_name": animal_name,
            "domain_id": domain_id
        }).execute()
        item_id = item_res.data[0]['id']
        
        feature_map = _get_feature_map()
        rows = []
        for fname, val in answered_features.items():
            if fname not in feature_map:
                print(f"Warning: feature '{fname}' not in map, skipping...")
                continue
            if pd.notna(val):
                rows.append({
                    "item_id": item_id,
                    "feature_id": feature_map[fname],
                    "value_sum": float(val),
                    "vote_count": 1
                })
        
        if rows:
            supabase.table("item_features").insert(rows).execute()
        
        print(f"✓ Inserted '{animal_name}' with {len(rows)} features")
        return "inserted"
    
    except Exception as e:
        print(f"✗ Error persisting '{animal_name}': {e}")
        return "error"


def persist_suggestion(animal_name: str, answered_features: dict, domain_name: str = "animals") -> str:
    """Update existing animal features with new votes."""
    if not supabase:
        return "error"
    
    try:
        domain_id = _get_domain_id(domain_name)
        if not domain_id:
            return "error_domain_not_found"
        
        item_id = _get_item_id(domain_id, animal_name)
        if not item_id:
            return "error_item_not_found"
        
        feature_map = _get_feature_map()
        votes_recorded = 0
        
        for fname, val in answered_features.items():
            if fname not in feature_map or pd.notna(val) is False:
                continue
            
            feature_id = feature_map[fname]
            
            try:
                # Use upsert to handle both insert and update in one go
                supabase.rpc('upsert_item_feature_vote', {
                    'p_item_id': item_id,
                    'p_feature_id': feature_id,
                    'p_value_sum': float(val),
                    'p_vote_count': 1
                }).execute()
                
                votes_recorded += 1
            except Exception as e:
                # This fallback is in case the 'upsert_item_feature_vote' RPC doesn't exist
                # For this to work, you should create a SQL function:
                # CREATE OR REPLACE FUNCTION upsert_item_feature_vote(p_item_id UUID, p_feature_id UUID, p_value_sum FLOAT, p_vote_count INT)
                # RETURNS void AS $$
                # INSERT INTO item_features (item_id, feature_id, value_sum, vote_count)
                # VALUES (p_item_id, p_feature_id, p_value_sum, p_vote_count)
                # ON CONFLICT (item_id, feature_id)
                # DO UPDATE SET
                #   value_sum = item_features.value_sum + p_value_sum,
                #   vote_count = item_features.vote_count + p_vote_count;
                # $$ LANGUAGE sql;
                print(f"✗ RPC failed, falling back to read-then-write for '{fname}': {e}")
                # Fallback logic
                existing = supabase.table("item_features").select("value_sum,vote_count").eq("item_id", item_id).eq("feature_id", feature_id).execute()
                if existing.data:
                    new_sum = existing.data[0]['value_sum'] + float(val)
                    new_count = existing.data[0]['vote_count'] + 1
                    supabase.table("item_features").update({"value_sum": new_sum, "vote_count": new_count}).eq("item_id", item_id).eq("feature_id", feature_id).execute()
                else:
                    supabase.table("item_features").insert({"item_id": item_id, "feature_id": feature_id, "value_sum": float(val), "vote_count": 1}).execute()
                votes_recorded += 1
        
        print(f"✓ Recorded {votes_recorded} votes for '{animal_name}'")
        return "suggestion_inserted"
    
    except Exception as e:
        print(f"✗ Error persisting suggestion for '{animal_name}': {e}")
        return "error"

# --- NEW FUNCTION ---
def suggest_new_feature(domain_name: str, feature_name: str, question_text: str, item_name: str, fuzzy_value: float) -> dict:
    """
    Suggests a new feature, links it to a domain, and adds
    the first vote for a given item.
    """
    if not supabase:
        return {"status": "error", "message": "Database not connected"}
    
    try:
        # 1. Get Domain ID
        domain_id = _get_domain_id(domain_name)
        if not domain_id:
            return {"status": "error", "message": f"Domain '{domain_name}' not found"}
        
        # 2. Get Item ID
        item_id = _get_item_id(domain_id, item_name)
        if not item_id:
            return {"status": "error", "message": f"Item '{item_name}' not found in domain '{domain_name}'"}
            
        # 3. Get or Create Feature ID
        feature_id = _get_or_create_feature_id(feature_name, question_text)
        if not feature_id:
            return {"status": "error", "message": "Could not create or find feature"}
        
        # 4. Link Feature to Domain (idempotent)
        _link_feature_to_domain(domain_id, feature_id)
        
        # 5. Add the vote
        try:
            supabase.rpc('upsert_item_feature_vote', {
                'p_item_id': item_id,
                'p_feature_id': feature_id,
                'p_value_sum': float(fuzzy_value),
                'p_vote_count': 1
            }).execute()
        except Exception:
            # Fallback if RPC doesn't exist (copy-paste from persist_suggestion)
            existing = supabase.table("item_features").select("value_sum,vote_count").eq("item_id", item_id).eq("feature_id", feature_id).execute()
            if existing.data:
                new_sum = existing.data[0]['value_sum'] + float(fuzzy_value)
                new_count = existing.data[0]['vote_count'] + 1
                supabase.table("item_features").update({"value_sum": new_sum, "vote_count": new_count}).eq("item_id", item_id).eq("feature_id", feature_id).execute()
            else:
                supabase.table("item_features").insert({"item_id": item_id, "feature_id": feature_id, "value_sum": float(fuzzy_value), "vote_count": 1}).execute()

        return {
            "status": "ok",
            "message": f"Suggestion recorded for new feature '{feature_name}' on item '{item_name}'."
        }
        
    except Exception as e:
        print(f"✗ Error in suggest_new_feature: {e}")
        return {"status": "error", "message": str(e)}