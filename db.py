import redis
import pandas as pd
import numpy as np
import msgpack
import msgpack_numpy as m
from supabase import create_client, Client
import config
from datetime import datetime, timezone
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
def _convert_to_serializable(state: dict) -> dict:
    state_copy = state.copy()
    if 'probabilities' in state_copy: del state_copy['probabilities']
    return state_copy

def _convert_from_serializable(state: dict) -> dict:
    for key in ['cumulative_scores', 'rejected_mask']:
        if key in state and isinstance(state[key], np.ndarray):
            if key == 'cumulative_scores':
                state[key] = state[key].astype(np.float32)
            elif key == 'rejected_mask':
                state[key] = state[key].astype(bool)
    if 'probabilities' in state: del state['probabilities']
    return state

def push_session_state(session_id: str, state: dict):
    if not redis_client: return
    try:
        key = f"session:{session_id}"
        packed = msgpack.packb(_convert_to_serializable(state), use_bin_type=True)
        redis_client.lpush(key, packed)
        redis_client.ltrim(key, 0, 59) 
        redis_client.expire(key, config.SESSION_TTL_SECONDS)
    except Exception as e:
        print(f"✗ Error pushing session {session_id}: {e}")

def get_current_session_state(session_id: str) -> dict | None:
    if not redis_client: return None
    try:
        key = f"session:{session_id}"
        packed = redis_client.lindex(key, 0)
        if not packed: return None
        redis_client.expire(key, config.SESSION_TTL_SECONDS)
        state = msgpack.unpackb(packed, raw=False)
        return _convert_from_serializable(state)
    except Exception as e:
        print(f"✗ Error getting current session {session_id}: {e}")
        return None

def pop_session_state(session_id: str) -> dict | None:
    if not redis_client: return None
    try:
        key = f"session:{session_id}"
        packed = redis_client.lpop(key)
        if not packed: return None
        redis_client.expire(key, config.SESSION_TTL_SECONDS)
        state = msgpack.unpackb(packed, raw=False)
        return _convert_from_serializable(state)
    except Exception as e:
        return None

def get_session_history_length(session_id: str) -> int:
    if not redis_client: return 0
    try:
        return redis_client.llen(f"session:{session_id}")
    except: return 0

def get_active_session_count() -> int:
    if not redis_client: return 0
    try:
        return len(list(redis_client.scan_iter("session:*")))
    except: return 0

def get_all_domains() -> list[str]:
    if not supabase: return ["animals"]
    try:
        result = supabase.table("domains").select("domain_name").execute()
        if not result.data: return ["animals"]
        return [d['domain_name'] for d in result.data]
    except: return ["animals"]


# --- Data Loading (Active + Suggested) ---

def load_data_from_supabase(domain_name: str = "animals") -> tuple[pd.DataFrame, list, dict]:
    if not supabase: raise Exception("Supabase missing")
    
    print(f"Loading domain '{domain_name}' (FULL: Active + Suggested)...")
    
    # 0. Get Domain ID
    domain = supabase.table("domains").select("id").ilike("domain_name", domain_name).limit(1).execute()
    if not domain.data: return pd.DataFrame(), [], {}
    domain_id = domain.data[0]['id']

    # 1. Get Features (ALL statuses)
    feat_res = supabase.table("domain_features").select("features(id, feature_name, question_text)")\
        .eq("domain_id", domain_id).execute()
    
    active_features = [r['features'] for r in feat_res.data if r['features']]
    feature_map = {f['id']: f['feature_name'] for f in active_features}
    questions_map = {f['feature_name']: f['question_text'] for f in active_features}
    feature_cols = list(feature_map.values())
    feature_ids = list(feature_map.keys())
    feature_count = len(feature_ids)
    print(f"✓ Features Loaded: {feature_count}")

    # 2. Get Items (ALL statuses)
    items_res = supabase.table("items").select("id,item_name").eq("domain_id", domain_id)\
        .limit(50000).execute()
    
    item_map = {i['id']: i['item_name'] for i in items_res.data}
    item_ids = list(item_map.keys())
    print(f"✓ Items Loaded: {len(item_ids)}")

    if not item_ids or not feature_ids:
        return pd.DataFrame(columns=['animal_name'] + feature_cols), feature_cols, questions_map

    # 3. Get Feature Values (Batched)
    max_rows_per_query = 9500
    batch_size = max(1, max_rows_per_query // max(1, feature_count))
    all_data = []
    total_batches = (len(item_ids) + batch_size - 1) // batch_size

    for i in range(0, len(item_ids), batch_size):
        batch_ids = item_ids[i:i+batch_size]
        raw_data = supabase.table("item_features").select("item_id,feature_id,value_sum,vote_count")\
            .in_("item_id", batch_ids).in_("feature_id", feature_ids).execute()
            
        for row in raw_data.data:
            if row['vote_count'] > 0:
                all_data.append({
                    'animal_name': item_map.get(row['item_id']),
                    'feature_name': feature_map.get(row['feature_id']),
                    'value': row['value_sum'] / row['vote_count']
                })
    
    print(f"\n✓ Retrieved {len(all_data)} feature values")

    # 4. Pivot to DataFrame (Optimized to avoid fragmentation)
    if not all_data:
        df_wide = pd.DataFrame({'animal_name': list(item_map.values())})
    else:
        df_long = pd.DataFrame(all_data)
        df_wide = df_long.pivot_table(index='animal_name', columns='feature_name', values='value', aggfunc='mean')
        df_wide = df_wide.reindex(list(item_map.values()))
        df_wide = df_wide.reset_index().rename(columns={'index': 'animal_name'})
    
    # Optimization: Efficiently add missing columns using concatenation instead of loop
    missing_cols = [c for c in feature_cols if c not in df_wide.columns]
    if missing_cols:
        # Create a DataFrame of NaNs for the missing columns
        df_missing = pd.DataFrame(np.nan, index=df_wide.index, columns=missing_cols)
        df_wide = pd.concat([df_wide, df_missing], axis=1)
    
    # Ensure numeric types
    # We use apply only on feature columns to be safe
    # But usually concat respects types. pd.to_numeric might be needed if 'value' was mixed.
    # We skip explicit loop for speed unless needed. Assuming float32 from engine cast.

    # Sort columns to match feature_cols list order for the Engine
    cols_to_keep = ['animal_name'] + feature_cols
    df_wide = df_wide[cols_to_keep]

    return df_wide, feature_cols, questions_map

def get_recent_updates(domain_name: str, since: datetime) -> list[dict]:
    if not supabase: return []
    
    ts_str = since.isoformat()
    
    try:
        query = supabase.table("item_features").select(
            "value_sum, vote_count, updated_at, items!inner(item_name, domain_id), features!inner(feature_name, question_text)"
        ).gt("updated_at", ts_str).limit(500)
        
        data = query.execute()
        updates = []
        target_domain_id = _get_domain_id(domain_name)

        for row in data.data:
            # Filter by domain since we can't easily do it in the join query yet
            if row['items']['domain_id'] != target_domain_id:
                continue

            if row['vote_count'] > 0:
                val = row['value_sum'] / row['vote_count']
                updates.append({
                    'item_name': row['items']['item_name'],
                    'feature_name': row['features']['feature_name'],
                    'value': val,
                    'question_text': row['features'].get('question_text')
                })
        return updates
    except Exception as e:
        return []

# --- Helper Functions ---
def _get_domain_id(domain_name):
    if not supabase: return None
    res = supabase.table("domains").select("id").ilike("domain_name", domain_name).limit(1).execute()
    return res.data[0]['id'] if res.data else None

def _get_item_id(domain_id, item_name):
    res = supabase.table("items").select("id").eq("domain_id", domain_id).ilike("item_name", item_name).limit(1).execute()
    return res.data[0]['id'] if res.data else None

def _get_or_create_feature_id(feature_name, question_text):
    res = supabase.table("features").select("id").eq("feature_name", feature_name).limit(1).execute()
    if res.data: return res.data[0]['id']
    if not question_text: question_text = f"Is it {feature_name}?"
    res = supabase.table("features").insert({"feature_name": feature_name, "question_text": question_text, "status": "suggested"}).execute()
    return res.data[0]['id'] if res.data else None

def persist_suggestion(animal_name, answered_features, domain_name="animals"):
    if not supabase: return "error"
    try:
        domain_id = _get_domain_id(domain_name)
        if not domain_id: return "error"
        item_id = _get_item_id(domain_id, animal_name)
        if not item_id: return "error"
        
        f_res = supabase.table("features").select("id,feature_name").execute()
        f_map = {x['feature_name']: x['id'] for x in f_res.data}
        
        current_time = datetime.now(timezone.utc).isoformat()

        for fname, val in answered_features.items():
            if fname in f_map and pd.notna(val):
                fid = f_map[fname]
                try:
                    supabase.rpc('upsert_item_feature_vote', {
                        'p_item_id': item_id, 'p_feature_id': fid,
                        'p_value_sum': float(val), 'p_vote_count': 1
                    }).execute()
                    
                    supabase.table("item_features").update({
                        "updated_at": current_time
                    }).eq("item_id", item_id).eq("feature_id", fid).execute()
                except Exception: pass
        return "ok"
    except Exception as e:
        print(f"Persist error: {e}")
        return "error"

def persist_new_animal(animal_name, answered_features, domain_name="animals"):
    if not supabase: return "error"
    try:
        domain_id = _get_domain_id(domain_name)
        if not domain_id: return "error"
        
        if _get_item_id(domain_id, animal_name):
            return persist_suggestion(animal_name, answered_features, domain_name)
            
        ires = supabase.table("items").insert({"item_name": animal_name, "domain_id": domain_id, "status": "suggested"}).execute()
        item_id = ires.data[0]['id']
        
        f_res = supabase.table("features").select("id,feature_name").execute()
        f_map = {x['feature_name']: x['id'] for x in f_res.data}
        
        rows = []
        current_time = datetime.now(timezone.utc).isoformat()
        
        for fname, val in answered_features.items():
            if fname in f_map and pd.notna(val):
                rows.append({
                    "item_id": item_id,
                    "feature_id": f_map[fname],
                    "value_sum": float(val),
                    "vote_count": 1,
                    "updated_at": current_time 
                })
        if rows: supabase.table("item_features").insert(rows).execute()
        return "ok"
    except Exception as e:
        print(f"New animal error: {e}")
        return "error"

def suggest_new_feature(domain_name, feature_name, question_text, item_name, fuzzy_value):
    if not supabase: return {"status": "error"}
    try:
        domain_id = _get_domain_id(domain_name)
        item_id = _get_item_id(domain_id, item_name)
        if not item_id: return {"status": "error", "message": "Item not found"}
        
        fid = _get_or_create_feature_id(feature_name, question_text)
        try:
            supabase.table("domain_features").insert({"domain_id": domain_id, "feature_id": fid}).execute()
        except: pass
        
        current_time = datetime.now(timezone.utc).isoformat()
        
        supabase.rpc('upsert_item_feature_vote', {
            'p_item_id': item_id, 'p_feature_id': fid,
            'p_value_sum': float(fuzzy_value), 'p_vote_count': 1
        }).execute()
        
        supabase.table("item_features").update({
            "updated_at": current_time
        }).eq("item_id", item_id).eq("feature_id", fid).execute()
        
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}