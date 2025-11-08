import os
import pandas as pd
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

with open("questions.json", "r") as f:
    question_mapping = json.load(f)

def main():
    csv_file = "countries.csv"
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} countries from CSV")

    # --- Step 1: Get or Create Domain ---
    print("\n1. Checking domain 'Places(Countries)'...")
    # Try to find it first
    existing_domain = supabase.table("domains").select("id").eq("domain_name", "Places(Countries)").execute()
    
    if existing_domain.data:
        domain_id = existing_domain.data[0]["id"]
        print(f"Found existing domain with ID: {domain_id}")
    else:
        domain_response = supabase.table("domains").insert({
            "domain_name": "Places(Countries)",
            "description": "Countries and their geographical, political, and cultural features"
        }).execute()
        domain_id = domain_response.data[0]["id"]
        print(f"Created new domain with ID: {domain_id}")

    # --- Step 2: Smart Feature Sync (Skip existing) ---
    print("\n2. Syncing features...")
    feature_columns = [col for col in df.columns if col != "country_name"]
    
    # A. Fetch ALL existing features first to build our initial map
    # We select only needed columns to save bandwidth
    print("Fetching existing features...")
    all_features_resp = supabase.table("features").select("id, feature_name").execute()
    feature_map = {item["feature_name"]: item["id"] for item in all_features_resp.data}
    
    # B. Identify which features are missing
    features_to_create = []
    for f_name in feature_columns:
        if f_name not in feature_map:
             features_to_create.append({
                "feature_name": f_name,
                "question_text": question_mapping.get(f_name, f"Does the country have {f_name}?"),
                "status": "active"
            })
            
    # C. Bulk insert ONLY missing features
    if features_to_create:
        print(f"Creating {len(features_to_create)} new features...")
        new_features_resp = supabase.table("features").insert(features_to_create).execute()
        # Update our map with the newly created IDs
        for item in new_features_resp.data:
            feature_map[item["feature_name"]] = item["id"]
    else:
        print("All features already exist. Skipping creation.")

    # --- Step 3: Sync Domain Links (Upsert preferred here) ---
    print("\n3. Syncing domain links...")
    # We use upsert here with ignore_duplicates=True because we don't need the return IDs,
    # we just need them to exist. Requires unique constraint on (domain_id, feature_id).
    domain_links = [{"domain_id": domain_id, "feature_id": feature_map[f_name]} for f_name in feature_columns]
    
    # Upsert in chunks just in case
    for i in range(0, len(domain_links), 1000):
         supabase.table("domain_features").upsert(
             domain_links[i:i+1000], 
             on_conflict="domain_id, feature_id", 
             ignore_duplicates=True
         ).execute()

    # --- Step 4: Smart Item Sync (Skip existing) ---
    print("\n4. Syncing country items...")
    
    # A. Fetch existing items for this domain
    print("Fetching existing items...")
    existing_items_resp = supabase.table("items").select("id, item_name").eq("domain_id", domain_id).execute()
    item_map = {item["item_name"]: item["id"] for item in existing_items_resp.data}
    
    # B. Identify missing items
    items_to_create = []
    for name in df["country_name"]:
        if name not in item_map:
            items_to_create.append({
                "domain_id": domain_id, 
                "item_name": name, 
                "status": "active"
            })
            
    # C. Bulk insert missing items
    if items_to_create:
        print(f"Creating {len(items_to_create)} new items...")
        new_items_resp = supabase.table("items").insert(items_to_create).execute()
        for item in new_items_resp.data:
            item_map[item["item_name"]] = item["id"]
    else:
        print("All items already exist. Skipping creation.")

# --- Step 5: Upsert Item Features ---
    print("\n5. Syncing feature values (Upserting)...")
    
    records_to_upsert = []
    for _, row in df.iterrows():
        country_name = row["country_name"]
        if country_name not in item_map: continue
        
        item_id = item_map[country_name]
        
        for feature_name in feature_columns:
            if feature_name not in feature_map: continue

            val = row[feature_name]
            # FIX: Check if the value is NaN (missing data) and skip if true
            if pd.isna(val):
                continue

            records_to_upsert.append({
                "item_id": item_id,
                "feature_id": feature_map[feature_name],
                "value_sum": float(val),
                "vote_count": 1
            })
    total = len(records_to_upsert)
    print(f"Prepared {total} records. performing bulk upsert (skipping duplicates)...")

    BATCH_SIZE = 5000
    for i in range(0, total, BATCH_SIZE):
        batch = records_to_upsert[i:i + BATCH_SIZE]
        # ignore_duplicates=True effectively means "INSERT IF NOT EXISTS"
        supabase.table("item_features").upsert(
            batch, 
            on_conflict="item_id, feature_id", # Crucial: specify your unique constraint columns here
            ignore_duplicates=True 
        ).execute()
        print(f"Processed {min(i + BATCH_SIZE, total)} / {total}")

    print("\n✅ Sync complete!")

if __name__ == "__main__":
    main()