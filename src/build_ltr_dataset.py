
import pandas as pd
import numpy as np
import random
import os
import joblib
from feature_engineering import FeatureEngineer
from data_processor import DataProcessor
from features_ltr import LTRFeatureEngineer
import sys

# Add root
sys.path.append(os.getcwd())

def build_dataset():
    print("="*60)
    print("🏗️  Building Learning-to-Rank Dataset")
    print("="*60)
    
    # 1. Load Data
    try:
        from config import DATASET_PATH
    except:
        DATASET_PATH = 'data/hr_star_trek_v4c_modernized_clean_modified_v4.csv'
        
    dp = DataProcessor()
    df_raw = pd.read_csv(DATASET_PATH)
    df_transitions = dp.create_transition_dataset(df_raw)
    
    print(f"Loaded {len(df_transitions)} transitions.")
    
    # 2. Enrich Officer Data (Standard Features)
    print("Enriching Officer Features...")
    fe = FeatureEngineer()
    df_transitions = fe.extract_features(df_transitions)
    
    # Format History String
    def format_history(snapshot_hist):
        if not snapshot_hist: return ""
        return " > ".join([h.get('title', '') for h in snapshot_hist])
    
    df_transitions['history_str'] = df_transitions['snapshot_history'].apply(format_history)
    
    # 3. Build Role Metadata (The "Candidate Store")
    print("Building Role Metadata Store...")
    role_meta = {}
    
    # Also Gather Transition Stats
    pool_transitions = {} # {Src_Pool: {Tgt_Pool: Count}}
    title_transitions = {} # {Src_Title: {Tgt_Title: Count}}
    
    # We use df_transitions to count transitions accurately (Source -> Target)
    for _, row in df_transitions.iterrows():
        # Pool Trans
        src_p = row.get('Pool', 'Unknown')
        # Target Pool is tricky, df_transitions usually has 'Target_Next_Role' but maybe not Target Pool directly?
        # We might need to look up Target Role's pool from df_raw or role_meta later.
        # Actually, let's build role_meta first to map Role -> Pool.
        pass

    # Build Role Meta First
    for _, row in df_raw.iterrows():
        r = str(row['current_appointment']).strip()
        if r not in role_meta:
            role_meta[r] = {'Branch': row['Branch'], 'Pool': row['Pool'], 'REQ_Ranks': set(), 'freq': 0}
        
        role_meta[r]['REQ_Ranks'].add(row['Rank'])
        role_meta[r]['freq'] += 1
        
    for r in role_meta:
        role_meta[r]['REQ_Ranks'] = list(role_meta[r]['REQ_Ranks'])
        role_meta[r]['Role'] = r
        
    all_roles = list(role_meta.keys())
    print(f"Found {len(all_roles)} unique roles.")
    
    # Now Compute Transitions using known pools
    print("Computing Transition Priors...")
    for _, row in df_transitions.iterrows():
        # Source
        src_p = row.get('Pool', 'Unknown')
        src_t = row.get('last_role_title', str(row.get('current_appointment'))) # Logic varies if using transitions vs raw
        
        # Target
        tgt_t = row['Target_Next_Role']
        if tgt_t not in role_meta: continue
        tgt_p = role_meta[tgt_t]['Pool']
        
        # Count Pool
        if src_p not in pool_transitions: pool_transitions[src_p] = {}
        pool_transitions[src_p][tgt_p] = pool_transitions[src_p].get(tgt_p, 0) + 1
        
        # Count Title
        if src_t not in title_transitions: title_transitions[src_t] = {}
        title_transitions[src_t][tgt_t] = title_transitions[src_t].get(tgt_t, 0) + 1
        
    # Convert Counts to Probs
    transition_stats = {'pool_trans': {}, 'title_trans': {}}
    
    for src, targets in pool_transitions.items():
        total = sum(targets.values())
        transition_stats['pool_trans'][src] = {t: c/total for t, c in targets.items()}
        
    for src, targets in title_transitions.items():
        total = sum(targets.values())
        transition_stats['title_trans'][src] = {t: c/total for t, c in targets.items()}
        
    print("Transition Stats Computed.")
    
    # 4. Initialize LTR Feature Engineer
    ltr_fe = LTRFeatureEngineer()
    print("Fitting Vectorizers...")
    ltr_fe.fit(pd.Series(all_roles))
    
    # 5. Generate Pairs
    print("Generating Positive & Negative Pairs...")
    
    X_rows = []
    y = []
    groups = []
    
    NEG_RATIO = 5
    
    for idx, row in df_transitions.iterrows():
        if idx % 500 == 0: print(f"Processed {idx} transitions...")
        
        officer_state = row.to_dict()
        target_role = row['Target_Next_Role']
        
        if target_role not in role_meta: continue
            
        # Feature Vector: Positive
        pos_cand = role_meta[target_role]
        pos_feats = ltr_fe.generate_pair_features(officer_state, pos_cand, transition_stats)
        
        X_rows.append(pos_feats)
        y.append(1)
        groups.append(idx)
        
        # NEGATIVES
        neg_candidates = []
        current_rank = row['Rank']
        
        attempts = 0
        while len(neg_candidates) < NEG_RATIO and attempts < 20:
            rand_role = random.choice(all_roles)
            if rand_role == target_role: continue
            
            cand_meta = role_meta[rand_role]
            is_hard = (current_rank in cand_meta['REQ_Ranks']) or (cand_meta['Branch'] == row['Branch'])
            
            if is_hard or attempts > 10:
                neg_candidates.append(cand_meta)
            
            attempts += 1
            
        for neg_cand in neg_candidates:
            neg_feats = ltr_fe.generate_pair_features(officer_state, neg_cand, transition_stats)
            X_rows.append(neg_feats)
            y.append(0)
            groups.append(idx)
            
    # 6. Create DataFrame
    print("Compiling Dataset...")
    df_ltr = pd.DataFrame(X_rows)
    df_ltr['target'] = y
    
    print(f"Final Dataset: {len(df_ltr)} rows.")
    
    # Save
    out_dir = 'data/ltr'
    os.makedirs(out_dir, exist_ok=True)
    
    df_ltr.to_csv(os.path.join(out_dir, 'train_pairs.csv'), index=False)
    
    # Save Artifacts
    joblib.dump(role_meta, os.path.join(out_dir, 'role_meta.pkl'))
    joblib.dump(transition_stats, os.path.join(out_dir, 'transition_stats.pkl'))
    ltr_fe.save(os.path.join(out_dir, 'ltr_fe.pkl'))
    
    print("✓ Dataset & Artifacts Saved.")

if __name__ == "__main__":
    build_dataset()
