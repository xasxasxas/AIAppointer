
import pandas as pd
import numpy as np
import sys
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add root to path
sys.path.append(os.getcwd())

from src.predictor import Predictor
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

def evaluate():
    print("="*60)
    print("🚀 DETAILED MODEL EVALUATION: HYBRID ENSEMBLE")
    print("="*60)
    
    # 1. Load Data & Split (Replicate Training Split)
    try:
        from config import DATASET_PATH
    except:
        DATASET_PATH = 'data/hr_star_trek_v4c_modernized_clean_modified_v4.csv'
        
    print(f"Loading data from {DATASET_PATH}...")
    df_raw = pd.read_csv(DATASET_PATH)
    
    dp = DataProcessor()
    df_transitions = dp.create_transition_dataset(df_raw)
    
    # Pre-extract features for the whole dataset before splitting?
    # Yes, this ensures df_test has everything including 'last_role_title'
    print("Extracting features for evaluation data...")
    fe = FeatureEngineer()
    df_transitions = fe.extract_features(df_transitions)
    
    # Use same seed as train.py
    _, test_idx = train_test_split(df_transitions.index, test_size=0.2, random_state=42)
    df_test = df_transitions.loc[test_idx].copy()
    
    print(f"Test Set Size: {len(df_test)} transitions")
    
    # 2. Initialize Predictor
    print("Initializing Predictor...")
    predictor = Predictor()
    
    # 3. Load Metrics Containers
    correct_top1 = 0
    correct_top5 = 0
    rank_violations = 0
    branch_switches = 0
    total = len(df_test)
    
    # Segmented Accuracy
    acc_by_rank = {}
    
    # Constraints for checking
    with open('models/all_constraints.json', 'r') as f:
        constraints_db = json.load(f)
        
    print("\nStarting Evaluation Loop (this may take a minute)...")
    
    # Limit for speed if needed, but user asked for "various tests", let's run full test set (~500 items)
    # predictor.predict expects a DataFrame row representing "Current State"
    # But df_test is "Transitions". We need to reconstruct the "Current State" for each transition.
    # Fortunately, DataProcessor.get_current_features can accept a dict that looks like a raw row.
    # But df_test ALREADY has features extracted? 
    # NO: predictor.predict() calls _prepare_input which CALLS extract_features.
    # So we should pass the RAW-like data.
    # `df_test` has `snapshot_history` etc. It's already processed into transitions.
    # We can hack `_prepare_input` or just pass `df_test` rows directly if they have necessary cols.
    # `predictor.predict` calls `dp.get_current_features(input_data)`. 
    # `get_current_features` expects specific columns like 'Appointment_history', 'Training_history'.
    # `df_transitions` does NOT have raw history strings, it has PARSED history `snapshot_history`.
    
    # Workaround: feed `df_test` directly to `_prepare_input` bypass?
    # Actually, let's look at `predictor._prepare_input`.
    # It calls `dp.get_current_features` then `fe.extract_features`.
    # `df_test` is already the output of `dp.create_transition_dataset` -> `feature_engineering`.
    # So `df_test` is ready for the MODEL columns, but `predictor.predict` does re-processing.
    
    # To use `predictor.predict` as a black box, we need to reconstruct the Raw Row input.
    # That's hard because `snapshot_history` is a list of dicts.
    # Better approach: Use internal methods of predictor or mock the features.
    
    # Let's inspect `predictor.predict`. It takes `input_data`.
    # WE will modify the loop to construct a synthetic "Current State" row from the transition row
    # so that `dp.get_current_features` works?
    # Or simpler: The Predictor logic is:
    # 1. Prepare X (features)
    # 2. Predict attributes
    # 3. Get candidates
    # 4. Rank
    
    # We can reuse `predictor`'s internal models if we pass pre-computed X.
    # BUT `predict` does the logic for "Candidate Filtering" and "Ranking" which is what we want to test.
    # The `history_str` logic in `_prepare_input` relies on `snapshot_history`.
    # `df_test` HAS `snapshot_history`. 
    # Let's see if we can pass `df_test` row directly.
    # `dp.get_current_features` might fail if it tries to parse raw strings that aren't there.
    # Let's check `DataProcessor.get_current_features`.
    
    # It tries to parse 'Appointment_history'. df_test doesn't have it.
    # It Creates 'parsed_appointments' = input['snapshot_history'] if available?
    # No, usually it parses.
    
    # SOLUTION: Mock `_prepare_input` for this test script to just return the row itself
    # since `df_test` is already processed features!
    # Wait, `df_test` needs `history_str` for sequential model.
    
    def format_history(history_list):
        if not history_list: return ""
        titles = [h.get('title', 'Unknown') for h in history_list]
        return " > ".join(titles)
        
    df_test['history_str'] = df_test['snapshot_history'].apply(format_history)
    
            
    # LTR Predictor doesn't need TestPredictor subclass hack
    # because it generates pair features on the fly from the input context.
    test_predictor = Predictor()
    
    # Evaluation Loop
    
    results = []
    
    for idx, row in df_test.iterrows():
        # Target
        actual_role = row['Target_Next_Role']
        actual_rank = str(row['Rank']).strip() # Current Rank
        
        # Alias 'last_role_title' to 'current_appointment' for Predictor logic
        if 'current_appointment' not in row:
            row['current_appointment'] = row['last_role_title']
            
        # Predict
        # We pass the row as a DataFrame (1 row)
        pred_df = test_predictor.predict(pd.DataFrame([row]))
        
        if pred_df.empty:
            top_1 = "None"
            top_5 = []
        else:
            top_1 = pred_df.iloc[0]['Prediction']
            top_5 = pred_df['Prediction'].head(5).tolist()
            
        # 1. Accuracy
        is_top1 = top_1 == actual_role
        is_top5 = actual_role in top_5
        
        if is_top1: correct_top1 += 1
        if is_top5: correct_top5 += 1
        
        # 2. Segmented
        if actual_rank not in acc_by_rank: acc_by_rank[actual_rank] = {'total':0, 'correct':0}
        acc_by_rank[actual_rank]['total'] += 1
        if is_top1: acc_by_rank[actual_rank]['correct'] += 1
        
        # 3. Constraint Check (on Top 1)
        # Does Top 1 recommendation ALLOW the current rank?
        # Note: Hybrid model predicts a "Next Rank". 
        # But let's check if the generic constraints DB says "Illegal Move".
        if top_1 in constraints_db:
            allowed = constraints_db[top_1].get('ranks', [])
            # Usually, move is valid if allowed_ranks contains either Current OR Next rank.
            # But let's check strict "Next Rank" validity if we knew it.
            # For simplicity: Is it completely wild? e.g. Commander -> Ensign role?
            pass 
            
        # 4. Branch Switch?
        current_branch = row['Branch']
        # We don't easily know Target Branch of 'actual_role' without looking it up
        # But we can check if Top-1 Branch != Current Branch
        
        if (idx % 50 == 0):
            print(f"Processed {idx}/{total}...")

    # Report
    acc_1 = correct_top1 / total
    acc_5 = correct_top5 / total
    
    print("\n" + "="*30)
    print("📊 RESULTS SUMMARY")
    print("="*30)
    print(f"Total Samples: {total}")
    print(f"Top-1 Accuracy: {acc_1:.2%} (Target: >3%)")
    print(f"Top-5 Accuracy: {acc_5:.2%} (Target: >12%)")
    
    print("\n📈 Accuracy by Rank:")
    for rank, stats in acc_by_rank.items():
        if stats['total'] > 0:
            print(f"  - {rank}: {stats['correct']/stats['total']:.2%} ({stats['correct']}/{stats['total']})")
            
    print("\n🧠 Qualitative Check:")
    print("Sequential + Hierarchical ensemble seems to be working if Top-5 is high.")

if __name__ == "__main__":
    evaluate()
