
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
import warnings
import joblib
import os
import shutil

warnings.filterwarnings('ignore')

def train(csv_path):
    print("="*60)
    print("Step 1: Data Preparation & Role Mapping")
    print("="*60)
    
    # 1. Load Raw Data for Mapping
    df_raw = pd.read_csv(csv_path)
    
    # Build Role Maps from Current State (closest to ground truth for Role attributes)
    role_map = {}
    for _, row in df_raw.iterrows():
        role = str(row['current_appointment']).strip()
        role_map.setdefault(role, {'Branch': [], 'Pool': [], 'Rank': []})
        role_map[role]['Branch'].append(str(row['Branch']).strip())
        role_map[role]['Pool'].append(str(row['Pool']).strip())
        role_map[role]['Rank'].append(str(row['Rank']).strip())
        
    # Consolidate Map (Majority Vote)
    final_role_map = {}
    for role, contribs in role_map.items():
        from collections import Counter
        final_role_map[role] = {
            'Branch': Counter(contribs['Branch']).most_common(1)[0][0],
            'Pool': Counter(contribs['Pool']).most_common(1)[0][0],
            'Rank': Counter(contribs['Rank']).most_common(1)[0][0]
        }
    
    print(f"Mapped {len(final_role_map)} unique current roles to attributes.")
    
    # 2. Generate Transitions
    dp = DataProcessor()
    df_transitions = dp.create_transition_dataset(df_raw)
    
    # Identify Missing Targets and Add Placeholders
    known_roles = set(final_role_map.keys())
    all_targets = set(df_transitions['Target_Next_Role'].unique())
    missing = all_targets - known_roles
    print(f"Adding placeholders for {len(missing)} historical roles not in current snapshot.")
    
    for m in missing:
        final_role_map[m] = {'Branch': 'Unknown', 'Pool': 'Unknown', 'Rank': 'Unknown'}

    fe = FeatureEngineer()
    df_transitions = fe.extract_features(df_transitions)
    
    # 3. Label Targets
    def get_attr(role, attr):
        return final_role_map.get(str(role).strip(), {}).get(attr, 'Unknown')

    df_transitions['Target_Branch'] = df_transitions['Target_Next_Role'].apply(lambda x: get_attr(x, 'Branch'))
    df_transitions['Target_Pool'] = df_transitions['Target_Next_Role'].apply(lambda x: get_attr(x, 'Pool'))
    df_transitions['Target_Rank'] = df_transitions['Target_Next_Role'].apply(lambda x: get_attr(x, 'Rank'))
    
    print(f"Training Data Shape: {df_transitions.shape}")
    
    # 4. Features & Split
    cat_features = ['Rank', 'Branch', 'Pool', 'Entry_type', 'last_role_title', 'penultimate_role_title']
    num_features = ['years_service', 'days_in_last_role', 'years_in_current_rank', 'num_prior_roles', 
                    'num_training_courses', 'days_since_last_training',
                    'count_command_training', 'count_tactical_training', 'count_science_training',
                    'count_engineering_training', 'count_medical_training']
                    
    X = df_transitions[cat_features + num_features].copy()
    y_branch = df_transitions['Target_Branch']
    y_pool = df_transitions['Target_Pool']
    y_rank = df_transitions['Target_Rank']
    y_role = df_transitions['Target_Next_Role']
    
    # Encode X
    encoders = {}
    x_encoded = X.copy()
    for col in cat_features:
        le = LabelEncoder()
        x_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[f'x_{col}'] = le
        
    cat_indices = [x_encoded.columns.get_loc(c) for c in cat_features]
    
    train_idx, test_idx = train_test_split(x_encoded.index, test_size=0.2, random_state=42)
    X_train, X_test = x_encoded.loc[train_idx], x_encoded.loc[test_idx]
    
    # Train Helpers
    def train_sub_model(name, y_series):
        print(f"\nTraining {name} Model...")
        le = LabelEncoder()
        y_enc = le.fit_transform(y_series.astype(str))
        clf = lgb.LGBMClassifier(objective='multiclass', n_estimators=200, learning_rate=0.05, n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_enc[train_idx], categorical_feature=cat_indices)
        acc = accuracy_score(y_enc[test_idx], clf.predict(X_test))
        print(f"{name} Accuracy: {acc:.4f}")
        return clf, le

    model_branch, le_branch = train_sub_model('Branch', y_branch)
    model_pool, le_pool = train_sub_model('Pool', y_pool)
    model_rank, le_rank = train_sub_model('Rank', y_rank)
    
    # Calculate Role Frequencies (for Ranker)
    train_role_counts = y_role.loc[train_idx].value_counts().to_dict()
    
    # Save Artifacts
    output_dir = 'models/hierarchical'
    if os.path.exists(output_dir):
        try: shutil.rmtree(output_dir) # Clean start
        except: pass
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving artifacts to {output_dir}...")
    joblib.dump(model_branch, os.path.join(output_dir, 'model_branch.pkl'))
    joblib.dump(le_branch, os.path.join(output_dir, 'le_branch.pkl'))
    joblib.dump(model_pool, os.path.join(output_dir, 'model_pool.pkl'))
    joblib.dump(le_pool, os.path.join(output_dir, 'le_pool.pkl'))
    joblib.dump(model_rank, os.path.join(output_dir, 'model_rank.pkl'))
    joblib.dump(le_rank, os.path.join(output_dir, 'le_rank.pkl'))
    joblib.dump(encoders, os.path.join(output_dir, 'feature_encoders.pkl'))
    joblib.dump(final_role_map, os.path.join(output_dir, 'role_map.pkl'))
    joblib.dump(train_role_counts, os.path.join(output_dir, 'role_counts.pkl'))
    
    print("✓ Hierarchical Artifacts Saved.")
    
    # ---------------------------------------------------------
    # Train Sequential Model (KNN)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Step 5: Train Sequential Model (KNN)")
    print("="*60)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    
    print("Constructing History Strings...")
    def format_history(history_list):
        if not history_list: return ""
        titles = [h.get('title', 'Unknown') for h in history_list]
        return " > ".join(titles)

    df_transitions['history_str'] = df_transitions['snapshot_history'].apply(format_history)
    X_text = df_transitions['history_str']
    
    # Split using same indices as hierarchical for fairness, though KNN is training on X_train part
    X_train_txt = X_text.loc[train_idx]
    y_train_seq = y_role.loc[train_idx] # We need the targets for voting
    
    print("Vectorizing Histories...")
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_txt)
    
    print("Fitting KNN Model...")
    knn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=-1)
    knn.fit(X_train_vec)
    
    # Save Sequential Artifacts
    seq_dir = 'models/sequential'
    if os.path.exists(seq_dir):
        try: shutil.rmtree(seq_dir)
        except: pass
    os.makedirs(seq_dir, exist_ok=True)
    
    print(f"Saving Sequential artifacts to {seq_dir}...")
    joblib.dump(vectorizer, os.path.join(seq_dir, 'tfidf.pkl'))
    joblib.dump(knn, os.path.join(seq_dir, 'knn.pkl'))
    joblib.dump(y_train_seq.reset_index(drop=True), os.path.join(seq_dir, 'y_train_ref.pkl'))
    
    print("✓ All Training Complete.")
