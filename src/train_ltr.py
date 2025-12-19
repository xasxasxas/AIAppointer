
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def train_ltr(data_path='data/ltr/train_pairs.csv', model_dir='models/ltr'):
    print("="*60)
    print("🚀 Training Learning-to-Rank Model (LightGBM)")
    print(f"Data: {data_path}")
    print(f"Output: {model_dir}")
    print("="*60)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print("Error: Training data not found. Run build_ltr_dataset.py first.")
        return
        
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Prepare Features
    # Drop non-feature cols
    drop_cols = ['target', 'group_id', 'officer_rank', 'target_role']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    X = df[feature_cols]
    y = df['target']
    
    # 3. Split
    # Important: Split by Group ID if using Lambdarank, but for Binary we can do random split 
    # IF we verify no leakage. (Group split is safer to avoid seeing same officer in train/test)
    # Let's do simple random split for now to mix positives/negatives.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train Size: {len(X_train)} | Val Size: {len(X_val)}")
    
    # 4. Train LightGBM
    print("Training LightGBM Classifier...")
    
    # Params for Ranking-style Binary
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    # 5. Evaluate
    print("\nEvaluating...")
    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Accuracy (Pairwise): {acc:.4f}")
    
    # 6. Feature Importance
    print("\n🔍 Top 10 Important Features:")
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    
    print(imp_df.head(10))
    
    # 7. Save
    # 7. Save
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving model to {model_dir}...")
    joblib.dump(model, os.path.join(model_dir, 'lgbm_ranker.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Save Metrics
    import json
    metrics = {'accuracy': acc, 'auc': auc}
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    print("✓ Training Complete.")

if __name__ == "__main__":
    train_ltr()
