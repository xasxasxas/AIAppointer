
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from datetime import datetime

def optimize_hyperparameters(
    data_path='data/ltr/train_pairs.csv',
    model_dir='models/ltr',
    n_trials=100,
    timeout=None,
    use_cv=True,
    n_folds=5
):
    """
    Optimize LightGBM hyperparameters using Optuna
    
    Args:
        data_path: Path to training data CSV
        model_dir: Directory to save optimized model
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (None for no limit)
        use_cv: Whether to use cross-validation (slower but more robust)
        n_folds: Number of CV folds if use_cv=True
    """
    print("="*60)
    print("🔬 LightGBM Hyperparameter Optimization with Optuna")
    print("="*60)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Run build_ltr_dataset.py first.")
        return None
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Prepare Features
    drop_cols = ['target', 'group_id', 'officer_rank', 'target_role']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Samples: {len(df)} rows")
    print(f"Positive rate: {df['target'].mean():.2%}")
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # 3. Train/Val Split (for final evaluation)
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train_full)} | Validation: {len(X_val)}")
    
    # 4. Define Objective Function
    def objective(trial):
        """
        Optuna objective function to maximize validation AUC
        """
        # Suggest hyperparameters
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 5),
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        
        if use_cv:
            # Cross-validation for more robust evaluation
            cv_scores = []
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
                X_train_fold = X_train_full[train_idx]
                y_train_fold = y_train_full[train_idx]
                X_val_fold = X_train_full[val_idx]
                y_val_fold = y_train_full[val_idx]
                
                dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
                dval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)
                
                # Train with pruning callback
                pruning_callback = LightGBMPruningCallback(trial, 'auc', valid_name='valid_0')
                
                model = lgb.train(
                    param,
                    dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    valid_names=['valid_0'],  # Changed from 'val' to 'valid_0'
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(0),  # Silent
                        pruning_callback
                    ]
                )
                
                # Predict and evaluate
                preds = model.predict(X_val_fold)
                auc = roc_auc_score(y_val_fold, preds)
                cv_scores.append(auc)
            
            # Return mean CV score
            mean_auc = np.mean(cv_scores)
            return mean_auc
        
        else:
            # Single train/val split (faster)
            X_train, X_val_opt, y_train, y_val_opt = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
            )
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val_opt, label=y_val_opt, reference=dtrain)
            
            pruning_callback = LightGBMPruningCallback(trial, 'auc', valid_name='valid_0')
            
            model = lgb.train(
                param,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                valid_names=['valid_0'],  # Changed from 'val' to 'valid_0'
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),
                    pruning_callback
                ]
            )
            
            preds = model.predict(X_val_opt)
            auc = roc_auc_score(y_val_opt, preds)
            return auc
    
    # 5. Run Optimization
    print(f"\n🚀 Starting Optuna optimization...")
    print(f"Trials: {n_trials}")
    print(f"Cross-validation: {use_cv} ({n_folds} folds)" if use_cv else "Single split validation")
    print(f"Timeout: {timeout}s" if timeout else "No timeout")
    print("\nThis may take a while...\n")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='lgbm_ltr_optimization',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # 6. Results
    print("\n" + "="*60)
    print("✅ Optimization Complete!")
    print("="*60)
    
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best AUC: {study.best_trial.value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 7. Train Final Model with Best Params
    print("\n" + "="*60)
    print("🏋️ Training final model with best hyperparameters...")
    print("="*60)
    
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    })
    
    dtrain_final = lgb.Dataset(X_train_full, label=y_train_full)
    dval_final = lgb.Dataset(X_val, label=y_val, reference=dtrain_final)
    
    final_model = lgb.train(
        best_params,
        dtrain_final,
        num_boost_round=1000,
        valid_sets=[dtrain_final, dval_final],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(50)
        ]
    )
    
    # 8. Final Evaluation
    print("\n" + "="*60)
    print("📊 Final Model Evaluation")
    print("="*60)
    
    preds_val = final_model.predict(X_val)
    final_auc = roc_auc_score(y_val, preds_val)
    final_acc = accuracy_score(y_val, (preds_val > 0.5).astype(int))
    
    print(f"\nValidation AUC: {final_auc:.4f}")
    print(f"Validation Accuracy: {final_acc:.4f}")
    
    # Feature Importance
    print("\n🔍 Top 10 Important Features:")
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    print(imp_df.head(10).to_string(index=False))
    
    # 9. Save Everything
    os.makedirs(model_dir, exist_ok=True)
    
    # Save optimized model
    model_path = os.path.join(model_dir, 'lgbm_ranker.pkl')
    joblib.dump(final_model, model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save feature columns
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Save optimization results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_auc': study.best_trial.value,
        'best_params': study.best_params,
        'final_validation_auc': final_auc,
        'final_validation_accuracy': final_acc,
        'optimization_history': [
            {'trial': t.number, 'auc': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ]
    }
    
    results_path = os.path.join(model_dir, 'optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {results_path}")
    
    # Save metrics (for compatibility with existing code)
    metrics = {'accuracy': final_acc, 'auc': final_auc}
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # 10. Visualization (optional - save plots)
    try:
        import matplotlib.pyplot as plt
        
        # Optimization history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Optimization history
        trials = [t.number for t in study.trials if t.value is not None]
        values = [t.value for t in study.trials if t.value is not None]
        ax1.plot(trials, values, 'o-', alpha=0.6)
        ax1.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.4f}')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('AUC')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())[:10]  # Top 10
            importances = [importance[p] for p in params]
            
            ax2.barh(params, importances)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importance')
            ax2.grid(True, alpha=0.3, axis='x')
        except:
            ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plot_path = os.path.join(model_dir, 'optimization_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plots saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("🎉 Optimization pipeline complete!")
    print("="*60)
    
    return study, final_model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize LightGBM hyperparameters with Optuna')
    parser.add_argument('--data', type=str, default='data/ltr/train_pairs.csv',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='models/ltr',
                       help='Output directory for models')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (None for no limit)')
    parser.add_argument('--cv', action='store_true',
                       help='Use cross-validation (slower but more robust)')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of CV folds')
    
    args = parser.parse_args()
    
    optimize_hyperparameters(
        data_path=args.data,
        model_dir=args.output,
        n_trials=args.trials,
        timeout=args.timeout,
        use_cv=args.cv,
        n_folds=args.folds
    )
