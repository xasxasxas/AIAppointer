"""
Evaluation Script for AI Appointer Model

This script evaluates the model performance using leave-one-out style evaluation:
For each officer, we use their career history to predict their current role
and check if the current role appears in the Top-K predictions.

Usage:
    python scripts/evaluate_model.py
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from datetime import datetime
import json

def evaluate_topk_accuracy(verbose=True):
    """
    Evaluates the model by predicting current roles from history.
    
    Returns:
        dict: Metrics including Top-1, Top-5, Top-10 accuracy
    """
    from src.predictor import Predictor
    from config import DATASET_PATH
    
    print("=" * 60)
    print("AI Appointer Model Evaluation")
    print("=" * 60)
    
    # Load predictor (suppress SHAP output for speed)
    print("\n📦 Loading predictor...")
    predictor = Predictor()
    
    if not predictor.ready:
        print("❌ Predictor not ready. Aborting.")
        return None
    
    # Load dataset
    print(f"\n📊 Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   Total officers: {len(df)}")
    
    # Metrics tracking
    top_1_hits = 0
    top_5_hits = 0
    top_10_hits = 0
    total_evaluated = 0
    
    # Track failures for analysis
    failures = []
    
    print("\n🔄 Evaluating predictions...")
    
    for idx, row in df.iterrows():
        try:
            # Get actual current appointment
            actual_role = row.get('current_appointment', '')
            if not actual_role or actual_role == 'Unknown':
                continue
            
            # Create input data for prediction
            input_data = {
                'Employee_ID': row['Employee_ID'],
                'Rank': row['Rank'],
                'Branch': row['Branch'],
                'Pool': row.get('Pool', 'Unknown'),
                'Entry_type': row.get('Entry_type', 'Unknown'),
                'Appointment_history': row.get('Appointment_history', ''),
                'Training_history': row.get('Training_history', ''),
                'Promotion_history': row.get('Promotion_history', ''),
                'current_appointment': actual_role,  # For reference
                '8_yr_avg_eval': row.get('8_yr_avg_eval', 70)
            }
            
            # Get predictions
            results = predictor.predict(input_data)
            
            if results.empty:
                continue
            
            # Extract predicted roles
            predicted_roles = results['Prediction'].tolist()
            
            # Check hits
            total_evaluated += 1
            
            if actual_role in predicted_roles[:1]:
                top_1_hits += 1
            if actual_role in predicted_roles[:5]:
                top_5_hits += 1
            if actual_role in predicted_roles[:10]:
                top_10_hits += 1
            else:
                # Track failures for analysis
                failures.append({
                    'Employee_ID': row['Employee_ID'],
                    'Rank': row['Rank'],
                    'Branch': row['Branch'],
                    'Actual': actual_role,
                    'Top_Prediction': predicted_roles[0] if predicted_roles else 'None'
                })
            
            # Progress update
            if verbose and (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} officers...")
                
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error on row {idx}: {e}")
            continue
    
    # Calculate metrics
    if total_evaluated == 0:
        print("❌ No officers evaluated. Check data format.")
        return None
    
    metrics = {
        'total_evaluated': total_evaluated,
        'top_1_accuracy': top_1_hits / total_evaluated,
        'top_5_accuracy': top_5_hits / total_evaluated,
        'top_10_accuracy': top_10_hits / total_evaluated,
        'top_1_hits': top_1_hits,
        'top_5_hits': top_5_hits,
        'top_10_hits': top_10_hits,
        'evaluation_date': datetime.now().isoformat()
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Officers Evaluated: {total_evaluated}")
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Top-1 Accuracy:  {metrics['top_1_accuracy']:.1%} ({top_1_hits}/{total_evaluated})")
    print(f"   Top-5 Accuracy:  {metrics['top_5_accuracy']:.1%} ({top_5_hits}/{total_evaluated})")
    print(f"   Top-10 Accuracy: {metrics['top_10_accuracy']:.1%} ({top_10_hits}/{total_evaluated})")
    
    # Context
    print(f"\n📊 Context:")
    print(f"   Random Baseline (1000 roles): ~0.1% Top-1, ~0.5% Top-5")
    print(f"   Improvement over Random: {metrics['top_5_accuracy'] / 0.005:.0f}x")
    
    # Show sample failures for debugging
    if failures and verbose:
        print(f"\n🔍 Sample Misses (for debugging):")
        for f in failures[:5]:
            print(f"   ID {f['Employee_ID']} ({f['Rank']}, {f['Branch']})")
            print(f"      Actual: {f['Actual']}")
            print(f"      Predicted: {f['Top_Prediction']}")
    
    # Save metrics
    metrics_path = 'models/evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")
    
    return metrics


def evaluate_billet_lookup(verbose=True):
    """
    Evaluates Billet Lookup by checking if the correct officer appears in top candidates.
    """
    from src.predictor import Predictor
    from config import DATASET_PATH
    
    print("\n" + "=" * 60)
    print("Billet Lookup Evaluation")
    print("=" * 60)
    
    predictor = Predictor()
    df = pd.read_csv(DATASET_PATH)
    
    # Sample some roles to test
    sample_roles = df['current_appointment'].value_counts().head(10).index.tolist()
    
    print(f"\n🎯 Testing {len(sample_roles)} popular roles...")
    
    hits = 0
    total = 0
    
    for role in sample_roles:
        # Get actual holders of this role
        actual_holders = df[df['current_appointment'] == role]['Employee_ID'].tolist()
        
        if not actual_holders or role not in predictor.valid_roles:
            continue
        
        # Get predictions
        results = predictor.predict_for_role(df, role)
        
        if results.empty:
            continue
        
        # Check if any actual holder is in top 10
        predicted_ids = results.head(10)['Employee_ID'].tolist()
        
        total += 1
        if any(h in predicted_ids for h in actual_holders):
            hits += 1
            if verbose:
                print(f"   ✓ {role}: Found actual holder in top 10")
        else:
            if verbose:
                print(f"   ✗ {role}: No actual holder in top 10")
    
    if total > 0:
        acc = hits / total
        print(f"\n📊 Billet Lookup Accuracy: {acc:.1%} ({hits}/{total} roles)")
    
    return {'billet_accuracy': hits/total if total > 0 else 0}


if __name__ == "__main__":
    print("\n" + "🚀 Starting AI Appointer Evaluation\n")
    
    # Run main evaluation
    metrics = evaluate_topk_accuracy(verbose=True)
    
    # Run billet lookup evaluation
    # billet_metrics = evaluate_billet_lookup(verbose=True)
    
    print("\n✅ Evaluation Complete!\n")
