
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.predictor import Predictor
from src.explainer import Explainer

def analyze():
    print("Loading Predictor...")
    predictor = Predictor()
    
    print("Loading Data...")
    df = pd.read_csv('data/hr_star_trek_v4c_modernized_clean_modified_v4.csv')
    
    officer_row = df[df['Employee_ID'] == 200016]
    if officer_row.empty:
        print("Employee 200016 not found!")
        return

    print("\n--- Officer Profile ---")
    print(officer_row.iloc[0].to_dict())

    print("\n--- Running Prediction ---")
    results = predictor.predict(officer_row)
    
    if results.empty:
        print("No predictions returned.")
        return

    top = results.iloc[0]
    print(f"\nTop Prediction: {top['Prediction']}")
    print(f"Confidence: {top['Confidence']:.2%}")
    print(f"Raw Score (Base+Sum): {top.get('_BaseVal', 0) + sum(top['_Contribs'].values()):.4f}")
    print(f"Base Value: {top.get('_BaseVal', 0)}")
    
    print("\n--- Features & SHAP ---")
    feats = top['_Feats']
    contribs = top['_Contribs']
    
    # Sort contribs
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for k, v in sorted_contribs:
        feat_val = feats.get(k, 'N/A')
        print(f"{k}: SHAP={v:.4f} | Value={feat_val}")
        
    print("\n--- Raw Context ---")
    print(feats.get('_Context', {}))

if __name__ == "__main__":
    analyze()
