
import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    def __init__(self, model):
        """
        Initializes the SHAP TreeExplainer for the given model.
        
        Args:
            model: Trained LightGBM (or other tree-based) model.
        """
        self.model = model
        print("Initializing SHAP TreeExplainer... (this may take a moment)")
        # TreeExplainer is optimized for fast calculation on tree models
        self.explainer = shap.TreeExplainer(model)
        
    def get_explanation_object(self, feature_df):
        """
        Returns a shap.Explanation object for the given features.
        Required for shap.plots.beeswarm/bar/force.
        """
        # SHAP calculation
        shap_values = self.explainer.shap_values(feature_df)
        
        # Handle Binary/Ranker Output
        vals = shap_values
        if isinstance(vals, list):
            vals = vals[1] # Positive class
            
        # Create Explanation Object
        # shap.Explanation(values, base_values, data, feature_names)
        # Note: explainer.expected_value might be list too
        base = self.explainer.expected_value
        if isinstance(base, list) or isinstance(base, np.ndarray):
             if len(base) > 1: base = base[1]
             else: base = base[0]
             
        explanation = shap.Explanation(
            values=vals,
            base_values=base,
            data=feature_df.values,
            feature_names=feature_df.columns
        )
        return explanation

    def get_contributions(self, feature_row):
        """
        Calculates feature contributions (SHAP values) for a single prediction row.
        
        Args:
            feature_row (pd.DataFrame): A single row of features (must match model input columns).
            
        Returns:
            dict: {feature_name: contribution_value}, sorted by absolute impact.
            float: base_value (the average score before features are applied).
        """
        if isinstance(feature_row, pd.Series):
            feature_row = feature_row.to_frame().T
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(feature_row)
        
        # Handle binary classification (SHAP returns list for each class, we want positive class index 1)
        # LightGBM binary/lambdarank usually returns single array if objective is simpler, 
        # but classifier returns [neg_shap, pos_shap].
        # We need to inspect shape.
        
        # If model is LGBMRanker or Classifier?
        # Let's assume standard behavior: values[1] for binary/ranker positive class
        
        vals = shap_values
        if isinstance(vals, list):
            # Binary classification usually returns list of arrays
            vals = vals[1] # Positive class
        
        # If it's a single row, dimensionality might be (1, n_features)
        if len(vals.shape) == 2:
            vals = vals[0]
            
        # Create dict
        contribs = {}
        feature_names = feature_row.columns
        
        for name, val in zip(feature_names, vals):
            if abs(val) > 0.001: # Filter noise
                contribs[name] = val
                
        # Sort by absolute impact
        sorted_contribs = dict(sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=True))
        
        return sorted_contribs, self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
