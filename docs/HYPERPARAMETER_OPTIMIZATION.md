# Hyperparameter Optimization Guide

## Quick Start

This guide shows you how to optimize your LightGBM model's hyperparameters using Optuna for improved accuracy.

## Prerequisites

1. **Install Optuna:**
   ```bash
   pip install optuna>=3.0.0
   ```

2. **Ensure training data exists:**
   ```bash
   python src/build_ltr_dataset.py
   ```
   This should create `data/ltr/train_pairs.csv`

## Usage

### Basic Optimization (Fast - 30 mins)

Run with default settings (100 trials, single validation split):

```bash
python src/optimize_hyperparameters.py
```

### Robust Optimization (Recommended - 2-3 hours)

Use cross-validation for more reliable results:

```bash
python src/optimize_hyperparameters.py --cv --trials 100 --folds 5
```

### Quick Test (5-10 mins)

Test the optimization pipeline with fewer trials:

```bash
python src/optimize_hyperparameters.py --trials 20
```

### Advanced Options

```bash
python src/optimize_hyperparameters.py \
  --data data/ltr/train_pairs.csv \
  --output models/ltr \
  --trials 150 \
  --timeout 7200 \
  --cv \
  --folds 5
```

**Parameters:**
- `--data`: Path to training data CSV
- `--output`: Directory to save optimized model
- `--trials`: Number of optimization trials (default: 100)
- `--timeout`: Maximum time in seconds (optional)
- `--cv`: Enable cross-validation (slower but more robust)
- `--folds`: Number of CV folds (default: 5)

## What Gets Optimized

The script optimizes these hyperparameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `boosting_type` | gbdt, dart | Boosting algorithm |
| `learning_rate` | 0.01 - 0.3 | Step size for updates |
| `num_leaves` | 20 - 150 | Max leaves per tree |
| `max_depth` | 3 - 12 | Maximum tree depth |
| `min_data_in_leaf` | 10 - 100 | Min samples per leaf |
| `feature_fraction` | 0.5 - 1.0 | Feature sampling ratio |
| `bagging_fraction` | 0.5 - 1.0 | Data sampling ratio |
| `bagging_freq` | 1 - 10 | Bagging frequency |
| `lambda_l1` | 0 - 10 | L1 regularization |
| `lambda_l2` | 0 - 10 | L2 regularization |
| `min_gain_to_split` | 0 - 5 | Min gain for split |

## Output Files

After optimization, you'll find:

```
models/ltr/
├── lgbm_ranker.pkl              # Optimized model (replaces old one)
├── feature_cols.pkl             # Feature column names
├── metrics.json                 # Final metrics
├── optimization_results.json    # Full optimization history
└── optimization_plots.png       # Visualization of optimization
```

### optimization_results.json

Contains:
- Best hyperparameters found
- Validation AUC score
- Complete trial history
- Timestamp

Example:
```json
{
  "best_params": {
    "learning_rate": 0.087,
    "num_leaves": 89,
    "max_depth": 8,
    ...
  },
  "final_validation_auc": 0.8542,
  "n_trials": 100
}
```

## Expected Results

**Before Optimization (Current):**
- Validation AUC: ~0.82-0.84
- Top-1 Accuracy: ~31.6%

**After Optimization (Expected):**
- Validation AUC: ~0.85-0.87 (+2-3%)
- Top-1 Accuracy: ~34-36% (+2-5%)

## Monitoring Progress

The script shows real-time progress:

```
Trial 0: AUC = 0.8234
Trial 1: AUC = 0.8301 ⬆️
Trial 2: AUC = 0.8189
...
Trial 47: AUC = 0.8542 ⬆️ (Best so far!)
```

## Troubleshooting

### "Training data not found"
Run the dataset builder first:
```bash
python src/build_ltr_dataset.py
```

### Out of Memory
Reduce trials or disable CV:
```bash
python src/optimize_hyperparameters.py --trials 50
```

### Taking too long
Set a timeout (in seconds):
```bash
python src/optimize_hyperparameters.py --timeout 3600  # 1 hour
```

## Integration with Existing System

The optimized model automatically replaces `models/ltr/lgbm_ranker.pkl`. 

To use it:
1. Restart your Streamlit app
2. Or click "🔄 Reload Models & Cache" in the sidebar

The predictor will automatically load the new optimized model!

## Next Steps

After optimization:

1. **Test the model:**
   - Use the Streamlit app to verify predictions
   - Check if confidence scores improved
   
2. **Compare metrics:**
   - Check `models/ltr/optimization_results.json`
   - Compare with previous `metrics.json`

3. **Deploy:**
   - If results are better, commit the new model
   - If not, restore backup from `models/backups/`

## Advanced: Optuna Dashboard

For interactive visualization:

```bash
# Install dashboard
pip install optuna-dashboard

# Run optimization with database
python src/optimize_hyperparameters.py --storage sqlite:///optuna.db

# Launch dashboard
optuna-dashboard sqlite:///optuna.db
```

Then open http://localhost:8080 in your browser.
