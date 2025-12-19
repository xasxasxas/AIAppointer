#!/bin/bash
# Quick Start Script for Hyperparameter Optimization
# This script runs the optimization with recommended settings

echo "========================================"
echo "AI Appointer - Hyperparameter Optimization"
echo "========================================"
echo ""

# Check if training data exists
if [ ! -f "data/ltr/train_pairs.csv" ]; then
    echo "ERROR: Training data not found!"
    echo ""
    echo "Please run the dataset builder first:"
    echo "  python src/build_ltr_dataset.py"
    echo ""
    exit 1
fi

echo "Training data found!"
echo ""
echo "Starting optimization with recommended settings:"
echo "- Trials: 100"
echo "- Cross-validation: 5 folds"
echo "- This will take approximately 2-3 hours"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "Starting optimization..."
echo ""

python src/optimize_hyperparameters.py --cv --trials 100 --folds 5

echo ""
echo "========================================"
echo "Optimization Complete!"
echo "========================================"
echo ""
echo "Check the results in:"
echo "  models/ltr/optimization_results.json"
echo "  models/ltr/optimization_plots.png"
echo ""
echo "The optimized model has been saved to:"
echo "  models/ltr/lgbm_ranker.pkl"
echo ""
echo "Restart your Streamlit app to use the new model!"
echo ""
