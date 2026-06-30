# Explainability and Fairness in ML Models for Credit Risk Prediction

MSc Machine Learning — Liverpool John Moores University

This repository contains the source code for a credit risk prediction study that
compares five classifiers, explains their predictions with SHAP, LIME, and DiCE
counterfactuals, and audits them for demographic fairness on the
[UCI German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) set.

## Contents

| File | Purpose |
|---|---|
| `download_dataset.py` | Downloads the German Credit dataset from UCI and saves it to `data/german_credit.csv` |
| `pipeline.py` | End-to-end pipeline: data prep, 5-fold CV, model training, SHAP/LIME explanations, DiCE counterfactuals, fairness audit |
| `train_test_comparison.py` | Standalone follow-up that reproduces `pipeline.py`'s training exactly and compares train-set vs. test-set performance to check for overfitting |
| `requirements.txt` | Python package dependencies |

Running the scripts generates `data/`, `results/`, `figures/`, and `latex_exports/`
locally (all gitignored — see [Generated output](#generated-output) below).

## Pipeline overview (`pipeline.py`)

1. **Data preparation** — loads `data/german_credit.csv`, extracts the protected
   attribute (gender, derived from `personal_status`), one-hot encodes categorical
   features, and scales numeric features.
2. **Train/test split** — stratified 80/20 split (`random_state=42`), with SMOTE
   applied to the training fold only.
3. **Models** — Logistic Regression, Decision Tree, Random Forest, Gradient
   Boosting, and XGBoost.
4. **5-fold stratified cross-validation** — AUC, accuracy, and F1 per model, with
   SMOTE re-applied inside each fold.
5. **Statistical significance** — Wilcoxon signed-rank test comparing the best
   black-box model against the best baseline model.
6. **Final training** — all 5 models retrained on the full SMOTE-resampled
   training set, evaluated on the held-out test set (ROC curves, confusion
   matrices).
7. **SHAP analysis** — global feature importance (beeswarm + bar plots) and a
   local waterfall explanation for a misclassified instance, for each
   tree-based model (Random Forest, Gradient Boosting, XGBoost).
8. **LIME analysis** — local explanations aggregated across 50 test instances
   per tree-based model.
9. **SHAP vs. LIME consistency** — Spearman rank correlation between the two
   explanation methods' feature rankings.
10. **DiCE counterfactuals** — actionable counterfactual explanations for 5
    bad-credit test instances using the best black-box model.
11. **Fairness audit** — Demographic Parity Difference and Equalized Odds
    Difference (by gender) for every model, against a 0.1 threshold.

## `train_test_comparison.py`

Reproduces the exact data preparation, split, SMOTE resampling, and model
definitions from `pipeline.py`, then evaluates each model on both the
*original* (pre-SMOTE) training instances and the test set side by side. It
includes a built-in sanity check against expected test-set AUCs to confirm the
retrained models match the ones used to produce `pipeline.py`'s output figures
before generating new train-vs-test comparison plots.

## Quick start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
python download_dataset.py

# 4. Run the full pipeline
python pipeline.py

# 5. (Optional) Train/test overfitting comparison
python train_test_comparison.py
```

## Generated output

These folders are created by the scripts above and are not tracked in git
(see `.gitignore`) — regenerate them by re-running the scripts:

- `data/` — `german_credit.csv` (via `download_dataset.py`)
- `results/` — CSV/text outputs: performance tables, Wilcoxon test, SHAP/LIME
  consistency, counterfactuals, fairness metrics
- `figures/` — PDF/PNG plots: ROC curves, confusion matrices, SHAP/LIME plots,
  fairness charts
- `latex_exports/` — `.tex` tables ready for inclusion in a LaTeX report

## Dependencies

See `requirements.txt`: pandas, numpy, matplotlib, seaborn, scikit-learn,
imbalanced-learn, shap, lime, fairlearn, dice-ml, xgboost, scipy.
