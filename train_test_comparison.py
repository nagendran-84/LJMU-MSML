"""
Train vs Test Performance Comparison
=====================================
MSc Machine Learning -- Liverpool John Moores University
Topic: Explainability and Fairness in ML Models for Credit Risk Prediction

Standalone follow-up to pipeline.py. Reproduces the same data preparation,
train/test split (random_state=42), SMOTE resampling, and model definitions
used in pipeline.py (Section 7), retrains the same 5 models on the
SMOTE-resampled training set, and evaluates each model on both:
  - the original (pre-SMOTE) training instances  (X_train_raw, y_train)
  - the held-out test set                         (X_test_raw,  y_test)

Train-set metrics are computed on the original (non-resampled) training
instances rather than the SMOTE-augmented set the model was fit on --
scoring against the SMOTE-augmented set would inflate train performance
with synthetic minority oversamples and wouldn't be comparable to the test
set, which keeps the natural imbalanced class distribution. This keeps the
train-vs-test gap a measure of generalisation rather than an oversampling
artefact.

Run from the same environment used for pipeline.py (so scikit-learn /
XGBoost / imbalanced-learn versions match and the retrained models reproduce
the same numbers as the existing test-set figures):

    cd "Source Code"
    python train_test_comparison.py

The script re-checks its test-set AUCs against the values already shown in
figures/roc_curves.png (Figure 5.1 in the thesis) and prints OK/MISMATCH per
model -- a MISMATCH means the environment differs from the one that produced
the original figures (most likely a scikit-learn / XGBoost version
difference), and the new train-vs-test outputs won't be consistent with the
existing figures.

Outputs (written into results/ and figures/, alongside pipeline.py's outputs):
    results/performance_train_vs_test.csv
    latex_exports/performance_train_vs_test.tex
    figures/roc_curves_train.pdf / .png
    figures/confusion_matrices_train.pdf / .png
    figures/train_vs_test_comparison.pdf / .png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive -- safe for scripts
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

for d in ["results", "figures", "latex_exports"]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION  (identical to pipeline.py Section 2)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Reproducing data preparation from pipeline.py")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "german_credit.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}.\n"
        "Please run  python download_dataset.py  first (same as pipeline.py)."
    )

df_raw = pd.read_csv(DATA_PATH)
print(f"Raw shape : {df_raw.shape}", flush=True)

y_raw = df_raw["target"].values          # 0 = good, 1 = bad
X_raw = df_raw.drop(columns=["target"])

X_encoded = pd.get_dummies(X_raw, drop_first=True)
feature_names = list(X_encoded.columns)
print(f"Features after encoding: {len(feature_names)}", flush=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded.values)

# ---------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT  (identical: stratified, 80/20, same seed)
#    NOTE: pipeline.py also splits a third 'protected' array in the same
#    call, but extra arrays don't change the index generation in
#    train_test_split -- the split indices for X_scaled/y_raw here are
#    identical to those used in pipeline.py.
# ---------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.20, stratify=y_raw, random_state=RANDOM_STATE
)

smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train_res = smote.fit_resample(X_train_raw, y_train)
print(f"Original train (pre-SMOTE): {X_train_raw.shape}, test: {X_test_raw.shape}")
print(f"SMOTE-resampled train (used for fitting): {X_train.shape}\n", flush=True)

# ---------------------------------------------------------------------------
# 3. MODEL DEFINITIONS  (identical to pipeline.py Section 4)
# ---------------------------------------------------------------------------
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        random_state=RANDOM_STATE
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        eval_metric="logloss",
        random_state=RANDOM_STATE, verbosity=0
    ),
}

# ---------------------------------------------------------------------------
# 4. TRAIN FINAL MODELS  (identical to pipeline.py Section 7)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Training final models on the SMOTE-resampled training set")
print("=" * 60)
trained_models = {}
for name, model in MODELS.items():
    print(f"  Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train_res)
    trained_models[name] = model
    print("done", flush=True)
print()

# ---------------------------------------------------------------------------
# 4b. SANITY CHECK vs the test-set AUCs already shown in the existing
#     figures/roc_curves.png (Figure 5.1 in the thesis). If these don't
#     match closely, the new train-vs-test comparison would be inconsistent
#     with the figures already embedded in the report -- stop and flag it.
# ---------------------------------------------------------------------------
EXPECTED_TEST_AUC = {
    "Logistic Regression": 0.800,
    "Decision Tree": 0.704,
    "Random Forest": 0.780,
    "Gradient Boosting": 0.790,
    "XGBoost": 0.799,
}
print("=" * 60)
print("Sanity check vs existing Figure 5.1 test-set AUCs")
print("=" * 60)
any_mismatch = False
for name, model in trained_models.items():
    got = roc_auc_score(y_test, model.predict_proba(X_test_raw)[:, 1])
    exp = EXPECTED_TEST_AUC[name]
    ok = abs(got - exp) < 0.01
    any_mismatch = any_mismatch or not ok
    print(f"  {name:<22} expected={exp:.3f}  got={got:.3f}  "
          f"[{'OK' if ok else 'MISMATCH -- check sklearn/XGBoost versions'}]")
print()
if any_mismatch:
    print("!! At least one model's test AUC differs from the existing Figure 5.1.")
    print("!! The figures generated below will not be perfectly consistent with")
    print("!! the existing test-set figures in the thesis -- check sklearn/XGBoost")
    print("!! versions before using these outputs.\n")

# ---------------------------------------------------------------------------
# 5. TRAIN vs TEST METRICS
# ---------------------------------------------------------------------------
print("=" * 60)
print("Computing Train vs Test metrics")
print("=" * 60)

rows = []
for name, model in trained_models.items():
    tr_proba = model.predict_proba(X_train_raw)[:, 1]
    tr_pred  = model.predict(X_train_raw)
    te_proba = model.predict_proba(X_test_raw)[:, 1]
    te_pred  = model.predict(X_test_raw)

    tr_auc, te_auc = roc_auc_score(y_train, tr_proba), roc_auc_score(y_test, te_proba)
    tr_acc, te_acc = accuracy_score(y_train, tr_pred), accuracy_score(y_test, te_pred)
    tr_f1, te_f1   = f1_score(y_train, tr_pred), f1_score(y_test, te_pred)

    rows.append({
        "Model": name,
        "Train AUC": tr_auc, "Test AUC": te_auc, "AUC Gap": tr_auc - te_auc,
        "Train Acc": tr_acc, "Test Acc": te_acc, "Acc Gap": tr_acc - te_acc,
        "Train F1": tr_f1, "Test F1": te_f1, "F1 Gap": tr_f1 - te_f1,
    })
    print(f"  {name:<22} Train AUC={tr_auc:.4f}  Test AUC={te_auc:.4f}  "
          f"Gap={tr_auc - te_auc:+.4f}", flush=True)

cmp_df = pd.DataFrame(rows).sort_values("Test AUC", ascending=False)
cmp_df.to_csv("results/performance_train_vs_test.csv", index=False)
with open("latex_exports/performance_train_vs_test.tex", "w") as f:
    f.write(cmp_df.to_latex(index=False, float_format="%.4f",
                             caption="Train vs Test Set Performance Comparison",
                             label="tab:train_vs_test"))
print("\n[Saved] results/performance_train_vs_test.csv  &  "
      "latex_exports/performance_train_vs_test.tex\n", flush=True)

# ---------------------------------------------------------------------------
# 6. ROC CURVES -- TRAIN SET  (mirrors pipeline.py Section 8, on train data)
# ---------------------------------------------------------------------------
print("[Plotting] ROC curves (train set)...", flush=True)
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for (name, model), color in zip(trained_models.items(), colors):
    proba = model.predict_proba(X_train_raw)[:, 1]
    fpr, tpr, _ = roc_curve(y_train, proba)
    auc = roc_auc_score(y_train, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves -- All Models (Train Set)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/roc_curves_train.pdf", bbox_inches="tight")
plt.savefig("figures/roc_curves_train.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/roc_curves_train.pdf\n", flush=True)

# ---------------------------------------------------------------------------
# 7. CONFUSION MATRICES -- TRAIN SET  (mirrors pipeline.py Section 9)
# ---------------------------------------------------------------------------
print("[Plotting] Confusion matrices (train set)...", flush=True)
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, model) in zip(axes, trained_models.items()):
    preds = model.predict(X_train_raw)
    cm = confusion_matrix(y_train, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(name, fontsize=9)

plt.suptitle("Confusion Matrices -- Train Set", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("figures/confusion_matrices_train.pdf", bbox_inches="tight")
plt.savefig("figures/confusion_matrices_train.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/confusion_matrices_train.pdf\n", flush=True)

# ---------------------------------------------------------------------------
# 8. TRAIN vs TEST GROUPED BAR CHART (AUC / Accuracy / F1 side-by-side)
# ---------------------------------------------------------------------------
print("[Plotting] Train vs Test comparison chart...", flush=True)
metrics = [("AUC", "Train AUC", "Test AUC"),
           ("Accuracy", "Train Acc", "Test Acc"),
           ("F1-Score", "Train F1", "Test F1")]
plot_df = cmp_df.set_index("Model")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(plot_df))
width = 0.35
for ax, (label, tr_col, te_col) in zip(axes, metrics):
    ax.bar(x - width / 2, plot_df[tr_col], width, label="Train", color="#4C72B0")
    ax.bar(x + width / 2, plot_df[te_col], width, label="Test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=9)
    ax.set_title(label, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Train vs Test Set Performance Comparison -- All Models", fontsize=13)
plt.tight_layout()
plt.savefig("figures/train_vs_test_comparison.pdf", bbox_inches="tight")
plt.savefig("figures/train_vs_test_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/train_vs_test_comparison.pdf\n", flush=True)

print("=" * 60)
print("DONE -- new outputs")
print("=" * 60)
print("  results/performance_train_vs_test.csv")
print("  latex_exports/performance_train_vs_test.tex")
print("  figures/roc_curves_train.pdf / .png")
print("  figures/confusion_matrices_train.pdf / .png")
print("  figures/train_vs_test_comparison.pdf / .png")
