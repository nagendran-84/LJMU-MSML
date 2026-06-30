"""
Enhanced Credit Risk Explainability Pipeline
=============================================
MSc Machine Learning — Liverpool John Moores University
Topic: Explainability and Fairness in ML Models for Credit Risk Prediction

Run:
    pip install -r requirements.txt
    python pipeline.py

Outputs (written to results/ and figures/):
    - performance_table.csv / .tex
    - roc_curves.pdf
    - confusion_matrices.pdf
    - shap_summary_<model>.pdf
    - lime_explanation_<model>.pdf
    - shap_lime_rank_comparison.csv / .tex
    - counterfactuals.csv / .tex
    - fairness_metrics.csv / .tex
"""

# ---------------------------------------------------------------------------
# 0. DEPENDENCIES
# ---------------------------------------------------------------------------
import subprocess, sys

REQUIRED = [
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "imbalanced-learn", "shap", "lime", "fairlearn", "dice-ml",
    "xgboost", "scipy"
]
for pkg in REQUIRED:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# ---------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# ---------------------------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon, spearmanr

import shap
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import dice_ml

# Output directories
for d in ["results", "figures", "latex_exports"]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 2. DATA LOADING & PREPARATION
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "german_credit.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}.\n"
        "Please run  python download_dataset.py  first."
    )

df_raw = pd.read_csv(DATA_PATH)
print(f"Raw shape : {df_raw.shape}", flush=True)
print(f"Columns   : {list(df_raw.columns)}", flush=True)
print(f"Target dist:\n{df_raw['target'].value_counts()}\n", flush=True)

# ── Identify the protected-attribute column (gender / personal status) ──
# ucimlrepo names it 'personal_status'; older versions may differ
PERSONAL_STATUS_COL = None
for candidate in ["personal_status", "personal_status_and_sex",
                   "Personal_status_and_sex", "Attribute9"]:
    if candidate in df_raw.columns:
        PERSONAL_STATUS_COL = candidate
        break

if PERSONAL_STATUS_COL:
    print(f"Protected attribute column found: '{PERSONAL_STATUS_COL}'")
    print(df_raw[PERSONAL_STATUS_COL].value_counts(), "\n")
else:
    print("Protected attribute column not found — fairness will use age proxy.\n")

# ── Separate features / target ──
y_raw = df_raw["target"].values          # 0 = good, 1 = bad
X_raw = df_raw.drop(columns=["target"])

# ── Extract protected attribute BEFORE encoding ──
if PERSONAL_STATUS_COL:
    # A91 / A93 / A94 → male (0) ;  A92 / A95 → female (1)
    gender_map = {
        "male div/sep":           0, "male single":    0, "male mar/wid":    0,
        "female div/dep/mar":     1,
        # fallback for code-style values
        "A91": 0, "A93": 0, "A94": 0, "A92": 1, "A95": 1,
    }
    raw_vals = X_raw[PERSONAL_STATUS_COL].astype(str)
    protected_raw = raw_vals.map(gender_map)
    if protected_raw.isna().all():
        # Try substring matching
        protected_raw = raw_vals.apply(
            lambda v: 1 if "female" in v.lower() else 0
        )
    print(f"Gender distribution (0=male, 1=female):\n{protected_raw.value_counts()}\n")
else:
    # Fallback: age ≤ 25 as disadvantaged group proxy
    age_col = [c for c in X_raw.columns if "age" in c.lower()]
    if age_col:
        protected_raw = (X_raw[age_col[0]] <= 25).astype(int)
    else:
        protected_raw = pd.Series(
            np.random.choice([0, 1], size=len(y_raw)), index=X_raw.index
        )

# ── One-hot encode categoricals ──
X_encoded = pd.get_dummies(X_raw, drop_first=True)
feature_names = list(X_encoded.columns)
print(f"Features after encoding: {len(feature_names)}", flush=True)

# ── Scale ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded.values)

# ── Align protected attribute to full dataset index ──
protected = protected_raw.values

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT  (stratified, 80/20)
# ---------------------------------------------------------------------------
(X_train_raw, X_test_raw,
 y_train, y_test,
 prot_train, prot_test) = train_test_split(
    X_scaled, y_raw, protected,
    test_size=0.20, stratify=y_raw, random_state=RANDOM_STATE
)

# ── Apply SMOTE only to training fold ──
smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train_res = smote.fit_resample(X_train_raw, y_train)
print(f"After SMOTE — train: {X_train.shape}, test: {X_test_raw.shape}\n", flush=True)

# ---------------------------------------------------------------------------
# 4. MODEL DEFINITIONS
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
# 5. CROSS-VALIDATION  (5-fold stratified, SMOTE per fold)
# ---------------------------------------------------------------------------
print("=" * 60, flush=True)
print("STEP 2 — 5-Fold Stratified Cross-Validation", flush=True)
print("=" * 60, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {name: {"auc": [], "acc": [], "f1": []} for name in MODELS}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_raw), 1):
    print(f"  Fold {fold}/5 — training models...", flush=True)
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y_raw[train_idx], y_raw[val_idx]

    # SMOTE on this fold's training data only
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    for name, model in MODELS.items():
        print(f"    {name}...", end=" ", flush=True)
        model.fit(X_tr_res, y_tr_res)
        proba = model.predict_proba(X_val)[:, 1]
        preds = model.predict(X_val)
        cv_results[name]["auc"].append(roc_auc_score(y_val, proba))
        cv_results[name]["acc"].append(accuracy_score(y_val, preds))
        cv_results[name]["f1"].append(f1_score(y_val, preds))
        print(f"AUC={cv_results[name]['auc'][-1]:.3f}", flush=True)

    print(f"  Fold {fold} complete.\n", flush=True)

# ── Summary table ──
rows = []
for name, scores in cv_results.items():
    rows.append({
        "Model":      name,
        "AUC Mean":   np.mean(scores["auc"]),
        "AUC Std":    np.std(scores["auc"]),
        "Acc Mean":   np.mean(scores["acc"]),
        "Acc Std":    np.std(scores["acc"]),
        "F1 Mean":    np.mean(scores["f1"]),
        "F1 Std":     np.std(scores["f1"]),
    })

perf_df = pd.DataFrame(rows).sort_values("AUC Mean", ascending=False)
print("\nCross-Validation Results:", flush=True)
print(perf_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

perf_df.to_csv("results/performance_table.csv", index=False)

# LaTeX table
with open("latex_exports/performance_table.tex", "w") as f:
    f.write(perf_df.to_latex(index=False, float_format="%.4f",
                              caption="5-Fold CV Performance Comparison",
                              label="tab:performance"))
print("\n[Saved] results/performance_table.csv  &  latex_exports/performance_table.tex", flush=True)

# ---------------------------------------------------------------------------
# 6. WILCOXON SIGNED-RANK TEST  (best black-box vs best baseline)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 3 — Statistical Significance (Wilcoxon Test)", flush=True)
print("=" * 60, flush=True)

best_bb   = max(["Random Forest", "Gradient Boosting", "XGBoost"],
                key=lambda m: np.mean(cv_results[m]["auc"]))
best_base = max(["Logistic Regression", "Decision Tree"],
                key=lambda m: np.mean(cv_results[m]["auc"]))

auc_bb   = cv_results[best_bb]["auc"]
auc_base = cv_results[best_base]["auc"]

stat, p_val = wilcoxon(auc_bb, auc_base)
print(f"  {best_bb}  vs  {best_base}", flush=True)
print(f"  Wilcoxon statistic = {stat:.4f},  p-value = {p_val:.4f}", flush=True)
sig = "significant (p < 0.05)" if p_val < 0.05 else "NOT significant (p >= 0.05)"
print(f"  Result: {sig}\n", flush=True)

with open("results/wilcoxon_test.txt", "w") as f:
    f.write(f"{best_bb} vs {best_base}\n")
    f.write(f"Wilcoxon statistic = {stat:.4f}\np-value = {p_val:.4f}\n{sig}\n")

# ---------------------------------------------------------------------------
# 7. TRAIN FINAL MODELS ON FULL TRAINING SET
# ---------------------------------------------------------------------------
print("=" * 60, flush=True)
print("STEP 4 — Training final models on full train split", flush=True)
print("=" * 60, flush=True)

trained_models = {}
for name, model in MODELS.items():
    print(f"  Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train_res)
    trained_models[name] = model
    preds = model.predict(X_test_raw)
    proba = model.predict_proba(X_test_raw)[:, 1]
    print(f"Test AUC={roc_auc_score(y_test, proba):.4f}"
          f"  Acc={accuracy_score(y_test, preds):.4f}"
          f"  F1={f1_score(y_test, preds):.4f}", flush=True)

# ---------------------------------------------------------------------------
# 8. ROC CURVES
# ---------------------------------------------------------------------------
print("\n[Plotting] ROC curves...", flush=True)
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for (name, model), color in zip(trained_models.items(), colors):
    proba = model.predict_proba(X_test_raw)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models (Test Set)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/roc_curves.pdf", bbox_inches="tight")
plt.savefig("figures/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/roc_curves.pdf", flush=True)

# ---------------------------------------------------------------------------
# 9. CONFUSION MATRICES
# ---------------------------------------------------------------------------
print("[Plotting] Confusion matrices...", flush=True)
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, model) in zip(axes, trained_models.items()):
    preds = model.predict(X_test_raw)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9)

plt.suptitle("Confusion Matrices — Test Set", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("figures/confusion_matrices.pdf", bbox_inches="tight")
plt.savefig("figures/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/confusion_matrices.pdf", flush=True)

# ---------------------------------------------------------------------------
# 10. SHAP ANALYSIS  (tree-based black-box models)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 5 — SHAP Analysis", flush=True)
print("=" * 60, flush=True)

BLACKBOX_MODELS = ["Random Forest", "Gradient Boosting", "XGBoost"]
shap_global_importance = {}   # name → sorted {feature: mean_abs_shap}

for name in BLACKBOX_MODELS:
    model = trained_models[name]
    print(f"  Computing SHAP values for {name}...", flush=True)

    explainer = shap.TreeExplainer(model)
    # Use test set for explanations — reflects real inference
    shap_vals = explainer.shap_values(X_test_raw)

    # shap_values() return shape differs by SHAP version / model type:
    # list of 2D arrays (old SHAP), 3D array (new SHAP), or plain 2D (XGBoost)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]                  # take positive class
    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]            # slice out positive class
    else:
        sv = shap_vals                     # already (n_samples, n_features)

    # Ensure sv is 2D float
    sv = np.array(sv, dtype=float)
    if sv.ndim != 2:
        sv = sv.reshape(len(X_test_raw), -1)

    # Global feature importance: mean |SHAP| — guaranteed scalar per feature
    mean_abs = np.abs(sv).mean(axis=0).tolist()
    importance = dict(zip(feature_names, mean_abs))
    importance_sorted = dict(
        sorted(importance.items(), key=lambda x: float(x[1]), reverse=True)
    )
    shap_global_importance[name] = importance_sorted

    # ── SHAP Summary (Beeswarm) plot ──
    fig_name = name.lower().replace(" ", "_")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test_raw,           # sv is now guaranteed 2D (n_samples, n_features)
        feature_names=feature_names,
        show=False, max_display=15
    )
    plt.title(f"SHAP Summary — {name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/shap_summary_{fig_name}.pdf", bbox_inches="tight")
    plt.savefig(f"figures/shap_summary_{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] figures/shap_summary_{fig_name}.pdf")

    # ── SHAP Bar plot (top 15) ──
    top_feats = list(importance_sorted.keys())[:15]
    top_vals  = [importance_sorted[f] for f in top_feats]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_feats[::-1], top_vals[::-1], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Top 15 Features — {name} (SHAP)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/shap_bar_{fig_name}.pdf", bbox_inches="tight")
    plt.savefig(f"figures/shap_bar_{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Local SHAP: waterfall for first misclassified bad-credit instance ──
    preds = trained_models[name].predict(X_test_raw)
    bad_misclassified = np.where((y_test == 1) & (preds == 0))[0]
    if len(bad_misclassified) > 0:
        try:
            idx = bad_misclassified[0]
            # Use pre-computed sv row for the local explanation (version-safe)
            local_vals = sv[idx].astype(float)
            base_val   = float(np.mean(explainer.expected_value)
                               if hasattr(explainer.expected_value, "__len__")
                               else explainer.expected_value)
            expl = shap.Explanation(
                values=local_vals,
                base_values=base_val,
                data=X_test_raw[idx],
                feature_names=feature_names
            )
            plt.figure()
            shap.waterfall_plot(expl, max_display=15, show=False)
            plt.title(f"SHAP Local — {name} (misclassified bad-credit)", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"figures/shap_local_{fig_name}.pdf", bbox_inches="tight")
            plt.savefig(f"figures/shap_local_{fig_name}.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"    [Warning] Local SHAP waterfall skipped: {e}", flush=True)

# ---------------------------------------------------------------------------
# 11. LIME ANALYSIS  (local explanations on test set)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 6 — LIME Analysis", flush=True)
print("=" * 60, flush=True)

lime_global_importance = {}   # name → {feature: mean_abs_weight}

# Build explainer from the SMOTE training data
lime_explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["Good Credit", "Bad Credit"],
    mode="classification",
    discretize_continuous=True,
    random_state=RANDOM_STATE
)

N_LIME_SAMPLES = min(50, len(X_test_raw))   # explain first 50 test instances

for name in BLACKBOX_MODELS:
    model = trained_models[name]
    print(f"  Computing LIME explanations for {name} ({N_LIME_SAMPLES} instances)...", flush=True)

    aggregated = {f: [] for f in feature_names}

    for i in range(N_LIME_SAMPLES):
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N_LIME_SAMPLES} instances explained...", flush=True)
        exp = lime_explainer.explain_instance(
            X_test_raw[i],
            model.predict_proba,
            num_features=len(feature_names),
            num_samples=500
        )
        exp_map = dict(exp.as_list())
        for feat in feature_names:
            # LIME feature names include bin ranges; match by substring
            matched = [v for k, v in exp_map.items() if feat in k]
            aggregated[feat].append(matched[0] if matched else 0.0)

    mean_abs_lime = {f: np.mean(np.abs(vals)) for f, vals in aggregated.items()}
    lime_sorted = dict(sorted(mean_abs_lime.items(), key=lambda x: x[1], reverse=True))
    lime_global_importance[name] = lime_sorted

    # ── LIME Bar plot (top 15) ──
    fig_name = name.lower().replace(" ", "_")
    top_feats = list(lime_sorted.keys())[:15]
    top_vals  = [lime_sorted[f] for f in top_feats]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_feats[::-1], top_vals[::-1], color="#ff7f0e")
    ax.set_xlabel("Mean |LIME Weight|", fontsize=11)
    ax.set_title(f"Top 15 Features — {name} (LIME)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/lime_bar_{fig_name}.pdf", bbox_inches="tight")
    plt.savefig(f"figures/lime_bar_{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] figures/lime_bar_{fig_name}.pdf", flush=True)

# ---------------------------------------------------------------------------
# 12. SHAP vs LIME CONSISTENCY CHECK (Spearman rank correlation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 7 — SHAP vs LIME Feature Rank Consistency", flush=True)
print("=" * 60, flush=True)

consistency_rows = []
for name in BLACKBOX_MODELS:
    shap_imp = shap_global_importance[name]
    lime_imp = lime_global_importance[name]
    common   = [f for f in feature_names if f in shap_imp and f in lime_imp]

    shap_ranks = pd.Series(shap_imp).reindex(common).rank(ascending=False)
    lime_ranks = pd.Series(lime_imp).reindex(common).rank(ascending=False)
    rho, p = spearmanr(shap_ranks, lime_ranks)

    consistency_rows.append({
        "Model": name,
        "Spearman rho": round(rho, 4),
        "p-value": round(p, 4),
        "Agreement": "High" if rho > 0.7 else ("Moderate" if rho > 0.4 else "Low")
    })
    print(f"  {name:<25}  Spearman rho = {rho:.4f}  (p={p:.4f})")

consistency_df = pd.DataFrame(consistency_rows)
consistency_df.to_csv("results/shap_lime_consistency.csv", index=False)
with open("latex_exports/shap_lime_consistency.tex", "w") as f:
    f.write(consistency_df.to_latex(index=False, float_format="%.4f",
                                    caption="SHAP vs LIME Feature Rank Consistency (Spearman)",
                                    label="tab:shap_lime"))
print("[Saved] results/shap_lime_consistency.csv")

# ── Side-by-side top-10 comparison plot for best black-box model ──
best_bb_model = max(BLACKBOX_MODELS,
                    key=lambda m: np.mean(cv_results[m]["auc"]))
shap_top10 = list(shap_global_importance[best_bb_model].keys())[:10]
lime_top10 = list(lime_global_importance[best_bb_model].keys())[:10]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(shap_top10[::-1],
             [shap_global_importance[best_bb_model][f] for f in shap_top10[::-1]],
             color="#1f77b4")
axes[0].set_title(f"SHAP Top 10 — {best_bb_model}", fontsize=11)
axes[0].set_xlabel("Mean |SHAP Value|")

axes[1].barh(lime_top10[::-1],
             [lime_global_importance[best_bb_model][f] for f in lime_top10[::-1]],
             color="#ff7f0e")
axes[1].set_title(f"LIME Top 10 — {best_bb_model}", fontsize=11)
axes[1].set_xlabel("Mean |LIME Weight|")

plt.suptitle(f"SHAP vs LIME Feature Importance — {best_bb_model}", fontsize=12)
plt.tight_layout()
plt.savefig("figures/shap_vs_lime_comparison.pdf", bbox_inches="tight")
plt.savefig("figures/shap_vs_lime_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/shap_vs_lime_comparison.pdf")

# ---------------------------------------------------------------------------
# 13. DiCE COUNTERFACTUAL EXPLANATIONS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 8 — DiCE Counterfactual Explanations", flush=True)
print("=" * 60, flush=True)

# Use best black-box model for counterfactuals
cf_model = trained_models[best_bb_model]

# Build a DataFrame of test instances with feature names
X_test_df = pd.DataFrame(X_test_raw, columns=feature_names)
X_test_df["target"] = y_test

# DiCE needs a data object
continuous_features = feature_names   # all scaled features treated as continuous

dice_data = dice_ml.Data(
    dataframe=X_test_df,
    continuous_features=continuous_features,
    outcome_name="target"
)
dice_model = dice_ml.Model(model=cf_model, backend="sklearn")
exp_dice = dice_ml.Dice(dice_data, dice_model, method="random")

# Generate CFs for the first 5 bad-credit test instances
bad_credit_idx = np.where(y_test == 1)[0][:5]
all_cfs = []

for idx in bad_credit_idx:
    query = X_test_df.iloc[idx].drop("target").to_dict()
    try:
        cf_result = exp_dice.generate_counterfactuals(
            query_instances=pd.DataFrame([query]),
            total_CFs=3,
            desired_class="opposite",
            verbose=False
        )
        # Extract CF DataFrame (works across DiCE versions)
        try:
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
        except AttributeError:
            cf_df = cf_result.visualize_as_dataframe(show_only_changes=True)

        if cf_df is not None and len(cf_df) > 0:
            cf_df["original_instance"] = idx
            cf_df["original_class"] = "Bad Credit"
            cf_df["cf_class"] = "Good Credit"
            all_cfs.append(cf_df)
            print(f"  Instance {idx}: {len(cf_df)} counterfactual(s) generated")
    except Exception as e:
        print(f"  Instance {idx}: DiCE error — {e}")

if all_cfs:
    cf_combined = pd.concat(all_cfs, ignore_index=True)
    cf_combined.to_csv("results/counterfactuals.csv", index=False)
    # Save a concise LaTeX table (top features only)
    top_cols = (["original_instance", "original_class", "cf_class"]
                + list(shap_global_importance[best_bb_model].keys())[:5])
    top_cols = [c for c in top_cols if c in cf_combined.columns]
    with open("latex_exports/counterfactuals.tex", "w") as f:
        f.write(cf_combined[top_cols].head(9).to_latex(
            index=False, float_format="%.3f",
            caption=f"DiCE Counterfactual Explanations — {best_bb_model}",
            label="tab:counterfactuals"
        ))
    print(f"[Saved] results/counterfactuals.csv  ({len(cf_combined)} rows)")
else:
    print("  No counterfactuals generated.")

# ---------------------------------------------------------------------------
# 14. FAIRNESS AUDIT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 9 — Fairness Audit", flush=True)
print("=" * 60, flush=True)

fairness_rows = []
for name, model in trained_models.items():
    preds = model.predict(X_test_raw)
    dpd = demographic_parity_difference(y_test, preds,
                                         sensitive_features=prot_test)
    eod = equalized_odds_difference(y_test, preds,
                                     sensitive_features=prot_test)
    fairness_rows.append({
        "Model": name,
        "Dem. Parity Diff.": round(dpd, 4),
        "Equal. Odds Diff.": round(eod, 4),
        "Fair (DPD<0.1)": "Yes" if abs(dpd) < 0.1 else "No",
        "Fair (EOD<0.1)": "Yes" if abs(eod) < 0.1 else "No",
    })
    print(f"  {name:<25}  DPD={dpd:.4f}  EOD={eod:.4f}")

fairness_df = pd.DataFrame(fairness_rows)
fairness_df.to_csv("results/fairness_metrics.csv", index=False)
with open("latex_exports/fairness_metrics.tex", "w") as f:
    f.write(fairness_df.to_latex(index=False, float_format="%.4f",
                                  caption="Fairness Metrics Across All Models",
                                  label="tab:fairness"))
print("[Saved] results/fairness_metrics.csv  &  latex_exports/fairness_metrics.tex")

# ── Fairness bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
model_names = fairness_df["Model"].tolist()

axes[0].bar(model_names, fairness_df["Dem. Parity Diff."].abs(),
            color=["#2ca02c" if abs(v) < 0.1 else "#d62728"
                   for v in fairness_df["Dem. Parity Diff."]])
axes[0].axhline(0.1, color="red", linestyle="--", label="Threshold (0.1)")
axes[0].set_title("Demographic Parity Difference (|DPD|)", fontsize=11)
axes[0].set_ylabel("|DPD|")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend()

axes[1].bar(model_names, fairness_df["Equal. Odds Diff."].abs(),
            color=["#2ca02c" if abs(v) < 0.1 else "#d62728"
                   for v in fairness_df["Equal. Odds Diff."]])
axes[1].axhline(0.1, color="red", linestyle="--", label="Threshold (0.1)")
axes[1].set_title("Equalized Odds Difference (|EOD|)", fontsize=11)
axes[1].set_ylabel("|EOD|")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend()

plt.suptitle("Fairness Audit — All Models", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fairness_metrics.pdf", bbox_inches="tight")
plt.savefig("figures/fairness_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] figures/fairness_metrics.pdf")

# ---------------------------------------------------------------------------
# 15. FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("results/, figures/, latex_exports/ have been populated.")
