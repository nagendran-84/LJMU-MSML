# Model Evaluation & Explainability Notebook

This repository contains a Jupyter notebook that implements a model evaluation and explainability pipeline.

Files created:
- `notebook.ipynb` - The notebook with sections for imports, data prep, baseline models, SHAP/LIME, DiCE counterfactuals, fairness checks, and reporting.
- `requirements.txt` - Minimal Python package list required to run the notebook.

Quick start (Windows PowerShell):

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r "c:\Users\nagen\OneDrive\MSML\Source Code\requirements.txt"

# 3. Open the notebook in JupyterLab (optional)
pip install jupyterlab
jupyter lab "c:\Users\nagen\OneDrive\MSML\Source Code\notebook.ipynb"
```

Notes:
- Replace `data/german_credit.csv` with your dataset path or update the notebook accordingly.
- Some visualization features (LIME `show_in_notebook`, DiCE visualizations) require a notebook frontend to render.
- I can add a `results/` folder, unit tests, or example data if you'd like.
