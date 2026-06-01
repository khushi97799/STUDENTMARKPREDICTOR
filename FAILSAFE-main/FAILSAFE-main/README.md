# FAILSAFE — Student Risk Intelligence Platform

> Early student failure detection using XGBoost + SHAP + personalised AI interventions.

## Live Demo
Login: `faculty@college.edu` / `faculty123`

## Quick Start
```bash
pip install -r requirements.txt
python ml/model.py student-mat.csv
python -m uvicorn backend.main:app --reload --port 8000
# Open http://localhost:8000
```
