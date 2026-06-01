"""
FAILSAFE FastAPI Backend
Endpoints: auth, upload/predict, dashboard, interventions
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import json, os, io, uuid
from datetime import datetime, timedelta
from passlib.context import CryptContext
import jwt

# ─── App Setup ────────────────────────────────
app = FastAPI(title="FAILSAFE API", version="1.0.0")
# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "failsafe_secret_2024_xgboost")
ALGORITHM  = "HS256"
TOKEN_EXP_HOURS = 24

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer()

# ─── In-memory store (replace with PostgreSQL) ─
USERS_DB = {
    "faculty@college.edu": {
        "id": "u1",
        "name": "Dr. Priya Sharma",
        "role": "faculty",
        "department": "Computer Science",
        "hashed_password": pwd_ctx.hash("faculty123")
    },
    "hod@college.edu": {
        "id": "u2",
        "name": "Prof. Rajesh Kumar",
        "role": "hod",
        "department": "Computer Science",
        "hashed_password": pwd_ctx.hash("hod123")
    }
}

SESSIONS = {}          # token → user_id
PREDICTIONS_STORE = {} # upload_id → list of predictions
INTERVENTIONS_LOG = {} # student_id → list of applied interventions

# ─── ML Model (lazy load) ──────────────────────
_model    = None
_scaler   = None
_features = None
_metrics  = None
_explainer = None

def get_model():
    global _model, _scaler, _features, _metrics, _explainer
    if _model is None:
        try:
            import sys; sys.path.insert(0, '.')
            from ml.model import load_artifacts
            _model, _scaler, _features, _metrics, _explainer = load_artifacts('ml/artifacts')
            print("[FAILSAFE] Model loaded from artifacts")
        except Exception as e:
            print(f"[FAILSAFE] Model not found: {e}. Using mock predictions.")
    return _model, _scaler, _features, _metrics, _explainer


# ─── Auth ─────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXP_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        data = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        uid  = data.get("sub")
        user = next((u for u in USERS_DB.values() if u["id"] == uid), None)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = USERS_DB.get(req.email)
    if not user or not pwd_ctx.verify(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"])
    return {
        "token": token,
        "user": {
            "id":         user["id"],
            "name":       user["name"],
            "role":       user["role"],
            "department": user["department"],
            "email":      req.email
        }
    }

@app.get("/api/auth/me")
def me(user=Depends(get_current_user)):
    return user


# ─── Upload & Predict ─────────────────────────

@app.post("/api/predict/upload")
async def upload_csv(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine='python')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    model, scaler, features, metrics, explainer = get_model()

    if model is not None:
        import sys; sys.path.insert(0, '.')
        from ml.model import batch_predict
        predictions = batch_predict(df, model, scaler, explainer, features)
    else:
        # Mock predictions for demo without trained model
        predictions = _mock_predictions(df)

    upload_id = str(uuid.uuid4())
    PREDICTIONS_STORE[upload_id] = predictions

    summary = {
        'upload_id':      upload_id,
        'total_students': len(predictions),
        'high_risk':      sum(1 for p in predictions if p['risk_level'] == 'High'),
        'medium_risk':    sum(1 for p in predictions if p['risk_level'] == 'Medium'),
        'low_risk':       sum(1 for p in predictions if p['risk_level'] == 'Low'),
        'avg_risk':       round(np.mean([p['risk_probability'] for p in predictions]), 3),
        'predictions':    predictions,
        'uploaded_at':    datetime.utcnow().isoformat(),
        'filename':       file.filename
    }
    return summary


@app.get("/api/predict/{upload_id}")
def get_predictions(upload_id: str, user=Depends(get_current_user)):
    data = PREDICTIONS_STORE.get(upload_id)
    if not data:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"upload_id": upload_id, "predictions": data}


@app.get("/api/predict/{upload_id}/student/{student_id}")
def get_student_detail(upload_id: str, student_id: int, user=Depends(get_current_user)):
    preds = PREDICTIONS_STORE.get(upload_id, [])
    student = next((p for p in preds if p['id'] == student_id), None)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


# ─── Interventions ────────────────────────────

class InterventionApply(BaseModel):
    upload_id:  str
    student_id: int
    category:   str
    action:     str
    note:       Optional[str] = ""

@app.post("/api/interventions/apply")
def apply_intervention(req: InterventionApply, user=Depends(get_current_user)):
    key = f"{req.upload_id}_{req.student_id}"
    if key not in INTERVENTIONS_LOG:
        INTERVENTIONS_LOG[key] = []
    entry = {
        "id":          str(uuid.uuid4()),
        "category":    req.category,
        "action":      req.action,
        "note":        req.note,
        "applied_by":  user["name"],
        "applied_at":  datetime.utcnow().isoformat(),
        "status":      "Applied"
    }
    INTERVENTIONS_LOG[key].append(entry)
    return {"success": True, "intervention": entry}

@app.get("/api/interventions/{upload_id}/{student_id}")
def get_student_interventions(upload_id: str, student_id: int, user=Depends(get_current_user)):
    key = f"{upload_id}_{student_id}"
    return {"interventions": INTERVENTIONS_LOG.get(key, [])}


# ─── Dashboard ────────────────────────────────

@app.get("/api/dashboard/stats")
def dashboard_stats(user=Depends(get_current_user)):
    all_preds = [p for batch in PREDICTIONS_STORE.values() for p in batch]
    if not all_preds:
        return _mock_dashboard_stats()

    risk_dist  = {'High': 0, 'Medium': 0, 'Low': 0}
    for p in all_preds:
        risk_dist[p['risk_level']] += 1

    # Feature importance aggregation
    factor_counts = {}
    for p in all_preds:
        for f in p.get('top_factors', []):
            if f['direction'] == 'risk':
                factor_counts[f['feature']] = factor_counts.get(f['feature'], 0) + 1

    top_risk_factors = sorted(factor_counts.items(), key=lambda x: -x[1])[:8]

    return {
        "total_students":    len(all_preds),
        "high_risk":         risk_dist['High'],
        "medium_risk":       risk_dist['Medium'],
        "low_risk":          risk_dist['Low'],
        "avg_risk":          round(np.mean([p['risk_probability'] for p in all_preds]), 3),
        "top_risk_factors":  [{"feature": k, "count": v} for k, v in top_risk_factors],
        "risk_distribution": risk_dist,
        "interventions_applied": sum(len(v) for v in INTERVENTIONS_LOG.values()),
        "uploads_processed": len(PREDICTIONS_STORE)
    }


@app.get("/api/dashboard/model-metrics")
def model_metrics(user=Depends(get_current_user)):
    _, _, _, metrics, _ = get_model()
    if metrics:
        return metrics
    return _mock_model_metrics()


# ─── Mock data helpers ────────────────────────

def _mock_predictions(df: pd.DataFrame) -> list:
    import random
    results = []
    names = df.get('name', pd.Series([f'Student {i}' for i in range(len(df))])) if 'name' in df.columns else [f'Student {i+1}' for i in range(len(df))]
    for i, row in df.iterrows():
        prob = random.uniform(0.05, 0.95)
        risk = 'High' if prob > 0.65 else 'Medium' if prob > 0.35 else 'Low'
        results.append({
            'id': i + 1,
            'name': names[i] if isinstance(names, list) else names.iloc[i],
            'risk_probability': round(prob, 4),
            'risk_level': risk,
            'top_factors': [
                {'feature': 'absences',    'impact': round(random.uniform(0.1,0.5),3), 'direction':'risk'},
                {'feature': 'G2',          'impact': round(random.uniform(-0.4,-0.1),3),'direction':'protective'},
                {'feature': 'failures',    'impact': round(random.uniform(0.05,0.3),3),'direction':'risk'},
                {'feature': 'studytime',   'impact': round(random.uniform(-0.3,-0.05),3),'direction':'protective'},
                {'feature': 'avg_grade',   'impact': round(random.uniform(-0.35,-0.05),3),'direction':'protective'},
            ],
            'interventions': [
                {'category':'Attendance',       'action':'Schedule attendance review.','priority':'High' if prob>0.65 else 'Medium'},
                {'category':'Academic Support', 'action':'Assign subject tutor.',      'priority':'Medium'},
            ],
            'grades':    {'G1': int(row.get('G1',8)), 'G2': int(row.get('G2',7))},
            'absences':  int(row.get('absences',5)),
            'studytime': int(row.get('studytime',2)),
            'failures':  int(row.get('failures',0)),
        })
    return results


def _mock_dashboard_stats():
    return {
        "total_students": 0,
        "high_risk": 0, "medium_risk": 0, "low_risk": 0,
        "avg_risk": 0,
        "top_risk_factors": [
            {"feature":"absences","count":45},{"feature":"G2","count":38},
            {"feature":"failures","count":31},{"feature":"studytime","count":27},
            {"feature":"avg_grade","count":22},{"feature":"Walc","count":18},
            {"feature":"health","count":15},{"feature":"goout","count":12}
        ],
        "risk_distribution": {"High":0,"Medium":0,"Low":0},
        "interventions_applied": 0,
        "uploads_processed": 0
    }


def _mock_model_metrics():
    return {
        "auc": 0.89,
        "report": {
            "0": {"precision":0.91,"recall":0.93,"f1-score":0.92,"support":80},
            "1": {"precision":0.84,"recall":0.79,"f1-score":0.81,"support":40},
            "accuracy": 0.88
        },
        "confusion_matrix": [[74,6],[9,31]]
    }


# ─── Health ───────────────────────────────────
@app.get("/api/health")
def health():
    m, *_ = get_model()
    return {"status": "ok", "model_loaded": m is not None, "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)