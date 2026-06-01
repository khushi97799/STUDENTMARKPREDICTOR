"""
FAILSAFE ML Pipeline
- XGBoost classifier for student failure prediction
- SHAP explainability
- Personalised intervention generation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import joblib
import json
import os

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess(filepath: str) -> tuple:
    # Auto-detect separator
    df = pd.read_csv(filepath, sep=None, engine='python')
    
    print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    
    # Strip whitespace from column names (common issue)
    df.columns = df.columns.str.strip()
    
    # Ensure G3 exists
    if 'G3' not in df.columns:
        raise ValueError(f"G3 column not found. Available columns: {df.columns.tolist()}")
    
    # Target: fail = 1 if final grade G3 < 10
    df['fail'] = (df['G3'] < 10).astype(int)

    # Feature engineering
    df['avg_grade']      = (df['G1'] + df['G2']) / 2
    df['grade_trend']    = df['G2'] - df['G1']
    df['total_absences'] = df['absences']
    df['study_score']    = df['studytime'] * df['age']

    bool_map = {'yes': 1, 'no': 0}
    df['support_score'] = (
        df.get('famsup',   pd.Series(['no']*len(df))).map(bool_map).fillna(0) +
        df.get('schoolsup',pd.Series(['no']*len(df))).map(bool_map).fillna(0) +
        df.get('paid',     pd.Series(['no']*len(df))).map(bool_map).fillna(0)
    )

    cat_cols = ['school','sex','address','famsize','Pstatus',
                'Mjob','Fjob','reason','guardian','schoolsup',
                'famsup','paid','activities','nursery','higher',
                'internet','romantic']

    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Only use feature cols that actually exist in the file
    desired_features = [
        'school','sex','age','address','famsize','Pstatus',
        'Medu','Fedu','Mjob','Fjob','reason','guardian',
        'traveltime','studytime','failures','schoolsup','famsup',
        'paid','activities','nursery','higher','internet','romantic',
        'famrel','freetime','goout','Dalc','Walc','health',
        'absences','G1','G2',
        'avg_grade','grade_trend','total_absences','study_score','support_score'
    ]
    feature_cols = [f for f in desired_features if f in df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols]
    y = df['fail']
    print(f"Class distribution — Pass: {(y==0).sum()}, Fail: {(y==1).sum()}")

    return X, y, feature_cols, encoders, df


# ─────────────────────────────────────────────
# 2. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Class imbalance weight
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        'auc':    round(roc_auc_score(y_test, y_prob), 4),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return model, scaler, metrics, X_test_s, y_test


# ─────────────────────────────────────────────
# 3. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def get_shap_values(model, X_scaled, feature_names):
    """Return SHAP values for all samples."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    return explainer, shap_values


def explain_student(model, scaler, explainer, student_row: dict, feature_names: list):
    """
    Get SHAP explanation for a single student.
    Returns top positive (risk-increasing) and negative (protective) factors.
    """
    df_row   = pd.DataFrame([student_row])[feature_names]
    scaled   = scaler.transform(df_row)
    sv       = explainer.shap_values(scaled)[0]

    factors = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    risk_factors = [
        {'feature': f, 'impact': round(v, 4), 'direction': 'risk' if v > 0 else 'protective'}
        for f, v in factors
    ]

    prob = model.predict_proba(scaled)[0][1]
    return {
        'risk_probability': round(float(prob), 4),
        'risk_level': 'High' if prob > 0.65 else 'Medium' if prob > 0.35 else 'Low',
        'top_factors': risk_factors
    }


# ─────────────────────────────────────────────
# 4. INTERVENTION ENGINE
# ─────────────────────────────────────────────

INTERVENTION_RULES = {
    'absences': {
        'threshold': 10,
        'high': "Schedule an immediate attendance review meeting. Contact parents/guardians regarding absence pattern. Consider assigning a peer mentor.",
        'medium': "Send a proactive attendance reminder. Monitor weekly and alert if absences exceed 15."
    },
    'G1': {
        'threshold': 9,
        'high': "Enrol student in remedial/extra classes immediately. Assign subject-specific tutor.",
        'medium': "Provide additional practice materials. Weekly check-in with subject teacher."
    },
    'G2': {
        'threshold': 9,
        'high': "Emergency academic support — assign study buddy + faculty mentoring sessions 2×/week.",
        'medium': "Review G1→G2 drop. One-on-one academic counselling recommended."
    },
    'failures': {
        'threshold': 1,
        'high': "Refer to student welfare committee. Create a structured academic recovery plan.",
        'medium': "Discuss prior failures with student. Identify learning gaps and address specifically."
    },
    'studytime': {
        'threshold': 2,
        'low': "Introduce study habit workshops. Provide a personalised weekly study timetable."
    },
    'Walc': {
        'threshold': 3,
        'high': "Referral to student counsellor for lifestyle and wellness guidance."
    },
    'health': {
        'threshold': 3,
        'low': "Recommend health services check-up. Adjust academic load if health impacts performance."
    },
    'goout': {
        'threshold': 4,
        'high': "Counsellor session on time management and academic priorities."
    },
    'internet': {
        'threshold': 0,
        'low': "Provide access to college computer lab / library resources. Share digital learning materials."
    },
    'higher': {
        'threshold': 0,
        'low': "Career counselling session to motivate long-term academic goals."
    },
    'support_score': {
        'threshold': 1,
        'low': "Connect student with college support services. Inform about available tutoring and paid classes."
    },
    'famrel': {
        'threshold': 3,
        'low': "Refer to student welfare for family support assessment."
    },
    'avg_grade': {
        'threshold': 9,
        'high': "Comprehensive academic intervention: extra classes + faculty mentoring + counselling."
    },
    'grade_trend': {
        'threshold': -2,
        'low': "Declining grade trend detected. Immediate academic review and motivation session needed."
    },
}

def generate_interventions(student_row: dict, risk_level: str, top_factors: list) -> dict:
    """Generate personalised interventions based on student data and SHAP factors."""
    interventions = []
    categories    = set()

    # Rule-based interventions
    for feature, rules in INTERVENTION_RULES.items():
        val = student_row.get(feature)
        if val is None:
            continue
        threshold = rules['threshold']
        if 'high' in rules and float(val) > threshold:
            interventions.append({'category': _categorize(feature), 'action': rules['high'], 'priority': 'High'})
            categories.add(_categorize(feature))
        elif 'medium' in rules and float(val) > threshold * 0.7:
            interventions.append({'category': _categorize(feature), 'action': rules['medium'], 'priority': 'Medium'})
            categories.add(_categorize(feature))
        elif 'low' in rules and float(val) <= threshold:
            interventions.append({'category': _categorize(feature), 'action': rules['low'], 'priority': 'Medium'})
            categories.add(_categorize(feature))

    # SHAP-driven priority boost
    shap_features = {f['feature'] for f in top_factors if f['direction'] == 'risk'}
    for inv in interventions:
        if any(f in inv['action'].lower() for f in shap_features):
            inv['priority'] = 'High'

    # Deduplicate by category keeping highest priority
    seen = {}
    for inv in interventions:
        cat = inv['category']
        if cat not in seen or (inv['priority'] == 'High' and seen[cat]['priority'] != 'High'):
            seen[cat] = inv

    final = list(seen.values())

    # Always add a general note for high risk
    if risk_level == 'High' and not any(i['category'] == 'Faculty Follow-up' for i in final):
        final.append({
            'category': 'Faculty Follow-up',
            'action': 'Schedule bi-weekly one-on-one check-ins with the student. Log progress in system.',
            'priority': 'High'
        })

    return {
        'risk_level': risk_level,
        'total_interventions': len(final),
        'interventions': sorted(final, key=lambda x: 0 if x['priority']=='High' else 1)
    }


def _categorize(feature: str) -> str:
    mapping = {
        'absences':'Attendance', 'total_absences':'Attendance',
        'G1':'Academic Support', 'G2':'Academic Support',
        'avg_grade':'Academic Support', 'grade_trend':'Academic Support',
        'failures':'Academic Recovery',
        'studytime':'Study Habits', 'study_score':'Study Habits',
        'Walc':'Wellness & Counselling', 'Dalc':'Wellness & Counselling',
        'health':'Health Services', 'goout':'Lifestyle Counselling',
        'internet':'Resource Access',
        'higher':'Career Counselling',
        'support_score':'Support Services',
        'famrel':'Family Support',
    }
    return mapping.get(feature, 'General Support')


# ─────────────────────────────────────────────
# 5. SAVE / LOAD
# ─────────────────────────────────────────────

def save_artifacts(model, scaler, feature_names, metrics, path='ml/artifacts'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model,         f'{path}/model.pkl')
    joblib.dump(scaler,        f'{path}/scaler.pkl')
    joblib.dump(feature_names, f'{path}/features.pkl')
    with open(f'{path}/metrics.json', 'w') as f:
        # Convert numpy types for JSON serialisation
        def convert(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        json.dump(metrics, f, default=convert, indent=2)
    print(f"[FAILSAFE] Artifacts saved to {path}/")


def load_artifacts(path='ml/artifacts'):
    model    = joblib.load(f'{path}/model.pkl')
    scaler   = joblib.load(f'{path}/scaler.pkl')
    features = joblib.load(f'{path}/features.pkl')
    with open(f'{path}/metrics.json') as f:
        metrics = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, metrics, explainer


# ─────────────────────────────────────────────
# 6. BATCH PREDICTION (for uploaded CSV)
# ─────────────────────────────────────────────

def batch_predict(df_raw: pd.DataFrame, model, scaler, explainer, feature_names: list) -> list:
    """
    Process a raw uploaded dataframe.
    Returns list of per-student prediction dicts.
    """
    results = []

    # Feature engineer same as training
    if 'G1' in df_raw.columns and 'G2' in df_raw.columns:
        df_raw['avg_grade']      = (df_raw['G1'] + df_raw['G2']) / 2
        df_raw['grade_trend']    = df_raw['G2'] - df_raw['G1']
        df_raw['total_absences'] = df_raw['absences']
        df_raw['study_score']    = df_raw['studytime'] * df_raw['age']

        bool_map = {'yes':1,'no':0}
        df_raw['support_score'] = (
            df_raw.get('famsup', pd.Series(['no']*len(df_raw))).map(bool_map).fillna(0) +
            df_raw.get('schoolsup', pd.Series(['no']*len(df_raw))).map(bool_map).fillna(0) +
            df_raw.get('paid', pd.Series(['no']*len(df_raw))).map(bool_map).fillna(0)
        )

    cat_cols = ['school','sex','address','famsize','Pstatus',
                'Mjob','Fjob','reason','guardian','schoolsup',
                'famsup','paid','activities','nursery','higher',
                'internet','romantic']
    for col in cat_cols:
        if col in df_raw.columns:
            le = LabelEncoder()
            df_raw[col] = le.fit_transform(df_raw[col].astype(str))

    for idx, row in df_raw.iterrows():
        try:
            row_dict = row.to_dict()
            available = {f: row_dict.get(f, 0) for f in feature_names}
            exp = explain_student(model, scaler, explainer, available, feature_names)
            interventions = generate_interventions(row_dict, exp['risk_level'], exp['top_factors'])

            student_name = row_dict.get('name', f'Student_{idx+1}')
            results.append({
                'id':              idx + 1,
                'name':            student_name,
                'risk_probability': exp['risk_probability'],
                'risk_level':      exp['risk_level'],
                'top_factors':     exp['top_factors'],
                'interventions':   interventions['interventions'],
                'grades':          {'G1': row_dict.get('G1',0), 'G2': row_dict.get('G2',0)},
                'absences':        row_dict.get('absences', 0),
                'studytime':       row_dict.get('studytime', 0),
                'failures':        row_dict.get('failures', 0),
            })
        except Exception as e:
            print(f"[FAILSAFE] Error processing student {idx}: {e}")

    return results


# ─────────────────────────────────────────────
# 7. TRAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'student-mat.csv'
    print(f"[FAILSAFE] Loading data from {filepath}...")
    X, y, features, encoders, df = load_and_preprocess(filepath)
    print(f"[FAILSAFE] Training on {len(X)} samples, {y.sum()} failures ({y.mean()*100:.1f}%)")
    model, scaler, metrics, X_test, y_test = train_model(X, y)
    print(f"[FAILSAFE] AUC: {metrics['auc']}")
    save_artifacts(model, scaler, features, metrics)
    print("[FAILSAFE] Training complete!")