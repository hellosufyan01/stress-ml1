from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle, json
import numpy as np
import pandas as pd

app = FastAPI(
    title="StressShield ML API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + config
with open('model/model.pkl','rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('model/config.json','r') as f:
    config = json.load(f)

FEATURES        = config['features']
THRESH_HIGH     = config['thresh_high']
THRESH_CRITICAL = config['thresh_critical']
RISK_LABELS     = ['LOW','MEDIUM','HIGH','CRITICAL']

print(f"✅ Model v{config['model_version']} loaded")
print(f"Sensitivity: {config['sensitivity']*100:.1f}%")
print(f"ROC AUC: {config['roc_auc']:.4f}")

class PredictRequest(BaseModel):
    personnel_id: str
    deployment_months: float
    days_since_last_leave: float
    leave_denial_rate_6mo: float
    avg_weekly_duty_hours: float
    night_shift_ratio: float
    transfer_count_1yr: float
    mood_score_avg: float
    sleep_hours_avg: float
    workload_rating_avg: float
    stress_self_rating_avg: float
    days_since_last_assessment: float
    years_of_service: float
    composite_stress: float = 0.0
    rest_deficit: float = 0.0
    leave_pressure: float = 0.0
    risk_multiplier: float = 1.0

class PredictResponse(BaseModel):
    personnel_id: str
    score: float
    risk_level: str
    top_factors: List[str]
    recommendation_hints: List[str]
    model_version: str
    sensitivity: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": config['model_version'],
        "sensitivity": f"{config['sensitivity']*100:.1f}%",
        "roc_auc": config['roc_auc']
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Auto compute composite features
        composite = (
            req.deployment_months * 0.25 +
            req.leave_denial_rate_6mo * 25 +
            (10 - req.mood_score_avg) * 2.5 +
            max(req.avg_weekly_duty_hours-48, 0) * 0.5
        )
        rest_def  = max(req.avg_weekly_duty_hours-48,0)
        leave_prs = (req.leave_denial_rate_6mo *
                     req.days_since_last_leave / 30)

        df = pd.DataFrame([{
            'deployment_months':
                req.deployment_months,
            'days_since_last_leave':
                req.days_since_last_leave,
            'leave_denial_rate_6mo':
                req.leave_denial_rate_6mo,
            'avg_weekly_duty_hours':
                req.avg_weekly_duty_hours,
            'night_shift_ratio':
                req.night_shift_ratio,
            'transfer_count_1yr':
                req.transfer_count_1yr,
            'mood_score_avg':
                req.mood_score_avg,
            'sleep_hours_avg':
                req.sleep_hours_avg,
            'workload_rating_avg':
                req.workload_rating_avg,
            'stress_self_rating_avg':
                req.stress_self_rating_avg,
            'days_since_last_assessment':
                req.days_since_last_assessment,
            'years_of_service':
                req.years_of_service,
            'composite_stress': composite,
            'rest_deficit':     rest_def,
            'leave_pressure':   leave_prs,
            'risk_multiplier':  req.risk_multiplier
        }])[FEATURES]

        scaled = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled,
                                  columns=FEATURES)
        proba = model.predict_proba(scaled_df)[0]

        # Threshold based prediction
        if proba[3] >= THRESH_CRITICAL:
            risk_class = 3
        elif proba[2] >= THRESH_HIGH:
            risk_class = 2
        elif proba[1] >= 0.30:
            risk_class = 1
        else:
            risk_class = 0

        risk_level = RISK_LABELS[risk_class]
        score = round(float(
            proba[0]*5 + proba[1]*30 +
            proba[2]*65 + proba[3]*100
        ), 1)

        # Top factors
        factors = []
        if req.deployment_months > 6:
            factors.append(
                f"Deployment: {req.deployment_months}"
                f" months (threshold: 6)"
            )
        if req.days_since_last_leave > 90:
            factors.append(
                f"No leave since "
                f"{int(req.days_since_last_leave)} days"
            )
        if req.leave_denial_rate_6mo > 0.5:
            factors.append(
                f"Leave denied "
                f"{int(req.leave_denial_rate_6mo*100)}%"
                f" of time"
            )
        if req.avg_weekly_duty_hours > 55:
            factors.append(
                f"Duty hours: "
                f"{req.avg_weekly_duty_hours}"
                f" hrs/week (norm: 48)"
            )
        if req.mood_score_avg < 4:
            factors.append(
                f"Low mood score: "
                f"{req.mood_score_avg}/10"
            )
        if req.sleep_hours_avg < 5:
            factors.append(
                f"Poor sleep: "
                f"{req.sleep_hours_avg} hrs/night"
            )
        if not factors:
            factors = ["All indicators normal"]

        # Recommendations
        hints = []
        if req.deployment_months > 8:
            hints.append("rotation_priority")
        if req.days_since_last_leave > 90:
            hints.append("immediate_leave")
        if req.leave_denial_rate_6mo > 0.5:
            hints.append("review_leave")
        if req.mood_score_avg < 4:
            hints.append("counseling")
        if req.avg_weekly_duty_hours > 60:
            hints.append("workload_reduction")
        if risk_level == "CRITICAL":
            hints.append("immediate_intervention")
        if not hints:
            hints = ["regular_checkin"]

        return PredictResponse(
            personnel_id=req.personnel_id,
            score=score,
            risk_level=risk_level,
            top_factors=factors[:3],
            recommendation_hints=hints,
            model_version=config['model_version'],
            sensitivity=config['sensitivity']
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )