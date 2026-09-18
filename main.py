from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import json
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="StressShield ML API",
    description="AI stress prediction for CAPF personnel",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Load ──────────────────────────────────
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/features.json', 'r') as f:
    FEATURES = json.load(f)

RISK_LABELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
print("✅ Model loaded successfully")

# ── Request Model ───────────────────────────────
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

class PredictResponse(BaseModel):
    personnel_id: str
    score: float
    risk_level: str
    top_factors: List[str]
    recommendation_hints: List[str]
    is_fallback: bool = False

# ── Endpoints ───────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "1.0.0",
        "features_count": len(FEATURES)
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # DataFrame banao — feature names ke saath
        df = pd.DataFrame([{
            'deployment_months':          request.deployment_months,
            'days_since_last_leave':      request.days_since_last_leave,
            'leave_denial_rate_6mo':      request.leave_denial_rate_6mo,
            'avg_weekly_duty_hours':      request.avg_weekly_duty_hours,
            'night_shift_ratio':          request.night_shift_ratio,
            'transfer_count_1yr':         request.transfer_count_1yr,
            'mood_score_avg':             request.mood_score_avg,
            'sleep_hours_avg':            request.sleep_hours_avg,
            'workload_rating_avg':        request.workload_rating_avg,
            'stress_self_rating_avg':     request.stress_self_rating_avg,
            'days_since_last_assessment': request.days_since_last_assessment,
            'years_of_service':           request.years_of_service
        }])[FEATURES]

        # Scale
        scaled = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=FEATURES)

        # Predict
        proba = model.predict_proba(scaled_df)[0]
        risk_class = int(np.argmax(proba))
        risk_level = RISK_LABELS[risk_class]

        # Score 0-100
        score = round(float(
            proba[0] * 10 +
            proba[1] * 35 +
            proba[2] * 70 +
            proba[3] * 100
        ), 1)

        # Top factors — simple version (no SHAP)
        feature_vals = {
            'deployment_months':          request.deployment_months,
            'days_since_last_leave':      request.days_since_last_leave,
            'leave_denial_rate_6mo':      request.leave_denial_rate_6mo,
            'avg_weekly_duty_hours':      request.avg_weekly_duty_hours,
            'night_shift_ratio':          request.night_shift_ratio,
            'transfer_count_1yr':         request.transfer_count_1yr,
            'mood_score_avg':             request.mood_score_avg,
            'sleep_hours_avg':            request.sleep_hours_avg,
            'workload_rating_avg':        request.workload_rating_avg,
            'stress_self_rating_avg':     request.stress_self_rating_avg,
            'days_since_last_assessment': request.days_since_last_assessment,
            'years_of_service':           request.years_of_service
        }

        # Risk rules — top factors
        top_factors = []
        if request.deployment_months > 6:
            top_factors.append(
                f"Deployment Duration: {request.deployment_months} months (high risk)"
            )
        if request.days_since_last_leave > 90:
            top_factors.append(
                f"Days Since Last Leave: {int(request.days_since_last_leave)} days"
            )
        if request.leave_denial_rate_6mo > 0.5:
            top_factors.append(
                f"Leave Denial Rate: {int(request.leave_denial_rate_6mo*100)}% denials"
            )
        if request.avg_weekly_duty_hours > 55:
            top_factors.append(
                f"Avg Duty Hours: {request.avg_weekly_duty_hours} hrs/week"
            )
        if request.mood_score_avg < 4:
            top_factors.append(
                f"Low Mood Score: {request.mood_score_avg}/10"
            )
        if request.sleep_hours_avg < 5:
            top_factors.append(
                f"Low Sleep: {request.sleep_hours_avg} hrs/night"
            )

        if not top_factors:
            top_factors = ["All indicators within normal range"]

        top_factors = top_factors[:3]

        # Recommendations
        hints = []
        if request.deployment_months > 8:
            hints.append("rotation_priority")
        if request.days_since_last_leave > 90:
            hints.append("immediate_leave")
        if request.leave_denial_rate_6mo > 0.5:
            hints.append("review_leave_applications")
        if request.mood_score_avg < 4:
            hints.append("counseling_recommended")
        if request.avg_weekly_duty_hours > 60:
            hints.append("workload_reduction")
        if risk_level == "CRITICAL":
            hints.append("immediate_welfare_intervention")
        if not hints:
            hints = ["regular_welfare_checkin"]

        return PredictResponse(
            personnel_id=request.personnel_id,
            score=score,
            risk_level=risk_level,
            top_factors=top_factors,
            recommendation_hints=hints,
            is_fallback=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))