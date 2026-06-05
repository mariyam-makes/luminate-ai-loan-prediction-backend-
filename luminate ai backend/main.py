import os
import json
import io
import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ReportLab Imports for Premium PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Initialize FastAPI App
app = FastAPI(
    title="Luminate Loan Eligibility & Risk Intelligence API",
    description="Enterprise decoupled underwriting backend exposing ML risk predictions, analytics, and automated PDF report downloads.",
    version="1.1.0"
)

# Enable CORS for Next.js (both local and Vercel deployments)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for clean cross-origin compatibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Directory and Asset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "model.joblib")
SCALER_PATH = os.path.join(ASSETS_DIR, "scaler.joblib")
META_PATH = os.path.join(ASSETS_DIR, "metadata.json")
DB_PATH = os.path.join(ASSETS_DIR, "predictions_db.json")

# Global Preprocessor & Model Variables
model = None
scaler = None
metadata = None
MOCK_MODE = False

@app.on_event("startup")
def load_assets():
    """
    On startup, load trained scikit-learn models and preprocessors.
    Gracefully fall back to Mock Mode if files are missing to ensure clean local runs.
    """
    global model, scaler, metadata, MOCK_MODE
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(META_PATH):
        print("[WARNING] Trained model assets not found in backend/assets/.")
        print("[MOCK] Running API in MOCK_MODE with dynamic rules-based ML classifier emulation.")
        MOCK_MODE = True
        return
        
    try:
        print("Loading serialized KNN model and preprocessors...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        print("All assets successfully loaded. ML Model is online.")
    except Exception as e:
        print(f"[ERROR] Error loading serialized models: {e}. Defaulting to MOCK_MODE.")
        MOCK_MODE = True


# --- API REQUEST / RESPONSE SCHEMAS ---

class ApplicantRequest(BaseModel):
    # Standard features matching the dataset and frontend forms
    Gender: Optional[str] = Field(default="Male", description="Gender: Male, Female")
    Married: Optional[str] = Field(default="Yes", description="Married: Yes, No")
    Dependents: Optional[str] = Field(default="0", description="Number of dependents: 0, 1, 2, 3+")
    Education: Optional[str] = Field(default="Graduate", description="Education level: Graduate, Not Graduate")
    Self_Employed: Optional[str] = Field(default="No", description="Self employed status: Yes, No")
    ApplicantIncome: Optional[float] = Field(default=None, description="Applicant monthly income (e.g. 5000)")
    CoapplicantIncome: Optional[float] = Field(default=0.0, description="Co-applicant monthly income (e.g. 1500)")
    LoanAmount: Optional[float] = Field(default=None, description="Loan amount in thousands (e.g. 120)")
    Loan_Amount_Term: Optional[float] = Field(default=360.0, description="Loan duration term in days/months (e.g. 360)")
    Credit_History: Optional[float] = Field(default=None, description="Credit history meets guidelines: 1.0 (Yes) or 0.0 (No)")
    Property_Area: Optional[str] = Field(default="Urban", description="Property Area: Rural, Semiurban, Urban")
    
    # Custom/Alternative metrics requested by user specification
    Income: Optional[float] = Field(default=None, description="Alternative field for Income")
    Credit_Score: Optional[float] = Field(default=None, description="Alternative field for Credit Score (300-850)")
    Employment_Duration: Optional[float] = Field(default=None, description="Employment duration in years")

class PredictionResponse(BaseModel):
    loan_status: str = Field(..., description="Approval status: Approved, Rejected")
    probability: float = Field(..., description="Probability of approval (0.0 to 1.0)")
    risk_score: float = Field(..., description="AI Risk Score mapping rejection probability (0 to 100)")
    risk_level: str = Field(..., description="Risk category: Low (0-40), Medium (41-70), High (71-100)")
    feature_impact: Dict[str, str] = Field(..., description="Explainer keys showing what factors influenced the risk score")
    suggestions: List[str] = Field(..., description="Actionable financial improvement suggestions for underwriting optimization")
    message: str = Field(..., description="Natural language explanation of the decision outcome")


# --- CORE ENGINES: SUGGESTIONS & FEATURE IMPACT ---

def calculate_feature_impact(req: ApplicantRequest, risk_score: float) -> Dict[str, str]:
    """
    Generate a simple rule-based feature impact breakdown explaining key input influences.
    """
    impacts = {}
    
    # Resolve values
    applicant_income = req.ApplicantIncome if req.ApplicantIncome is not None else (req.Income or 5000.0)
    coapplicant_income = req.CoapplicantIncome or 0.0
    total_income = applicant_income + coapplicant_income
    loan_amount = req.LoanAmount if req.LoanAmount is not None else 120.0
    
    credit_hist = req.Credit_History
    if credit_hist is None:
        credit_hist = 1.0 if (req.Credit_Score is not None and req.Credit_Score >= 650) else 0.0
        
    # Analyze Credit guidelines status
    if credit_hist == 0.0:
        impacts["Credit guidelines status"] = "Poor credit history status strongly inflated the underwriting rejection probability."
    else:
        impacts["Credit guidelines status"] = "Established credit history meets underwriting criteria, mitigating default risk."
        
    # Analyze Debt-to-Income / size
    annual_income = total_income * 12
    loan_actual = loan_amount * 1000
    if annual_income > 0:
        dti_ratio = loan_actual / annual_income
        if dti_ratio > 5.0:
            impacts["Debt-to-Income profile"] = f"Extremely high loan amount compared to income ({dti_ratio:.1f}x annual earnings) elevated risk."
        elif dti_ratio > 3.0:
            impacts["Debt-to-Income profile"] = f"Moderate loan size ({dti_ratio:.1f}x annual earnings) contributed to medium-risk indicators."
        else:
            impacts["Debt-to-Income profile"] = "Requested loan amount aligns well within safe debt-to-income bounds."
            
    # Employment stability influence
    if req.Employment_Duration is not None:
        if req.Employment_Duration < 2.0:
            impacts["Employment history"] = f"Short employment tenure ({req.Employment_Duration} years) indicates income volatility risk."
        else:
            impacts["Employment history"] = "Solid job longevity suggests robust and stable earnings."
            
    # Coapplicant safety net
    if coapplicant_income > 0:
        impacts["Income structure"] = "Presence of a co-applicant provides an additional debt repayment buffer."
        
    return impacts

def generate_improvement_suggestions(req: ApplicantRequest) -> List[str]:
    """
    Implement a rule-based logic engine to generate actionable suggestions to reduce loan default risk.
    """
    suggestions = []
    
    # Resolve values
    applicant_income = req.ApplicantIncome if req.ApplicantIncome is not None else (req.Income or 5000.0)
    coapplicant_income = req.CoapplicantIncome or 0.0
    total_income = applicant_income + coapplicant_income
    loan_amount = req.LoanAmount if req.LoanAmount is not None else 120.0
    
    credit_hist = req.Credit_History
    if credit_hist is None:
        credit_hist = 1.0 if (req.Credit_Score is not None and req.Credit_Score >= 650) else 0.0
        
    # Credit History Suggestion
    if credit_hist == 0.0 or (req.Credit_Score is not None and req.Credit_Score < 650):
        suggestions.append("Credit Score: Establish automated bill payments and settle outstanding debts to boost your credit profile.")
        suggestions.append("Credit Profile: Review your credit report for disputes or errors that might suppress your score.")
        
    # Income/DTI Ratio Suggestion
    annual_income = total_income * 12
    loan_actual = loan_amount * 1000
    if annual_income > 0:
        dti_ratio = loan_actual / annual_income
        if dti_ratio > 5.0:
            target_amount = int((annual_income * 4.0) / 1000)
            suggestions.append(f"Debt-to-Income: Apply for a lower loan amount (e.g., under ${target_amount}k) to align with a 4.0x income ratio.")
            
    # Co-applicant Suggestion
    if coapplicant_income == 0.0:
        suggestions.append("Borrowing Power: Apply with a co-applicant with a solid credit score to distribute risk and improve status.")
        
    # Employment Tenure Suggestion
    if req.Employment_Duration is not None and req.Employment_Duration < 2.0:
        suggestions.append("Employment Stability: Maintain your current role to establish at least 24 months of steady job tenure.")
        
    # Term adjustment Suggestion
    if req.Loan_Amount_Term is not None and req.Loan_Amount_Term < 180:
        suggestions.append("Payment Terms: Opt for a longer loan term (e.g. 360 days/months) to decrease the monthly installment stress.")
        
    # Self employment Suggestion
    if req.Self_Employed == "Yes":
        suggestions.append("Verification: Prepare audited profit/loss statements and two years of IRS tax filings to verify self-employed stability.")
        
    # Add a generic fallback if no issues found
    if not suggestions:
        suggestions.append("Maintain your strong financial status and keep credit card balances under 30% of their limits.")
        
    return suggestions


# --- DYNAMIC MOCK PREDICTOR ---

def run_mock_prediction(req: ApplicantRequest) -> tuple[float, float]:
    """
    Simulates ML classification outputs based on logical financial checks.
    Returns: (approval_probability, risk_score)
    """
    # Resolve inputs
    applicant_income = req.ApplicantIncome if req.ApplicantIncome is not None else (req.Income or 5000.0)
    coapplicant_income = req.CoapplicantIncome or 0.0
    total_income = applicant_income + coapplicant_income
    loan_amount = req.LoanAmount if req.LoanAmount is not None else 120.0
    
    credit_hist = req.Credit_History
    if credit_hist is None:
        credit_hist = 1.0 if (req.Credit_Score is not None and req.Credit_Score >= 650) else 0.0
        
    # Start with baseline risk score
    risk = 30.0
    
    # 1. Credit History Impact (strongest factor)
    if credit_hist == 0.0:
        risk += 50.0
    elif req.Credit_Score is not None:
        if req.Credit_Score < 580:
            risk += 45.0
        elif req.Credit_Score < 660:
            risk += 20.0
        elif req.Credit_Score > 750:
            risk -= 15.0
            
    # 2. Debt-to-Income / Loan Size Impact
    annual_income = total_income * 12
    loan_actual = loan_amount * 1000
    if annual_income > 0:
        dti = loan_actual / annual_income
        if dti > 6.0:
            risk += 30.0
        elif dti > 4.5:
            risk += 15.0
        elif dti < 2.0:
            risk -= 10.0
            
    # 3. Employment Duration Impact
    if req.Employment_Duration is not None:
        if req.Employment_Duration < 1.0:
            risk += 15.0
        elif req.Employment_Duration < 3.0:
            risk += 5.0
        elif req.Employment_Duration > 5.0:
            risk -= 8.0
            
    # 4. Other categorical factors
    if req.Education == "Not Graduate":
        risk += 5.0
    if req.Self_Employed == "Yes":
        risk += 5.0
    if req.Property_Area == "Rural":
        risk += 5.0
        
    # Clamp risk score between 5.0 and 98.0
    risk = max(5.0, min(98.0, risk))
    
    # Approval Probability is the inverse of Rejection Risk
    approval_proba = 1.0 - (risk / 100.0)
    
    return approval_proba, risk


# --- DATABASE LOGGER (ANALYTICS ENGINE) ---

def log_prediction_to_db(status: str, risk_level: str, risk_score: float):
    """
    Persist prediction entries to predictions_db.json to drive real-time analytics charts.
    Executes in a background thread to prevent endpoint latency.
    """
    try:
        # Determine month based on current local date
        current_month = datetime.datetime.now().strftime("%b")
        
        # Load or initialize database list
        records = []
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, "r") as f:
                    records = json.load(f)
            except Exception:
                records = []
                
        # Append new entry
        records.append({
            "month": current_month,
            "status": status,
            "risk_level": risk_level,
            "risk_score": float(risk_score)
        })
        
        # Write back safely
        os.makedirs(ASSETS_DIR, exist_ok=True)
        with open(DB_PATH, "w") as f:
            json.dump(records, f, indent=2)
            
    except Exception as e:
        print(f"[WARNING] Failed to log prediction entry to database: {e}")


def apply_business_rules(req: ApplicantRequest, approval_proba: float, risk_score: float, loan_status: str) -> tuple[float, float, str, Optional[str]]:
    """
    Applies strict hardcoded Business Rule Overrides to handle impossible/highly risky borrowing requests.
    """
    # Resolve values
    applicant_income = req.ApplicantIncome if req.ApplicantIncome is not None else (req.Income or 0.0)
    coapplicant_income = req.CoapplicantIncome or 0.0
    total_income = applicant_income + coapplicant_income
    loan_amount = req.LoanAmount if req.LoanAmount is not None else 0.0
    
    # 1. Income < 500 and loan > 10,000 (checked first for precedence)
    if total_income < 500.0 and loan_amount > 10.0:
        return 0.05, 95.0, "Rejected", "Automated Reject: Insufficient monthly income for requested loan amount."
        
    # 2. Loan exceeds 5x annual income
    annual_income = total_income * 12
    loan_amount_actual = loan_amount * 1000
    if loan_amount_actual > annual_income * 5:
        return 0.05, 95.0, "Rejected", "Automated Reject: Requested loan amount drastically exceeds the applicant's income threshold."
        
    return approval_proba, risk_score, loan_status, None


# --- PREPROCESSING & INFERENCE PIPELINE ---

def preprocess_and_predict(req: ApplicantRequest) -> tuple[float, float, str]:
    """
    Transforms and scales inputs, runs inference, and returns (approval_probability, risk_score, loan_status).
    """
    global MOCK_MODE, model, scaler, metadata
    
    # Auto-load assets if not initialized
    if not MOCK_MODE and (model is None or scaler is None or metadata is None):
        load_assets()
        
    # Check fallback mock mode
    if MOCK_MODE:
        approval_proba, risk_score = run_mock_prediction(req)
        loan_status = "Approved" if risk_score <= 50.0 else "Rejected"
        return approval_proba, risk_score, loan_status
        
    try:
        # Resolve metrics and fallbacks
        gender = req.Gender if req.Gender is not None else metadata["modes"]["Gender"]
        married = req.Married if req.Married is not None else metadata["modes"]["Married"]
        dependents = req.Dependents if req.Dependents is not None else metadata["modes"]["Dependents"]
        education = req.Education if req.Education is not None else metadata["modes"]["Education"]
        self_employed = req.Self_Employed if req.Self_Employed is not None else metadata["modes"]["Self_Employed"]
        
        # Income and guidelines fallback
        income_val = req.ApplicantIncome if req.ApplicantIncome is not None else (req.Income or metadata["medians"]["ApplicantIncome"])
        co_income_val = req.CoapplicantIncome if req.CoapplicantIncome is not None else metadata["medians"]["CoapplicantIncome"]
        loan_val = req.LoanAmount if req.LoanAmount is not None else metadata["medians"]["LoanAmount"]
        term_val = req.Loan_Amount_Term if req.Loan_Amount_Term is not None else metadata["medians"]["Loan_Amount_Term"]
        
        credit_history = req.Credit_History
        if credit_history is None:
            if req.Credit_Score is not None:
                credit_history = 1.0 if req.Credit_Score >= 650 else 0.0
            else:
                credit_history = metadata["medians"]["Credit_History"]
                
        property_area = req.Property_Area if req.Property_Area is not None else metadata["modes"]["Property_Area"]
        
        # Label encode categories matching training mapping
        def encode_category(val: str, field_name: str) -> int:
            mappings = metadata["label_mappings"][field_name]
            clean_val = str(val).strip()
            if clean_val in mappings:
                return mappings.index(clean_val)
            # Default fallback to modes index
            mode_val = metadata["modes"][field_name]
            return mappings.index(mode_val)
            
        gender_enc = encode_category(gender, "Gender")
        married_enc = encode_category(married, "Married")
        dependents_enc = encode_category(dependents, "Dependents")
        education_enc = encode_category(education, "Education")
        self_employed_enc = encode_category(self_employed, "Self_Employed")
        property_area_enc = encode_category(property_area, "Property_Area")
        
        # Build features array in strict column training order
        features_dict = {
            "Gender": gender_enc,
            "Married": married_enc,
            "Dependents": dependents_enc,
            "Education": education_enc,
            "Self_Employed": self_employed_enc,
            "ApplicantIncome": float(income_val),
            "CoapplicantIncome": float(co_income_val),
            "LoanAmount": float(loan_val),
            "Loan_Amount_Term": float(term_val),
            "Credit_History": float(credit_history),
            "Property_Area": property_area_enc
        }
        
        ordered_features = [features_dict[col] for col in metadata["feature_cols"]]
        features_df = pd.DataFrame([ordered_features], columns=metadata["feature_cols"])
        
        # Apply standard scaler transform
        scaled_features = scaler.transform(features_df)
        
        # Predict Probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        approval_proba = float(probabilities[1])  # Class 1 probability (Approved)
        rejection_proba = float(probabilities[0]) # Class 0 probability (Rejected)
        
        # Calculate Risk Score (0-100) based directly on rejection probability
        risk_score = rejection_proba * 100.0
        
        # Predict class label
        pred_label = int(model.predict(scaled_features)[0])
        loan_status = "Approved" if pred_label == 1 else "Rejected"
        
        return approval_proba, risk_score, loan_status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WARNING] Inference failed: {e}. Falling back to dynamic mock prediction.")
        approval_proba, risk_score = run_mock_prediction(req)
        loan_status = "Approved" if risk_score <= 50.0 else "Rejected"
        return approval_proba, risk_score, loan_status


# --- FASTAPI ENDPOINTS ---

@app.get("/")
def read_root():
    """
    Base server health and environment information check.
    """
    global MOCK_MODE, metadata
    if metadata is None and not MOCK_MODE:
        load_assets()
    return {
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "mock_mode": MOCK_MODE,
        "model_type": "KNN Classifier (Scikit-Learn)" if not MOCK_MODE else "Rules-based ML Classifier Emulator",
        "model_accuracy": metadata["accuracy"] if metadata else 0.7886
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_loan(request: ApplicantRequest, background_tasks: BackgroundTasks):
    """
    POST `/predict` evaluates the loan parameters and outputs underwriting intelligence:
    rejection probability-mapped Risk Score, Risk levels, features impact, and action suggestions.
    """
    try:
        # Run preprocess and predict pipeline
        approval_proba, risk_score, loan_status = preprocess_and_predict(request)
        
        # Apply business rules overrides
        approval_proba, risk_score, loan_status, override_msg = apply_business_rules(
            request, approval_proba, risk_score, loan_status
        )
        
        # Categorize risk score to Risk Levels
        if risk_score <= 40.0:
            risk_level = "Low"
        elif risk_score <= 70.0:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        # Calculate rule-based feature impacts and improvement recommendations
        feature_impact = calculate_feature_impact(request, risk_score)
        suggestions = generate_improvement_suggestions(request)
        
        # Build explanation message
        if override_msg:
            message = override_msg
        elif loan_status == "Approved":
            message = f"Congratulations! Your loan request is predicted to be approved with {approval_proba*100:.1f}% approval probability (AI Risk Score: {risk_score:.0f}/100 - {risk_level} Risk)."
        else:
            message = f"Sorry, based on our models your loan request is predicted to be rejected. Rejection risk probability is {risk_score:.1f}% ({risk_level} Risk)."
            
        # Log this run to the analytics database file asynchronously
        background_tasks.add_task(log_prediction_to_db, loan_status, risk_level, risk_score)
        
        return PredictionResponse(
            loan_status=loan_status,
            probability=approval_proba,
            risk_score=risk_score,
            risk_level=risk_level,
            feature_impact=feature_impact,
            suggestions=suggestions,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/analytics")
def get_analytics():
    """
    GET `/analytics` aggregates database metrics to feed frontend dashboard graphs.
    Provides total prediction counts, approvals vs rejections, risk distribution, and historical monthly trends.
    """
    if not os.path.exists(DB_PATH):
        return {
            "total_predictions": 0,
            "approval_ratio": 0.0,
            "approvals": 0,
            "rejections": 0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0},
            "monthly_trends": []
        }
        
    try:
        with open(DB_PATH, "r") as f:
            records = json.load(f)
            
        total = len(records)
        if total == 0:
            return {
                "total_predictions": 0,
                "approval_ratio": 0.0,
                "approvals": 0,
                "rejections": 0,
                "risk_distribution": {"Low": 0, "Medium": 0, "High": 0},
                "monthly_trends": []
            }
            
        approvals = sum(1 for r in records if r["status"] == "Approved")
        rejections = total - approvals
        approval_ratio = approvals / total
        
        risk_dist = {"Low": 0, "Medium": 0, "High": 0}
        for r in records:
            lvl = r.get("risk_level", "Low")
            risk_dist[lvl] = risk_dist.get(lvl, 0) + 1
            
        # Group monthly trends
        # Map months to compile orderly outputs
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_map = {}
        
        for r in records:
            m = r.get("month", "Jun")
            if m not in monthly_map:
                monthly_map[m] = {"Approved": 0, "Rejected": 0}
            status = r.get("status", "Approved")
            monthly_map[m][status] = monthly_map[m].get(status, 0) + 1
            
        trends = []
        for m in month_order:
            if m in monthly_map:
                trends.append({
                    "month": m,
                    "approvals": monthly_map[m]["Approved"],
                    "rejections": monthly_map[m]["Rejected"]
                })
                
        return {
            "total_predictions": total,
            "approval_ratio": round(approval_ratio, 3),
            "approvals": approvals,
            "rejections": rejections,
            "risk_distribution": risk_dist,
            "monthly_trends": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile analytics dashboard details: {str(e)}")


# --- PREMIUM PDF UNDERWRITING REPORT GENERATOR ---

def generate_pdf_report(req: ApplicantRequest, res: PredictionResponse) -> bytes:
    """
    Builds a beautifully styled corporate underwriting assessment report using ReportLab.
    Returns: In-memory PDF byte stream.
    """
    buffer = io.BytesIO()
    
    # Page setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    story = []
    
    # Custom styles definition
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#1E1B4B'),  # indigo-950
        spaceAfter=4,
        leading=28
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#4F46E5'),  # indigo-600
        spaceAfter=15,
        leading=12
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),  # slate-800
        spaceBefore=14,
        spaceAfter=8,
        leading=18
    )
    
    label_style = ParagraphStyle(
        'CellLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#475569'),  # slate-600
        leading=11
    )
    
    value_style = ParagraphStyle(
        'CellValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#0F172A'),  # slate-900
        leading=11
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6,
        leading=14
    )
    
    # Color coding risk configurations
    if res.risk_level == "Low":
        risk_bg = colors.HexColor('#E6F4EA')
        risk_text = colors.HexColor('#137333')
        risk_border = colors.HexColor('#34A853')
    elif res.risk_level == "Medium":
        risk_bg = colors.HexColor('#FEF7E0')
        risk_text = colors.HexColor('#B06000')
        risk_border = colors.HexColor('#FBBC04')
    else:
        risk_bg = colors.HexColor('#FCE8E6')
        risk_text = colors.HexColor('#C5221F')
        risk_border = colors.HexColor('#EA4335')
        
    decision_box_style = ParagraphStyle(
        'DecisionBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=risk_text,
        alignment=TA_CENTER,
        leading=20
    )
    
    risk_score_style = ParagraphStyle(
        'RiskScoreBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),
        alignment=TA_CENTER,
        leading=18
    )
    
    # 1. Header Section
    story.append(Paragraph("LUMINATE UNDERWRITING & RISK REPORT", title_style))
    story.append(Paragraph("Automated Credit Risk Intelligence & Eligibility Verification System", subtitle_style))
    story.append(Spacer(1, 5))
    
    # 2. Decision & AI Score Block
    decision_content = [
        [
            Paragraph(f"UNDERWRITING DECISION: {res.loan_status.upper()}", decision_box_style),
            Paragraph(f"AI RISK SCORE: {res.risk_score:.0f} / 100 ({res.risk_level.upper()} RISK)", risk_score_style)
        ]
    ]
    decision_table = Table(decision_content, colWidths=[260, 260])
    decision_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), risk_bg),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 15),
        ('BOTTOMPADDING', (0,0), (-1,-1), 15),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 2, risk_border),
    ]))
    story.append(decision_table)
    story.append(Spacer(1, 15))
    
    # 3. Applicant Profile Summary Section
    story.append(Paragraph("Applicant Financial Parameters", section_heading))
    
    # Resolve values for PDF table display
    inc = req.ApplicantIncome if req.ApplicantIncome is not None else (req.Income or 0.0)
    coinc = req.CoapplicantIncome or 0.0
    loan = req.LoanAmount if req.LoanAmount is not None else 0.0
    term = req.Loan_Amount_Term or 360.0
    credit_hist = req.Credit_History
    if credit_hist is None:
        credit_hist = 1.0 if (req.Credit_Score is not None and req.Credit_Score >= 650) else 0.0
        
    credit_str = "Meets Credit Guidelines" if credit_hist == 1.0 else "No / Poor Credit History"
    if req.Credit_Score is not None:
        credit_str += f" (Score: {req.Credit_Score:.0f})"
        
    emp_str = "Salaried" if req.Self_Employed == "No" else "Self-Employed"
    if req.Employment_Duration is not None:
        emp_str += f" ({req.Employment_Duration} years tenure)"
        
    summary_data = [
        [Paragraph("Applicant Monthly Income", label_style), Paragraph(f"${inc:,.2f}", value_style),
         Paragraph("Marital Status", label_style), Paragraph(str(req.Married), value_style)],
         
        [Paragraph("Co-Applicant Income", label_style), Paragraph(f"${coinc:,.2f}", value_style),
         Paragraph("Number of Dependents", label_style), Paragraph(str(req.Dependents), value_style)],
         
        [Paragraph("Requested Loan Amount", label_style), Paragraph(f"${loan:,.0f}k (${loan*1000:,.2f})", value_style),
         Paragraph("Education Level", label_style), Paragraph(str(req.Education), value_style)],
         
        [Paragraph("Loan Term Duration", label_style), Paragraph(f"{term:.0f} days", value_style),
         Paragraph("Property Area", label_style), Paragraph(str(req.Property_Area), value_style)],
         
        [Paragraph("Credit History Rating", label_style), Paragraph(credit_str, value_style),
         Paragraph("Employment Status", label_style), Paragraph(emp_str, value_style)]
    ]
    
    summary_table = Table(summary_data, colWidths=[130, 130, 130, 130])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # 4. AI Feature Impact Breakdown Table
    story.append(Paragraph("AI Underwriting Risk Breakdown", section_heading))
    
    impact_headers = [Paragraph("Key Risk Factor Evaluated", label_style), Paragraph("Underwriting Impact Explanation", label_style)]
    impact_rows = [impact_headers]
    for key, explainer in res.feature_impact.items():
        impact_rows.append([
            Paragraph(key, value_style),
            Paragraph(explainer, value_style)
        ])
        
    impact_table = Table(impact_rows, colWidths=[160, 360])
    impact_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(impact_table)
    story.append(Spacer(1, 15))
    
    # 5. Actionable Financial Improvement Suggestions
    story.append(Paragraph("Actionable Optimization Suggestions", section_heading))
    for sug in res.suggestions:
        story.append(Paragraph(f"• {sug}", bullet_style))
        
    story.append(Spacer(1, 20))
    
    # 6. Corporate Sign-off Block
    sign_off_data = [
        [Paragraph("Report Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), value_style),
         Paragraph("Luminate Automated Underwriting Systems", value_style)]
    ]
    sign_off_table = Table(sign_off_data, colWidths=[260, 260])
    sign_off_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.5, colors.HexColor('#94A3B8')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(sign_off_table)
    
    # Build Document
    doc.build(story)
    
    # Extract binary stream bytes
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

@app.post("/report/download")
def download_underwriting_report(request: ApplicantRequest):
    """
    POST `/report/download` generates and serves a beautifully structured PDF document 
    re-running ML classification and compiling underwriting risk suggestions.
    """
    try:
        # 1. Run preprocess and predict
        approval_proba, risk_score, loan_status = preprocess_and_predict(request)
        
        # Apply business rules overrides
        approval_proba, risk_score, loan_status, override_msg = apply_business_rules(
            request, approval_proba, risk_score, loan_status
        )
        
        # 2. Risk level categorization
        if risk_score <= 40.0:
            risk_level = "Low"
        elif risk_score <= 70.0:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        # 3. Create prediction response structure
        feature_impact = calculate_feature_impact(request, risk_score)
        suggestions = generate_improvement_suggestions(request)
        
        if override_msg:
            message = override_msg
        elif loan_status == "Approved":
            message = f"Loan approved with {approval_proba*100:.1f}% confidence."
        else:
            message = f"Loan rejected. Rejection risk is {risk_score:.1f}%."
            
        res_data = PredictionResponse(
            loan_status=loan_status,
            probability=approval_proba,
            risk_score=risk_score,
            risk_level=risk_level,
            feature_impact=feature_impact,
            suggestions=suggestions,
            message=message
        )
        
        # 4. Generate premium PDF bytes
        pdf_bytes = generate_pdf_report(request, res_data)
        
        # 5. Return StreamingResponse
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=loan_underwriting_report.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF generation error: {str(e)}")

# Run command check: `uvicorn main:app --host 127.0.0.1 --port 8000 --reload`
