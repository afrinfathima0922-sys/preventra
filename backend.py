from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Preventra Health API",
    description="AI-Powered Health Risk Prediction API",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model and scaler at startup
try:
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    model_name = metadata['model_name'].replace(" ", "_").lower()
    model = joblib.load(f'best_model_{model_name}.pkl')
    scaler = joblib.load('scaler.pkl')
    print(f"✅ Model loaded: {metadata['model_name']}")
    print(f"✅ Accuracy: {metadata['accuracy']:.2%}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    metadata = None

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class HealthData(BaseModel):
    """Input data for health risk prediction"""
    age: int = Field(..., ge=18, le=100, description="Age in years")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    glucose: int = Field(..., ge=50, le=400, description="Blood Glucose (mg/dL)")
    blood_pressure: int = Field(..., ge=60, le=200, description="Systolic Blood Pressure (mmHg)")
    cholesterol: int = Field(..., ge=100, le=500, description="Total Cholesterol (mg/dL)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking status (0=No, 1=Yes)")
    alcohol: int = Field(..., ge=0, le=2, description="Alcohol consumption (0=Never, 1=Occasional, 2=Regular)")
    physical_activity: int = Field(..., ge=0, le=2, description="Physical activity (0=Sedentary, 1=Moderate, 2=Active)")
    sleep_hours: float = Field(..., ge=3, le=12, description="Average sleep hours per night")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level (1-10)")
    family_history: int = Field(..., ge=0, le=1, description="Family history of chronic disease (0=No, 1=Yes)")

    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "bmi": 28.5,
                "glucose": 120,
                "blood_pressure": 130,
                "cholesterol": 210,
                "smoking": 0,
                "alcohol": 1,
                "physical_activity": 1,
                "sleep_hours": 7.0,
                "stress_level": 6,
                "family_history": 1
            }
        }

class DiseaseRisks(BaseModel):
    """Individual disease risk percentages"""
    diabetes: float = Field(..., description="Diabetes risk percentage")
    heart_attack: float = Field(..., description="Heart attack risk percentage")
    obesity: float = Field(..., description="Obesity risk percentage")

class RiskPrediction(BaseModel):
    """Complete risk prediction response"""
    overall_risk: str = Field(..., description="Overall risk level (Low/Medium/High)")
    overall_risk_index: int = Field(..., description="Risk index (0=Low, 1=Medium, 2=High)")
    confidence: float = Field(..., description="Confidence level (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each risk level")
    disease_risks: DiseaseRisks = Field(..., description="Individual disease risk percentages")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="Health recommendations")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_info: Dict[str, str] = Field(..., description="Model metadata")

class HealthStatus(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    accuracy: Optional[float]
    timestamp: str

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_disease_risks(health_data: dict) -> DiseaseRisks:
    """Calculate individual disease risk percentages"""
    
    # Diabetes Risk
    diabetes_score = 0
    if health_data['glucose'] > 140: diabetes_score += 40
    elif health_data['glucose'] > 126: diabetes_score += 30
    elif health_data['glucose'] > 100: diabetes_score += 15
    
    if health_data['bmi'] > 30: diabetes_score += 25
    elif health_data['bmi'] > 25: diabetes_score += 15
    
    if health_data['age'] > 45: diabetes_score += 15
    if health_data['family_history'] == 1: diabetes_score += 20
    
    diabetes_risk = min(diabetes_score, 95)
    
    # Heart Attack Risk
    heart_score = 0
    if health_data['blood_pressure'] > 140: heart_score += 35
    elif health_data['blood_pressure'] > 130: heart_score += 20
    elif health_data['blood_pressure'] > 120: heart_score += 10
    
    if health_data['cholesterol'] > 240: heart_score += 30
    elif health_data['cholesterol'] > 200: heart_score += 15
    
    if health_data['smoking'] == 1: heart_score += 25
    if health_data['age'] > 55: heart_score += 15
    if health_data['stress_level'] >= 7: heart_score += 10
    
    heart_risk = min(heart_score, 95)
    
    # Obesity Risk
    obesity_score = 0
    if health_data['bmi'] > 35: obesity_score += 90
    elif health_data['bmi'] > 30: obesity_score += 70
    elif health_data['bmi'] > 25: obesity_score += 40
    elif health_data['bmi'] > 23: obesity_score += 20
    
    if health_data['physical_activity'] == 0: obesity_score += 15
    if health_data['sleep_hours'] < 6: obesity_score += 10
    
    obesity_risk = min(obesity_score, 95)
    
    return DiseaseRisks(
        diabetes=round(diabetes_risk, 1),
        heart_attack=round(heart_risk, 1),
        obesity=round(obesity_risk, 1)
    )

def identify_risk_factors(health_data: dict) -> List[str]:
    """Identify risk factors from health data"""
    risk_factors = []
    
    if health_data['bmi'] > 30:
        risk_factors.append("High BMI - Obesity Range")
    
    if health_data['glucose'] > 126:
        risk_factors.append("Elevated Blood Glucose - Diabetic Range")
    elif health_data['glucose'] > 100:
        risk_factors.append("Pre-diabetic Glucose Levels")
    
    if health_data['blood_pressure'] > 140:
        risk_factors.append("Hypertension Detected")
    elif health_data['blood_pressure'] > 130:
        risk_factors.append("Elevated Blood Pressure")
    
    if health_data['cholesterol'] > 240:
        risk_factors.append("High Cholesterol Levels")
    elif health_data['cholesterol'] > 200:
        risk_factors.append("Borderline High Cholesterol")
    
    if health_data['smoking'] == 1:
        risk_factors.append("Active Tobacco Use")
    
    if health_data['physical_activity'] == 0:
        risk_factors.append("Sedentary Lifestyle")
    
    if health_data['family_history'] == 1:
        risk_factors.append("Family History of Chronic Disease")
    
    if health_data['stress_level'] >= 7:
        risk_factors.append("High Stress Levels")
    
    if health_data['sleep_hours'] < 6:
        risk_factors.append("Sleep Deprivation")
    
    if not risk_factors:
        risk_factors.append("No Major Risk Factors Detected")
    
    return risk_factors

def get_recommendations(risk_level: int) -> List[str]:
    """Get health recommendations based on risk level"""
    recommendations = {
        0: [
            "Continue regular physical activity (30 minutes daily)",
            "Maintain balanced nutrition and healthy eating habits",
            "Ensure 7-8 hours of quality sleep each night",
            "Stay properly hydrated throughout the day",
            "Practice regular stress management techniques",
            "Schedule annual preventive health check-ups"
        ],
        1: [
            "Increase physical activity to 45 minutes daily",
            "Significantly reduce sugar and processed food intake",
            "Monitor blood pressure and glucose levels regularly",
            "Eliminate tobacco use if applicable",
            "Limit alcohol consumption",
            "Schedule comprehensive health screening within 3 months",
            "Consult with a registered nutritionist or dietitian"
        ],
        2: [
            "CONSULT HEALTHCARE PROFESSIONAL IMMEDIATELY",
            "Undergo comprehensive medical screening and testing",
            "Establish regular monitoring with healthcare provider",
            "Follow prescribed dietary guidelines strictly",
            "Begin medically supervised exercise program",
            "Cease tobacco use immediately",
            "Eliminate alcohol consumption completely",
            "Prioritize quality sleep and stress reduction",
            "Consider medication or intervention as prescribed"
        ]
    }
    return recommendations[risk_level]

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=HealthStatus)
async def root():
    """API health check endpoint"""
    return HealthStatus(
        status="healthy",
        model_loaded=model is not None,
        model_name=metadata['model_name'] if metadata else None,
        accuracy=metadata['accuracy'] if metadata else None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Detailed health check endpoint"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure training has been completed."
        )
    
    return HealthStatus(
        status="operational",
        model_loaded=True,
        model_name=metadata['model_name'],
        accuracy=metadata['accuracy'],
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=RiskPrediction)
async def predict_risk(health_data: HealthData):
    """
    Predict health risk based on input data
    
    Returns comprehensive risk assessment including:
    - Overall risk level (Low/Medium/High)
    - Confidence score
    - Individual disease risks
    - Identified risk factors
    - Personalized recommendations
    """
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run training script first."
        )
    
    try:
        # Convert input to DataFrame
        data_dict = {
            'Age': health_data.age,
            'BMI': health_data.bmi,
            'Glucose': health_data.glucose,
            'BloodPressure': health_data.blood_pressure,
            'Cholesterol': health_data.cholesterol,
            'Smoking': health_data.smoking,
            'Alcohol': health_data.alcohol,
            'PhysicalActivity': health_data.physical_activity,
            'SleepHours': health_data.sleep_hours,
            'StressLevel': health_data.stress_level,
            'FamilyHistory': health_data.family_history
        }
        
        df = pd.DataFrame([data_dict])
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = int(model.predict(scaled_data)[0])
        prediction_proba = model.predict_proba(scaled_data)[0]
        
        # Calculate disease-specific risks
        disease_risks = calculate_disease_risks(health_data.dict())
        
        # Identify risk factors
        risk_factors = identify_risk_factors(health_data.dict())
        
        # Get recommendations
        recommendations = get_recommendations(prediction)
        
        # Map prediction to label
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Build response
        return RiskPrediction(
            overall_risk=risk_labels[prediction],
            overall_risk_index=prediction,
            confidence=float(prediction_proba[prediction]),
            probabilities={
                'low': float(prediction_proba[0]),
                'medium': float(prediction_proba[1]),
                'high': float(prediction_proba[2])
            },
            disease_risks=disease_risks,
            risk_factors=risk_factors,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            model_info={
                'name': metadata['model_name'],
                'accuracy': f"{metadata['accuracy']:.2%}",
                'version': '1.0.0'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not available"
        )
    
    return {
        "model_name": metadata['model_name'],
        "accuracy": metadata['accuracy'],
        "precision": metadata.get('precision', 'N/A'),
        "recall": metadata.get('recall', 'N/A'),
        "f1_score": metadata.get('f1_score', 'N/A'),
        "features": metadata['features'],
        "target_classes": metadata['target_classes'],
        "training_date": metadata['training_date'],
        "training_samples": metadata.get('training_samples', 'N/A'),
        "test_samples": metadata.get('test_samples', 'N/A')
    }

@app.post("/batch-predict")
async def batch_predict(health_data_list: List[HealthData]):
    """
    Batch prediction for multiple patients
    
    Useful for processing multiple assessments at once
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    results = []
    for health_data in health_data_list:
        try:
            prediction = await predict_risk(health_data)
            results.append(prediction)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {
        "total": len(health_data_list),
        "successful": sum(1 for r in results if "error" not in r),
        "results": results
    }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 PREVENTRA HEALTH API")
    print("=" * 50)
    print("📡 Starting server on http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative Docs: http://localhost:8000/redoc")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )