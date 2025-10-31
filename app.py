import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import time
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import io
import base64

# Page Configuration
st.set_page_config(
    page_title="Preventra - AI Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Blue Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a1128 0%, #1a2332 50%, #0f1b2d 100%);
    }
    
    .main {
        background: transparent;
    }
    
    /* Splash Screen */
    .splash-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
        text-align: center;
    }
    
    .splash-logo {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .splash-tagline {
        font-size: 1.5rem;
        color: #93c5fd;
        margin-bottom: 3rem;
        animation: fadeInUp 1.2s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .spinner {
        border: 4px solid rgba(59, 130, 246, 0.3);
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Login Screen */
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 3rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .login-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        font-size: 1rem;
        color: #93c5fd;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Header Styling */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-tagline {
        font-size: 1.2rem;
        color: #93c5fd;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
    }
    
    .metric-card h3 {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #bfdbfe;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Risk Cards */
    .risk-card {
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 2px solid;
        text-align: center;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-color: #34d399;
    }
    
    .medium-risk {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        border-color: #fbbf24;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        border-color: #f87171;
    }
    
    .risk-card h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .risk-card h2 {
        color: #f0f9ff;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Info Boxes with better visibility */
    .info-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-left: 5px solid #60a5fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .info-box strong {
        color: #ffffff;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #b45309 0%, #d97706 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .warning-box strong {
        color: #ffffff;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #047857 0%, #059669 100%);
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .success-box strong {
        color: #ffffff;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .error-box {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%);
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .error-box strong {
        color: #ffffff;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        box-shadow: 0 6px 25px rgba(37, 99, 235, 0.6);
        transform: translateY(-2px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Input Fields */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stTextInput>div>div>input {
        background: rgba(30, 58, 138, 0.3) !important;
        color: #ffffff !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #60a5fa;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-weight: 600;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    /* Subheaders */
    .stMarkdown h3 {
        color: #93c5fd !important;
    }
    
    /* Progress Steps */
    .step-complete {
        color: #10b981;
        font-weight: 600;
        padding: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .step-current {
        color: #3b82f6;
        font-weight: 600;
        padding: 0.5rem;
        background: rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .step-pending {
        color: #64748b;
        font-weight: 500;
        padding: 0.5rem;
        background: rgba(100, 116, 139, 0.1);
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    /* Divider */
    hr {
        border-color: rgba(59, 130, 246, 0.3);
        margin: 2rem 0;
    }
    
    /* Radio and Slider */
    .stRadio>label, .stSlider>label {
        color: #e2e8f0 !important;
    }
    
    /* Section Container */
    .section-container {
        background: rgba(30, 58, 138, 0.2);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 1rem 0;
    }
    
    /* Factor Card */
    .factor-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .factor-card h4 {
        color: #60a5fa;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Download Button Special Styling */
    .download-btn {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    }
    
    .download-btn:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        model_name = metadata['model_name'].replace(" ", "_").lower()
        model = joblib.load(f'best_model_{model_name}.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.info("Please run the training script first to generate the model files!")
        st.code("python train_model.py", language="bash")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Calculate BMI
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

# Get BMI category
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#60a5fa"
    elif 18.5 <= bmi < 25:
        return "Normal", "#10b981"
    elif 25 <= bmi < 30:
        return "Overweight", "#f59e0b"
    else:
        return "Obese", "#ef4444"

# Calculate individual disease risks
def calculate_disease_risks(user_data):
    diabetes_score = 0
    if user_data['Glucose'] > 140: diabetes_score += 40
    elif user_data['Glucose'] > 126: diabetes_score += 30
    elif user_data['Glucose'] > 100: diabetes_score += 15
    
    if user_data['BMI'] > 30: diabetes_score += 25
    elif user_data['BMI'] > 25: diabetes_score += 15
    
    if user_data['Age'] > 45: diabetes_score += 15
    if user_data['FamilyHistory'] == 1: diabetes_score += 20
    
    diabetes_risk = min(diabetes_score, 95)
    
    heart_score = 0
    if user_data['BloodPressure'] > 140: heart_score += 35
    elif user_data['BloodPressure'] > 130: heart_score += 20
    elif user_data['BloodPressure'] > 120: heart_score += 10
    
    if user_data['Cholesterol'] > 240: heart_score += 30
    elif user_data['Cholesterol'] > 200: heart_score += 15
    
    if user_data['Smoking'] == 1: heart_score += 25
    if user_data['Age'] > 55: heart_score += 15
    if user_data['StressLevel'] >= 7: heart_score += 10
    
    heart_risk = min(heart_score, 95)
    
    obesity_score = 0
    if user_data['BMI'] > 35: obesity_score += 90
    elif user_data['BMI'] > 30: obesity_score += 70
    elif user_data['BMI'] > 25: obesity_score += 40
    elif user_data['BMI'] > 23: obesity_score += 20
    
    if user_data['PhysicalActivity'] == 0: obesity_score += 15
    if user_data['SleepHours'] < 6: obesity_score += 10
    
    obesity_risk = min(obesity_score, 95)
    
    return {
        'Diabetes': diabetes_risk,
        'Heart Attack': heart_risk,
        'Obesity': obesity_risk
    }

# Get risk recommendations
def get_recommendations(risk_level, user_data):
    recommendations = {
        0: {
            'title': 'Low Risk - Maintain Your Healthy Lifestyle',
            'tips': [
                'Continue regular physical activity (30 minutes daily)',
                'Maintain balanced nutrition and healthy eating habits',
                'Ensure 7-8 hours of quality sleep each night',
                'Stay properly hydrated throughout the day',
                'Practice regular stress management techniques',
                'Schedule annual preventive health check-ups'
            ]
        },
        1: {
            'title': 'Medium Risk - Immediate Lifestyle Changes Recommended',
            'tips': [
                'Increase physical activity to 45 minutes daily',
                'Significantly reduce sugar and processed food intake',
                'Monitor blood pressure and glucose levels regularly',
                'Eliminate tobacco use if applicable',
                'Limit alcohol consumption',
                'Schedule comprehensive health screening within 3 months',
                'Consult with a registered nutritionist or dietitian'
            ]
        },
        2: {
            'title': 'High Risk - Urgent Medical Attention Required',
            'tips': [
                'CONSULT HEALTHCARE PROFESSIONAL IMMEDIATELY',
                'Undergo comprehensive medical screening and testing',
                'Establish regular monitoring with healthcare provider',
                'Follow prescribed dietary guidelines strictly',
                'Begin medically supervised exercise program',
                'Cease tobacco use immediately',
                'Eliminate alcohol consumption completely',
                'Prioritize quality sleep and stress reduction',
                'Consider medication or intervention as prescribed'
            ]
        }
    }
    return recommendations[risk_level]

# PDF Generation Function
def generate_pdf_report(user_data, prediction, prediction_proba, disease_risks, recommendations, metadata):
    """Generate comprehensive PDF health report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        fontName='Helvetica'
    )
    
    # Title
    elements.append(Paragraph("PREVENTRA", title_style))
    elements.append(Paragraph("Comprehensive Health Risk Assessment Report", styles['Heading3']))
    elements.append(Spacer(1, 20))
    
    # Report Info
    report_info = [
        ['Report Date:', datetime.now().strftime('%B %d, %Y')],
        ['Report Time:', datetime.now().strftime('%I:%M %p')],
        ['Patient Name:', st.session_state.username],
        ['Report ID:', f"PREV-{datetime.now().strftime('%Y%m%d%H%M%S')}"]
    ]
    
    info_table = Table(report_info, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f2fe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 20))
    
    # Overall Risk Assessment
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_colors_hex = ['#10b981', '#f59e0b', '#ef4444']
    
    elements.append(Paragraph("OVERALL RISK ASSESSMENT", heading_style))
    
    risk_data = [
        ['Risk Level:', risk_labels[prediction]],
        ['Confidence:', f"{prediction_proba[prediction]:.1%}"],
        ['Assessment Model:', metadata['model_name']],
        ['Model Accuracy:', f"{metadata['accuracy']:.1%}"]
    ]
    
    risk_table = Table(risk_data, colWidths=[2*inch, 4*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 20))
    
    # Risk Probabilities
    elements.append(Paragraph("RISK PROBABILITY BREAKDOWN", heading_style))
    
    prob_data = [['Risk Category', 'Probability']]
    for i, label in enumerate(risk_labels):
        prob_data.append([label, f"{prediction_proba[i]:.1%}"])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 20))
    
    # Disease-Specific Risks
    elements.append(Paragraph("DISEASE-SPECIFIC RISK ANALYSIS", heading_style))
    
    disease_data = [['Disease', 'Risk Percentage', 'Risk Level']]
    for disease, risk in disease_risks.items():
        if risk < 30:
            level = 'Low'
        elif risk < 60:
            level = 'Medium'
        else:
            level = 'High'
        disease_data.append([disease, f"{risk:.0f}%", level])
    
    disease_table = Table(disease_data, colWidths=[2*inch, 2*inch, 2*inch])
    disease_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(disease_table)
    elements.append(Spacer(1, 20))
    
    # Health Metrics
    elements.append(Paragraph("YOUR HEALTH METRICS", heading_style))
    
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Age', f"{user_data['Age']} years", 'Recorded'],
        ['BMI', f"{user_data['BMI']:.1f}", 'Recorded'],
        ['Blood Glucose', f"{user_data['Glucose']} mg/dL", 'Normal' if user_data['Glucose'] < 100 else 'Elevated'],
        ['Blood Pressure', f"{user_data['BloodPressure']} mmHg", 'Normal' if user_data['BloodPressure'] < 120 else 'Elevated'],
        ['Cholesterol', f"{user_data['Cholesterol']} mg/dL", 'Normal' if user_data['Cholesterol'] < 200 else 'Elevated'],
        ['Stress Level', f"{user_data['StressLevel']}/10", 'Low' if user_data['StressLevel'] < 5 else 'High'],
        ['Sleep Hours', f"{user_data['SleepHours']:.1f} hrs", 'Adequate' if user_data['SleepHours'] >= 7 else 'Insufficient']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    # Recommendations
    elements.append(Paragraph("PERSONALIZED HEALTH RECOMMENDATIONS", heading_style))
    elements.append(Paragraph(recommendations['title'], normal_style))
    elements.append(Spacer(1, 10))
    
    for i, tip in enumerate(recommendations['tips'], 1):
        elements.append(Paragraph(f"{i}. {tip}", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # Disclaimer
    elements.append(Paragraph("MEDICAL DISCLAIMER", heading_style))
    disclaimer_text = """This report is generated by an AI-powered risk assessment tool and is intended 
    for informational purposes only. It does NOT constitute medical diagnosis, advice, or treatment. 
    Always consult with qualified healthcare professionals for medical decisions. This assessment is 
    based on the information provided and may not reflect your complete health status."""
    elements.append(Paragraph(disclaimer_text, normal_style))
    
    elements.append(Spacer(1, 20))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("Generated by Preventra - AI-Powered Health Intelligence", footer_style))
    elements.append(Paragraph(f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Splash Screen
def show_splash_screen():
    st.markdown("""
    <div class='splash-container'>
        <div class='splash-logo'>PREVENTRA</div>
        <div class='splash-tagline'>AI-Powered Health Intelligence</div>
        <div class='spinner'></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.session_state.splash_shown = True
    st.rerun()

# Login Screen
def show_login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class='login-container'>
            <div class='login-title'>PREVENTRA</div>
            <div class='login-subtitle'>Your Partner in Proactive Health Management</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your name")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                login_btn = st.form_submit_button("LOGIN", use_container_width=True)
            
            with col_b:
                guest_btn = st.form_submit_button("CONTINUE AS GUEST", use_container_width=True)
            
            if login_btn:
                if username and password:
                    # Simple validation (in production, use proper authentication)
                    if password == "preventra123" or len(password) >= 6:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success(f"Welcome, {username}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Password must be at least 6 characters")
                else:
                    st.error("Please enter both username and password")
            
            if guest_btn:
                st.session_state.logged_in = True
                st.session_state.username = "Guest User"
                st.info("Continuing as Guest...")
                time.sleep(1)
                st.rerun()
        
        st.markdown("""
        <div class='info-box' style='margin-top: 2rem;'>
            <strong>Demo Credentials</strong>
            Username: demo<br>
            Password: preventra123<br><br>
            Or click "Continue as Guest" to explore without login.
        </div>
        """, unsafe_allow_html=True)

# Main Application Logic
if not st.session_state.splash_shown:
    show_splash_screen()
elif not st.session_state.logged_in:
    show_login_screen()
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #60a5fa; font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem;'>PREVENTRA</h1>
            <p style='color: #93c5fd; font-size: 0.9rem; margin: 0;'>AI-Powered Health Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class='info-box'>
            <strong>Welcome</strong>
            {st.session_state.username}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            st.markdown(f"""
            <div class='success-box'>
                <strong>Model Active</strong>
                {metadata['model_name']}<br>
                Accuracy: {metadata['accuracy']:.1%}
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class='warning-box'>
                <strong>Model Status</strong>
                Not Loaded
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class='warning-box'>
            <strong>Medical Disclaimer</strong>
            This tool provides risk assessment for informational purposes only. 
            It does not constitute medical diagnosis or advice. 
            Always consult qualified healthcare professionals for medical decisions.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("<h3 style='color: #60a5fa; margin-bottom: 1rem;'>Assessment Progress</h3>", unsafe_allow_html=True)
        
        steps = ["Introduction", "Basic Information", "Health Metrics", "Lifestyle Factors", "Risk Analysis"]
        for i, step in enumerate(steps):
            if i < st.session_state.step:
                st.markdown(f"<div class='step-complete'>✓ {step}</div>", unsafe_allow_html=True)
            elif i == st.session_state.step:
                st.markdown(f"<div class='step-current'>→ {step}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='step-pending'>○ {step}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("RESET ASSESSMENT"):
            st.session_state.step = 0
            st.session_state.user_data = {}
            st.session_state.prediction_made = False
            st.session_state.prediction_results = None
            st.rerun()
        
        if st.button("LOGOUT"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.step = 0
            st.session_state.user_data = {}
            st.session_state.prediction_made = False
            st.session_state.prediction_results = None
            st.rerun()

    # Main content
    if st.session_state.step == 0:
        # Home Page
        st.markdown("""
        <div class='app-header'>
            <h1 class='app-title'>PREVENTRA</h1>
            <p class='app-tagline'>Your Partner in Proactive Health Management</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Rapid Assessment</h3>
                <p>Complete evaluation in under 5 minutes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>AI-Powered Analysis</h3>
                <p>Advanced machine learning algorithms</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Personalized Insights</h3>
                <p>Customized health recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Comprehensive Risk Assessment</strong>
            • Diabetes Risk Evaluation<br>
            • Cardiovascular Health Analysis<br>
            • Overall Wellness Score<br>
            • Evidence-Based Recommendations<br>
            • Actionable Health Strategies<br>
            • Downloadable PDF Report
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("BEGIN ASSESSMENT"):
            st.session_state.step = 1
            st.rerun()

    elif st.session_state.step == 1:
        # Basic Info
        st.markdown("<h1>Basic Information</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Please provide your personal details</h3>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, key="age_input")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_input")
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="height_input")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, key="weight_input")
            
            if height > 0 and weight > 0:
                bmi = calculate_bmi(weight, height)
                category, color = get_bmi_category(bmi)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Body Mass Index (BMI)", f"{bmi}", f"{category}")
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("CONTINUE TO HEALTH METRICS"):
            st.session_state.user_data.update({
                'Age': age,
                'Gender': gender,
                'Height': height,
                'Weight': weight,
                'BMI': bmi
            })
            st.session_state.step = 2
            st.rerun()

    elif st.session_state.step == 2:
        # Health Data
        st.markdown("<h1>Health Metrics</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Current physiological measurements</h3>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Cardiovascular Metrics</h4>", unsafe_allow_html=True)
            
            systolic = st.number_input("Blood Pressure - Systolic (mmHg)", min_value=80, max_value=200, value=120, key="systolic_input")
            diastolic = st.number_input("Blood Pressure - Diastolic (mmHg)", min_value=50, max_value=130, value=80, key="diastolic_input")
            
            if systolic < 120 and diastolic < 80:
                st.markdown(f"<div class='success-box'><strong>Normal Range</strong>{systolic}/{diastolic} mmHg</div>", unsafe_allow_html=True)
            elif systolic < 140 and diastolic < 90:
                st.markdown(f"<div class='warning-box'><strong>Elevated</strong>{systolic}/{diastolic} mmHg</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='error-box'><strong>High</strong>{systolic}/{diastolic} mmHg</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Metabolic Indicators</h4>", unsafe_allow_html=True)
            
            glucose = st.number_input("Blood Glucose (mg/dL)", min_value=70, max_value=300, value=100, key="glucose_input")
            if glucose < 100:
                st.markdown("<div class='success-box'><strong>Normal</strong>Fasting glucose level</div>", unsafe_allow_html=True)
            elif glucose < 126:
                st.markdown("<div class='warning-box'><strong>Pre-diabetic</strong>Range detected</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-box'><strong>Diabetic</strong>Consult physician</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Lipid Profile</h4>", unsafe_allow_html=True)
            
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180, key="cholesterol_input")
            if cholesterol < 200:
                st.markdown("<div class='success-box'><strong>Desirable</strong>Cholesterol level</div>", unsafe_allow_html=True)
            elif cholesterol < 240:
                st.markdown("<div class='warning-box'><strong>Borderline</strong>High cholesterol</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-box'><strong>High</strong>Cholesterol detected</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Family Medical History</h4>", unsafe_allow_html=True)
            
            family_history = st.selectbox(
                "Chronic Disease in Immediate Family",
                ["None", "Diabetes", "Heart Disease", "Both Diabetes and Heart Disease"],
                key="family_input"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("CONTINUE TO LIFESTYLE ASSESSMENT"):
            st.session_state.user_data.update({
                'BloodPressure': systolic,
                'Glucose': glucose,
                'Cholesterol': cholesterol,
                'FamilyHistory': 1 if family_history != "None" else 0
            })
            st.session_state.step = 3
            st.rerun()

    elif st.session_state.step == 3:
        # Lifestyle
        st.markdown("<h1>Lifestyle Assessment</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Daily habits and behavioral patterns</h3>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Substance Use</h4>", unsafe_allow_html=True)
            
            smoking = st.radio("Tobacco Use", ["Non-Smoker", "Current Smoker"], key="smoking_input")
            alcohol = st.select_slider(
                "Alcohol Consumption Frequency",
                options=["Never", "Occasional", "Regular"],
                key="alcohol_input"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Physical Activity</h4>", unsafe_allow_html=True)
            
            physical_activity = st.select_slider(
                "Exercise Level",
                options=["Sedentary", "Moderate Activity", "Highly Active"],
                key="activity_input"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Sleep Pattern</h4>", unsafe_allow_html=True)
            
            sleep_hours = st.slider("Average Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.5, key="sleep_input")
            if sleep_hours < 6:
                st.markdown("<div class='warning-box'><strong>Insufficient</strong>Sleep detected</div>", unsafe_allow_html=True)
            elif sleep_hours > 9:
                st.markdown("<div class='info-box'><strong>Extended</strong>Sleep duration noted</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'><strong>Optimal</strong>Sleep duration</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #60a5fa;'>Stress Assessment</h4>", unsafe_allow_html=True)
            
            stress_level = st.slider("Stress Level (1-10 scale)", 1, 10, 5, key="stress_input")
            if stress_level >= 7:
                st.markdown("<div class='warning-box'><strong>Elevated</strong>Stress - Intervention recommended</div>", unsafe_allow_html=True)
            elif stress_level >= 4:
                st.markdown("<div class='info-box'><strong>Moderate</strong>Stress levels</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'><strong>Low</strong>Stress maintained</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("GENERATE RISK ASSESSMENT"):
            activity_map = {"Sedentary": 0, "Moderate Activity": 1, "Highly Active": 2}
            alcohol_map = {"Never": 0, "Occasional": 1, "Regular": 2}
            
            st.session_state.user_data.update({
                'Smoking': 1 if smoking == "Current Smoker" else 0,
                'Alcohol': alcohol_map[alcohol],
                'PhysicalActivity': activity_map[physical_activity],
                'SleepHours': sleep_hours,
                'StressLevel': stress_level
            })
            st.session_state.step = 4
            st.rerun()

    elif st.session_state.step == 4:
        # Results
        st.markdown("<h1>Comprehensive Risk Analysis</h1>", unsafe_allow_html=True)
        
        model, scaler, metadata = load_model_and_scaler()
        
        if model is None:
            st.markdown("<div class='error-box'><strong>Model Error:</strong> Unable to load predictive model. Please ensure training has been completed.</div>", unsafe_allow_html=True)
            if st.button("RETURN TO START"):
                st.session_state.step = 0
                st.rerun()
            st.stop()
        
        # Prepare data for prediction
        user_df = pd.DataFrame([st.session_state.user_data])
        
        required_features = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Cholesterol', 
                            'Smoking', 'Alcohol', 'PhysicalActivity', 'SleepHours', 
                            'StressLevel', 'FamilyHistory']
        
        try:
            user_df = user_df[required_features]
        except KeyError as e:
            st.markdown(f"<div class='error-box'><strong>Data Error:</strong> Missing required information: {e}</div>", unsafe_allow_html=True)
            if st.button("RETURN TO START"):
                st.session_state.step = 0
                st.rerun()
            st.stop()
        
        # Scale and predict
        user_scaled = scaler.transform(user_df)
        prediction = int(model.predict(user_scaled)[0])
        prediction_proba = model.predict_proba(user_scaled)[0]
        
        # Calculate individual disease risks
        disease_risks = calculate_disease_risks(st.session_state.user_data)
        
        # Store results for PDF generation
        st.session_state.prediction_results = {
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'disease_risks': disease_risks
        }
        
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_colors = ['#10b981', '#f59e0b', '#ef4444']
        risk_classes = ['low-risk', 'medium-risk', 'high-risk']
        
        # Display main result
        st.markdown(f"""
        <div class="risk-card {risk_classes[prediction]}">
            <h1>{risk_labels[prediction]}</h1>
            <h2>Confidence Level: {prediction_proba[prediction]:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[prediction] * 100,
            title={'text': "Risk Score", 'font': {'size': 24, 'color': '#ffffff'}},
            number={'font': {'size': 48, 'color': '#ffffff'}, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#cbd5e1'},
                'bar': {'color': risk_colors[prediction], 'thickness': 0.8},
                'bgcolor': 'rgba(30, 58, 138, 0.3)',
                'borderwidth': 2,
                'bordercolor': '#3b82f6',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(16, 185, 129, 0.3)'},
                    {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.3)'},
                    {'range': [66, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': '#ffffff', 'width': 4},
                    'thickness': 0.8,
                    'value': 75
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff', 'family': 'Inter'},
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk breakdown
        st.markdown("<h2>Detailed Risk Breakdown</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.metric("Low Risk Probability", f"{prediction_proba[0]:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.metric("Medium Risk Probability", f"{prediction_proba[1]:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.metric("High Risk Probability", f"{prediction_proba[2]:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Specific Disease Risk Assessment
        st.markdown("<h2>Specific Disease Risk Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>Diabetes Risk</h4>", unsafe_allow_html=True)
            diabetes_color = '#ef4444' if disease_risks['Diabetes'] > 60 else '#f59e0b' if disease_risks['Diabetes'] > 30 else '#10b981'
            st.markdown(f"<h1 style='text-align: center; color: {diabetes_color};'>{disease_risks['Diabetes']:.0f}%</h1>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>Heart Attack Risk</h4>", unsafe_allow_html=True)
            heart_color = '#ef4444' if disease_risks['Heart Attack'] > 60 else '#f59e0b' if disease_risks['Heart Attack'] > 30 else '#10b981'
            st.markdown(f"<h1 style='text-align: center; color: {heart_color};'>{disease_risks['Heart Attack']:.0f}%</h1>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>Obesity Risk</h4>", unsafe_allow_html=True)
            obesity_color = '#ef4444' if disease_risks['Obesity'] > 60 else '#f59e0b' if disease_risks['Obesity'] > 30 else '#10b981'
            st.markdown(f"<h1 style='text-align: center; color: {obesity_color};'>{disease_risks['Obesity']:.0f}%</h1>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Key factors analysis
        st.markdown("<h2>Key Health Indicators</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Your Current Metrics</h4>", unsafe_allow_html=True)
            
            metrics_data = {
                'Age': st.session_state.user_data['Age'],
                'Body Mass Index': st.session_state.user_data['BMI'],
                'Blood Glucose': st.session_state.user_data['Glucose'],
                'Blood Pressure': st.session_state.user_data['BloodPressure'],
                'Cholesterol': st.session_state.user_data['Cholesterol'],
                'Stress Level': st.session_state.user_data['StressLevel']
            }
            
            for metric, value in metrics_data.items():
                st.metric(metric, f"{value:.1f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='factor-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Risk Factor Analysis</h4>", unsafe_allow_html=True)
            
            risk_factors_found = False
            
            if st.session_state.user_data['BMI'] > 30:
                st.markdown("<div class='error-box'><strong>High BMI</strong>Obesity Range</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['Glucose'] > 126:
                st.markdown("<div class='error-box'><strong>Elevated Glucose</strong>Diabetic range</div>", unsafe_allow_html=True)
                risk_factors_found = True
            elif st.session_state.user_data['Glucose'] > 100:
                st.markdown("<div class='warning-box'><strong>Pre-diabetic</strong>Glucose levels</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['BloodPressure'] > 140:
                st.markdown("<div class='error-box'><strong>Hypertension</strong>Detected</div>", unsafe_allow_html=True)
                risk_factors_found = True
            elif st.session_state.user_data['BloodPressure'] > 130:
                st.markdown("<div class='warning-box'><strong>Elevated</strong>Blood pressure</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['Cholesterol'] > 240:
                st.markdown("<div class='error-box'><strong>High</strong>Cholesterol levels</div>", unsafe_allow_html=True)
                risk_factors_found = True
            elif st.session_state.user_data['Cholesterol'] > 200:
                st.markdown("<div class='warning-box'><strong>Borderline</strong>High cholesterol</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['Smoking'] == 1:
                st.markdown("<div class='error-box'><strong>Active</strong>Tobacco use</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['PhysicalActivity'] == 0:
                st.markdown("<div class='warning-box'><strong>Sedentary</strong>Lifestyle</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['FamilyHistory'] == 1:
                st.markdown("<div class='warning-box'><strong>Family History</strong>Present</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['StressLevel'] >= 7:
                st.markdown("<div class='warning-box'><strong>High Stress</strong>Levels</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if st.session_state.user_data['SleepHours'] < 6:
                st.markdown("<div class='warning-box'><strong>Sleep</strong>Deprivation</div>", unsafe_allow_html=True)
                risk_factors_found = True
            
            if not risk_factors_found:
                st.markdown("<div class='success-box'><strong>No Major</strong>Risk factors detected</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("<h2>Personalized Health Recommendations</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        recommendations = get_recommendations(prediction, st.session_state.user_data)
        
        if prediction == 0:
            box_class = 'success-box'
        elif prediction == 1:
            box_class = 'warning-box'
        else:
            box_class = 'error-box'
        
        st.markdown(f"<div class='{box_class}'><strong>{recommendations['title']}</strong></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        for i, tip in enumerate(recommendations['tips'], 1):
            st.markdown(f"**{i}.** {tip}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model information
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>
            <strong>Analysis Details</strong><br>
            Model: {metadata['model_name']} | Accuracy: {metadata['accuracy']:.1%} | 
            Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate PDF
            if st.session_state.prediction_results:
                pdf_buffer = generate_pdf_report(
                    st.session_state.user_data,
                    st.session_state.prediction_results['prediction'],
                    st.session_state.prediction_results['prediction_proba'],
                    st.session_state.prediction_results['disease_risks'],
                    recommendations,
                    metadata
                )
                
                st.download_button(
                    label="DOWNLOAD PDF REPORT",
                    data=pdf_buffer,
                    file_name=f"Preventra_Health_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col2:
            if st.button("NEW ASSESSMENT", use_container_width=True):
                st.session_state.step = 0
                st.session_state.user_data = {}
                st.session_state.prediction_made = False
                st.session_state.prediction_results = None
                st.rerun()