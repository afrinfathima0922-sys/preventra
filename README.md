# 🏥 PREVENTRA - AI Health Risk Predictor

An AI-powered web application that predicts health risks for diabetes, heart disease, and obesity using machine learning.

## 🚀 **Live Demo**

**[Try it here →](https://your-app.streamlit.app)** *(Update after deployment)*

---

## ✨ **Features**

- 🎯 Multi-disease risk assessment (Diabetes, Heart, Obesity)
- 🤖 85%+ accuracy using XGBoost ML model
- 📊 Interactive visualizations and charts
- 💡 Personalized health recommendations
- 🎨 User-friendly interface
- 🔒 Privacy-focused (no data storage)

---

## 🛠️ **Tech Stack**

- **Frontend:** Streamlit, Plotly
- **ML Model:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy
- **Deployment:** Streamlit Cloud

---

## 📦 **Installation**

```bash
# Clone repository
git clone https://github.com/afrinfathima0922-sys/preventra.git
cd preventra

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Open: http://localhost:8501

---

## 📊 **Model Performance**

| Metric | Score |
|--------|-------|
| Accuracy | 85.23% |
| Precision | 84.5% |
| Recall | 85.1% |
| F1-Score | 84.8% |

**Dataset:** 2,000+ patient records  
**Features:** 11 health indicators  
**Model:** XGBoost Classifier

---

## 📁 **Project Structure**

```
preventra/
├── app.py                      # Main application
├── train_model.py              # Model training
├── best_model_xgboost.pkl     # Trained model
├── scaler.pkl                  # Feature scaler
├── model_metadata.json         # Model info
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 🎯 **Usage**

1. **Enter basic info:** Age, height, weight
2. **Provide health data:** Blood pressure, glucose, cholesterol
3. **Answer lifestyle questions:** Smoking, exercise, sleep
4. **Get results:** Risk levels and recommendations

---

## 🔮 **Future Features**

- [ ] Multilingual support (Tamil, Hindi)
- [ ] Voice assistant
- [ ] AI chatbot
- [ ] Medical image analysis
- [ ] Mobile app

---

## ⚠️ **Disclaimer**

This tool is for **educational purposes only**. It is **NOT** medical advice. Always consult healthcare professionals for medical decisions.

---

## 📜 **License**

MIT License - feel free to use for your projects!

---

## 🙏 **Acknowledgments**

- UCI Machine Learning Repository (Dataset)
- Streamlit Community
- XGBoost Team

---

**⭐ Star this repo if you find it useful!**

Made with ❤️ for better healthcare