import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set professional styling for plots
plt.style.use('default')
sns.set_palette("husl")

print("=" * 70)
print("PREVENTRA - AI-POWERED HEALTH RISK PREDICTION MODEL TRAINING")
print("=" * 70)
print("\nLibrary Initialization: SUCCESS")

# ============================================
# PREPROCESSING FUNCTIONS FOR EACH DATASET
# ============================================

def preprocess_diabetes_csv(filepath='diabetes.csv'):
    """
    Preprocess Pima Indians Diabetes Dataset
    Columns: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[DIABETES] Loaded: {len(df)} samples")
        
        processed_df = pd.DataFrame()
        processed_df['Age'] = df['Age']
        processed_df['BMI'] = df['BMI']
        processed_df['Glucose'] = df['Glucose']
        processed_df['BloodPressure'] = df['BloodPressure']
        
        # Estimate cholesterol from other metrics
        processed_df['Cholesterol'] = (df['BMI'] * 5 + 100 + np.random.randint(-20, 20, len(df))).clip(150, 300)
        
        # Create lifestyle features
        np.random.seed(42)
        processed_df['Smoking'] = np.where(df['Age'] > 40, 
                                           np.random.choice([0, 1], len(df), p=[0.6, 0.4]),
                                           np.random.choice([0, 1], len(df), p=[0.8, 0.2]))
        
        processed_df['Alcohol'] = np.random.choice([0, 1, 2], len(df), p=[0.5, 0.3, 0.2])
        
        processed_df['PhysicalActivity'] = np.where(df['BMI'] > 30, 
                                                     np.random.choice([0, 1, 2], len(df), p=[0.5, 0.3, 0.2]),
                                                     np.random.choice([0, 1, 2], len(df), p=[0.2, 0.3, 0.5]))
        
        processed_df['SleepHours'] = np.random.uniform(5, 9, len(df))
        processed_df['StressLevel'] = np.random.randint(3, 9, len(df))
        processed_df['FamilyHistory'] = (df['DiabetesPedigreeFunction'] > 0.5).astype(int)
        
        # Convert binary outcome to risk levels
        processed_df['DiabetesRisk'] = df['Outcome']
        high_risk_mask = (df['Glucose'] > 140) & (df['BMI'] > 35)
        medium_risk_mask = ((df['Glucose'] > 100) | (df['BMI'] > 30)) & (~high_risk_mask)
        
        processed_df.loc[high_risk_mask, 'DiabetesRisk'] = 2
        processed_df.loc[medium_risk_mask & (processed_df['DiabetesRisk'] == 0), 'DiabetesRisk'] = 1
        
        # Replace zeros with NaN for medical measurements
        medical_cols = ['Glucose', 'BloodPressure', 'BMI']
        for col in medical_cols:
            processed_df[col] = processed_df[col].replace(0, np.nan)
        
        return processed_df
        
    except FileNotFoundError:
        print(f"[DIABETES] File not found: {filepath}")
        return None

def preprocess_heart_csv(filepath='heart.csv'):
    """
    Preprocess Heart Disease Dataset
    Columns: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[HEART] Loaded: {len(df)} samples")
        
        processed_df = pd.DataFrame()
        processed_df['Age'] = df['age']
        processed_df['BloodPressure'] = df['trestbps']
        processed_df['Cholesterol'] = df['chol']
        
        # Estimate BMI from age and other factors
        processed_df['BMI'] = 22 + (df['age'] - 30) * 0.15 + np.random.uniform(-3, 5, len(df))
        processed_df['BMI'] = processed_df['BMI'].clip(18, 45)
        
        # Estimate glucose from cholesterol
        processed_df['Glucose'] = 70 + (df['chol'] - 200) * 0.3 + np.random.randint(-20, 40, len(df))
        processed_df['Glucose'] = processed_df['Glucose'].clip(70, 200)
        
        # Map lifestyle features
        processed_df['Smoking'] = np.where(df['sex'] == 1,
                                           np.random.choice([0, 1], len(df), p=[0.6, 0.4]),
                                           np.random.choice([0, 1], len(df), p=[0.8, 0.2]))
        
        processed_df['Alcohol'] = np.random.choice([0, 1, 2], len(df), p=[0.5, 0.3, 0.2])
        
        processed_df['PhysicalActivity'] = np.where(df['exang'] == 0, 
                                                     np.random.choice([1, 2], len(df), p=[0.5, 0.5]),
                                                     np.random.choice([0, 1], len(df), p=[0.7, 0.3]))
        
        processed_df['SleepHours'] = np.random.uniform(5, 9, len(df))
        processed_df['StressLevel'] = np.where(df['cp'] > 0, 
                                                np.random.randint(6, 10, len(df)),
                                                np.random.randint(3, 7, len(df)))
        
        processed_df['FamilyHistory'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
        
        # Convert heart disease to diabetes risk
        processed_df['DiabetesRisk'] = df['target']
        high_risk = (df['target'] == 1) & ((df['chol'] > 240) | (df['fbs'] == 1))
        processed_df.loc[high_risk, 'DiabetesRisk'] = 2
        
        return processed_df
        
    except FileNotFoundError:
        print(f"[HEART] File not found: {filepath}")
        return None

def preprocess_s1_csv(filepath='s1.csv'):
    """
    Preprocess Survey Dataset 1
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[SURVEY-1] Loaded: {len(df)} samples")
        
        # Sample if too large
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            print(f"[SURVEY-1] Sampled to: 10,000 samples")
        
        processed_df = pd.DataFrame()
        processed_df['Age'] = df['Age']
        processed_df['BMI'] = df['BMI']
        
        # Estimate glucose from diabetes status
        processed_df['Glucose'] = np.where(
            df['Diabetes_012'] == 2,
            np.random.randint(140, 200, len(df)),
            np.where(df['Diabetes_012'] == 1,
                     np.random.randint(100, 140, len(df)),
                     np.random.randint(70, 100, len(df)))
        )
        
        # Estimate blood pressure
        processed_df['BloodPressure'] = np.where(
            df['HighBP'] == 1,
            np.random.randint(130, 180, len(df)),
            np.random.randint(80, 130, len(df))
        )
        
        # Estimate cholesterol
        processed_df['Cholesterol'] = np.where(
            df['HighChol'] == 1,
            np.random.randint(240, 300, len(df)),
            np.random.randint(150, 240, len(df))
        )
        
        processed_df['Smoking'] = df['Smoker']
        processed_df['Alcohol'] = np.where(df['HvyAlcoholConsump'] == 1, 2,
                                           np.random.choice([0, 1], len(df), p=[0.6, 0.4]))
        processed_df['PhysicalActivity'] = np.where(df['PhysActivity'] == 1, 
                                                     np.random.choice([1, 2], len(df), p=[0.5, 0.5]),
                                                     0)
        
        processed_df['SleepHours'] = np.where(
            df['GenHlth'] <= 2,
            np.random.uniform(7, 9, len(df)),
            np.random.uniform(5, 7, len(df))
        )
        
        processed_df['StressLevel'] = (df['MentHlth'] / 3).clip(1, 10).astype(int)
        processed_df['FamilyHistory'] = np.random.choice([0, 1], len(df), p=[0.5, 0.5])
        
        # Map diabetes status
        processed_df['DiabetesRisk'] = df['Diabetes_012'].clip(0, 2)
        
        return processed_df
        
    except FileNotFoundError:
        print(f"[SURVEY-1] File not found: {filepath}")
        return None

def preprocess_s2_csv(filepath='s2.csv'):
    """
    Preprocess Survey Dataset 2
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[SURVEY-2] Loaded: {len(df)} samples")
        
        # Sample if too large
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            print(f"[SURVEY-2] Sampled to: 10,000 samples")
        
        processed_df = pd.DataFrame()
        processed_df['Age'] = df['Age']
        processed_df['BMI'] = df['BMI']
        
        # Estimate glucose
        processed_df['Glucose'] = np.where(
            df['Diabetes_binary'] == 1,
            np.random.randint(126, 200, len(df)),
            np.random.randint(70, 125, len(df))
        )
        
        # Estimate blood pressure
        processed_df['BloodPressure'] = np.where(
            df['HighBP'] == 1,
            np.random.randint(130, 180, len(df)),
            np.random.randint(80, 130, len(df))
        )
        
        # Estimate cholesterol
        processed_df['Cholesterol'] = np.where(
            df['HighChol'] == 1,
            np.random.randint(240, 300, len(df)),
            np.random.randint(150, 240, len(df))
        )
        
        processed_df['Smoking'] = df['Smoker']
        processed_df['Alcohol'] = np.where(df['HvyAlcoholConsump'] == 1, 2,
                                           np.random.choice([0, 1], len(df), p=[0.6, 0.4]))
        processed_df['PhysicalActivity'] = np.where(df['PhysActivity'] == 1,
                                                     np.random.choice([1, 2], len(df), p=[0.5, 0.5]),
                                                     0)
        
        processed_df['SleepHours'] = np.where(
            df['GenHlth'] <= 2,
            np.random.uniform(7, 9, len(df)),
            np.random.uniform(5, 7, len(df))
        )
        
        processed_df['StressLevel'] = (df['MentHlth'] / 3).clip(1, 10).astype(int)
        processed_df['FamilyHistory'] = np.random.choice([0, 1], len(df), p=[0.5, 0.5])
        
        # Convert binary to risk levels
        processed_df['DiabetesRisk'] = df['Diabetes_binary']
        medium_risk = (df['Diabetes_binary'] == 0) & ((df['HighBP'] == 1) | (df['HighChol'] == 1) | (df['BMI'] > 30))
        processed_df.loc[medium_risk, 'DiabetesRisk'] = 1
        high_risk = (df['Diabetes_binary'] == 1) & ((df['HighBP'] == 1) & (df['HighChol'] == 1))
        processed_df.loc[high_risk, 'DiabetesRisk'] = 2
        
        return processed_df
        
    except FileNotFoundError:
        print(f"[SURVEY-2] File not found: {filepath}")
        return None

# ============================================
# LOAD AND COMBINE ALL DATASETS
# ============================================

print("\n" + "=" * 70)
print("PHASE 1: DATA LOADING AND PREPROCESSING")
print("=" * 70)

datasets = []
dataset_names = []

# Load each dataset
df_diabetes = preprocess_diabetes_csv('diabetes.csv')
if df_diabetes is not None:
    datasets.append(df_diabetes)
    dataset_names.append('diabetes.csv')

df_heart = preprocess_heart_csv('heart.csv')
if df_heart is not None:
    datasets.append(df_heart)
    dataset_names.append('heart.csv')

df_s1 = preprocess_s1_csv('s1.csv')
if df_s1 is not None:
    datasets.append(df_s1)
    dataset_names.append('s1.csv')

df_s2 = preprocess_s2_csv('s2.csv')
if df_s2 is not None:
    datasets.append(df_s2)
    dataset_names.append('s2.csv')

# Combine all datasets
if len(datasets) > 0:
    print(f"\n[SUCCESS] Loaded {len(datasets)} datasets: {', '.join(dataset_names)}")
    print("[PROCESS] Combining datasets...")
    
    # Ensure all have same columns in same order
    standard_columns = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Cholesterol',
                       'Smoking', 'Alcohol', 'PhysicalActivity', 'SleepHours',
                       'StressLevel', 'FamilyHistory', 'DiabetesRisk']
    
    for i, ds in enumerate(datasets):
        datasets[i] = ds[standard_columns]
    
    df = pd.concat(datasets, ignore_index=True)
    print(f"[SUCCESS] Combined dataset shape: {df.shape}")
    
else:
    print("\n[WARNING] No datasets found. Creating synthetic data...")
    def create_sample_diabetes_data(n_samples=2000):
        np.random.seed(42)
        
        data = {
            'Age': np.random.randint(20, 80, n_samples),
            'BMI': np.random.uniform(18, 45, n_samples),
            'Glucose': np.random.randint(70, 200, n_samples),
            'BloodPressure': np.random.randint(60, 140, n_samples),
            'Cholesterol': np.random.randint(150, 300, n_samples),
            'Smoking': np.random.choice([0, 1], n_samples),
            'Alcohol': np.random.choice([0, 1, 2], n_samples),
            'PhysicalActivity': np.random.choice([0, 1, 2], n_samples),
            'SleepHours': np.random.uniform(4, 10, n_samples),
            'StressLevel': np.random.randint(1, 11, n_samples),
            'FamilyHistory': np.random.choice([0, 1], n_samples),
        }
        
        risk = []
        for i in range(n_samples):
            score = 0
            if data['Age'][i] > 45: score += 1
            if data['BMI'][i] > 30: score += 2
            if data['Glucose'][i] > 140: score += 3
            if data['BloodPressure'][i] > 130: score += 1
            if data['Smoking'][i] == 1: score += 2
            if data['FamilyHistory'][i] == 1: score += 2
            if data['PhysicalActivity'][i] == 0: score += 1
            
            if score <= 3: risk.append(0)
            elif score <= 6: risk.append(1)
            else: risk.append(2)
        
        data['DiabetesRisk'] = risk
        return pd.DataFrame(data)
    
    df = create_sample_diabetes_data(2000)
    dataset_names.append('synthetic_data')

# ============================================
# DATA CLEANING
# ============================================

print("\n" + "=" * 70)
print("PHASE 2: DATA CLEANING AND QUALITY CONTROL")
print("=" * 70)

print(f"\n[INITIAL] Dataset shape: {df.shape}")

# Handle missing values
print("\n[ANALYSIS] Checking for missing values...")
missing_before = df.isnull().sum().sum()
print(f"[FOUND] Total missing values: {missing_before}")

if missing_before > 0:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"[IMPUTED] {col}: Filled with median")

missing_after = df.isnull().sum().sum()
print(f"[RESULT] Missing values after imputation: {missing_after}")

# Remove outliers using IQR method
print("\n[ANALYSIS] Detecting and removing outliers...")
numeric_cols = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Cholesterol', 'SleepHours']
initial_len = len(df)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

outliers_removed = initial_len - len(df)
print(f"[REMOVED] {outliers_removed} outlier rows ({outliers_removed/initial_len*100:.2f}%)")

# Remove duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"[REMOVED] {duplicates} duplicate rows")

print(f"\n[FINAL] Clean dataset: {len(df):,} samples, {df.shape[1]} features")

# Save combined dataset
df.to_csv('healthcare_data.csv', index=False)
print("[SAVED] healthcare_data.csv")

# ============================================
# EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\n[DISTRIBUTION] Target Variable (DiabetesRisk):")
risk_dist = df['DiabetesRisk'].value_counts().sort_index()
for risk_level, count in risk_dist.items():
    risk_name = ['Low Risk', 'Medium Risk', 'High Risk'][int(risk_level)]
    percentage = (count / len(df)) * 100
    print(f"  {risk_name:15s}: {count:6,} samples ({percentage:5.2f}%)")

# Create visualizations with professional styling
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor('#0a1128')

# Risk distribution
ax1 = axes[0,0]
risk_dist.plot(kind='bar', ax=ax1, color=['#10b981', '#f59e0b', '#ef4444'])
ax1.set_title('Risk Level Distribution', fontsize=14, fontweight='bold', color='white')
ax1.set_xlabel('Risk Level (0=Low, 1=Medium, 2=High)', color='white')
ax1.set_ylabel('Count', color='white')
ax1.tick_params(colors='white')
ax1.set_facecolor('#1a2332')
for spine in ax1.spines.values():
    spine.set_edgecolor('#3b82f6')

# Age distribution
ax2 = axes[0,1]
df.boxplot(column='Age', by='DiabetesRisk', ax=ax2)
ax2.set_title('Age Distribution by Risk Level', fontsize=14, fontweight='bold', color='white')
ax2.set_xlabel('Risk Level', color='white')
ax2.set_ylabel('Age', color='white')
ax2.tick_params(colors='white')
ax2.set_facecolor('#1a2332')
for spine in ax2.spines.values():
    spine.set_edgecolor('#3b82f6')

# BMI distribution
ax3 = axes[1,0]
df.boxplot(column='BMI', by='DiabetesRisk', ax=ax3)
ax3.set_title('BMI Distribution by Risk Level', fontsize=14, fontweight='bold', color='white')
ax3.set_xlabel('Risk Level', color='white')
ax3.set_ylabel('BMI', color='white')
ax3.tick_params(colors='white')
ax3.set_facecolor('#1a2332')
for spine in ax3.spines.values():
    spine.set_edgecolor('#3b82f6')

# Glucose distribution
ax4 = axes[1,1]
df.boxplot(column='Glucose', by='DiabetesRisk', ax=ax4)
ax4.set_title('Glucose Distribution by Risk Level', fontsize=14, fontweight='bold', color='white')
ax4.set_xlabel('Risk Level', color='white')
ax4.set_ylabel('Glucose', color='white')
ax4.tick_params(colors='white')
ax4.set_facecolor('#1a2332')
for spine in ax4.spines.values():
    spine.set_edgecolor('#3b82f6')

plt.suptitle('')
plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight', facecolor='#0a1128')
print("[SAVED] eda_analysis.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8), facecolor='#0a1128')
ax = plt.gca()
ax.set_facecolor('#1a2332')
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, 
            cbar_kws={'label': 'Correlation'}, ax=ax)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', color='white', pad=20)
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='#0a1128')
print("[SAVED] correlation_heatmap.png")
plt.close()

# ============================================
# DATA PREPROCESSING FOR MODELING
# ============================================

print("\n" + "=" * 70)
print("PHASE 4: DATA PREPROCESSING FOR MACHINE LEARNING")
print("=" * 70)

X = df.drop('DiabetesRisk', axis=1)
y = df['DiabetesRisk']

print(f"\n[FEATURES] Shape: {X.shape}")
print(f"[TARGET] Shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[SPLIT] Training set: {X_train.shape}")
print(f"[SPLIT] Testing set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[SCALING] Feature normalization completed")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n[SMOTE] Balanced training set: {X_train_balanced.shape}")
print("[SMOTE] Class distribution after balancing:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for cls, count in zip(unique, counts):
    risk_name = ['Low Risk', 'Medium Risk', 'High Risk'][int(cls)]
    print(f"  {risk_name:15s}: {count:6,} samples")

joblib.dump(scaler, 'scaler.pkl')
print("\n[SAVED] scaler.pkl")

# ============================================
# TRAIN MODELS
# ============================================

print("\n" + "=" * 70)
print("PHASE 5: MODEL TRAINING AND EVALUATION")
print("=" * 70)

models = {}
results = {}

# Logistic Regression
print("\n[MODEL 1/5] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr_model.fit(X_train_balanced, y_train_balanced)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"[RESULT] Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
models['Logistic Regression'] = lr_model
results['Logistic Regression'] = lr_accuracy

# Random Forest
print("\n[MODEL 2/5] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"[RESULT] Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
models['Random Forest'] = rf_model
results['Random Forest'] = rf_accuracy

# Gradient Boosting
print("\n[MODEL 3/5] Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42
)
gb_model.fit(X_train_balanced, y_train_balanced)
gb_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"[RESULT] Accuracy: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")
models['Gradient Boosting'] = gb_model
results['Gradient Boosting'] = gb_accuracy

# XGBoost
print("\n[MODEL 4/5] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=7, min_child_weight=3,
    gamma=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"[RESULT] Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
models['XGBoost'] = xgb_model
results['XGBoost'] = xgb_accuracy

# SVM
print("\n[MODEL 5/5] Training Support Vector Machine...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_model.fit(X_train_balanced, y_train_balanced)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"[RESULT] Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
models['SVM'] = svm_model
results['SVM'] = svm_accuracy

# ============================================
# MODEL COMPARISON & EVALUATION
# ============================================

print("\n" + "=" * 70)
print("PHASE 6: MODEL COMPARISON AND SELECTION")
print("=" * 70)

print("\n[RANKINGS] Model Performance:")
for rank, (model_name, accuracy) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"  {rank}. {model_name:25s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n[SELECTED] Best Model: {best_model_name}")
print(f"[ACCURACY] {results[best_model_name]:.4f} ({results[best_model_name]*100:.2f}%)")

# Visualization with professional theme
plt.figure(figsize=(12, 7), facecolor='#0a1128')
ax = plt.gca()
ax.set_facecolor('#1a2332')

model_names = list(results.keys())
accuracies = list(results.values())
colors = ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']
bars = ax.bar(model_names, accuracies, color=colors, edgecolor='#3b82f6', linewidth=2)

ax.set_title('Model Accuracy Comparison', fontsize=18, fontweight='bold', color='white', pad=20)
ax.set_xlabel('Models', fontsize=14, color='white', fontweight='600')
ax.set_ylabel('Accuracy', fontsize=14, color='white', fontweight='600')
ax.set_ylim([min(accuracies) - 0.05, 1.0])
ax.tick_params(colors='white', labelsize=11)
plt.xticks(rotation=45, ha='right')

for spine in ax.spines.values():
    spine.set_edgecolor('#3b82f6')
    spine.set_linewidth(2)

for bar, v in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
            f'{v:.4f}', ha='center', va='bottom', fontweight='bold', 
            color='white', fontsize=12)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='#0a1128')
print("\n[SAVED] model_comparison.png")
plt.close()

# ============================================
# DETAILED EVALUATION
# ============================================

print("\n" + "=" * 70)
print(f"PHASE 7: DETAILED EVALUATION - {best_model_name.upper()}")
print("=" * 70)

best_pred = best_model.predict(X_test_scaled)

print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_test, best_pred, 
                          target_names=['Low Risk', 'Medium Risk', 'High Risk']))

# Confusion Matrix with professional styling
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10, 8), facecolor='#0a1128')
ax = plt.gca()
ax.set_facecolor('#1a2332')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar_kws={'label': 'Count'},
            ax=ax, annot_kws={'size': 14, 'weight': 'bold'})

plt.title(f'Confusion Matrix - {best_model_name}', 
          fontsize=16, fontweight='bold', color='white', pad=20)
plt.ylabel('True Label', fontsize=13, color='white', fontweight='600')
plt.xlabel('Predicted Label', fontsize=13, color='white', fontweight='600')
plt.xticks(color='white', fontsize=11)
plt.yticks(color='white', fontsize=11, rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.tick_params(labelcolor='white')
cbar.set_label('Count', color='white', fontweight='600')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='#0a1128')
print("\n[SAVED] confusion_matrix.png")
plt.close()

# Calculate detailed metrics
precision = precision_score(y_test, best_pred, average='weighted')
recall = recall_score(y_test, best_pred, average='weighted')
f1 = f1_score(y_test, best_pred, average='weighted')

print("\n[PERFORMANCE METRICS]")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n[FEATURE IMPORTANCE] Top Features:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")
    
    plt.figure(figsize=(12, 8), facecolor='#0a1128')
    ax = plt.gca()
    ax.set_facecolor('#1a2332')
    
    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], 
                   color='#3b82f6', edgecolor='#60a5fa', linewidth=2)
    
    ax.set_xlabel('Importance Score', fontsize=13, color='white', fontweight='600')
    ax.set_title('Feature Importance Analysis', fontsize=16, fontweight='bold', 
                 color='white', pad=20)
    ax.tick_params(colors='white', labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#3b82f6')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight', facecolor='#0a1128')
    print("\n[SAVED] feature_importance.png")
    plt.close()

# ============================================
# SAVE MODEL & METADATA
# ============================================

print("\n" + "=" * 70)
print("PHASE 8: MODEL EXPORT AND METADATA GENERATION")
print("=" * 70)

model_filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model, model_filename)
print(f"\n[SAVED] {model_filename}")

metadata = {
    'model_name': best_model_name,
    'accuracy': float(results[best_model_name]),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'features': list(X.columns),
    'target_classes': ['Low Risk', 'Medium Risk', 'High Risk'],
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'total_samples': len(df),
    'datasets_used': len(datasets) if len(datasets) > 0 else 1,
    'dataset_sources': dataset_names,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_version': '1.0',
    'framework': 'Preventra AI Health Risk Predictor'
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("[SAVED] model_metadata.json")

# ============================================
# TEST SAVED MODEL
# ============================================

print("\n" + "=" * 70)
print("PHASE 9: MODEL VALIDATION AND TESTING")
print("=" * 70)

loaded_model = joblib.load(model_filename)
loaded_scaler = joblib.load('scaler.pkl')

print("\n[VALIDATION] Testing with sample patient data...")

sample_patient = {
    'Age': 55,
    'BMI': 32.5,
    'Glucose': 150,
    'BloodPressure': 135,
    'Cholesterol': 240,
    'Smoking': 1,
    'Alcohol': 1,
    'PhysicalActivity': 0,
    'SleepHours': 6.0,
    'StressLevel': 8,
    'FamilyHistory': 1
}

sample_df = pd.DataFrame([sample_patient])
print("\n[SAMPLE] Patient Profile:")
for feature, value in sample_patient.items():
    print(f"  {feature:20s}: {value}")

sample_scaled = loaded_scaler.transform(sample_df)
prediction = loaded_model.predict(sample_scaled)[0]
prediction_proba = loaded_model.predict_proba(sample_scaled)[0]

risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
print(f"\n[PREDICTION] Risk Level: {risk_labels[int(prediction)]}")
print("\n[CONFIDENCE] Probability Distribution:")
for i, prob in enumerate(prediction_proba):
    print(f"  {risk_labels[i]:15s}: {prob:.2%}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE - PREVENTRA MODEL READY")
print("=" * 70)

print(f"""
MODEL SUMMARY
─────────────────────────────────────────────────────────────────────

Selected Algorithm    : {best_model_name}
Model Accuracy        : {results[best_model_name]:.2%}
Precision Score       : {precision:.2%}
Recall Score          : {recall:.2%}
F1 Score              : {f1:.2%}

TRAINING DATASET
─────────────────────────────────────────────────────────────────────

Total Samples         : {len(df):,}
Training Samples      : {len(X_train):,}
Testing Samples       : {len(X_test):,}
Data Sources          : {len(dataset_names)}
Source Files          : {', '.join(dataset_names)}

RISK DISTRIBUTION
─────────────────────────────────────────────────────────────────────

Low Risk              : {risk_dist.get(0, 0):,} samples ({risk_dist.get(0, 0)/len(df)*100:.1f}%)
Medium Risk           : {risk_dist.get(1, 0):,} samples ({risk_dist.get(1, 0)/len(df)*100:.1f}%)
High Risk             : {risk_dist.get(2, 0):,} samples ({risk_dist.get(2, 0)/len(df)*100:.1f}%)

GENERATED FILES
─────────────────────────────────────────────────────────────────────

Model File            : {model_filename}
Scaler File           : scaler.pkl
Metadata File         : model_metadata.json
Dataset File          : healthcare_data.csv
EDA Visualizations    : eda_analysis.png
Correlation Heatmap   : correlation_heatmap.png
Model Comparison      : model_comparison.png
Confusion Matrix      : confusion_matrix.png""")

if hasattr(best_model, 'feature_importances_'):
    print("Feature Importance    : feature_importance.png")

print(f"""
DEPLOYMENT
─────────────────────────────────────────────────────────────────────

Training Date         : {metadata['training_date']}
Model Version         : {metadata['model_version']}
Framework             : {metadata['framework']}
Status                : READY FOR PRODUCTION

─────────────────────────────────────────────────────────────────────

The Preventra AI model is now ready for integration with the 
Streamlit application. Run the following command to start the app:

    streamlit run app.py

─────────────────────────────────────────────────────────────────────
""")

print("=" * 70)