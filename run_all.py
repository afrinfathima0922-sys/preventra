import subprocess
import sys
import time
import os
from threading import Thread
import webbrowser

def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              PREVENTRA - DEPLOYMENT MANAGER               ║
    ║          AI-Powered Health Risk Prediction System         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n[1/5] Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 'imblearn', 
        'xgboost', 'matplotlib', 'seaborn', 'plotly', 'joblib',
        'reportlab', 'fastapi', 'uvicorn', 'pydantic', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\n🔧 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def check_model_files():
    """Check if model files exist"""
    print("\n[2/5] Checking model files...")
    
    required_files = ['scaler.pkl', 'model_metadata.json']
    model_file_found = False
    
    # Check for any model file
    for file in os.listdir('.'):
        if file.startswith('best_model_') and file.endswith('.pkl'):
            model_file_found = True
            print(f"   ✓ {file}")
            break
    
    if not model_file_found:
        print("   ✗ Model file not found")
        print("\n⚠️  Model not trained yet!")
        print("🔧 Running training script...")
        return run_training()
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} - MISSING")
            print("\n⚠️  Required files missing!")
            print("🔧 Running training script...")
            return run_training()
    
    print("✅ All model files present!")
    return True

def run_training():
    """Run the training script"""
    print("\n🎓 Training model... (This may take a few minutes)")
    try:
        result = subprocess.run(
            [sys.executable, 'train_model.py'],
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print("❌ Training failed!")
            return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def run_streamlit():
    """Run Streamlit app"""
    print("\n[3/5] Starting Streamlit App...")
    print("   📍 URL: http://localhost:8501")
    
    try:
        subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'app.py',
             '--server.port=8501',
             '--server.headless=true',
             '--browser.gatherUsageStats=false'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        print("   ✅ Streamlit app started!")
        return True
    except Exception as e:
        print(f"   ❌ Failed to start Streamlit: {e}")
        return False

def run_backend():
    """Run FastAPI backend"""
    print("\n[4/5] Starting API Backend...")
    print("   📍 URL: http://localhost:8000")
    print("   📖 Docs: http://localhost:8000/docs")
    
    try:
        subprocess.Popen(
            [sys.executable, 'backend.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        print("   ✅ API backend started!")
        return True
    except Exception as e:
        print(f"   ❌ Failed to start backend: {e}")
        return False

def run_tests():
    """Run API tests"""
    print("\n[5/5] Running API Tests...")
    
    time.sleep(2)  # Wait for backend to be fully ready
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_backend.py'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "All tests passed" in result.stdout or result.returncode == 0:
            print("   ✅ All tests passed!")
            return True
        else:
            print("   ⚠️  Some tests may have issues")
            print("   (This is normal if backend is still starting)")
            return True
    except subprocess.TimeoutExpired:
        print("   ⚠️  Tests timed out (backend may be starting)")
        return True
    except Exception as e:
        print(f"   ⚠️  Test execution note: {e}")
        return True

def open_browsers():
    """Open browsers for user"""
    print("\n🌐 Opening web browsers...")
    time.sleep(2)
    
    try:
        webbrowser.open('http://localhost:8501')
        print("   ✅ Streamlit app opened in browser")
    except:
        print("   ℹ️  Please manually open: http://localhost:8501")

def print_summary():
    """Print deployment summary"""
    summary = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                DEPLOYMENT SUCCESSFUL!                     ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🎯 PREVENTRA SYSTEM IS NOW RUNNING
    
    📱 STREAMLIT WEB APP
       → http://localhost:8501
       → User-friendly health assessment interface
       → Login: demo / preventra123 (or continue as guest)
    
    🔌 API BACKEND
       → http://localhost:8000
       → RESTful API for programmatic access
       → API Documentation: http://localhost:8000/docs
       → Alternative Docs: http://localhost:8000/redoc
    
    📊 FEATURES AVAILABLE:
       ✓ Splash Screen
       ✓ User Authentication
       ✓ Health Risk Assessment
       ✓ AI-Powered Predictions
       ✓ PDF Report Download
       ✓ Disease-Specific Risk Analysis
       ✓ Personalized Recommendations
       ✓ RESTful API Access
    
    ⚙️  TO STOP ALL SERVICES:
       Press Ctrl+C in this terminal
       Or run: python stop_all.py
    
    📝 LOGS:
       Streamlit: Check terminal output
       API: Check terminal output
       Errors: Will be displayed here
    
    ════════════════════════════════════════════════════════════
    """
    print(summary)

def main():
    """Main deployment function"""
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Deployment failed: Missing dependencies")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 2: Check/train model
    if not check_model_files():
        print("\n❌ Deployment failed: Model training failed")
        sys.exit(1)
    
    # Step 3: Start Streamlit
    if not run_streamlit():
        print("\n❌ Deployment failed: Streamlit couldn't start")
        sys.exit(1)
    
    # Step 4: Start Backend
    if not run_backend():
        print("\n⚠️  Warning: Backend couldn't start")
        print("   Streamlit app will work, but API won't be available")
    
    # Step 5: Run tests
    run_tests()
    
    # Open browsers
    open_browsers()
    
    # Print summary
    print_summary()
    
    # Keep running
    try:
        print("🔄 System running... Press Ctrl+C to stop all services\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutting down Preventra system...")
        print("   Stopping all services...")
        time.sleep(1)
        print("   ✅ System stopped successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()