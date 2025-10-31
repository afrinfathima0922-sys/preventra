"""
Test script for Preventra Health API
Run this after starting backend.py to verify everything works
"""

import requests
import json
from datetime import datetime

# API Base URL
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health_check():
    """Test 1: Health Check Endpoint"""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running!")
            print(f"   Model: {data['model_name']}")
            print(f"   Accuracy: {data['accuracy']:.2%}")
            print(f"   Status: {data['status']}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend!")
        print("   Make sure backend.py is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_low_risk():
    """Test 2: Prediction - Low Risk Patient"""
    print_section("TEST 2: Prediction - Low Risk Patient")
    
    # Healthy patient data
    patient_data = {
        "age": 30,
        "bmi": 22.5,
        "glucose": 90,
        "blood_pressure": 115,
        "cholesterol": 180,
        "smoking": 0,
        "alcohol": 0,
        "physical_activity": 2,
        "sleep_hours": 8.0,
        "stress_level": 3,
        "family_history": 0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"\n   Overall Risk: {result['overall_risk']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"\n   Disease Risks:")
            print(f"      Diabetes: {result['disease_risks']['diabetes']:.1f}%")
            print(f"      Heart Attack: {result['disease_risks']['heart_attack']:.1f}%")
            print(f"      Obesity: {result['disease_risks']['obesity']:.1f}%")
            print(f"\n   Risk Factors: {len(result['risk_factors'])}")
            for factor in result['risk_factors'][:3]:
                print(f"      • {factor}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_high_risk():
    """Test 3: Prediction - High Risk Patient"""
    print_section("TEST 3: Prediction - High Risk Patient")
    
    # High-risk patient data
    patient_data = {
        "age": 65,
        "bmi": 35.0,
        "glucose": 180,
        "blood_pressure": 160,
        "cholesterol": 280,
        "smoking": 1,
        "alcohol": 2,
        "physical_activity": 0,
        "sleep_hours": 5.0,
        "stress_level": 9,
        "family_history": 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"\n   Overall Risk: {result['overall_risk']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"\n   Probabilities:")
            print(f"      Low: {result['probabilities']['low']:.1%}")
            print(f"      Medium: {result['probabilities']['medium']:.1%}")
            print(f"      High: {result['probabilities']['high']:.1%}")
            print(f"\n   Disease Risks:")
            print(f"      Diabetes: {result['disease_risks']['diabetes']:.1f}%")
            print(f"      Heart Attack: {result['disease_risks']['heart_attack']:.1f}%")
            print(f"      Obesity: {result['disease_risks']['obesity']:.1f}%")
            print(f"\n   Recommendations: {len(result['recommendations'])}")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"      {i}. {rec}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test 4: Get Model Information"""
    print_section("TEST 4: Model Information")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved!")
            print(f"\n   Model Name: {data['model_name']}")
            print(f"   Accuracy: {data['accuracy']}")
            print(f"   Training Date: {data['training_date']}")
            print(f"   Training Samples: {data['training_samples']}")
            print(f"   Features: {len(data['features'])}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_prediction():
    """Test 5: Batch Prediction"""
    print_section("TEST 5: Batch Prediction")
    
    # Multiple patients
    patients = [
        {
            "age": 40,
            "bmi": 25.0,
            "glucose": 100,
            "blood_pressure": 120,
            "cholesterol": 200,
            "smoking": 0,
            "alcohol": 1,
            "physical_activity": 1,
            "sleep_hours": 7.0,
            "stress_level": 5,
            "family_history": 0
        },
        {
            "age": 55,
            "bmi": 32.0,
            "glucose": 140,
            "blood_pressure": 145,
            "cholesterol": 240,
            "smoking": 1,
            "alcohol": 2,
            "physical_activity": 0,
            "sleep_hours": 6.0,
            "stress_level": 8,
            "family_history": 1
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json=patients
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            print(f"\n   Total Patients: {result['total']}")
            print(f"   Successful: {result['successful']}")
            print(f"\n   Results:")
            for i, patient_result in enumerate(result['results'], 1):
                if 'error' not in patient_result:
                    print(f"      Patient {i}: {patient_result['overall_risk']}")
                else:
                    print(f"      Patient {i}: Error - {patient_result['error']}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_data():
    """Test 6: Error Handling - Invalid Data"""
    print_section("TEST 6: Error Handling")
    
    # Invalid data (age out of range)
    invalid_data = {
        "age": 150,  # Invalid age
        "bmi": 25.0,
        "glucose": 100,
        "blood_pressure": 120,
        "cholesterol": 200,
        "smoking": 0,
        "alcohol": 1,
        "physical_activity": 1,
        "sleep_hours": 7.0,
        "stress_level": 5,
        "family_history": 0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data
        )
        
        if response.status_code == 422:  # Validation error expected
            print("✅ Validation working correctly!")
            print("   Backend properly rejected invalid data")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 " + "="*58)
    print("  PREVENTRA HEALTH API - TESTING SUITE")
    print("  " + "="*58)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  " + "="*58)
    
    tests = [
        ("Health Check", test_health_check),
        ("Low Risk Prediction", test_prediction_low_risk),
        ("High Risk Prediction", test_prediction_high_risk),
        ("Model Information", test_model_info),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_invalid_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%\n")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print("  🎉 All tests passed! Backend is working perfectly!")
    else:
        print("  ⚠️  Some tests failed. Check the errors above.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()