import requests
import json

API_URL = "https://fraud-detection-87in.onrender.com"

def test_api_health():
    try:
        response = requests.get(f"{API_URL}/docs")
        if response.status_code == 200:
            print("✅ API is running and accessible")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing API: {str(e)}")
        return False

def test_single_prediction():
    # Test data for a single claim
    test_data = {
        "INSURANCE_TYPE": "Vehicle",
        "MARITAL_STATUS": "Single",
        "EMPLOYMENT_STATUS": "Employed",
        "RISK_SEGMENTATION": "Low",
        "HOUSE_TYPE": "Rented",
        "SOCIAL_CLASS": "Middle",
        "CUSTOMER_EDUCATION_LEVEL": "Graduate",
        "CLAIM_STATUS": "Pending",
        "INCIDENT_SEVERITY": "Minor",
        "PREMIUM_AMOUNT": 5000,
        "CLAIM_AMOUNT": 3000,
        "AGE": 35,
        "TENURE": 2,
        "NO_OF_FAMILY_MEMBERS": 2,
        "days_to_loss": 30,
        "claim_premium_ratio": 0.6,
        "INCIDENT_HOUR_OF_THE_DAY": 14,
        "ANY_INJURY": "No"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print("\nSingle Prediction Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error in single prediction test: {str(e)}")
        return False

def test_batch_prediction():
    # Test data for batch prediction
    test_data = {
        "claims": [
            {
                "INSURANCE_TYPE": "Vehicle",
                "MARITAL_STATUS": "Single",
                "EMPLOYMENT_STATUS": "Employed",
                "RISK_SEGMENTATION": "Low",
                "HOUSE_TYPE": "Rented",
                "SOCIAL_CLASS": "Middle",
                "CUSTOMER_EDUCATION_LEVEL": "Graduate",
                "CLAIM_STATUS": "Pending",
                "INCIDENT_SEVERITY": "Minor",
                "PREMIUM_AMOUNT": 5000,
                "CLAIM_AMOUNT": 3000,
                "AGE": 35,
                "TENURE": 2,
                "NO_OF_FAMILY_MEMBERS": 2,
                "days_to_loss": 30,
                "claim_premium_ratio": 0.6,
                "INCIDENT_HOUR_OF_THE_DAY": 14,
                "ANY_INJURY": "No"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict_batch",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print("\nBatch Prediction Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error in batch prediction test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing deployed API...")
    
    # Test API health
    health_check = test_api_health()
    
    # Test single prediction
    single_pred = test_single_prediction()
    
    # Test batch prediction
    batch_pred = test_batch_prediction()
    
    print("\nTest Summary:")
    print(f"Health Check: {'✅' if health_check else '❌'}")
    print(f"Single Prediction: {'✅' if single_pred else '❌'}")
    print(f"Batch Prediction: {'✅' if batch_pred else '❌'}") 