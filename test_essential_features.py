import requests
import json

API_URL = "http://localhost:1248"

def test_single_prediction():
    # Test data with only essential features
    test_data = {
        "CLAIM_AMOUNT": 50000,  # High claim amount
        "days_to_loss": 15,     # Quick claim
        "claim_premium_ratio": 2.5,  # High ratio
        "avg_claim_amount": 45000,  # High average
        "quick_claim": 1        # Quick claim flag
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
    # Test data for batch prediction with only essential features
    test_data = {
        "claims": [
            {
                "CLAIM_AMOUNT": 50000,  # High risk claim
                "days_to_loss": 15,
                "claim_premium_ratio": 2.5,
                "avg_claim_amount": 45000,
                "quick_claim": 1
            },
            {
                "CLAIM_AMOUNT": 15000,  # Low risk claim
                "days_to_loss": 180,
                "claim_premium_ratio": 0.8,
                "avg_claim_amount": 12000,
                "quick_claim": 0
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
    print("Testing API with essential features...")
    
    # Test single prediction
    single_pred = test_single_prediction()
    
    # Test batch prediction
    batch_pred = test_batch_prediction()
    
    print("\nTest Summary:")
    print(f"Single Prediction: {'✅' if single_pred else '❌'}")
    print(f"Batch Prediction: {'✅' if batch_pred else '❌'}") 