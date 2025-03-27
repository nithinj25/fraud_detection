import requests
import json
import pandas as pd
import time

def test_api():
    # API endpoint - Update this with your Render URL after deployment
    BASE_URL = "https://fraud-detection-87in.onrender.com"  # Replace with your actual Render URL
    
    # Test single prediction with low-risk case
    print("\nTesting single claim prediction (Low Risk Case)...")
    low_risk_claim = {
        "INSURANCE_TYPE": "Home",
        "MARITAL_STATUS": "Married",
        "EMPLOYMENT_STATUS": "Employed",
        "RISK_SEGMENTATION": "Low",
        "HOUSE_TYPE": "Owned",
        "SOCIAL_CLASS": "Middle",
        "CUSTOMER_EDUCATION_LEVEL": "Graduate",
        "CLAIM_STATUS": "Approved",
        "INCIDENT_SEVERITY": "Low",
        "PREMIUM_AMOUNT": 3000,
        "CLAIM_AMOUNT": 5000,
        "AGE": 45,
        "TENURE": 5,
        "NO_OF_FAMILY_MEMBERS": 4,
        "days_to_loss": 180,
        "claim_premium_ratio": 1.67,
        "INCIDENT_HOUR_OF_THE_DAY": 14,
        "ANY_INJURY": 0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=low_risk_claim,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("\nSingle Claim Test Results:")
        print(f"Claim ID: {result['claim_id']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Predicted Fraud: {'Yes' if result['prediction'] else 'No'}")
        print(f"Fraud Probability: {result['fraud_probability']:.2%}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error testing single prediction: {str(e)}")
    
    # Test batch prediction with three different risk levels
    print("\nTesting batch prediction with different risk levels...")
    batch_claims = {
        "claims": [
            # High Risk Case
            {
                "INSURANCE_TYPE": "Auto",
                "MARITAL_STATUS": "Single",
                "EMPLOYMENT_STATUS": "Employed",
                "RISK_SEGMENTATION": "High",
                "HOUSE_TYPE": "Owned",
                "SOCIAL_CLASS": "Upper",
                "CUSTOMER_EDUCATION_LEVEL": "Graduate",
                "CLAIM_STATUS": "Pending",
                "INCIDENT_SEVERITY": "High",
                "PREMIUM_AMOUNT": 5000,
                "CLAIM_AMOUNT": 25000,
                "AGE": 35,
                "TENURE": 2,
                "NO_OF_FAMILY_MEMBERS": 3,
                "days_to_loss": 15,
                "claim_premium_ratio": 5,
                "INCIDENT_HOUR_OF_THE_DAY": 23,
                "ANY_INJURY": 1
            },
            # Medium Risk Case
            {
                "INSURANCE_TYPE": "Home",
                "MARITAL_STATUS": "Married",
                "EMPLOYMENT_STATUS": "Self-employed",
                "RISK_SEGMENTATION": "Medium",
                "HOUSE_TYPE": "Rented",
                "SOCIAL_CLASS": "Middle",
                "CUSTOMER_EDUCATION_LEVEL": "High School",
                "CLAIM_STATUS": "Approved",
                "INCIDENT_SEVERITY": "Medium",
                "PREMIUM_AMOUNT": 3000,
                "CLAIM_AMOUNT": 15000,
                "AGE": 45,
                "TENURE": 5,
                "NO_OF_FAMILY_MEMBERS": 4,
                "days_to_loss": 180,
                "claim_premium_ratio": 3,
                "INCIDENT_HOUR_OF_THE_DAY": 14,
                "ANY_INJURY": 0
            },
            # Low Risk Case
            {
                "INSURANCE_TYPE": "Health",
                "MARITAL_STATUS": "Married",
                "EMPLOYMENT_STATUS": "Employed",
                "RISK_SEGMENTATION": "Low",
                "HOUSE_TYPE": "Owned",
                "SOCIAL_CLASS": "Upper",
                "CUSTOMER_EDUCATION_LEVEL": "Graduate",
                "CLAIM_STATUS": "Approved",
                "INCIDENT_SEVERITY": "Low",
                "PREMIUM_AMOUNT": 4000,
                "CLAIM_AMOUNT": 6000,
                "AGE": 50,
                "TENURE": 8,
                "NO_OF_FAMILY_MEMBERS": 2,
                "days_to_loss": 365,
                "claim_premium_ratio": 1.5,
                "INCIDENT_HOUR_OF_THE_DAY": 10,
                "ANY_INJURY": 0
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json=batch_claims,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("\nBatch Test Results:")
        print(f"Total Claims Processed: {result['total_claims']}")
        print("\nDetailed Results:")
        for claim_result in result['results']:
            print(f"\nClaim ID: {claim_result['claim_id']}")
            print(f"Risk Level: {claim_result['risk_level']}")
            print(f"Predicted Fraud: {'Yes' if claim_result['prediction'] else 'No'}")
            print(f"Fraud Probability: {claim_result['fraud_probability']:.2%}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error testing batch prediction: {str(e)}")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running on the deployed URL")
    print("\nWaiting for 2 seconds to ensure server is ready...")
    time.sleep(2)
    test_api() 