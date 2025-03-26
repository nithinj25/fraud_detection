import requests
import json
import time

def test_legitimate_claims():
    # API endpoint
    BASE_URL = "http://localhost:8005"  # Updated to use port 8005
    
    # Test Case 1: Legitimate Home Insurance Claim (Low Risk)
    print("\nTest Case 1: Legitimate Home Insurance Claim (Low Risk)")
    home_claim = {
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
    
    # Test Case 2: Legitimate Health Insurance Claim (Medium Risk)
    print("\nTest Case 2: Legitimate Health Insurance Claim (Medium Risk)")
    health_claim = {
        "INSURANCE_TYPE": "Health",
        "MARITAL_STATUS": "Married",
        "EMPLOYMENT_STATUS": "Employed",
        "RISK_SEGMENTATION": "Medium",
        "HOUSE_TYPE": "Owned",
        "SOCIAL_CLASS": "Upper",
        "CUSTOMER_EDUCATION_LEVEL": "Graduate",
        "CLAIM_STATUS": "Pending",
        "INCIDENT_SEVERITY": "Medium",
        "PREMIUM_AMOUNT": 4000,
        "CLAIM_AMOUNT": 12000,
        "AGE": 35,
        "TENURE": 3,
        "NO_OF_FAMILY_MEMBERS": 3,
        "days_to_loss": 90,
        "claim_premium_ratio": 3,
        "INCIDENT_HOUR_OF_THE_DAY": 10,
        "ANY_INJURY": 1
    }
    
    # Test Case 3: Legitimate Auto Insurance Claim (Medium Risk)
    print("\nTest Case 3: Legitimate Auto Insurance Claim (Medium Risk)")
    auto_claim = {
        "INSURANCE_TYPE": "Auto",
        "MARITAL_STATUS": "Single",
        "EMPLOYMENT_STATUS": "Employed",
        "RISK_SEGMENTATION": "Medium",
        "HOUSE_TYPE": "Rented",
        "SOCIAL_CLASS": "Middle",
        "CUSTOMER_EDUCATION_LEVEL": "High School",
        "CLAIM_STATUS": "Approved",
        "INCIDENT_SEVERITY": "Medium",
        "PREMIUM_AMOUNT": 2500,
        "CLAIM_AMOUNT": 8000,
        "AGE": 28,
        "TENURE": 2,
        "NO_OF_FAMILY_MEMBERS": 1,
        "days_to_loss": 150,
        "claim_premium_ratio": 3.2,
        "INCIDENT_HOUR_OF_THE_DAY": 16,
        "ANY_INJURY": 0
    }
    
    # Test Case 4: Legitimate Life Insurance Claim (Low Risk)
    print("\nTest Case 4: Legitimate Life Insurance Claim (Low Risk)")
    life_claim = {
        "INSURANCE_TYPE": "Life",
        "MARITAL_STATUS": "Married",
        "EMPLOYMENT_STATUS": "Self-employed",
        "RISK_SEGMENTATION": "Low",
        "HOUSE_TYPE": "Owned",
        "SOCIAL_CLASS": "Upper",
        "CUSTOMER_EDUCATION_LEVEL": "Graduate",
        "CLAIM_STATUS": "Approved",
        "INCIDENT_SEVERITY": "Low",
        "PREMIUM_AMOUNT": 5000,
        "CLAIM_AMOUNT": 15000,
        "AGE": 50,
        "TENURE": 10,
        "NO_OF_FAMILY_MEMBERS": 4,
        "days_to_loss": 365,
        "claim_premium_ratio": 3,
        "INCIDENT_HOUR_OF_THE_DAY": 8,
        "ANY_INJURY": 0
    }
    
    # Test Case 5: Legitimate Property Insurance Claim (Low Risk)
    print("\nTest Case 5: Legitimate Property Insurance Claim (Low Risk)")
    property_claim = {
        "INSURANCE_TYPE": "Property",
        "MARITAL_STATUS": "Married",
        "EMPLOYMENT_STATUS": "Employed",
        "RISK_SEGMENTATION": "Low",
        "HOUSE_TYPE": "Owned",
        "SOCIAL_CLASS": "Middle",
        "CUSTOMER_EDUCATION_LEVEL": "Graduate",
        "CLAIM_STATUS": "Approved",
        "INCIDENT_SEVERITY": "Low",
        "PREMIUM_AMOUNT": 3500,
        "CLAIM_AMOUNT": 7000,
        "AGE": 40,
        "TENURE": 7,
        "NO_OF_FAMILY_MEMBERS": 3,
        "days_to_loss": 200,
        "claim_premium_ratio": 2,
        "INCIDENT_HOUR_OF_THE_DAY": 12,
        "ANY_INJURY": 0
    }
    
    # Test all claims
    claims = [home_claim, health_claim, auto_claim, life_claim, property_claim]
    
    for i, claim in enumerate(claims, 1):
        try:
            print(f"\nTesting Claim {i}:")
            print(f"Insurance Type: {claim['INSURANCE_TYPE']}")
            print(f"Claim Amount: ${claim['CLAIM_AMOUNT']}")
            
            response = requests.post(
                f"{BASE_URL}/predict",
                json=claim,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            print("\nPrediction Results:")
            print(f"Claim ID: {result['claim_id']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Predicted Fraud: {'Yes' if result['prediction'] else 'No'}")
            print(f"Fraud Probability: {result['fraud_probability']:.2%}")
            
            # Add a small delay between requests
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error testing claim {i}: {str(e)}")
    
    # Test batch prediction with all legitimate claims
    print("\nTesting Batch Prediction with All Legitimate Claims")
    try:
        batch_response = requests.post(
            f"{BASE_URL}/predict_batch",
            json={"claims": claims},
            headers={"Content-Type": "application/json"}
        )
        batch_response.raise_for_status()
        batch_result = batch_response.json()
        
        print(f"\nTotal Claims Processed: {batch_result['total_claims']}")
        print("\nDetailed Results:")
        for claim_result in batch_result['results']:
            print(f"\nClaim ID: {claim_result['claim_id']}")
            print(f"Risk Level: {claim_result['risk_level']}")
            print(f"Predicted Fraud: {'Yes' if claim_result['prediction'] else 'No'}")
            print(f"Fraud Probability: {claim_result['fraud_probability']:.2%}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error in batch prediction: {str(e)}")

if __name__ == "__main__":
    print("Starting Legitimate Claims Test...")
    print("Make sure the API server is running")
    print("\nWaiting for 2 seconds to ensure server is ready...")
    time.sleep(2)
    test_legitimate_claims() 