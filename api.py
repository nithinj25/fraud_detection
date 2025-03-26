from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="API for detecting potential insurance fraud using machine learning",
    version="1.0.0"
)

# Define the input data model
class InsuranceClaim(BaseModel):
    INSURANCE_TYPE: str
    MARITAL_STATUS: str
    EMPLOYMENT_STATUS: str
    RISK_SEGMENTATION: str
    HOUSE_TYPE: str
    SOCIAL_CLASS: str
    CUSTOMER_EDUCATION_LEVEL: str
    CLAIM_STATUS: str
    INCIDENT_SEVERITY: str
    PREMIUM_AMOUNT: float
    CLAIM_AMOUNT: float
    AGE: int
    TENURE: int
    NO_OF_FAMILY_MEMBERS: int
    days_to_loss: int
    claim_premium_ratio: float
    INCIDENT_HOUR_OF_THE_DAY: int
    ANY_INJURY: int

class BatchInsuranceClaims(BaseModel):
    claims: List[InsuranceClaim]

# Load the model at startup
model = None
try:
    model = joblib.load('fraud_detection_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Insurance Fraud Detection API"}

@app.post("/predict", response_model=dict)
async def predict_fraud(claim: InsuranceClaim):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame with correct feature names
        input_data = pd.DataFrame([claim.dict()])
        
        # Ensure feature names match training data
        feature_columns = [
            'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 'RISK_SEGMENTATION',
            'HOUSE_TYPE', 'SOCIAL_CLASS', 'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS',
            'INCIDENT_SEVERITY', 'PREMIUM_AMOUNT', 'CLAIM_AMOUNT', 'AGE', 'TENURE',
            'NO_OF_FAMILY_MEMBERS', 'days_to_loss', 'claim_premium_ratio',
            'INCIDENT_HOUR_OF_THE_DAY', 'ANY_INJURY'
        ]
        
        # Convert categorical variables to numeric
        categorical_columns = [
            'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 'RISK_SEGMENTATION',
            'HOUSE_TYPE', 'SOCIAL_CLASS', 'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS',
            'INCIDENT_SEVERITY'
        ]
        
        for col in categorical_columns:
            input_data[col] = pd.Categorical(input_data[col]).codes
        
        # Select only the required features in the correct order
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Get probability value
        if probability.shape[1] == 1:
            fraud_probability = float(probability[0][0])
        else:
            fraud_probability = float(probability[0][-1])
        
        return {
            "claim_id": "1",
            "risk_level": "High" if claim.RISK_SEGMENTATION == "High" else "Medium" if claim.RISK_SEGMENTATION == "Medium" else "Low",
            "prediction": bool(prediction[0]),
            "fraud_probability": fraud_probability
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=dict)
async def predict_fraud_batch(claims: BatchInsuranceClaims):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame with correct feature names
        input_data = pd.DataFrame([claim.dict() for claim in claims.claims])
        
        # Ensure feature names match training data
        feature_columns = [
            'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 'RISK_SEGMENTATION',
            'HOUSE_TYPE', 'SOCIAL_CLASS', 'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS',
            'INCIDENT_SEVERITY', 'PREMIUM_AMOUNT', 'CLAIM_AMOUNT', 'AGE', 'TENURE',
            'NO_OF_FAMILY_MEMBERS', 'days_to_loss', 'claim_premium_ratio',
            'INCIDENT_HOUR_OF_THE_DAY', 'ANY_INJURY'
        ]
        
        # Convert categorical variables to numeric
        categorical_columns = [
            'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 'RISK_SEGMENTATION',
            'HOUSE_TYPE', 'SOCIAL_CLASS', 'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS',
            'INCIDENT_SEVERITY'
        ]
        
        for col in categorical_columns:
            input_data[col] = pd.Categorical(input_data[col]).codes
        
        # Select only the required features in the correct order
        input_data = input_data[feature_columns]
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Process results
        results = []
        for i, (pred, prob, claim) in enumerate(zip(predictions, probabilities, claims.claims)):
            if prob.shape[0] == 1:
                fraud_probability = float(prob[0])
            else:
                fraud_probability = float(prob[-1])
            
            results.append({
                "claim_id": str(i + 1),
                "risk_level": "High" if claim.RISK_SEGMENTATION == "High" else "Medium" if claim.RISK_SEGMENTATION == "Medium" else "Low",
                "prediction": bool(pred),
                "fraud_probability": fraud_probability
            })
        
        return {
            "total_claims": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8005))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 