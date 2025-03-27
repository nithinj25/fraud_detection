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

# Define the essential features
ESSENTIAL_FEATURES = [
    'CLAIM_AMOUNT',
    'days_to_loss',
    'claim_premium_ratio',
    'avg_claim_amount',
    'quick_claim'
]

# Load the model and preprocessor
model = None
scaler = None

try:
    model = joblib.load('fraud_detection_model.joblib')
    scaler = joblib.load('fraud_detection_preprocessor.joblib')
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

class EssentialClaim(BaseModel):
    CLAIM_AMOUNT: float
    days_to_loss: int
    claim_premium_ratio: float
    avg_claim_amount: float
    quick_claim: int

class BatchEssentialClaims(BaseModel):
    claims: List[EssentialClaim]

@app.get("/")
async def root():
    return {"message": "Insurance Fraud Detection API is running"}

@app.post("/predict", response_model=dict)
async def predict_fraud(claim: EssentialClaim):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame with correct feature names
        input_data = pd.DataFrame([claim.dict()])
        
        # Ensure feature names match training data
        input_data = input_data[ESSENTIAL_FEATURES]
        
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
            "risk_level": "High" if fraud_probability > 0.7 else "Medium" if fraud_probability > 0.3 else "Low",
            "prediction": bool(prediction[0]),
            "fraud_probability": fraud_probability
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=dict)
async def predict_fraud_batch(claims: BatchEssentialClaims):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame with correct feature names
        input_data = pd.DataFrame([claim.dict() for claim in claims.claims])
        
        # Ensure feature names match training data
        input_data = input_data[ESSENTIAL_FEATURES]
        
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
                "risk_level": "High" if fraud_probability > 0.7 else "Medium" if fraud_probability > 0.3 else "Low",
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
    port = int(os.environ.get("PORT", 1248))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 