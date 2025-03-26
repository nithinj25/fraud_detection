# Insurance Fraud Detection API

A machine learning-based API for detecting potential insurance fraud using FastAPI and scikit-learn.

## Features

- Single claim prediction endpoint
- Batch prediction endpoint
- Risk level assessment
- Fraud probability scoring
- RESTful API with OpenAPI documentation

## Prerequisites

- Python 3.12.0 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python api.py
```

2. Access the API documentation:
- Open your browser and navigate to: http://localhost:8005/docs
- Or use the alternative documentation at: http://localhost:8005/redoc

3. Test the API:
```bash
python test_legitimate_claims.py
```

## API Endpoints

### Single Prediction
- **Endpoint**: `/predict`
- **Method**: POST
- **Description**: Predict fraud probability for a single insurance claim

### Batch Prediction
- **Endpoint**: `/predict_batch`
- **Method**: POST
- **Description**: Predict fraud probability for multiple insurance claims

## Model Features

The model uses the following features for prediction:
- Insurance Type
- Marital Status
- Employment Status
- Risk Segmentation
- House Type
- Social Class
- Customer Education Level
- Claim Status
- Incident Severity
- Premium Amount
- Claim Amount
- Age
- Tenure
- Number of Family Members
- Days to Loss
- Claim Premium Ratio
- Incident Hour of the Day
- Any Injury

## License

This project is licensed under the MIT License - see the LICENSE file for details. 