import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
def load_data(file_path):
    """
    Load the insurance dataset and perform initial preprocessing
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Data Cleaning and Preprocessing
    # Convert date columns to datetime
    date_columns = ['TXN_DATE_TIME', 'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Feature Engineering
    # Calculate days between policy effective date and loss date
    df['days_to_loss'] = (df['LOSS_DT'] - df['POLICY_EFF_DT']).dt.days
    
    # Calculate claim amount to premium ratio
    df['claim_premium_ratio'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
    
    # Create a fraud flag (for demonstration, we'll use some heuristics)
    def flag_potential_fraud(row):
        # Potential fraud indicators
        suspicious_conditions = [
            # Extremely short policy tenure before claim
            row['days_to_loss'] < 30 and row['CLAIM_AMOUNT'] > row['PREMIUM_AMOUNT'] * 2,
            # Unreasonably high claim amount compared to premium
            row['claim_premium_ratio'] > 5,
            # Suspicious incident timing (late night)
            row['INCIDENT_HOUR_OF_THE_DAY'] > 22 or row['INCIDENT_HOUR_OF_THE_DAY'] < 5,
            # Incomplete documentation
            row['POLICE_REPORT_AVAILABLE'] == 0,
            # High severity incidents
            row['INCIDENT_SEVERITY'] == 'High'
        ]
        return int(any(suspicious_conditions))
    
    df['FRAUD_FLAG'] = df.apply(flag_potential_fraud, axis=1)
    
    return df

# Prepare features for machine learning
def prepare_features(df):
    """
    Prepare features for fraud detection model
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
    
    Returns:
        tuple: X (features), y (fraud flag)
    """
    # Select features for the model
    categorical_features = [
        'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 
        'RISK_SEGMENTATION', 'HOUSE_TYPE', 'SOCIAL_CLASS',
        'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS', 'INCIDENT_SEVERITY'
    ]
    
    numerical_features = [
        'PREMIUM_AMOUNT', 'CLAIM_AMOUNT', 'AGE', 'TENURE', 
        'NO_OF_FAMILY_MEMBERS', 'days_to_loss', 'claim_premium_ratio',
        'INCIDENT_HOUR_OF_THE_DAY', 'ANY_INJURY'
    ]
    
    # Prepare the feature matrix and target variable
    X = df[categorical_features + numerical_features]
    y = df['FRAUD_FLAG']
    
    return X, y

# Create and train the fraud detection model
def create_fraud_detection_model(X, y):
    """
    Create and train a fraud detection model
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Fraud flag
    
    Returns:
        tuple: Trained model, preprocessor
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), 
             ['PREMIUM_AMOUNT', 'CLAIM_AMOUNT', 'AGE', 'TENURE', 
              'NO_OF_FAMILY_MEMBERS', 'days_to_loss', 'claim_premium_ratio',
              'INCIDENT_HOUR_OF_THE_DAY', 'ANY_INJURY']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), 
             ['INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 
              'RISK_SEGMENTATION', 'HOUSE_TYPE', 'SOCIAL_CLASS',
              'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS', 'INCIDENT_SEVERITY'])
        ])
    
    # Create a pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42
        ))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, preprocessor

# Main execution function
def main():
    # Load the data
    df = load_data('insurance_data.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Create and train the model
    fraud_model, preprocessor = create_fraud_detection_model(X, y)
    
    # Save the model and preprocessor
    joblib.dump(fraud_model, 'fraud_detection_model.joblib')
    joblib.dump(preprocessor, 'fraud_detection_preprocessor.joblib')
    
    # Provide insights into fraud indicators
    fraud_cases = df[df['FRAUD_FLAG'] == 1]
    print("\nFraud Detection Insights:")
    print(f"Total Potential Fraud Cases: {len(fraud_cases)}")
    print("\nTop Fraud Indicators:")
    
    # Analyze fraud indicators
    indicator_columns = [
        'INSURANCE_TYPE', 'RISK_SEGMENTATION', 
        'INCIDENT_SEVERITY', 'EMPLOYMENT_STATUS'
    ]
    for col in indicator_columns:
        print(f"\n{col} Distribution in Fraud Cases:")
        print(fraud_cases[col].value_counts(normalize=True))

# Prediction function for new data
def predict_fraud(new_data):
    """
    Predict fraud for new insurance claims
    
    Args:
        new_data (pd.DataFrame): New insurance claims data
    
    Returns:
        np.array: Fraud predictions
    """
    # Load the saved model and preprocessor
    model = joblib.load('fraud_detection_model.joblib')
    
    # Preprocess and predict
    predictions = model.predict(new_data)
    return predictions

if __name__ == '__main__':
    main()