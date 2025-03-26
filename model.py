import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_fraud_target(df):
    # Create a target variable based on multiple indicators with balanced criteria
    
    # Calculate the 90th percentile of claim-to-premium ratio (less strict)
    high_claim_ratio = df['CLAIM_AMOUNT'].div(df['PREMIUM_AMOUNT']).quantile(0.90)
    
    # Calculate suspicious hours (late night/early morning)
    suspicious_hours = (df['INCIDENT_HOUR_OF_THE_DAY'] >= 22) | (df['INCIDENT_HOUR_OF_THE_DAY'] <= 5)
    
    fraud_indicators = (
        # Condition 1: High claim amount relative to premium
        (df['CLAIM_AMOUNT'] > df['PREMIUM_AMOUNT'] * high_claim_ratio) |
        
        # Condition 2: Multiple risk factors
        ((df['days_to_loss'] < 90) &  # Quick claim
         (df['INCIDENT_SEVERITY'] == 'High') & 
         (df['RISK_SEGMENTATION'] == 'High')) |
        
        # Condition 3: Suspicious timing with risk factors
        (suspicious_hours & 
         (df['INCIDENT_SEVERITY'] == 'High') & 
         (df['CLAIM_STATUS'] == 'Pending')) |
        
        # Condition 4: Very high claim amount for high risk
        ((df['RISK_SEGMENTATION'] == 'High') & 
         (df['CLAIM_AMOUNT'] > df['PREMIUM_AMOUNT'] * 3))
    )
    return fraud_indicators.astype(int)

def prepare_features(df):
    # Calculate days to loss
    df['days_to_loss'] = (pd.to_datetime(df['LOSS_DT']) - pd.to_datetime(df['POLICY_EFF_DT'])).dt.days
    
    # Calculate claim to premium ratio
    df['claim_premium_ratio'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
    
    # Select features for the model
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
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()
    
    for col in categorical_columns:
        df_encoded[col] = pd.Categorical(df[col]).codes
    
    return df_encoded[feature_columns]

def main():
    # Load the data
    print("Loading data...")
    df = pd.read_csv('insurance_data.csv')
    
    # Prepare features and target
    X = prepare_features(df)
    y = create_fraud_target(df)
    
    # Print class distribution
    print("\nClass Distribution:")
    print(pd.Series(y).value_counts(normalize=True))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and fit the preprocessor
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Create and train the model with balanced class weights
    print("\nTraining the model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train_scaled)
    y_train_prob = model.predict_proba(X_train_scaled)
    
    # Print training data performance
    print("\nTraining Data Performance:")
    print(classification_report(y_train, y_train_pred))
    print("\nTraining Data Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    
    # Make predictions on test data
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)
    
    # Print test data performance
    print("\nTest Data Performance:")
    print(classification_report(y_test, y_test_pred))
    print("\nTest Data Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save the model and preprocessor
    print("\nSaving model and preprocessor...")
    joblib.dump(model, 'fraud_detection_model.joblib')
    joblib.dump(preprocessor, 'fraud_detection_preprocessor.joblib')
    print("Model and preprocessor saved successfully!")

if __name__ == "__main__":
    main()