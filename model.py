import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the insurance data"""
    print("Loading data...")
    df = pd.read_csv('insurance_data.csv')
    
    # Print column names to verify
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Print missing values information
    print("\nMissing values before handling:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Handle missing values in numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            print(f"Filling missing values in {col} with mean")
            df[col] = df[col].fillna(df[col].mean())
    
    # Convert categorical variables to numeric
    categorical_columns = [
        'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 'RISK_SEGMENTATION',
        'HOUSE_TYPE', 'SOCIAL_CLASS', 'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS',
        'INCIDENT_SEVERITY'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            if df[col].isnull().any():
                print(f"Filling missing values in {col} with mode")
                df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = pd.Categorical(df[col]).codes
        else:
            print(f"Warning: Column {col} not found in the dataset")
    
    # Calculate additional features
    df['claim_premium_ratio'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
    df['claim_premium_ratio'] = df['claim_premium_ratio'].replace([np.inf, -np.inf], 0)
    df['claim_premium_ratio'] = df['claim_premium_ratio'].fillna(0)
    
    # Handle date columns
    date_columns = {
        'LOSS_DT': 'days_to_loss',
        'POLICY_EFF_DT': 'policy_tenure',
        'INCIDENT_TIME': 'INCIDENT_HOUR_OF_THE_DAY'
    }
    
    for date_col, new_col in date_columns.items():
        if date_col in df.columns:
            if date_col == 'INCIDENT_TIME':
                df[new_col] = pd.to_datetime(df[date_col], errors='coerce').dt.hour
            else:
                df[new_col] = pd.to_datetime(df[date_col], errors='coerce').dt.dayofyear
            if df[new_col].isnull().any():
                print(f"Filling missing values in {new_col} with median")
                df[new_col] = df[new_col].fillna(df[new_col].median())
        else:
            print(f"Warning: Date column {date_col} not found in the dataset")
            if new_col == 'days_to_loss':
                df[new_col] = 30  # Default to 30 days
            elif new_col == 'policy_tenure':
                df[new_col] = 365  # Default to 1 year
            elif new_col == 'INCIDENT_HOUR_OF_THE_DAY':
                df[new_col] = 12  # Default to noon
    
    # Convert ANY_INJURY to numeric if it exists
    if 'ANY_INJURY' in df.columns:
        if df['ANY_INJURY'].isnull().any():
            print("Filling missing values in ANY_INJURY with 'No'")
            df['ANY_INJURY'] = df['ANY_INJURY'].fillna('No')
        df['ANY_INJURY'] = df['ANY_INJURY'].map({'Yes': 1, 'No': 0})
    else:
        print("Warning: ANY_INJURY column not found in the dataset")
        df['ANY_INJURY'] = 0
    
    # Add more sophisticated features
    df['age_risk'] = df['AGE'].apply(lambda x: 1 if x < 25 or x > 65 else 0)
    
    # Calculate customer-level features if CUSTOMER_ID exists
    if 'CUSTOMER_ID' in df.columns:
        df['claim_frequency'] = df.groupby('CUSTOMER_ID')['CLAIM_AMOUNT'].transform('count')
        df['avg_claim_amount'] = df.groupby('CUSTOMER_ID')['CLAIM_AMOUNT'].transform('mean')
        df['claim_amount_std'] = df.groupby('CUSTOMER_ID')['CLAIM_AMOUNT'].transform('std')
        df['claim_amount_std'] = df['claim_amount_std'].fillna(0)
    else:
        print("Warning: CUSTOMER_ID column not found in the dataset")
        df['claim_frequency'] = 1
        df['avg_claim_amount'] = df['CLAIM_AMOUNT']
        df['claim_amount_std'] = 0
    
    # Calculate suspicious patterns
    df['suspicious_timing'] = ((df['INCIDENT_HOUR_OF_THE_DAY'] >= 22) | 
                             (df['INCIDENT_HOUR_OF_THE_DAY'] <= 5)).astype(int)
    df['high_claim_ratio'] = (df['claim_premium_ratio'] > df['claim_premium_ratio'].quantile(0.95)).astype(int)
    df['quick_claim'] = (df['days_to_loss'] < 30).astype(int)
    
    # Create fraud target variable
    df['FRAUD_FLAG'] = create_fraud_target(df)
    
    # Select features for modeling
    feature_columns = [
        'CLAIM_AMOUNT',
        'days_to_loss',
        'claim_premium_ratio',
        'avg_claim_amount',
        'quick_claim'
    ]
    
    # Filter only existing columns
    feature_columns = [col for col in feature_columns if col in df.columns]
    print("\nUsing features:", feature_columns)
    
    X = df[feature_columns]
    y = df['FRAUD_FLAG']
    
    # Final check for any remaining NaN values
    if X.isnull().any().any():
        print("\nWarning: Still have missing values in features:")
        print(X.isnull().sum()[X.isnull().sum() > 0])
        print("Filling remaining missing values with 0")
        X = X.fillna(0)
    
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))
    
    return X, y

def create_fraud_target(df):
    """Create more sophisticated fraud target based on multiple indicators"""
    # Calculate percentiles for thresholds
    claim_premium_threshold = df['claim_premium_ratio'].quantile(0.75)  # More lenient threshold
    claim_amount_threshold = df['CLAIM_AMOUNT'].quantile(0.90)
    
    # Define suspicious hours (10 PM to 5 AM)
    suspicious_hours = ((df['INCIDENT_HOUR_OF_THE_DAY'] >= 22) | 
                       (df['INCIDENT_HOUR_OF_THE_DAY'] <= 5))
    
    # Calculate claim amount statistics
    claim_amount_mean = df['CLAIM_AMOUNT'].mean()
    claim_amount_std = df['CLAIM_AMOUNT'].std()
    
    # Create individual risk factors
    high_claim_ratio = df['claim_premium_ratio'] > claim_premium_threshold
    quick_claim = df['days_to_loss'] < 30
    high_risk_policy = df['RISK_SEGMENTATION'] == 'High'
    major_incident = df['INCIDENT_SEVERITY'] == 'Major'
    new_policy = df['TENURE'] < 90
    very_high_claim = df['CLAIM_AMOUNT'] > claim_amount_threshold
    suspicious_timing = ((df['INCIDENT_HOUR_OF_THE_DAY'] >= 22) | 
                        (df['INCIDENT_HOUR_OF_THE_DAY'] <= 5))
    pending_claim = df['CLAIM_STATUS'] == 'Pending'
    high_age_risk = df['age_risk'] == 1
    
    # Create fraud indicators with more combinations
    fraud_indicators = (
        # Condition 1: High claim ratio with quick claim
        (high_claim_ratio & quick_claim)
        |
        # Condition 2: High risk policy with major incident
        (high_risk_policy & major_incident)
        |
        # Condition 3: Suspicious timing with pending claim
        (suspicious_timing & pending_claim & high_claim_ratio)
        |
        # Condition 4: Very high claim amount with new policy
        (very_high_claim & new_policy)
        |
        # Condition 5: Multiple medium risk factors
        ((high_claim_ratio | quick_claim | suspicious_timing).astype(int) +
         (high_risk_policy | major_incident | pending_claim).astype(int) +
         (new_policy | high_age_risk).astype(int) >= 4)
        |
        # Condition 6: Extreme claim amount
        (df['CLAIM_AMOUNT'] > claim_amount_mean + 3 * claim_amount_std)
    )
    
    # Convert to integer and print distribution
    fraud_flags = fraud_indicators.astype(int)
    fraud_percentage = (fraud_flags.sum() / len(fraud_flags)) * 100
    print(f"\nFraud Detection Results:")
    print(f"Total cases: {len(fraud_flags)}")
    print(f"Fraud cases: {fraud_flags.sum()}")
    print(f"Fraud percentage: {fraud_percentage:.2f}%")
    
    return fraud_flags

def train_model(X, y):
    """Train an ensemble of models with advanced techniques"""
    print("\nTraining model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Create and train multiple models
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_balanced, y_train_balanced)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluate on test set
        test_score = model.score(X_test_scaled, y_test)
        print(f"Test set score: {test_score:.4f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
    
    # Feature selection using the best model
    selector = SelectFromModel(best_model, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()
    print("\nSelected features:", selected_features)
    
    return best_model, scaler, selected_features

def main():
    # Load and prepare data
    X, y = load_data()
    
    # Train model
    model, scaler, selected_features = train_model(X, y)
    
    # Save model and preprocessor
    print("\nSaving model and preprocessor...")
    joblib.dump(model, 'fraud_detection_model.joblib')
    joblib.dump(scaler, 'fraud_detection_preprocessor.joblib')
    joblib.dump(selected_features, 'selected_features.joblib')
    print("Model and preprocessor saved successfully!")

if __name__ == "__main__":
    main()