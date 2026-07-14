"""
Titanic Survival Prediction - Advanced Model with Ensemble Methods
Higher Accuracy Version with Hyperparameter Tuning and Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"Training: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def advanced_feature_engineering(train_df, test_df):
    """Create advanced features for better prediction"""
    print("\nAdvanced Feature Engineering...")
    
    full_data = [train_df, test_df]
    
    for dataset in full_data:
        # Extract Title from Name
        dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                      'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                      'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        
        # Map titles to numbers
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        
        # Fill Age based on Title and Pclass
        dataset['Age'].fillna(dataset.groupby(['Title', 'Pclass'])['Age'].transform('median'), inplace=True)
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        
        # Map Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        
        # Fill Embarked
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        
        # Fill Fare
        dataset['Fare'].fillna(dataset.groupby('Pclass')['Fare'].transform('median'), inplace=True)
        
        # Create FamilySize
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        
        # Create IsAlone
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        
        # Create Age*Class interaction
        dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
        
        # Create FareBin
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=False, duplicates='drop')
        
        # Create AgeBin
        dataset['AgeBin'] = pd.cut(dataset['Age'], 5, labels=False)
        
        # Cabin feature - has cabin or not
        dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
        
        # Extract Deck from Cabin
        dataset['Deck'] = dataset['Cabin'].apply(lambda s: s[0] if pd.notna(s) else 'M')
        deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'M': 0}
        dataset['Deck'] = dataset['Deck'].map(deck_mapping)
        
        # Create family survival feature
        dataset['Family'] = dataset['SibSp'] + dataset['Parch']
        dataset['Family'].loc[dataset['Family'] > 0] = 1
        dataset['Family'].loc[dataset['Family'] == 0] = 0
        
        # Fare per person
        dataset['Fare_Per_Person'] = dataset['Fare'] / (dataset['FamilySize'] + 1)
        
        # Drop unnecessary columns
        dataset.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True, errors='ignore')
    
    print("Features created successfully!")
    return train_df, test_df

def prepare_data(train_df, test_df):
    """Prepare features and target"""
    print("\nPreparing data for modeling...")
    
    # Save PassengerId for submission
    passenger_ids = pd.read_csv('test.csv')['PassengerId']
    
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df
    
    # Ensure same columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, passenger_ids

def train_ensemble_model(X_train, y_train):
    """Train ensemble model with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODEL")
    print("="*60)
    
    # Define models
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svc = SVC(probability=True, random_state=42)
    
    # Hyperparameter tuning for Random Forest
    print("\nTuning Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }
    
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Best RF score: {rf_grid.best_score_:.4f}")
    
    # Hyperparameter tuning for Gradient Boosting
    print("\nTuning Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4]
    }
    
    gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_
    print(f"Best GB params: {gb_grid.best_params_}")
    print(f"Best GB score: {gb_grid.best_score_:.4f}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr.fit(X_train, y_train)
    lr_score = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"LR CV score: {lr_score:.4f}")
    
    # Train SVC
    print("\nTraining SVC...")
    svc.fit(X_train, y_train)
    svc_score = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"SVC CV score: {svc_score:.4f}")
    
    # Create Voting Ensemble
    print("\nCreating Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('lr', lr),
            ('svc', svc)
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    voting_score = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"\nVoting Ensemble CV score: {voting_score:.4f}")
    
    print("\n" + "="*60)
    print(f"FINAL MODEL ACCURACY: {voting_score:.4f}")
    print("="*60)
    
    return voting_clf

def generate_submission(model, X_test, passenger_ids):
    """Generate submission file"""
    print("\nGenerating predictions...")
    
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    submission.to_csv('submission_advanced.csv', index=False)
    print(f"\nSubmission file created: submission_advanced.csv")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted survival rate: {predictions.mean():.2%}")
    
    return submission

def main():
    """Main execution"""
    print("="*60)
    print("TITANIC ADVANCED PREDICTION")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Feature engineering
    train_df, test_df = advanced_feature_engineering(train_df, test_df)
    
    # Prepare data
    X_train, X_test, y_train, passenger_ids = prepare_data(train_df, test_df)
    
    # Train ensemble model
    model = train_ensemble_model(X_train, y_train)
    
    # Generate submission
    submission = generate_submission(model, X_test, passenger_ids)
    
    print("\n" + "="*60)
    print("COMPLETE! 🚢")
    print("="*60)
    print("\nUpload submission_advanced.csv to Kaggle!")
    print("Expected accuracy: 80-84% on public leaderboard")

if __name__ == "__main__":
    main()
