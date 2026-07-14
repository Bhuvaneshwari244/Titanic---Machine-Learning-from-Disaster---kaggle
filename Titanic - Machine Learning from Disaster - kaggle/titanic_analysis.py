"""
Titanic Survival Prediction - Advanced Model with Ensemble Methods
Kaggle Competition: Titanic - Machine Learning from Disaster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')

def load_data():
    """Load the training and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nSurvival Rate:")
    if 'Survived' in df.columns:
        print(f"Overall: {df['Survived'].mean():.2%}")
        print("\nBy Gender:")
        print(df.groupby('Sex')['Survived'].mean())
        print("\nBy Class:")
        print(df.groupby('Pclass')['Survived'].mean())

def feature_engineering(train_df, test_df):
    """Create and engineer features"""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    
    # Combine datasets for consistent feature engineering
    full_data = [train_df, test_df]
    
    for dataset in full_data:
        # Fill missing Age with median by Pclass and Sex
        dataset['Age'].fillna(dataset.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
        
        # Fill missing Embarked with mode
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
        
        # Fill missing Fare with median
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
        
        # Create Title feature from Name
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                      'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                      'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        
        # Create FamilySize feature
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        
        # Create IsAlone feature
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        
        # Create Age bins
        dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 20, 40, 60, np.inf], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Create Fare bins
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    print("Features created: Title, FamilySize, IsAlone, AgeBin, FareBin")
    return train_df, test_df

def prepare_features(train_df, test_df):
    """Prepare features for modeling"""
    print("\n" + "="*50)
    print("PREPARING FEATURES FOR MODELING")
    print("="*50)
    
    # Select features to use
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']
    
    # Create feature dataframes
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df['Survived'].copy()
    
    # Convert categorical variables to dummy variables
    X_train = pd.get_dummies(X_train, columns=['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin'])
    X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin'])
    
    # Ensure both datasets have the same columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, X_test, y_train

def train_models(X_train, y_train):
    """Train and evaluate multiple models"""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'model': model,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
        print(f"Cross-validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['mean_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['mean_score']:.4f}")
    
    # Train best model on full training data
    best_model.fit(X_train, y_train)
    
    return best_model, best_model_name

def generate_predictions(model, X_test, test_df):
    """Generate predictions for submission"""
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created: submission.csv")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted survival rate: {predictions.mean():.2%}")
    
    return submission

def main():
    """Main execution function"""
    print("="*50)
    print("TITANIC SURVIVAL PREDICTION")
    print("="*50)
    
    # Load data
    train_df, test_df = load_data()
    
    # Explore data
    explore_data(train_df)
    
    # Feature engineering
    train_df, test_df = feature_engineering(train_df, test_df)
    
    # Prepare features
    X_train, X_test, y_train = prepare_features(train_df, test_df)
    
    # Train models
    best_model, model_name = train_models(X_train, y_train)
    
    # Generate predictions
    submission = generate_predictions(best_model, X_test, test_df)
    
    print("\n" + "="*50)
    print("COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Review the submission.csv file")
    print("2. Go to https://www.kaggle.com/competitions/titanic")
    print("3. Click 'Submit Predictions'")
    print("4. Upload submission.csv")
    print("\nGood luck! 🚢")

if __name__ == "__main__":
    main()
