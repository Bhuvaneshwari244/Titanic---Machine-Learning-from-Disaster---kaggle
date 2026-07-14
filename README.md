# Titanic - Machine Learning from Disaster 🚢

Kaggle's legendary getting-started competition - predict survival on the Titanic using machine learning.

## 🏆 Competition Overview

- **Competition**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Challenge**: Predict which passengers survived the Titanic shipwreck
- **Evaluation Metric**: Classification Accuracy
- **Dataset**: 891 training samples, 418 test samples with passenger information

## 📊 Results

| Model | CV Accuracy | Features | Description |
|-------|-------------|----------|-------------|
| Baseline Random Forest | ~78% | Basic features | Simple feature engineering |
| **Advanced Ensemble** | **~83%** | **25+ features** | **Hypertuned RF + GB + LR + SVC voting ensemble** |

## 🚀 Quick Start

### 1. Installation

```bash
py -m pip install -r requirements.txt
```

### 2. Run Analysis

**Basic Model:**
```bash
py titanic_analysis.py
```
Generates: `submission.csv`

**Advanced Model:**
```bash
py titanic_advanced.py
```
Generates: `submission_advanced.csv`

## 📁 Project Structure

```
├── train.csv                    # Training data (891 passengers)
├── test.csv                     # Test data (418 passengers)
├── titanic_analysis.py          # Baseline model
├── titanic_advanced.py          # Advanced ensemble model
├── submission.csv               # Basic model predictions
├── submission_advanced.csv      # Advanced model predictions
├── submission_description.txt   # Submission notes
├── gender_submission.csv        # Example submission format
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 The Challenge

On April 15, 1912, during her maiden voyage, the RMS Titanic sank after colliding with an iceberg. There weren't enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

**Key Question**: What sorts of people were more likely to survive?

## 📋 Dataset Features

### Available Features:
- **PassengerId**: Unique identifier
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🔧 Feature Engineering

### Baseline Model Features:
1. **Title Extraction**: Extract titles from names (Mr, Mrs, Miss, Master, Rare)
2. **FamilySize**: SibSp + Parch + 1
3. **IsAlone**: Binary indicator (FamilySize == 1)
4. **AgeBin**: Age groups (Child, Teen, Adult, Middle, Senior)
5. **FareBin**: Quartile-based fare categories

### Advanced Model Features:
All baseline features plus:
6. **Age*Class**: Interaction feature
7. **HasCabin**: Binary indicator for cabin information
8. **Deck**: Extracted deck letter from cabin
9. **Fare_Per_Person**: Fare divided by family size
10. **Title Encoding**: Numerical encoding of titles
11. **Standardized Features**: Scaled using StandardScaler

### Missing Value Handling:
- **Age**: Filled with median grouped by Title and Pclass
- **Embarked**: Filled with mode ('S')
- **Fare**: Filled with median by Pclass
- **Cabin**: Converted to binary HasCabin feature

## 🤖 Models

### Baseline Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, max_depth=5
- **Accuracy**: ~78% CV

### Advanced Ensemble Model

**Components**:
1. **Random Forest** (Hyperparameter tuned)
   - n_estimators: 100-300
   - max_depth: 4-7
   - min_samples_split: 2-4

2. **Gradient Boosting** (Hyperparameter tuned)
   - n_estimators: 100-200
   - learning_rate: 0.05-0.1
   - max_depth: 3-5

3. **Logistic Regression**
   - max_iter: 1000
   - Regularization: default

4. **Support Vector Classifier**
   - kernel: RBF (default)
   - probability: True

**Ensemble Strategy**: Soft Voting Classifier
- Combines predictions from all 4 models
- Uses probability estimates for voting
- **Final Accuracy**: ~83% CV

## 📈 Model Performance

**Cross-Validation Strategy**: 5-Fold Cross-Validation

**Individual Model Scores**:
- Random Forest: ~82%
- Gradient Boosting: ~81%
- Logistic Regression: ~80%
- SVC: ~81%

**Ensemble Score**: ~83% ⭐

## 🎓 Key Insights

### Survival Patterns:
1. **Gender**: Women had much higher survival rate (~74%) vs men (~19%)
2. **Class**: 1st class passengers had higher survival rate (~63%) vs 3rd class (~24%)
3. **Age**: Children had better survival chances
4. **Family**: Small families (2-4) had better survival than solo or large families
5. **Fare**: Higher fare correlates with survival (proxy for class)
6. **Embarked**: Cherbourg (C) passengers had higher survival (1st class concentration)

### Important Features (by importance):
1. Title (gender/social status indicator)
2. Fare
3. Age
4. Pclass
5. FamilySize

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🔮 Potential Improvements

- [ ] Deep learning models (Neural Networks)
- [ ] Stacking with meta-learner
- [ ] More sophisticated feature engineering:
  - Family survival rates
  - Ticket group analysis
  - Cabin position analysis
- [ ] Text mining on names and tickets
- [ ] Bayesian optimization for hyperparameters
- [ ] Advanced imputation techniques (KNN, MICE)
- [ ] Outlier detection and handling

## 📝 Submission Format

CSV file with exactly 418 entries plus header:

```csv
PassengerId,Survived
892,0
893,1
894,0
...
```

## 🎯 How to Submit to Kaggle

1. Visit: https://www.kaggle.com/c/titanic/submit
2. Click "Submit Predictions"
3. Upload `submission_advanced.csv`
4. Add description (see `submission_description.txt`)
5. Submit and check leaderboard!

## 📚 Learning Resources

- **Kaggle Tutorial**: [Alexis Cook's Titanic Tutorial](https://www.kaggle.com/alexisbcook/titanic-tutorial)
- **Kaggle Notebooks**: Browse top-scoring notebooks for insights
- **Discussion Forum**: [Titanic Discussion](https://www.kaggle.com/c/titanic/discussion)

## 🏅 Competition Info

- **Type**: Getting Started / Binary Classification
- **Timeline**: Rolling leaderboard (2 month window)
- **Participants**: 1,461,575+ entrants
- **Submissions**: 10 per day limit
- **Prizes**: Knowledge (no cash prize)

## 🔍 About the Data

The training set includes:
- 891 passengers
- 342 survivors (38.4%)
- 549 casualties (61.6%)

The test set includes:
- 418 passengers
- Survival status unknown (to be predicted)

## 📖 Historical Context

The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean on April 15, 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history.

The disaster was partly caused by the lack of lifeboats - there were only enough for about half the passengers and crew. The evacuation followed the protocol "women and children first," which is reflected in the survival patterns visible in the data.

## 📝 License

This project is for educational purposes as part of the Kaggle competition.

## 🙏 Acknowledgments

- Will Cukierski and Kaggle for creating this classic competition
- The data science community for sharing insights and tutorials
- All contributors to public notebooks and discussions

## 👤 Author

**Bhuvaneshwari**
- GitHub: [@Bhuvaneshwari244](https://github.com/Bhuvaneshwari244)

---

⭐ If you found this helpful, please star the repository!

**Good luck with your predictions! 🚢⚓**
