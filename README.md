# Titanic Dataset Analysis and Machine Learning Project

## Project Overview
This project demonstrates the use of the Titanic dataset to explore data preprocessing techniques, feature engineering, and machine learning modeling. The dataset contains passenger information, such as demographics and survival status, aboard the RMS Titanic. The project involves:

- Loading and exploring the Titanic dataset.
- Cleaning and preprocessing the data.
- Performing exploratory data analysis (EDA).
- Feature engineering for improved model performance.
- Training and evaluating machine learning models using Random Forest and Logistic Regression.

## Objective
The main objective is to build a robust classification model to predict the survival of passengers on the Titanic using features such as age, sex, class, fare, and family size.

## Dataset
The Titanic dataset contains the following columns:

- **PassengerId**: Unique identifier for each passenger.
- **Pclass**: Passenger class (1st, 2nd, 3rd).
- **Name**: Full name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Ticket fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
- **Survived**: Survival status (0 = No; 1 = Yes).

## Key Steps

### 1. Data Loading and Exploration
- The dataset is loaded from a compressed zip file and read into a pandas DataFrame.
- Basic statistics and data types are analyzed to understand the dataset structure.
- Missing values and unique values are identified for key columns.

### 2. Data Cleaning
- Missing values in numerical columns (`Age`, `Fare`) are filled with their respective medians.
- Missing values in the `Embarked` column are filled with the mode.
- The `Sex` column is encoded into numerical values (`male` = 0, `female` = 1).
- The `Embarked` column is one-hot encoded into `Embarked_Q` and `Embarked_S`.
- Irrelevant columns such as `Name`, `Ticket`, `Cabin`, and `PassengerId` are dropped.

### 3. Feature Engineering
- **FamilySize**: Created by combining `SibSp` and `Parch` to represent family size.
- **AgeGroup**: Binned `Age` into categories such as `Child`, `Teen`, `Adult`, `Middle Aged`, and `Senior`. These categories were one-hot encoded.

### 4. Exploratory Data Analysis (EDA)
- Visualized survival rates by:
  - Gender
  - Family size
  - Passenger class
- Analyzed the distribution of fare values.
- Generated a correlation heatmap to identify relationships between features.

### 5. Model Training and Evaluation
- **Feature Selection**: 
  Selected features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked_Q`, `Embarked_S`, `FamilySize`, and `AgeGroup` (encoded).

- **Data Splitting**: 
  The dataset was split into training (80%) and testing (20%) sets using stratified sampling.

- **Scaling**:
  Applied `StandardScaler` to standardize feature values.

- **Model Training**:
  - Logistic Regression: Trained with scaled features to establish a baseline.
  - Random Forest Classifier: Hyperparameter tuning was performed using `GridSearchCV` to optimize parameters like `n_estimators`, `max_depth`, and `min_samples_split`.

- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Score

### 6. Feature Importance
The importance of features was visualized using a bar plot from the Random Forest model.

##### Feature Importance

![screenshot-localhost_8889-2025 01 27-12_59_28](https://github.com/user-attachments/assets/b428c2ef-7238-4667-b4cb-d8758a655bdc)

## Results
### Best Random Forest Model:
- **Best Parameters**: `{'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}`
- **Accuracy**: 54.5%
- **Confusion Matrix**:
  ```
  [[71 31]
   [60 38]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             0       0.54      0.70      0.61       102
             1       0.55      0.39      0.46        98

      accuracy                           0.55       200
     macro avg       0.55      0.54      0.53       200
  weighted avg       0.55      0.55      0.53       200
  ```
- **ROC-AUC Score**: 0.518

## Challenges
1. **Class Imbalance**: 
   The survival classes (0 and 1) were imbalanced, impacting the model's performance.

2. **Feature Importance**: 
   Some features, like `Pclass` and `Sex`, were found to be more predictive than others.

3. **Hyperparameter Tuning**: 
   Optimizing parameters using GridSearchCV improved the model slightly, but the accuracy remained moderate.

## Future Improvements
- **Advanced Feature Engineering**:
  - Combine features like `FamilySize` and `Pclass` into interaction terms.
  - Explore polynomial transformations of numeric features.

- **Model Experimentation**:
  - Experiment with ensemble models like Gradient Boosting, XGBoost, and CatBoost.
  - Use SMOTE or other resampling techniques to address class imbalance.

- **Evaluation**:
  - Incorporate additional metrics like F2-score for imbalanced datasets.
  - Use cross-validation to ensure model robustness.

## Source

Dataset: [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/waqi786/titanic-dataset)
