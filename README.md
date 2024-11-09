# Titanic Dataset Analysis and Logistic Regression Model

### Project Overview

This project demonstrates the use of data loading, exploration, basic data analysis techniques, and machine learning modeling using the Titanic dataset. The dataset provides various features related to passengers aboard the RMS Titanic, such as demographics, survival status, and other factors. The project involves:

- Loading and exploring the Titanic dataset.
- Cleaning the data, handling missing values, and encoding categorical variables.
- Visualizing data trends.
- Training a Logistic Regression model to predict the survival status of passengers.

### Objective

The main objective of this project is to demonstrate the application of machine learning techniques, particularly logistic regression, to predict passenger survival on the Titanic based on various features like age, sex, class, and fare.

### Dataset

The dataset used for this project is the Titanic Dataset. It contains the following columns:

- PassengerId: Unique identifier for each passenger.
- Pclass: Passenger class (1st, 2nd, 3rd).
- Name: Full name of the passenger.
- Sex: Gender of the passenger.
- Age: Age of the passenger.
- SibSp: Number of siblings/spouses aboard.
- Parch: Number of parents/children aboard.
- Ticket: Ticket number.
- Fare: Ticket fare.
- Embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
- Survived: Survival status (0 = No; 1 = Yes).

### Key Steps

1. Data Loading and Exploration

- The dataset is loaded using pandas.read_csv() from a zipped file.
- We perform basic data exploration such as displaying the first few rows of the dataset and generating summary statistics using describe().
- We check for missing values and explore unique values in key columns like Sex and Embarked.

2. Data Cleaning

- Missing Values: The Age and Fare columns are filled with their median values. Any rows with missing values in the Embarked column are dropped.
- Categorical Variables: The Sex column is encoded into numerical values, where 'male' is mapped to 0 and 'female' to 1. The Embarked column is one-hot encoded into Embarked_Q and Embarked_S.
- Feature Selection: Irrelevant columns such as Name, Ticket, and PassengerId are dropped from the dataset.

3. Model Training

- Logistic Regression: The cleaned dataset is split into training and testing sets (80-20 split). A Logistic Regression model is trained to predict survival status based on the selected features.
- Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report, which provide insights into the precision, recall, and F1-score for each class.

4. Model Evaluation

The logistic regression model has the following evaluation results: Accuracy: 55.5%

### Challenges

- **Convergence Warning**: During training, a convergence warning was encountered, which was addressed by scaling the features and increasing the number of iterations in the Logistic Regression model.
- **Data Quality**: Handling missing data and properly encoding categorical variables were critical steps in preparing the dataset for modeling.

### Conclusion

This project demonstrates the process of loading, exploring, cleaning, and preparing a dataset for machine learning. Using logistic regression, we built a predictive model to estimate the survival probability of Titanic passengers based on features such as age, sex, and passenger class. While the model's performance (accuracy of 55.5%) can be improved, it serves as a strong foundation for learning about data preprocessing, feature engineering, and classification tasks.

### Future Improvements

- Feature Engineering: Explore additional feature combinations and polynomial features for potential improvements in model performance.
- Model Selection: Experiment with other classification models such as Random Forest, Gradient Boosting, or XGBoost.
- Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to optimize hyperparameters and improve model performance.

### Source

https://www.kaggle.com/datasets/waqi786/titanic-dataset
