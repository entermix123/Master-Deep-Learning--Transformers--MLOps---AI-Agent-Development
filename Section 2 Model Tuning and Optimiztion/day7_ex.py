## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd
## Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, StandardScaler
## Splits data into training and testing sets
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
## random forest classifier - used for human activity recognization
from sklearn.ensemble import RandomForestClassifier
## Metrics to evaluate the model's performance
from sklearn.metrics import accuracy_score, classification_report
## matrix operation and mathematical functions library
import numpy as np

 # Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
 
# Display dataset info
print("Dataset Info:\n")
print(df.info())
print("\n Class Distribution: \n")
print(df['Churn'].value_counts())
print("\n Sample Data:\n", df.head())

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])

# Encode target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Scale numerical features
## Initialize the scaler
scaler = StandardScaler()
## Scale numerical features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Features and Target
## Drop the target from the features
X = df.drop(columns=['Churn'])
## Define the target
y = df['Churn']

# Splitting the data set into training and testing data testing sets
## X and y are features and target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# EValuate inital model
y_pred = rf_model.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred)

print(f"Initial Model Accuracy: {accuracy_initial:.4f}")
print("\n Classification Report: \n", classification_report(y_test, y_pred))

# Define parameter grid
param_dist = {
    'n_estimators': np.arange(50, 200, 10),     # Number of trees
    'max_depth': [None, 5, 10, 15],             # Maximum depth of the tree
    'min_samples_split': [2, 5, 10, 20],        # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]               # Minimum number of samples required to be at a leaf node
}

# Initialze RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),      # Random Forest classifier
    param_distributions=param_dist,                         # Parameter distribution
    n_iter=20,                                              # Number of random parameter combinations
    cv=5,                                                   # Cross-validation
    scoring='accuracy',                                     # Accuracy
    n_jobs=-1,                                              # Use all processors
    random_state=42                                         # Random seed
)

# Perform Randomized Search - This will create our random search
random_search.fit(X_train, y_train)

# Get best parameters
best_params = random_search.best_params_
print(f"Best Parameters (RandomizedSearchCV): {best_params}")

# Train best model
best_model = random_search.best_estimator_

# Predict and Evaluate
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tunes Model Accuracy: {accuracy_tuned:.4f}")
print("\n Classification Report (Tuned Model):\n", classification_report(y_test, y_pred_tuned))

# Evaluate using cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")

