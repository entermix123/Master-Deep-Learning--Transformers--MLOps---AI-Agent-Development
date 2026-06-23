# import libraries
from sklearn.datasets import load_breast_cancer
## Splits data into training and testing sets
from sklearn.model_selection import train_test_split
## Data manipulation library allowing you to load and manipulate data in a structured DataFrame format
import pandas as pd
## random forest classifier - used for human activity recognization
from sklearn.ensemble import RandomForestClassifier
# Metrics to evaluate the model's performance
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class
from sklearn.metrics import  accuracy_score, classification_report


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## Setting a random state ensures reproducibility
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset info
print(f"Feature Names: {data.feature_names}")
print(f"Class Names: {data.target_names}")

# Train Random Forest with default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# Predict and evaluate
y_predict_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_predict_default)

print(f"Default Model Accuracy: {accuracy_default:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_predict_default))

# Train Random Forest with adjusted hyperparameters
## Modify some key hyperparameters such as n estimators and maxdepth to observe their impact on the model performance
rf_tuned = RandomForestClassifier(
    n_estimators=400,                   # giving the maximum is better
    max_depth=5,                        # limit of the tree depth
    random_state=42                     # generates the same data every time we run it
)
rf_tuned.fit(X_train, y_train)

# Predict and evaluate
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Model Accuracy: {accuracy_tuned:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred_tuned))