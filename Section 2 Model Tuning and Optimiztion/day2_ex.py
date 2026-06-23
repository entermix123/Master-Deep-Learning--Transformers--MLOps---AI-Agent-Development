from sklearn.datasets import load_iris
## Splits data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
## random forest classifier - ensemble method for classification
from sklearn.ensemble import RandomForestClassifier
# Imports performance metrics to evaluate the model's performance
## Accuracy score which calculates how often the model's predictions are correct
from sklearn.metrics import accuracy_score
## matrix operation and mathematical function
import numpy as np

# Load dataset
data = load_iris()
## X are the features and y is the target
X, y = data.data, data.target

# Split Dataset
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## Setting a random state ensures reproducibility
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset info
print(f"Feature Names: {data.feature_names}")
print(f"Class Names: {data.target_names}")

## Define a grid of hyperparameters for random forest model and perform exhaustive search using the grid search CV
# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Initialize Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                       # ross validation
    scoring='accuracy',
    n_jobs=-1                   # use all the processors on the PC
)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Evaluate best model
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f"Best Hyperparameters (Grid Search): {grid_search.best_params_}")
print(f"Grid Search Accuracy: {accuracy_grid:.4f}")


## Implement a random search
## Define a parameter distribution for random search and evaluate its performance using the Randomizedsearchcv.

# Define hyperparameter distribution
param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 20]
}

# Initialize Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,                        # number of random combinations to try
    cv=5,                             # ross validation
    scoring='accuracy',
    n_jobs=-1,                        # use all the processors on the PC
    random_state=42
)

# Perform Random Search
# That will create our random search
random_search.fit(X_train, y_train)

# Evaluate best model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Best Hyperparameters (Random Search): {random_search.best_params_}")
print(f"Random Search Accuracy: {accuracy_random:.4f}")