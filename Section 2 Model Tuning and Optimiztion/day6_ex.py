# Load Dataset
from sklearn.datasets import load_iris
# Splits data into training and testing sets, Import GridSearchCV and RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# Gradient Boosting Classifier - ensemble method for classification
from sklearn.ensemble import GradientBoostingClassifier
# Imports performance metrics to evaluate the model's performance
from sklearn.metrics import accuracy_score, classification_report
# Support Vector Machine
from sklearn.svm import SVC
## matrix operation and mathematical functions library
import numpy as np


# Load dataset
data = load_iris()
# X are the features and y is the target
X, y = data.data, data.target

# Splitting the data set into training and testing data testing sets
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset Loaded and Split Successfully")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],         # Number of trees
    'learning_rate': [0.01, 0.1, 0.2],      # Learning rate
    'max_depth': [3, 5, 7]                  # Depth of each tree
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),      # Model
    param_grid=param_grid,      # Parameter grid
    scoring='accuracy',         # Accuracy
    cv=5,                       # Cross-validation
    n_jobs=-1                   # Use all processors
)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Get best parameters and score
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print(f"Best Parameters (GridSeachCV): {best_params_grid}")
print(f"Best Cross-Validation Accuracy (GridSearchCV): {best_score_grid:.4f}")

# Get best model
best_grid_model = grid_search.best_estimator_

# Predict and evaluate
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f"Test Accuracy (GridSearchCV): {accuracy_grid:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred_grid))

# Define parameter distribution
param_dist = {
    'C': np.logspace(-3, 3, 10),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Initalize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist, 
    n_iter=20,          # Number of random combinations
    scoring='accuracy', # Accuracy
    cv=5,               # Cross-validation
    n_jobs=-1,          # Use all processors
    random_state=42     # Random seed
)

# Perform Randomized Search
random_search.fit(X_train, y_train)

# Get best parameters and score
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f"Best Parameters (RandomizedSearchCV): {best_params_random}")
print(f"Bestr Cross-Validation Accuracy (RandomizedSearchCV): {best_score_random:.4f}")

# Get best model
best_random_model = random_search.best_estimator_

# Predict and evaluate
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Test Accuracy (RandomizedSeachCV): {accuracy_random:.4f}")
print("\n Classsification Report:\n", classification_report(y_test, y_pred_random))
