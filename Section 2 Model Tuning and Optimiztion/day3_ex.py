# Import libraries
## pip install optuna xgboost sklearn
# Dataset
from sklearn.datasets import load_breast_cancer
## Splits data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
## This scales features to have a mean of zero and standard deviation of one helping improve model stability.
from sklearn.preprocessing import StandardScaler
## Xgboost classifier
from xgboost import XGBClassifier
# Imports performance metrics to evaluate the model's performance
## Accuracy score which calculates how often the model's predictions are correct
from sklearn.metrics import accuracy_score
## Flexible and user-friendly library for hyperparameter optimization
## Support dynamic search spaces and pruning of unpromising trails
import optuna


# Load the dataset
data = load_breast_cancer()
## Set features (X) and target (y)
X, y = data.data, data.target

## Split the data set into training and testing
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
## Fit the scaler to the training data
X_train = scaler.fit_transform(X_train)
## Apply the scaler to the test data
X_test = scaler.transform(X_test)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train a baseline XGBoost model
## Initialuze the model
baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)
## Train the model
baseline_model.fit(X_train, y_train)

# Evaluate the model
## Predict on the test data
baseline_preds = baseline_model.predict(X_test)
## Calculate accuracy
baseline_accuracy = accuracy_score(y_test, baseline_preds)
## Print accuracy
print(f"Baseline XGBoost Accuracy: {baseline_accuracy:.4f}")

# Define the objective function for Optuna
def objective(trial):
    params = {
        ## number of trees
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        ## tree depth
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        ## learning rate
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        ## subsample
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        ## colsample by tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        ## gamma parameter
        'gamma': trial.suggest_float('gamma', 0, 5),
        ## L1 regularization - Lasso regression penalty parameter - controls the amount of shrinkage applied to the model
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        ## L2 regularization - Ridge regression penalty parameter - controls the amount of shrinkage applied to the model
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    
    # Train XGBoost model with suggested params
    ## Initialize the model
    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    ## Train the model
    model.fit(X_train, y_train)
    
    # Evaluate model on validation set
    ## Predict on the test data
    preds = model.predict(X_test)
    ## Calculate accuracy
    accuracy = accuracy_score(y_test, preds)
    
    # Return accuracy
    return accuracy

# Create an Optuna study
## Optimize the objective function
study = optuna.create_study(direction="maximize")
## Run the optimization
study.optimize(objective, n_trials=50)

# Best hyperparameters
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy: ", study.best_value)

# Define parameter grid
## Initialize parameter grid
param_grid = {
    ## Number of trees in the ensemble - controls the complexity of the model
    'n_estimators': [100, 200, 300],
    ## Depth of each tree - controls the complexity of the model
    'max_depth': [3, 5, 7],
    ## Learning rate - controls the amount of shrinkage applied to the model
    'learning_rate': [0.01, 0.1, 0.2],
    ## Subsample ratio of training instances - controls the amount of data used for training each tree
    'subsample': [0.6, 0.8, 1.0]
}

# Train XGBoost with Grid Search
## Initialize Grid Search
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss',random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)
# Train Grid Search
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\n\n\nGrid Search Best Parameters: ", grid_search.best_params_)
print("Grid Search Best Accuracy:", grid_search.best_score_)

# Define parameter distributions
param_dist = {
    'n_estimators': [50,100,200,300,400],               # Number of trees
    'max_depth': [3, 5, 7, 9],                          # Depth of each tree
    'learning_rate': [0.01, 0.05, 0.1, 0.2],            # Learning rate
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],             # Subsample ratio
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]       # Feature subsample ratio
}

# Train XGBoost with Random Search
## Initialize Random Search
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=50,                          # Number of random combinations
    scoring='accuracy',                 # Scoring metric
    cv=3,                               # Cross-validation
    verbose=1,                          # Verbosity
    random_state=42                     # Random seed 
)

## Train Random Search
random_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\n\n\nRandom Search Best Parameters:", random_search.best_params_)
print("Random Search Best Accuracy:", random_search.best_score_)


# Baseline XGBoost Accuracy: 0.9561
# Optuna Best Accuracy:  0.9649122807017544
# Grid Search Best Accuracy: 0.9757900546067155
# Random Search Best Accuracy: 0.9758045776693388