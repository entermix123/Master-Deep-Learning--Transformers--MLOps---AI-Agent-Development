## Load data
from sklearn.datasets import fetch_california_housing
## Splits data into training and testing sets
from sklearn.model_selection import train_test_split
# import libraries
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd
## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression, Ridge, Lasso
## mean_squared_error and R2_score are metrics used to evaluate the performance of the model
## mean_squared_error measures the mean squared error of prediction
from sklearn.metrics import mean_squared_error

# Load dataset
california = fetch_california_housing()
## X are the features and y is the target
X, y = california.data, california.target
## feature_names are the names of the features
feature_names = california.feature_names

# Split the data set into training and testing
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset info
print("Feature Names:\n", feature_names)
print("\n Sample Data:\n", pd.DataFrame(X, columns=feature_names).head())

# Train linear regression model without regularization
## Initlialize the model
lr_model = LinearRegression()
## Train the model on the training data
lr_model.fit(X_train, y_train)

# Predict and evaluate
## y_pred are the predicted values on the test data
y_pred = lr_model.predict(X_test)
## mean_squared_error measures the mean squared error of prediction
mse_lr = mean_squared_error(y_test, y_pred)

## print the mean squared error
print(f"Linear Regression MSE (No Regularization): {mse_lr:.4f}")
## print the coefficients
print("Coefficients:\n", lr_model.coef_)

# Train Ridge regression model
## Initialize the ridge model, alpha is the regularization parameter
ridge_model = Ridge(alpha=0.1)
## Train the model on the training data
ridge_model.fit(X_train, y_train)

# Predict and evaluate
## y_pred are the predicted values on the test data
y_pred_ridge = ridge_model.predict(X_test)
## mean_squared_error measures the mean squared error of prediction
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

## print the mean squared error
print(f"Ridge Regression MSE: {mse_ridge:.4f}")
## print the coefficients
print("Coefficients:\n", ridge_model.coef_)

# Train Lasso regression model
## Initialize the lasso model, alpha is the regularization parameter
lasso_model = Lasso(alpha=0.1)
## Train the model on the training data
lasso_model.fit(X_train, y_train)

# Predict and evaluate the lasso model
## y_pred are the predicted values on the test data
y_pred_lasso = lasso_model.predict(X_test)
## mean_squared_error measures the mean squared error of prediction
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

## print the mean squared error
print(f"Lasso Regression MSE: {mse_lasso:.4f}")
## print the coefficients
print("Coefficients:\n", lasso_model.coef_)

# Compare models
## Print the mean squared error for each model
print(f"Linear Regression MSE (No Regularization): {mse_lr:.4f}")
print(f"Ridge Regression MSE: {mse_ridge:.4f}")
print(f"Lasso Regression MSE: {mse_lasso:.4f}")

## Print the coefficients for each model
print("Linear Regression Coefficients:\n", lr_model.coef_)
print("Ridge Regression Coefficients:\n", ridge_model.coef_)
print("Lasso Regression Coefficients:\n", lasso_model.coef_)

## Print the number of features for each model
print("Number of Features (Linear Regression):", lr_model.n_features_in_)
print("Number of Features (Ridge Regression):", ridge_model.n_features_in_)
print("Number of Features (Lasso Regression):", lasso_model.n_features_in_)
