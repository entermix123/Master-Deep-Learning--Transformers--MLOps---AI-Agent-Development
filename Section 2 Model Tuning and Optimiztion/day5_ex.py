## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd
## Splits data into training and testing sets, performs cross-validation and evaluates model performance
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
## Random Forest classifier
from sklearn.ensemble import RandomForestClassifier


# Load Dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Display dataset info
print("Dataset Info:\n")
print(df.info)
print("\n Class Distribution:\n")
print(df['Class'].value_counts())           # 'Class' is the fraud or not fraud column in the dataset

# Define Features and target
X = df.drop(columns=['Class'])     # drop the target column and leave the features
y = df['Class']                  # define the target

# Splitting the data set into training and testing data testing sets
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate model
rf_model = RandomForestClassifier(random_state=42)
scores_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

# Print results
print(f"K-fold cross validation scores: {scores_kfold}")
print(f"Mean Accuracy (K-Fold): {scores_kfold.mean():.2f}")

# Initialize Stratified K-Fold validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate model
scores_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

# Compare results
print(f"Stratified K-fold cross validation scores: {scores_stratified}")
print(f"Mean Accuracy (Stratified K-Fold): {scores_stratified.mean():.2f}")
