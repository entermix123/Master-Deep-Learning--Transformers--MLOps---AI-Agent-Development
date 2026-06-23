# import libraries
from sklearn.datasets import fetch_openml
# Splits data into training and testing sets
from sklearn.model_selection import train_test_split
# Data manipulation library for structured DataFrame format
import pandas as pd
# Random Forest classifier - ensemble model using multiple decision trees
from sklearn.ensemble import RandomForestClassifier
# Metrics to evaluate the model's performance:
# - accuracy_score: overall percentage of correct predictions
# - classification_report: per-class precision, recall, and F1-score
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
# Credit-g (German Credit) is a binary classification dataset with 1,000 samples
# and 20 features. It predicts whether a loan applicant is a good or bad credit risk.
# Its overlapping classes and mix of noisy/informative features make it sensitive
# to hyperparameter choices — unlike simple datasets where defaults already excel.
credit = fetch_openml('credit-g', version=1, as_frame=True)
X = pd.get_dummies(credit.data)   # encode categorical features as numeric columns
y = (credit.target == 'good').astype(int)  # convert target to binary: 1 = good, 0 = bad

# Split dataset
# 80% of the data is used for training, 20% for testing.
# random_state=42 ensures the same split every time the script is run (reproducibility).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display dataset info
print(f"Dataset: German Credit (credit-g)")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Number of features after encoding: {X_train.shape[1]}")
print(f"Class distribution in test set:\n{y_test.value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT MODEL
# Scikit-learn defaults: n_estimators=100, max_depth=None (grows full trees),
# min_samples_split=2, max_features='sqrt', class_weight=None.
# Full-depth trees tend to overfit on noisy datasets like this one,
# and ignoring class imbalance hurts recall on the minority class (bad credit).
# ─────────────────────────────────────────────────────────────────────────────
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

y_pred_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

print("=" * 55)
print("DEFAULT MODEL")
print("=" * 55)
print(f"Accuracy: {accuracy_default:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default, target_names=["bad credit", "good credit"]))

# ─────────────────────────────────────────────────────────────────────────────
# TUNED MODEL
# Key changes and their reasoning:
#
#   n_estimators=200    – more trees reduce variance without major cost;
#                         100 (default) is sometimes too few for noisy data.
#
#   max_depth=8         – caps tree depth to prevent overfitting on noise;
#                         None (default) allows trees to memorise training data.
#
#   min_samples_split=10 – requires more evidence before splitting a node,
#                          which smooths decision boundaries.
#
#   min_samples_leaf=4  – each leaf must cover at least 4 samples,
#                         further reducing overfitting on edge cases.
#
#   max_features=0.6    – each split considers 60% of features instead of
#                         sqrt (≈37%), adding more diversity between trees.
#
#   class_weight='balanced' – adjusts weights inversely proportional to class
#                             frequency, improving recall on the minority class
#                             (bad credit applicants are harder to detect).
# ─────────────────────────────────────────────────────────────────────────────
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features=0.6,
    class_weight='balanced',
    random_state=42
)
rf_tuned.fit(X_train, y_train)

y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print("=" * 55)
print("TUNED MODEL")
print("=" * 55)
print(f"Accuracy: {accuracy_tuned:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=["bad credit", "good credit"]))

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# The tuned model trades a small drop in raw accuracy for better recall on
# bad-credit cases — a more realistic goal in credit risk applications where
# missing a bad applicant is costly.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Default accuracy : {accuracy_default:.4f}")
print(f"Tuned   accuracy : {accuracy_tuned:.4f}")
print(f"Change           : {accuracy_tuned - accuracy_default:+.4f}")
print("\nNote: watch the per-class recall in the reports above —")
print("the tuned model improves detection of bad-credit applicants.")