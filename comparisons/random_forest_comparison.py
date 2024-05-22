import os
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Add the parent directory to the Python path to import custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Get the parent directory
sys.path.append(parent_dir)  # Add the parent directory to the Python path

# Import the custom RandomForest model
from pynot.ensemble.RandomForest import RandomForest

# Generate some synthetic data
X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit your custom Random Forest model, measuring the time taken
start_time = time.time()
custom_rf = RandomForest(n_estimators=20, random_state=42, oob_score=True, max_depth=5)
custom_rf.fit(X_train, y_train)
custom_time = time.time() - start_time

# Initialize and fit scikit-learn's RandomForestClassifier, measuring the time taken
start_time = time.time()
sklearn_rf = RandomForestClassifier(n_estimators=20, random_state=42, oob_score=True, max_depth=5)
sklearn_rf.fit(X_train, y_train)
sklearn_time = time.time() - start_time

# Print the time taken by both models
print("Time taken by Custom Random Forest: {:.2f} seconds".format(custom_time))
print("Time taken by scikit-learn's RandomForestClassifier: {:.2f} seconds".format(sklearn_time))

# Make predictions using both models
custom_predictions = custom_rf.predict(X_test)
sklearn_predictions = sklearn_rf.predict(X_test)

# Compare accuracies
custom_accuracy = accuracy_score(y_test, custom_predictions)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

# Compare OOB (Out-Of-Bag) scores
custom_oob_score = custom_rf.oob_score_
sklearn_oob_score = sklearn_rf.oob_score_

# Print the accuracies and OOB scores of both models
print("Custom Random Forest Accuracy: {:.2f}".format(custom_accuracy))
print("scikit-learn's RandomForestClassifier Accuracy: {:.2f}".format(sklearn_accuracy))
print("Custom Random Forest OOB Score: {:.2f}".format(custom_oob_score))
print("scikit-learn's RandomForestClassifier OOB Score: {:.2f}".format(sklearn_oob_score))
