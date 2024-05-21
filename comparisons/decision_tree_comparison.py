# Import necessary libraries
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Add the parent directory to the Python path to import custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Get the parent directory
sys.path.append(parent_dir)  # Add the parent directory to the Python path

# Import the custom DecisionTree model
from pyatch.trees.DecisionTree import DecisionTree

# Load the breast cancer dataset from scikit-learn
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train your DecisionTree implementation
clf = DecisionTree(criterion="gini")  # Initialize your DecisionTree model
clf.fit(X_train, y_train)  # Train the model on the training data

# Make predictions using your DecisionTree
preds = clf.predict(X_test)  # Use the trained model to make predictions on the test data

# Calculate accuracy of your DecisionTree
acc_my_model = accuracy_score(y_test, preds)  # Calculate the accuracy of your model
print("Accuracy of my DecisionTree implementation:", acc_my_model)

# Initialize and fit the scikit-learn DecisionTreeClassifier
sklearn_dt = DecisionTreeClassifier(criterion="gini", random_state=1234)  # Initialize the scikit-learn DecisionTreeClassifier
sklearn_dt.fit(X_train, y_train)  # Train the scikit-learn model on the training data

# Make predictions using scikit-learn's DecisionTreeClassifier
sklearn_preds = sklearn_dt.predict(X_test)  # Use the trained scikit-learn model to make predictions on the test data

# Calculate accuracy of scikit-learn's DecisionTreeClassifier
acc_sklearn_model = accuracy_score(y_test, sklearn_preds)  # Calculate the accuracy of the scikit-learn model
print("Accuracy of scikit-learn's DecisionTreeClassifier:", acc_sklearn_model)
