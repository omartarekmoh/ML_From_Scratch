import time
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# Add the parent directory to the Python path to import custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Import the custom LogisticRegression model
from pynot.linear_models.LogisticRegression import LogisticRegression

# Generate synthetic dataset with NumPy
np.random.seed(42)
n_samples = 1000
n_features = 10
X = np.random.randn(n_samples, n_features)
# Generate binary classification labels
y = np.random.randint(0, 4, size=n_samples)

# Standardize the features (important for gradient descent convergence)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and fit your custom LogisticRegression model
my_model = LogisticRegression(lr=0.01, n_epochs=1000, verbose=False)
start = time.time()
my_model.fit(X_train, y_train)
end = time.time()

# Predict on the test set using your custom model
my_predictions = my_model.predict(X_test)

# Calculate the accuracy on the test set using your custom model
my_accuracy = accuracy_score(y_test, my_predictions)

# Calculate and print the time taken for training in milliseconds
time_taken_ms = (end - start) * 1000
print(f"Time taken for training (my model): {time_taken_ms:.2f} ms")
print(f"Accuracy on the test set (my model): {my_accuracy:.2f}")

# Initialize and fit the sklearn LogisticRegression model
sklearn_model = SklearnLogisticRegression()
start = time.time()
sklearn_model.fit(X_train, y_train)
end = time.time()

# Predict on the test set using the sklearn model
sklearn_predictions = sklearn_model.predict(X_test)

# Calculate the accuracy on the test set using the sklearn model
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

# Calculate and print the time taken for training in milliseconds
time_taken_ms = (end - start) * 1000
print(f"Time taken for training (sklearn model): {time_taken_ms:.2f} ms")
print(f"Accuracy on the test set (sklearn model): {sklearn_accuracy:.2f}")

