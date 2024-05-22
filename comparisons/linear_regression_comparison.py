import os
import sys
import time
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Add the parent directory to the Python path to import custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Import the custom LinearRegression model
from pydok.linear_models.LinearRegression import LinearRegression

# Load the diabetes dataset
data = load_diabetes()
X, y = data.data, data.target

# Standardize the features (important for gradient descent convergence)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Reshape y_train and y_test to match the expected shape
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Initialize and fit your custom LinearRegression model
my_model = LinearRegression(
    lr=0.01, n_epochs=1000, method="batch", verbose=False, batch_size=64
)
start = time.time()
my_model.fit(X_train, y_train)
end = time.time()

# Predict on the test set using your custom model
my_predictions = my_model.predict(X_test)

# Calculate the Mean Squared Error on the test set using your custom model
my_mse = mean_squared_error(y_test, my_predictions)

# Calculate and print the time taken for training in milliseconds
time_taken_ms = (end - start) * 1000
print(f"Time taken for training (my model): {time_taken_ms:.2f} ms")
print(f"Mean Squared Error on the test set (my model): {my_mse}")

# Initialize and fit the sklearn LinearRegression model
sklearn_model = SklearnLinearRegression()
start = time.time()
sklearn_model.fit(X_train, y_train)
end = time.time()

# Predict on the test set using the sklearn model
sklearn_predictions = sklearn_model.predict(X_test)

# Calculate the Mean Squared Error on the test set using the sklearn model
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

# Calculate and print the time taken for training in milliseconds
time_taken_ms = (end - start) * 1000
print(f"Time taken for training (sklearn model): {time_taken_ms:.2f} ms")
print(f"Mean Squared Error on the test set (sklearn model): {sklearn_mse}")

# Print the learned coefficients and intercept from both models
print("Coefficients (my model):", my_model.coef_)
print("Intercept (my model):", my_model.intercept_)

print("Coefficients (sklearn model):", sklearn_model.coef_)
print("Intercept (sklearn model):", sklearn_model.intercept_)
