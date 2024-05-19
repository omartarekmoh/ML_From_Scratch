# Machine Learning Algorithms from Scratch

This repository contains implementations of various machine learning algorithms written from scratch in Python. The aim is to provide a clear understanding of how these algorithms work under the hood.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)

## Introduction

This project is designed for educational purposes to help you understand the inner workings of popular machine learning algorithms. By implementing these algorithms from scratch, you will gain a deeper insight into their mechanics, which is often abstracted away by high-level libraries such as scikit-learn or TensorFlow.

## Algorithms Implemented

The following algorithms are currently implemented in this repository:

1. **Linear Regression**
2. **Logistic Regression**
3. **K-Nearest Neighbors (KNN) - On Going**
4. **Decision Tree**
5. **Support Vector Machine (SVM)**
9. **Principal Component Analysis (PCA) - On Going**
10. **Neural Networks - On Going**

Each algorithm is implemented in its own module with a corresponding example script demonstrating its usage.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/omartarekmoh/ML_From_Scratch.git
cd ml-algos-from-scratch
pip install -r requirements.txt
```

## Usage

Each algorithm can be used by importing the respective module. Below is an example of how to use the Linear Regression implementation.

```python
from algorithms.linear_regression import LinearRegression
import numpy as np

# Generate some example data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(predictions)
```

## Examples

You can find example scripts for each algorithm in the `examples` directory. These scripts demonstrate how to use the algorithms on various datasets. To run an example, navigate to the `examples` directory and execute the script.

```bash
cd examples
python linear_regression_example.py
```

## Contributing

Contributions are welcome! If you would like to add a new algorithm, improve existing implementations, or fix bugs, please open a pull request. Make sure to follow the contribution guidelines.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request


