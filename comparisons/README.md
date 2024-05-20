# Model Comparison Results Documentation

This documentation provides details about a comparison between your custom model and the Sklearn model, including MSE comparison and coefficients.

## Clone the Project

To reproduce these results, follow these steps to clone the project:

1. Open your terminal.
2. Navigate to the directory where you want to clone the project.
3. Run the following command:

```bash
git clone https://github.com/omartarekmoh/ML_From_Scratch
```
### Installing Dependencies

Before running the comparison script, make sure you have the necessary dependencies installed:

- **Scikit-learn**: You can install it via pip:

  ```bash
  pip install scikit-learn
  ```

## Running the Comparison

After cloning the project and installing the dependencies, follow these steps to run the comparison:

1. Navigate to the comparisons directory within the project.
2. Run the `linear_comparison.py` script.

```bash
cd comparisons
python linear_comparison.py
```
## Results

### Mean Squared Error Comparison
- My Model: 2885.669
- Sklearn Model: 2900.194

### Coefficients Comparison

#### My Model
- Coefficients: [1.992, -11.431, 26.464, 16.321, -9.888, -2.272, -7.769, 8.176, 21.877, 2.562]
- Intercept: 151.308

#### Sklearn Model
- Coefficients: [1.803, -11.509, 25.801, 16.539, -44.306, 24.642, 7.773, 13.096, 35.017, 2.315]
- Intercept: 151.346
