import numpy as np


class LinearRegression:
    """
    Linear Regression model using gradient descent.

    Parameters
    ----------
    lr : float, optional (default=0.01)
        Learning rate for gradient descent.
    n_epochs : int, optional (default=100)
        Number of epochs for training the model.
    method : str, optional (default="batch_grad")
        Method for gradient descent. Currently, only "batch_grad" is implemented.

    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients of the linear model.
    intercept_ : float
        Intercept of the linear model.
    """

    def __init__(
        self, lr=0.01, n_epochs=100, *args, method="batch_grad", verbose=False
    ):
        # Initialize the learning rate, number of epochs, and method for gradient descent
        self.lr = lr
        self.n_epochs = n_epochs
        self.method = method
        self.verbose = verbose

    def getShape(self, X):
        """
        Get the shape of the input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        tuple
            Number of rows and columns in X.
        """
        # Return the shape of the input data
        X = np.array(X)
        shape = X.shape
        return shape[0], shape[1]

    def initialize(self, X):
        """
        Initialize the weights of the model.

        Parameters
        ----------
        X : np.ndarray
            Input data with bias term added.

        Returns
        -------
        np.ndarray
            Initialized weights.
        """
        # Get the number of columns (features) in the input data
        _, cols = self.getShape(X)
        # Initialize weights with small random values
        w = np.random.randn(1, cols + 1) * 0.01
        return w

    def addOnes(self, X):
        """
        Add a column of ones to the input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Input data with a column of ones added.
        """
        # Get the number of rows in the input data
        rows, _ = self.getShape(X)
        # Create a column of ones
        ones = np.ones((rows, 1))
        # Add the column of ones to the input data
        return np.hstack((ones, X))

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : np.ndarray
            Input data with bias term added.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        # Add a column of ones to the input data
        X_added_ones = self.addOnes(X)

        # Compute the predicted values by multiplying input data with weights
        preds = np.dot(X_added_ones, self.w.T)
        return preds

    def mse(self, preds, y):
        """
        Calculate the Mean Squared Error (MSE) loss.

        Parameters
        ----------
        preds : np.ndarray
            Predicted values.
        y : np.ndarray
            True values.

        Returns
        -------
        float
            MSE loss.
        """

        # Get the number of samples
        m, _ = self.getShape(y)
        # Compute the difference between predicted and true values
        loss = preds - y
        # Compute the Mean Squared Error
        cost = (1 / m) * np.sum(loss**2)
        return cost

    def compute_grads(self, X, y, preds):
        m, _ = self.getShape(X)
        # Compute the loss
        loss = preds - y
        # Add a column of ones to the input data
        X_added_ones = self.addOnes(X)
        # Compute the gradients
        grads = (2 / m) * np.dot(loss.T, X_added_ones)

        return grads

    def batch_grad(self, X, y):
        """
        Train the model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True values.
        """
        # Initialize the weights
        self.w = self.initialize(X)
        # Get the number of samples (m) and features (n)
        m, n = self.getShape(X)

        # Iterate over the number of epochs
        for epoch in range(self.n_epochs):
            # Predict the values using the current weights
            preds = self.predict(X)

            grads = self.compute_grads(X, y, preds)
            # Update the weights using the gradients
            self.w -= self.lr * grads

            # Print the cost every 100 epochs
            if epoch % 100 == 0 and self.verbose:
                cost = self.mse(preds, y)
                print(f"total loss = {cost}")

    def fit(self, X, y):
        """
        Fit the linear model to the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Check the method and apply batch gradient descent
        if self.method == "batch_grad":
            self.batch_grad(X, y)

        # Extract the coefficients (weights for features) and intercept (bias term)
        self.coef_ = self.w[:, 1:]
        self.intercept_ = self.w[:, 0]
