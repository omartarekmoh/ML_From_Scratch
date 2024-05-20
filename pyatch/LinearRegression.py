import numpy as np
import math

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
    method : str, optional (default="batch")
        Method for gradient descent. "batch" for Batch Gradient Descent and "mini_batch" for Mini-batch Gradient Descent.
    verbose : bool, optional (default=False)
        If True, prints the loss and coefficients every 50 epochs for batch gradient and 100 epochs for mini-batch gradient.
    batch_size : int, optional (default=64)
        Batch size for mini-batch gradient descent.

    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients of the linear model.
    intercept_ : float
        Intercept of the linear model.
    """

    def __init__(self, lr=0.01, n_epochs=100, method="batch", verbose=False, batch_size=64):
        # Initialize the learning rate, number of epochs, method for gradient descent, verbosity, and batch size
        self.lr = lr
        self.n_epochs = n_epochs
        self.method = method
        self.verbose = verbose
        self.batch_size = batch_size

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
        X = np.array(X)
        return X.shape

    def initialize(self, X):
        """
        Initialize the weights of the model.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Initialized weights.
        """
        _, cols = self.getShape(X)
        w = np.random.randn(1, cols + 1) * 0.01  # Initialize weights with small random values
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
        rows, _ = self.getShape(X)
        ones = np.ones((rows, 1))
        return np.hstack((ones, X))

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        X_added_ones = self.addOnes(X)
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
        m, _ = self.getShape(y)
        loss = preds - y
        cost = (1 / m) * np.sum(loss**2)
        return cost

    def compute_grads(self, X, y, preds):
        """
        Compute the gradients of the loss with respect to the weights.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True values.
        preds : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Gradients of the weights.
        """
        m, _ = self.getShape(X)
        loss = preds - y
        X_added_ones = self.addOnes(X)
        grads = (2 / m) * np.dot(loss.T, X_added_ones)
        return grads

    def get_coef_(self):
        """
        Get the coefficients and intercept of the model.

        Returns
        -------
        tuple
            Coefficients and intercept.
        """
        coef_ = self.w[:, 1:]
        intercept_ = self.w[:, 0]
        return coef_, intercept_

    def _print(self, epoch, preds, y):
        """
        Print the current state of the model during training.

        Parameters
        ----------
        epoch : int
            Current epoch.
        preds : np.ndarray
            Predicted values.
        y : np.ndarray
            True values.
        """
        cost = self.mse(preds, y)
        coefficient, intercept = self.get_coef_()
        coefs_str = ", ".join([f"{coef:.6f}" for coef in coefficient.flatten()])
        intercept_str = f"{intercept[0]:.6f}"

        print(
            f"Iteration {epoch} Summary:\n"
            f"-------------------------\n"
            f"Total Loss: {cost:.6f}\n"
            f"Coefficients: [{coefs_str}]\n"
            f"Intercept: {intercept_str}\n"
        )

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
        self.w = self.initialize(X)
        for epoch in range(self.n_epochs + 1):
            preds = self.predict(X)
            grads = self.compute_grads(X, y, preds)
            self.w -= self.lr * grads

            if epoch % 50 == 0 and self.verbose:
                self._print(epoch, preds, y)

    def generate_mini_batches(self, X, y, batch_size):
        """
        Generate mini-batches from the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True values.
        batch_size : int
            Size of each mini-batch.

        Yields
        ------
        tuple
            Mini-batch of input data and true values.
        """
        num_examples = len(y)
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        for start_idx in range(0, num_examples, batch_size):
            end_idx = min(start_idx + batch_size, num_examples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def mini_batch_grad(self, X, y):
        """
        Train the model using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True values.
        """
        self.w = self.initialize(X)
        for epoch in range(self.n_epochs + 1):
            mini_batch_generator = self.generate_mini_batches(X, y, self.batch_size)
            for X_mini_batch, y_mini_batch in mini_batch_generator:
                preds = self.predict(X_mini_batch)
                grads = self.compute_grads(X_mini_batch, y_mini_batch, preds)
                self.w -= self.lr * grads

            if epoch % 100 == 0 and self.verbose:
                self._print(epoch, self.predict(X), y)

    def fit(self, X, y):
        """
        Fit the linear model to the input data.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        """
        if self.method == "batch":
            self.batch_grad(X, y)
        elif self.method == "mini_batch":
            self.mini_batch_grad(X, y)
        else:
            raise ValueError("Method not recognized. Use 'batch' or 'mini_batch'.")
        self.coef_, self.intercept_ = self.get_coef_()
