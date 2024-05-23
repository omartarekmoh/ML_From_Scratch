import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, *, lr=0.01, n_epochs=1000, verbose=False):
        """
        Logistic Regression model.

        Parameters:
        - lr (float): Learning rate for gradient descent.
        - n_epochs (int): Number of epochs for gradient descent.
        - verbose (bool): Whether to print training progress.
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose

    def initialize(self, shape, n_class):
        """
        Initialize weights for the model.

        Parameters:
        - shape (tuple): Shape of input data.
        - n_class (int): Number of classes.

        Returns:
        - w (numpy.ndarray): Initialized weights.
        """
        _, cols = shape
        w = np.random.randn(n_class, cols + 1) * 0.01
        return w

    def _inverse_mapper(self, y_mapped):
        """
        Inverse mapper to convert encoded labels back to original labels.

        Parameters:
        - y_mapped (numpy.ndarray): Encoded labels.

        Returns:
        - numpy.ndarray: Decoded labels.
        """
        inverse_mapper = {v: k for k, v in self.mapper.items()}
        return np.vectorize(inverse_mapper.get)(y_mapped)

    def _mapper(self, y):
        """
        Mapper to encode labels.

        Parameters:
        - y (numpy.ndarray): Original labels.

        Returns:
        - numpy.ndarray: Encoded labels.
        """
        vectorized_mapping = np.vectorize(self.mapper.get)
        return vectorized_mapping(y)

    def _get_mapper(self, y):
        """
        Generate mapping of labels to integers.

        Parameters:
        - y (numpy.ndarray): Original labels.

        Returns:
        - dict: Mapping of labels to integers.
        """
        self.unique = np.unique(y)
        self._range = np.arange(len(self.unique))
        return dict(zip(self.unique, self._range))

    def _one_hot(self, y):
        """
        Convert labels to one-hot encoding.

        Parameters:
        - y (numpy.ndarray): Encoded labels.

        Returns:
        - numpy.ndarray: One-hot encoded labels.
        """
        n_rows = y.shape[0]
        n_class = len(np.unique(y))
        one_hot = np.zeros((n_rows, n_class))
        one_hot[np.arange(len(y)), y.ravel()] = 1
        return one_hot

    def _vec_oh(self, y):
        """
        Vectorize one-hot encoding process.

        Parameters:
        - y (numpy.ndarray): Original labels.

        Returns:
        - numpy.ndarray: One-hot encoded labels.
        - numpy.ndarray: Encoded labels.
        """
        self.mapper = self._get_mapper(y)
        y_mapped = self._mapper(y)
        y_oh = self._one_hot(y_mapped)
        return y_oh, y_mapped

    def softmax(self, X):
        """
        Compute softmax probabilities.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Softmax probabilities.
        """
        z = np.dot(X, self.w.T)
        pk = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return pk

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Predicted probabilities.
        """
        X_added_ones = self.addOnes(X)
        return self.softmax(X_added_ones)

    def addOnes(self, X):
        """
        Add a column of ones to input data.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Input data with column of ones added.
        """
        rows, _ = X.shape
        ones = np.ones((rows, 1))
        return np.hstack((ones, X))

    def _cost(self, y, preds):
        """
        Calculate cross-entropy loss.

        Parameters:
        - y (numpy.ndarray): True labels.
        - preds (numpy.ndarray): Predicted probabilities.

        Returns:
        - float: Cross-entropy loss.
        """
        return -np.mean(np.sum(y * np.log(preds + 1e-9), axis=1))

    def _init_data(self, X, y):
        """
        Initialize data for training.

        Parameters:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.

        Returns:
        - numpy.ndarray: One-hot encoded labels.
        """
        n_class = len(np.unique(y))
        self.w = self.initialize(X.shape, n_class)
        y_oh, _ = self._vec_oh(y)
        return y_oh

    def grad_desc(self, X, y):
        """
        Perform gradient descent.

        Parameters:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.
        """
        m = X.shape[0]
        y_oh = self._init_data(X, y)

        for epoch in range(self.n_epochs):
            X_added_ones = self.addOnes(X)
            pk = self.softmax(X_added_ones)
            error = pk - y_oh
            w_gradient = np.dot(X_added_ones.T, error) / m
            self.w -= self.lr * w_gradient.T

            if self.verbose and epoch % 100 == 0:
                loss = self._cost(y_oh, pk)
                print(f"Epoch {epoch}: loss = {loss}")

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): True labels.
        """
        X = np.array(X)
        y = np.array(y)
        self.grad_desc(X, y)
        

    def predict(self, X):
        """
        Predict labels for new data.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Predicted labels.
        """
        probabilities = self.predict_proba(X)
        predictions_encoded = np.argmax(probabilities, axis=1)
        predictions = self._inverse_mapper(predictions_encoded)
        return predictions