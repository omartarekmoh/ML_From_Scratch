import numpy as np
import os
import sys
from collections import Counter

# Add the parent directory to the Python path to import custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Get the parent directory
sys.path.append(parent_dir)  # Add the parent directory to the Python path

# Import the custom DecisionTree model
from trees.DecisionTree import DecisionTree

class RandomForest:
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=np.inf,
        min_samples_split=2,
        random_state=None,
        oob_score=False,
    ):
        """
        Initialize the RandomForest classifier.

        Parameters:
        - n_estimators (int, default=100): Number of trees in the forest.
        - criterion (str, default="gini"): The function to measure the quality of a split.
          Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
        - max_depth (int or float, default=np.inf): The maximum depth of the tree.
        - min_samples_split (int, default=2): The minimum number of samples required to split an internal node.
        - random_state (int, optional): Controls the randomness of the bootstrap sampling and feature selection.
        - oob_score (bool, default=False): Whether to use out-of-bag samples to estimate the generalization accuracy.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.criterion = criterion
        self.root = None
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
        Fit the random forest model to the training data.

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The training input samples.
        - y (array-like of shape (n_samples,)): The target values (class labels).
        """
        self.trees = []
        self.oob_scores_ = []
        
        for _ in range(self.n_estimators):
            samp, oob = self._bootstrap(X)
            new_x, new_y = X[samp], y[samp]
            dt = self._generate_tree(new_x, new_y)
            self.trees.append((dt, oob)) 
            
        if self.oob_score:
            self.oob_score_ = self._calculate_oob(X, y)
    
    def _calculate_oob(self, X, y):
        """
        Calculate the out-of-bag (OOB) score if oob_score is set to True.

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The training input samples.
        - y (array-like of shape (n_samples,)): The target values (class labels).

        Returns:
        - accuracy (float): The out-of-bag accuracy score.
        """
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, len(np.unique(y))))
        
        for tree, oob_idx in self.trees:
            if len(oob_idx) > 0:  # Check if there are OOB samples
                predictions = tree.predict(X[oob_idx])
                for idx, pred in zip(oob_idx, predictions):
                    oob_predictions[idx, pred] += 1
        
        oob_counts = np.sum(oob_predictions, axis=1)
        oob_valid = oob_counts > 0
        oob_labels = np.argmax(oob_predictions[oob_valid], axis=1)
        accuracy = np.mean(oob_labels == y[oob_valid])
        return accuracy
    
    def _generate_tree(self, X, y):
        """
        Generate a single decision tree using the training data.

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The training input samples.
        - y (array-like of shape (n_samples,)): The target values (class labels).

        Returns:
        - dt (DecisionTree): The fitted decision tree.
        """
        dt = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
        dt.fit(X, y)
        return dt
    
    def _most_common_label(self, y):
        """
        Find the most common label in the array y.

        Parameters:
        - y (array-like): Array of labels.

        Returns:
        - label (int): The most common label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
            
    def _bootstrap(self, X):
        """
        Generate a bootstrap sample of the dataset X.

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The training input samples.

        Returns:
        - samp_idx (array-like): The indices of the bootstrap sample.
        - left_out_idxs (array-like): The indices of the samples left out (out-of-bag samples).
        """
        n_samples = X.shape[0]
        idx = np.arange(len(X))
        samp_idx = np.random.choice(idx, n_samples, replace=True)
        mask = np.isin(idx, samp_idx, invert=True)
        left_out_idxs = idx[mask]

        return samp_idx, left_out_idxs
    
    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The input samples.

        Returns:
        - aggregated_predictions (array-like of shape (n_samples,)): The predicted classes.
        """
        # Make predictions by aggregating predictions from individual trees
        predictions = np.array([tree.predict(X) for tree, _ in self.trees])
        # Use majority voting to determine the final prediction
        aggregated_predictions = np.apply_along_axis(self._most_common_label, axis=0, arr=predictions)
        
        return aggregated_predictions
