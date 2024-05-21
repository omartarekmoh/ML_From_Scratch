import numpy as np
from collections import Counter

class Node:
    """
    Decision tree node representation.

    Attributes:
        right (Node): Right child node.
        left (Node): Left child node.
        feature (int): Index of the feature to split on.
        threshold (float): Threshold value for the feature split.
        value (int): Value (class label) of the leaf node.
        cost (float): Impurity or information gain of the node.
    """

    def __init__(self, right=None, left=None, feature=None, threshold=None, *, value, cost=np.inf):
        """
        Initializes a Node object.

        Args:
            right (Node): Right child node.
            left (Node): Left child node.
            feature (int): Index of the feature to split on.
            threshold (float): Threshold value for the feature split.
            value (int): Value (class label) of the leaf node.
            cost (float): Impurity or information gain of the node.
        """
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.cost = cost

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, False otherwise.
        """
        return np.isinf(self.cost)

class DecisionTree:
    """
    A decision tree classifier implementation.

    Parameters:
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the tree.
        n_features (int): The number of features to consider when looking for the best split.
        criterion (str): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.

    Attributes:
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the tree.
        n_features (int): The number of features to consider when looking for the best split.
        criterion (str): The function to measure the quality of a split.
        root (Node): The root node of the decision tree.
    """

    def __init__(self, *, min_samples_split=2, max_depth=100, n_features=None, criterion="gini"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the training data.

        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target values.

        Returns:
            None
        """
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _split(self, X, threshold):
        """
        Split the input data based on a threshold value.

        Parameters:
            X (array-like): The input data to be split.
            threshold (float): The threshold value for splitting.

        Returns:
            tuple: A tuple containing boolean arrays representing the split indices.
        """
        left_idxs = X <= threshold
        right_idxs = X > threshold
        return left_idxs, right_idxs

    def _information_gain(self, y, left_idxs, right_idxs):
        """
        Calculate the information gain from a split.

        Parameters:
            y (array-like): The target values.
            left_idxs (array-like): Boolean array representing the indices of the left split.
            right_idxs (array-like): Boolean array representing the indices of the right split.

        Returns:
            float: The information gain.
        """
        parent_entropy = self._entropy(y)
        n = len(y)
        n_l, n_r = left_idxs.sum(), right_idxs.sum()
        if n_l == 0 or n_r == 0:
            return 0
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _entropy(self, y):
        """
        Calculate the entropy of a set of target values.

        Parameters:
            y (array-like): The target values.

        Returns:
            float: The entropy.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        """
        Calculate the Gini impurity of a set of target values.

        Parameters:
            y (array-like): The target values.

        Returns:
            float: The Gini impurity.
        """
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / m
        return 1 - np.sum(p ** 2)

    def _impurity(self, y, left_idxs, right_idxs):
        """
        Calculate the impurity of a split using Gini impurity.

        Parameters:
            y (array-like): The target values.
            left_idxs (array-like): Boolean array representing the indices of the left split.
            right_idxs (array-like): Boolean array representing the indices of the right split.

        Returns:
            float: The impurity.
        """
        m = len(y)
        n_l, n_r = left_idxs.sum(), right_idxs.sum()
        if n_l == 0 or n_r == 0:
            return np.inf
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        return (n_l / m) * g_l + (n_r / m) * g_r

    def _find_best_split(self, X, y, feat_idx):
        """
        Find the best split for a given feature.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target values.
            feat_idx (array-like): The indices of the features to consider.

        Returns:
            tuple: A tuple containing the best threshold, best feature index, and the corresponding gain.
        """
        best_gain = -1
        gain = 0
        best_thresh, best_feature = None, None

        for feat in feat_idx:
            X_column = X[:, feat]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)
                if self.criterion == "gini":
                    impurity = self._impurity(y, left_idxs, right_idxs)
                    gain = -impurity  # Negative impurity to maximize gain
                elif self.criterion == "entropy":
                    gain = self._information_gain(y, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    best_thresh = threshold
                    best_feature = feat

        return best_thresh, best_feature, best_gain

    def _most_common_label(self, y):
        """
        Find the most common label in a set of target values.

        Parameters:
            y (array-like): The target values.

        Returns:
            object: The most common label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target values.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the subtree.
        """
        n_samples, n_features = X.shape
        n_label = len(np.unique(y))

        if depth >= self.max_depth or n_label == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_features, self.n_features, replace=False)
        best_thresh, best_feature, best_gain = self._find_best_split(X, y, feat_idx)

        if best_gain == -1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(left=left, right=right, feature=best_feature, threshold=best_thresh, value=self._most_common_label(y), cost=best_gain)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array-like: The predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to predict the class label for a sample.

        Parameters:
            x (array-like): The input sample.
            node (Node): The current node in the decision tree.

        Returns:
            object: The predicted class label.
        """
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value