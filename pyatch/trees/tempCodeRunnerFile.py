import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, right=None, left=None, feature=None, threshold=None, *, value, cost=np.inf):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.cost = cost

    def is_leaf_node(self):
        return np.isinf(self.cost)

class DecisionTree:
    def __init__(self, *, min_samples_split=2, max_depth=100, n_features=None, criterion="gini"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _split(self, X, threshold):
        left_idxs = X <= threshold
        right_idxs = X > threshold
        return left_idxs, right_idxs

    def _information_gain(self, y, left_idxs, right_idxs):
        parent_entropy = self._entropy(y)
        n = len(y)
        n_l, n_r = left_idxs.sum(), right_idxs.sum()
        if n_l == 0 or n_r == 0:
            return 0
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / m
        return 1 - np.sum(p ** 2)

    def _impurity(self, y, left_idxs, right_idxs):
        m = len(y)
        n_l, n_r = left_idxs.sum(), right_idxs.sum()
        if n_l == 0 or n_r == 0:
            return np.inf
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        return (n_l / m) * g_l + (n_r / m) * g_r

    def _find_best_split(self, X, y, feat_idx):
        best_gain = -1
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
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _grow_tree(self, X, y, depth=0):
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
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

# Testing the optimized DecisionTree implementation
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree(criterion="gini")
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

def accuracy(preds, y):
    return np.sum(preds == y) / len(y)

acc = accuracy(preds, y_test)
print("Accuracy:", acc)
