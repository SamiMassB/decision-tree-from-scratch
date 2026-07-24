import numpy as np

from .node import Node


class DecisionTreeClassifierScratch:

    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = [count / len(y) for count in counts]
        return sum(-p * np.log2(p) for p in probs)

    def information_gain(self, y, y_left, y_right):
        parent_entropy = self.entropy(y)
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)
        child_entropy = left_weight * self.entropy(y_left) + right_weight * self.entropy(y_right)
        return parent_entropy - child_entropy

    def majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def find_best_split(self, X, y):
        best_gain_per_feature = []
        best_threshold_per_feature = []
        features = X.columns

        for feature in features:
            sorted_values = np.sort(X[feature].values)
            best_gain = -1
            best_threshold = None

            for i in range(len(sorted_values) - 1):
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                y_left = y[X[feature] <= threshold]
                y_right = y[X[feature] > threshold]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = self.information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold

            best_gain_per_feature.append(best_gain)
            best_threshold_per_feature.append(best_threshold)

        best_index = np.argmax(best_gain_per_feature)
        return features[best_index], best_threshold_per_feature[best_index]

    def determine_leaf_value(self, y):
        return 1 if np.mean(y.values) > 0.5 else 0

    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y.values)) == 1 or len(y) < self.min_samples_split:
            return Node(value=self.determine_leaf_value(y))

        feature, threshold = self.find_best_split(X, y)
        if feature is None or threshold is None:
            return Node(value=self.determine_leaf_value(y))

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        node = Node(feature=feature, threshold=threshold)
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        return self

    def _predict_one(self, row):
        node = self.root
        while not node.is_leaf():
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return X.apply(self._predict_one, axis=1)
