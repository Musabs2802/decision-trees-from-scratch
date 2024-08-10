# Steps
# Training:
# Given the whole dataset:
# Calculate Information Gain with each information split
# Divide the set with that feature and value that gives the most Information Gain
# Divide tree and continue the same for all branches
# Continue until stopping creiteria is reached

# Testing
# Given a data point:
# Follow the tree until you reach the leaf node
# Return the most common class label if leaf node is not pure

# Terms
# IG: E(parent) - weighted_average[E(children)]
# Entropy: The measure of disorder
# E = - sum[p(X) * log2(p(X))]
# p(X) = num_of(X) / n
# Stopping criteria: maximum depth -> max num of layer of nodes
#                  : minimum num of samples -> min num of samples a node can have
#                  : minimum impurity descrease -> min entropy change required for a split


import numpy as np
from collections import Counter

class Node:
    def __init__(self, features=None, threshold=None, left=None, right=None, *, value=None) -> None:
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def _is_leaf_node(self) -> bool:
        return self.value is not None

class DecisionTrees:
    def __init__(self, max_depth=100, min_samples=2, n_features=None) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.root = None

    def fit(self, X, y) -> None:
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.__grow_tree(X, y)

    def __grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth >= self.max_depth or n_labels==1 or n_samples<self.min_samples):
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
            
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find best split
        best_threshold, best_feat_idx = self.__find_best_split(X, y, feat_idxs)

        # Create child node
        left_idxs, right_idxs = self.__split(X[:, best_feat_idx], best_threshold)

        left = self.__grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.__grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(features=best_feat_idx, threshold=best_threshold, left=left, right=right)

    def __find_best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_threshold, best_feat_idx = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Calculate information gain
                gain = self.__calculate_information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feat_idx = feat_idx

        return best_threshold, best_feat_idx
    
    def __calculate_information_gain(self, X_column, y, threshold) -> float:
        # Parent entropy
        parent_entropy = self.__calculate_entropy(y)

        # Create children
        left_idxs, right_idxs = self.__split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate weighted average entropy of children
        y_num = len(y)
        left_num, right_num = len(left_idxs), len(right_idxs)
        left_entropy = self.__calculate_entropy(y[left_idxs])
        right_entropy = self.__calculate_entropy(y[right_idxs])
        
        child_entropy = (left_num/y_num) * left_entropy + (right_num/y_num) * right_entropy
        
        # Calculate IG
        return parent_entropy - child_entropy

    def __split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs
    
    def __calculate_entropy(self, x):
        px = np.bincount(x) / len(x)
        return -np.sum(p * np.log2(p) for p in px if p>0)

    def __most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        return [self.__traverse_tree(x, self.root) for x in X]

    def __traverse_tree(self, x, node:Node):
        if node._is_leaf_node():
            return node.value
        
        if x[node.features] <= node.threshold:
            return self.__traverse_tree(x, node.left)
        else:
            return self.__traverse_tree(x, node.right)