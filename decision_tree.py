import numpy as np

# Node class for decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value for leaf nodes (class label)

# Decision tree classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        
        node = Node(value=predicted_class)
        
        if depth < self.max_depth:
            best_feature_index, best_threshold = self._find_best_split(X, y)
            if best_feature_index is not None:
                left_indices = X[:, best_feature_index] < best_threshold
                X_left, y_left = X[left_indices], y[left_indices]
                X_right, y_right = X[~left_indices], y[~left_indices]
                
                node.feature_index = best_feature_index
                node.threshold = best_threshold
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node

    def _find_best_split(self, X, y):
        best_gini = 1.0
        best_feature_index = None
        best_threshold = None
        
        for feature_index in range(self.num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini = self._compute_gini_index(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold

    def _compute_gini_index(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] < threshold
        y_left = y[left_indices]
        y_right = y[~left_indices]
        
        gini_left = self._compute_gini(y_left)
        gini_right = self._compute_gini(y_right)
        
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)
        
        gini_index = (left_weight * gini_left) + (right_weight * gini_right)
        return gini_index

    def _compute_gini(self, y):
        if len(y) == 0:
            return 0
        
        num_samples = len(y)
        counts = np.bincount(y)
        probabilities = counts / num_samples
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
       
