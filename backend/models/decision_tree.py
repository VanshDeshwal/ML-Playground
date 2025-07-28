import numpy as np
import time
from typing import Dict, Any, Optional
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import Counter

# Import our base algorithm interface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm_registry import BaseAlgorithm

# ⟢ Helper Functions for Decision Tree ⟣
def entropy(y):
    """Calculate entropy for classification"""
    if len(y) == 0:
        return 0
    
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-15))

def gini_impurity(y):
    """Calculate Gini impurity for classification"""
    if len(y) == 0:
        return 0
    
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def mse_impurity(y):
    """Calculate MSE for regression"""
    if len(y) == 0:
        return 0
    return np.var(y)

def information_gain(y, y_left, y_right, criterion='entropy'):
    """Calculate information gain from a split"""
    if criterion == 'entropy':
        impurity_func = entropy
    elif criterion == 'gini':
        impurity_func = gini_impurity
    elif criterion == 'mse':
        impurity_func = mse_impurity
    else:
        raise ValueError("criterion must be 'entropy', 'gini', or 'mse'")
    
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    parent_impurity = impurity_func(y)
    weighted_impurity = (n_left / n) * impurity_func(y_left) + (n_right / n) * impurity_func(y_right)
    
    return parent_impurity - weighted_impurity

def find_best_split(X, y, criterion='entropy'):
    """Find the best feature and threshold to split on"""
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    for feature in range(n_features):
        feature_values = X[:, feature]
        unique_values = np.unique(feature_values)
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            gain = information_gain(y, y_left, y_right, criterion)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

# ⟢ Decision Tree Node Class ⟣
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Feature index to split on
        self.threshold = threshold    # Threshold value for split
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.value = value           # Prediction value (for leaf nodes)

# ⟢ Building a Decision Tree Class ⟣
class MyDecisionTree(BaseAlgorithm):
    """Enhanced Decision Tree with BaseAlgorithm interface"""
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 criterion='entropy', task='classification'):
        """
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split
        min_samples_leaf: Minimum samples required in a leaf
        criterion: 'entropy', 'gini' (classification) or 'mse' (regression)
        task: 'classification' or 'regression'
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.root = None
        self.feature_importances_ = None
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return algorithm metadata"""
        return {
            "id": "decision_tree",
            "name": "Decision Tree (From Scratch)",
            "type": "classification",  # Default, can handle both
            "description": "Tree-based model for classification and regression using recursive binary splits",
            "hyperparameters": {
                "max_depth": {
                    "type": "int",
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "description": "Maximum depth of the tree"
                },
                "min_samples_split": {
                    "type": "int",
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "description": "Minimum samples required to split an internal node"
                },
                "min_samples_leaf": {
                    "type": "int",
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "description": "Minimum samples required to be at a leaf node"
                },
                "criterion": {
                    "type": "string",
                    "default": "entropy",
                    "options": ["entropy", "gini", "mse"],
                    "description": "Function to measure split quality"
                },
                "task": {
                    "type": "string",
                    "default": "classification",
                    "options": ["classification", "regression"],
                    "description": "Type of prediction task"
                }
            }
        }
    
    async def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the decision tree model"""
        start_time = time.time()
        
        # Store data and initialize
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_features = self.X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        
        # Build the tree
        self.root = self._build_tree(self.X, self.y, depth=0)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        
        training_time = time.time() - start_time
        
        # Calculate metrics on training data
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            accuracy = accuracy_score(y, y_pred)
            unique_classes = len(np.unique(y))
            metrics = {
                "accuracy": float(accuracy),
                "n_classes": int(unique_classes)
            }
        else:  # regression
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            metrics = {
                "mse": float(mse),
                "r2_score": float(r2),
                "rmse": float(np.sqrt(mse))
            }
        
        # Calculate tree statistics
        tree_depth = self._get_tree_depth(self.root)
        n_nodes = self._count_nodes(self.root)
        n_leaves = self._count_leaves(self.root)
        
        return {
            "training_time": float(training_time),
            "metrics": metrics,
            "metadata": {
                "max_depth": int(self.max_depth),
                "actual_depth": int(tree_depth),
                "min_samples_split": int(self.min_samples_split),
                "min_samples_leaf": int(self.min_samples_leaf),
                "criterion": str(self.criterion),
                "task": str(self.task),
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0]),
                "n_nodes": int(n_nodes),
                "n_leaves": int(n_leaves),
                "feature_importances": [float(x) for x in self.feature_importances_]
            }
        }
    
    def _get_tree_depth(self, node, depth=0):
        """Calculate actual depth of the tree"""
        if node is None or node.value is not None:
            return depth
        return max(self._get_tree_depth(node.left, depth + 1),
                  self._get_tree_depth(node.right, depth + 1))
    
    def _count_nodes(self, node):
        """Count total number of nodes in the tree"""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def _count_leaves(self, node):
        """Count number of leaf nodes in the tree"""
        if node is None:
            return 0
        if node.value is not None:  # Leaf node
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data"""
        if self.root is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = np.array(X)
        predictions = []
        
        for sample in X:
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def get_sklearn_equivalent(self):
        """Return equivalent sklearn model for comparison"""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        if self.task == 'classification':
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion if self.criterion != 'entropy' else 'entropy',
                random_state=42
            )
        else:
            criterion = 'squared_error' if self.criterion == 'mse' else self.criterion
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=criterion,
                random_state=42
            )

    def fit(self, X, y):
        """Legacy interface - calls async train method"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(self.train(X, y))
        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = find_best_split(X, y, self.criterion)
        
        if best_feature is None or best_gain <= 0:
            return self._create_leaf(y)
        
        # Update feature importance
        self.feature_importances_[best_feature] += best_gain * n_samples
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples in leaf
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._create_leaf(y)
        
        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(feature=best_feature, threshold=best_threshold, 
                       left=left_child, right=right_child)

    def _create_leaf(self, y):
        """Create a leaf node with prediction value"""
        if self.task == 'classification':
            # Most common class
            value = Counter(y).most_common(1)[0][0]
        else:  # regression
            # Mean value
            value = np.mean(y)
        
        return TreeNode(value=value)

    def predict(self, X):
        """Make predictions for input data"""
        if self.root is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        predictions = []
        
        for sample in X:
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)

    def _predict_sample(self, sample, node):
        """Predict for a single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def score(self, X, y):
        """Calculate accuracy (classification) or R² score (regression)"""
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return accuracy_score(y, y_pred)
        else:  # regression
            return r2_score(y, y_pred)

    def get_params(self):
        """Return model parameters"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'task': self.task
        }

    def get_feature_importances(self):
        """Return feature importances"""
        return self.feature_importances_

    def get_depth(self):
        """Get the depth of the tree"""
        return self._get_node_depth(self.root)

    def _get_node_depth(self, node):
        """Get depth of a node"""
        if node is None or node.value is not None:
            return 0
        
        left_depth = self._get_node_depth(node.left)
        right_depth = self._get_node_depth(node.right)
        
        return 1 + max(left_depth, right_depth)

# Demo/Testing code (only runs when script is executed directly)
if __name__ == "__main__":
    print("=== Testing Decision Tree Classification ===")
    
    # Generate classification data
    X_class, y_class = make_classification(n_samples=500, n_features=10, n_classes=3,
                                         n_informative=8, n_redundant=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    # Test our decision tree classifier
    model_class = MyDecisionTree(max_depth=10, criterion='entropy', task='classification')
    model_class.fit(X_train, y_train)
    
    y_pred_class = model_class.predict(X_test)
    accuracy = model_class.score(X_test, y_test)
    
    print(f"Decision Tree Classification Accuracy: {accuracy:.4f}")
    print(f"Tree depth: {model_class.get_depth()}")
    
    # Compare with sklearn
    from sklearn.tree import DecisionTreeClassifier
    sklearn_class = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)
    sklearn_class.fit(X_train, y_train)
    sklearn_accuracy = sklearn_class.score(X_test, y_test)
    
    print(f"Sklearn Decision Tree Accuracy: {sklearn_accuracy:.4f}")
    
    print("\n=== Testing Decision Tree Regression ===")
    
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Test our decision tree regressor
    model_reg = MyDecisionTree(max_depth=10, criterion='mse', task='regression')
    model_reg.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = model_reg.predict(X_test_reg)
    r2 = model_reg.score(X_test_reg, y_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    
    print(f"Decision Tree Regression R² Score: {r2:.4f}")
    print(f"Decision Tree Regression MSE: {mse:.4f}")
    print(f"Tree depth: {model_reg.get_depth()}")
    
    # Compare with sklearn
    from sklearn.tree import DecisionTreeRegressor
    sklearn_reg = DecisionTreeRegressor(max_depth=10, criterion='squared_error', random_state=42)
    sklearn_reg.fit(X_train_reg, y_train_reg)
    sklearn_r2 = sklearn_reg.score(X_test_reg, y_test_reg)
    
    print(f"Sklearn Decision Tree R² Score: {sklearn_r2:.4f}")
    
    print(f"\nTop 3 feature importances:")
    importances = model_class.get_feature_importances()
    top_features = np.argsort(importances)[::-1][:3]
    for i, feature in enumerate(top_features):
        print(f"  Feature {feature}: {importances[feature]:.4f}")
