import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import Counter

# ⟢ Helper Functions for k-Nearest Neighbors ⟣
def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    """Calculate Manhattan distance between two points"""
    return np.sum(np.abs(point1 - point2))

def get_neighbors(X_train, y_train, x_query, k, distance_func=euclidean_distance):
    """Find k nearest neighbors for a query point"""
    distances = []
    
    for i, x_train in enumerate(X_train):
        dist = distance_func(x_query, x_train)
        distances.append((dist, y_train[i]))
    
    # Sort by distance and return k nearest
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def predict_classification(neighbors):
    """Predict class based on majority vote of neighbors"""
    neighbor_labels = [neighbor[1] for neighbor in neighbors]
    return Counter(neighbor_labels).most_common(1)[0][0]

def predict_regression(neighbors):
    """Predict value based on average of neighbors"""
    neighbor_values = [neighbor[1] for neighbor in neighbors]
    return np.mean(neighbor_values)

def predict_regression_weighted(neighbors):
    """Predict value based on distance-weighted average of neighbors"""
    if len(neighbors) == 1:
        return neighbors[0][1]
    
    weights = []
    values = []
    
    for distance, value in neighbors:
        if distance == 0:
            return value  # Exact match
        weight = 1 / distance
        weights.append(weight)
        values.append(value)
    
    weights = np.array(weights)
    values = np.array(values)
    
    return np.sum(weights * values) / np.sum(weights)

# ⟢ Building a k-Nearest Neighbors Class ⟣
class MyKNN:
    def __init__(self, k=5, task='classification', distance_metric='euclidean', weights='uniform'):
        """
        k: number of neighbors
        task: 'classification' or 'regression'
        distance_metric: 'euclidean' or 'manhattan'
        weights: 'uniform' or 'distance'
        """
        self.k = k
        self.task = task
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
        # Set distance function
        if distance_metric == 'euclidean':
            self.distance_func = euclidean_distance
        elif distance_metric == 'manhattan':
            self.distance_func = manhattan_distance
        else:
            raise ValueError("distance_metric must be 'euclidean' or 'manhattan'")

    def fit(self, X, y):
        """Store training data (lazy learning)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        """Make predictions for input data"""
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        predictions = []
        
        for x_query in X:
            neighbors = get_neighbors(self.X_train, self.y_train, x_query, 
                                    self.k, self.distance_func)
            
            if self.task == 'classification':
                pred = predict_classification(neighbors)
            elif self.task == 'regression':
                if self.weights == 'uniform':
                    pred = predict_regression(neighbors)
                else:  # distance-weighted
                    pred = predict_regression_weighted(neighbors)
            else:
                raise ValueError("task must be 'classification' or 'regression'")
            
            predictions.append(pred)
        
        return np.array(predictions)

    def predict_proba(self, X):
        """Predict class probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        probabilities = []
        
        # Get unique classes
        unique_classes = np.unique(self.y_train)
        
        for x_query in X:
            neighbors = get_neighbors(self.X_train, self.y_train, x_query, 
                                    self.k, self.distance_func)
            
            # Count votes for each class
            neighbor_labels = [neighbor[1] for neighbor in neighbors]
            class_counts = Counter(neighbor_labels)
            
            # Convert to probabilities
            proba = []
            for cls in unique_classes:
                proba.append(class_counts.get(cls, 0) / self.k)
            
            probabilities.append(proba)
        
        return np.array(probabilities)

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
            'k': self.k,
            'task': self.task,
            'distance_metric': self.distance_metric,
            'weights': self.weights,
            'n_training_samples': len(self.X_train) if self.X_train is not None else 0
        }

# Demo/Testing code (only runs when script is executed directly)
if __name__ == "__main__":
    print("=== Testing k-NN Classification ===")
    
    # Generate classification data
    X_class, y_class = make_classification(n_samples=500, n_features=10, n_classes=3,
                                         n_informative=8, n_redundant=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    # Test our k-NN classifier
    model_class = MyKNN(k=5, task='classification', distance_metric='euclidean')
    model_class.fit(X_train, y_train)
    
    y_pred_class = model_class.predict(X_test)
    accuracy = model_class.score(X_test, y_test)
    
    print(f"k-NN Classification Accuracy: {accuracy:.4f}")
    
    # Compare with sklearn
    from sklearn.neighbors import KNeighborsClassifier
    sklearn_class = KNeighborsClassifier(n_neighbors=5)
    sklearn_class.fit(X_train, y_train)
    sklearn_accuracy = sklearn_class.score(X_test, y_test)
    
    print(f"Sklearn k-NN Accuracy: {sklearn_accuracy:.4f}")
    
    print("\n=== Testing k-NN Regression ===")
    
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Test our k-NN regressor
    model_reg = MyKNN(k=5, task='regression', weights='distance')
    model_reg.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = model_reg.predict(X_test_reg)
    r2 = model_reg.score(X_test_reg, y_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    
    print(f"k-NN Regression R² Score: {r2:.4f}")
    print(f"k-NN Regression MSE: {mse:.4f}")
    
    # Compare with sklearn
    from sklearn.neighbors import KNeighborsRegressor
    sklearn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
    sklearn_reg.fit(X_train_reg, y_train_reg)
    sklearn_r2 = sklearn_reg.score(X_test_reg, y_test_reg)
    
    print(f"Sklearn k-NN R² Score: {sklearn_r2:.4f}")
    
    print(f"\nModel parameters: {model_class.get_params()}")
