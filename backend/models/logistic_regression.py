import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Import our base algorithm interface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm_registry import BaseAlgorithm

# ⟢ Helper Functions for Logistic Regression ⟣
def sigmoid(z):
    """Sigmoid activation function with numerical stability"""
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def add_bias(X):
    """Add bias column (column of ones) to the feature matrix."""
    return np.column_stack((np.ones(X.shape[0]), X))

def predict_proba(X_b, beta):
    """Predict probabilities using logistic model"""
    z = X_b @ beta
    return sigmoid(z)

def predict_binary(X_b, beta, threshold=0.5):
    """Make binary predictions"""
    probabilities = predict_proba(X_b, beta)
    return (probabilities >= threshold).astype(int)

def compute_logistic_loss(y_true, y_pred_proba):
    """Compute logistic loss (cross-entropy)"""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

def compute_logistic_gradient(X_b, y_true, y_pred_proba):
    """Compute gradient of logistic loss with respect to beta"""
    m = X_b.shape[0]
    return (1 / m) * X_b.T @ (y_pred_proba - y_true)

# ⟢ Building a Logistic Regression Class ⟣
class MyLogisticRegression(BaseAlgorithm):
    """Enhanced Logistic Regression with BaseAlgorithm interface"""
    
    def __init__(self, alpha=0.01, n_iters=1000, threshold=0.5):
        super().__init__()
        self.alpha = alpha
        self.n_iters = n_iters
        self.threshold = threshold
        self.beta = None
        self.loss_history = []
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return algorithm metadata"""
        return {
            "id": "logistic_regression",
            "name": "Logistic Regression (From Scratch)",
            "type": "classification",
            "description": "Binary classification using logistic regression with gradient descent",
            "hyperparameters": {
                "alpha": {
                    "type": "float",
                    "default": 0.01,
                    "min": 0.001,
                    "max": 1.0,
                    "description": "Learning rate for gradient descent"
                },
                "n_iters": {
                    "type": "int", 
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "description": "Number of gradient descent iterations"
                },
                "threshold": {
                    "type": "float",
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "description": "Classification threshold for predictions"
                }
            }
        }
    
    async def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the logistic regression model"""
        start_time = time.time()
        
        # Add bias and convert inputs
        X_b = add_bias(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        
        # Initialize beta
        m, n = X_b.shape
        self.beta = np.zeros(n)
        self.loss_history = []
        
        # Gradient descent
        for i in range(self.n_iters):
            y_pred_proba = predict_proba(X_b, self.beta)
            grad = compute_logistic_gradient(X_b, y_arr, y_pred_proba)
            self.beta -= self.alpha * grad
            
            # Store loss every 10% of iterations
            if i % max(1, (self.n_iters // 10)) == 0:
                loss = compute_logistic_loss(y_arr, y_pred_proba)
                self.loss_history.append(loss)
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        final_loss = compute_logistic_loss(y_arr, y_pred_proba)
        accuracy = accuracy_score(y_arr, y_pred)
        
        return {
            "training_time": float(training_time),
            "metrics": {
                "accuracy": float(accuracy),
                "final_loss": float(final_loss),
                "convergence_loss": float(self.loss_history[-1] if self.loss_history else final_loss)
            },
            "metadata": {
                "coefficients": [float(x) for x in self.beta],
                "loss_history": [float(x) for x in self.loss_history],
                "n_iterations": int(self.n_iters),
                "learning_rate": float(self.alpha),
                "threshold": float(self.threshold),
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0])
            }
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        if self.beta is None:
            raise ValueError("Model must be trained before making predictions")
        X_b = add_bias(X)
        return predict_binary(X_b, self.beta, self.threshold)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.beta is None:
            raise ValueError("Model must be trained before making predictions")
        X_b = add_bias(X)
        return predict_proba(X_b, self.beta)
    
    def get_sklearn_equivalent(self):
        """Return equivalent sklearn model for comparison"""
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=self.n_iters, random_state=42)

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

    def get_params(self):
        """Return learned weight vector (including bias)."""
        return self.beta
    
    def get_loss_history(self):
        """Return the loss history during training."""
        return self.loss_history
    
    def score(self, X, y):
        """Calculate accuracy score for the model."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Demo/Testing code (only runs when script is executed directly)
if __name__ == "__main__":
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, 
                             n_informative=8, n_redundant=2, random_state=42)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Using our class
    model = MyLogisticRegression(alpha=0.1, n_iters=1000)
    model.fit(X_train, y_train)
    print("Learned beta from class:", model.get_params())
    
    y_pred_class = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred_class):.4f}")
    print(f"Final training loss: {model.get_loss_history()[-1]:.4f}")

    # Compare with sklearn's LogisticRegression
    from sklearn.linear_model import LogisticRegression
    sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    print(f"Sklearn accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print(f"Our model accuracy: {model.score(X_test, y_test):.4f}")
    
    print("\nClassification Report (Our Model):")
    print(classification_report(y_test, y_pred_class))
