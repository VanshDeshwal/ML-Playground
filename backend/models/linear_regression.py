import numpy as np
import time
from typing import Dict, Any, Optional
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
import os

# Add parent directory to path to import base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm_registry import BaseAlgorithm

# ⟢ Helper Functions for Linear Regression ⟣
def add_bias(X):
    """Add bias column (column of ones) to the feature matrix."""
    return np.column_stack((np.ones(X.shape[0]), X))

def predict(X_b, beta):
    """Make predictions using linear model: y = X_b @ beta"""
    return X_b @ beta

def compute_loss(y_true, y_pred):
    """Compute Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)

def compute_gradient(X_b, y_true, y_pred):
    """Compute gradient of MSE loss with respect to beta."""
    m = X_b.shape[0]  # number of samples
    return (2 / m) * X_b.T @ (y_pred - y_true)

# ⟢ Enhanced Linear Regression with Base Class ⟣
class MyLinearRegression(BaseAlgorithm):
    """Linear Regression with gradient descent - from scratch implementation"""
    
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.beta = None
        self.loss_history = []
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return algorithm metadata"""
        return {
            "id": "linear_regression",
            "name": "Linear Regression (From Scratch)",
            "type": "regression",
            "description": "Linear regression using gradient descent to find the best fit line through data points",
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
                    "max": 5000,
                    "description": "Number of iterations for gradient descent"
                }
            }
        }
    
    def __init__(self, alpha=0.01, n_iters=1000):
        super().__init__()
        self.alpha = alpha
        self.n_iters = n_iters
        self.beta = None
        self.loss_history = []

    async def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the linear regression model"""
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
            y_pred = predict(X_b, self.beta)
            loss = compute_loss(y_arr, y_pred)
            grad = compute_gradient(X_b, y_arr, y_pred)
            self.beta -= self.alpha * grad
            
            # Store loss history for visualization
            if i % max(1, (self.n_iters // 50)) == 0:  # Store 50 points max
                self.loss_history.append(loss)
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        y_pred_final = self.predict(X)
        final_loss = compute_loss(y_arr, y_pred_final)
        
        # Calculate R² score
        ss_res = np.sum((y_arr - y_pred_final) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "training_time": training_time,
            "metrics": {
                "mse": float(final_loss),
                "r2_score": float(r2_score),
                "rmse": float(np.sqrt(final_loss))
            },
            "metadata": {
                "loss_history": [float(x) for x in self.loss_history],
                "final_loss": float(final_loss),
                "coefficients": [float(x) for x in self.beta[1:]] if len(self.beta) > 1 else [],
                "intercept": float(self.beta[0]) if len(self.beta) > 0 else 0.0,
                "n_iterations": int(self.n_iters),
                "learning_rate": float(self.alpha),
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0]),
                "converged": bool(len(self.loss_history) > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6)
            }
        }

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

    def predict(self, X):
        """Make predictions on new data"""
        if self.beta is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        X_b = add_bias(X)
        return predict(X_b, self.beta)
    
    def get_sklearn_equivalent(self):
        """Get equivalent sklearn model for comparison"""
        return LinearRegression()
    
    def get_training_metadata(self) -> dict:
        """Get training metadata including loss history"""
        return {
            "loss_history": self.loss_history,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "coefficients": self.beta[1:].tolist() if self.beta is not None else None,
            "intercept": self.beta[0] if self.beta is not None else None,
            "iterations": len(self.loss_history),
            "converged": len(self.loss_history) > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6
        }

    def get_params(self):
        """Return learned weight vector (including bias)."""
        return self.beta
    
    def get_loss_history(self):
        """Return the loss history during training."""
        return self.loss_history
    
    def score(self, X, y):
        """Calculate R² score for the model."""
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)

# Demo/Testing code (only runs when script is executed directly)
if __name__ == "__main__":
    # Generate sample data
    X, y = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=42)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Using our class
    model = MyLinearRegression(alpha=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    print("Learned beta from class:", model.get_params())
    y_pred_class = model.predict(X_test)
    print(f"Class MSE on test set: {mean_squared_error(y_test, y_pred_class):.4f}")

    # Additional analysis
    print(f"R² Score on test set: {model.score(X_test, y_test):.4f}")
    print(f"Final training loss: {model.get_loss_history()[-1]:.4f}")

    # Compare with sklearn's LinearRegression for validation
    from sklearn.linear_model import LinearRegression
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    print(f"Sklearn MSE on test set: {mean_squared_error(y_test, y_pred_sklearn):.4f}")
    print(f"Sklearn R² Score: {sklearn_model.score(X_test, y_test):.4f}")

    print("\nCoefficient comparison (our model vs sklearn):")
    print(f"Our bias: {model.get_params()[0]:.4f}, Sklearn bias: {sklearn_model.intercept_:.4f}")
    print(f"Max coefficient difference: {np.max(np.abs(model.get_params()[1:] - sklearn_model.coef_)):.6f}")