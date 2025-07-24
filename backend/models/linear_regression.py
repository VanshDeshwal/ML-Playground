import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

# ⟢ Building a LinearRegression Class ⟣
class MyLinearRegression:
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.beta = None
        self.loss_history = []

    def fit(self, X, y):
        # Add bias and convert inputs
        X_b = add_bias(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        # Initialize beta
        m, n = X_b.shape
        self.beta = np.zeros(n)
        for i in range(self.n_iters):
            y_pred = predict(X_b, self.beta)
            grad   = compute_gradient(X_b, y_arr, y_pred)
            self.beta -= self.alpha * grad
            if i % max(1, (self.n_iters // 10)) == 0:
                self.loss_history.append(compute_loss(y_arr, y_pred))
        return self

    def predict(self, X):
        X_b = add_bias(X)
        return predict(X_b, self.beta)

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