import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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
class MyLogisticRegression:
    def __init__(self, alpha=0.01, n_iters=1000, threshold=0.5):
        self.alpha = alpha
        self.n_iters = n_iters
        self.threshold = threshold
        self.beta = None
        self.loss_history = []

    def fit(self, X, y):
        # Add bias and convert inputs
        X_b = add_bias(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        
        # Initialize beta
        m, n = X_b.shape
        self.beta = np.zeros(n)
        
        # Gradient descent
        for i in range(self.n_iters):
            y_pred_proba = predict_proba(X_b, self.beta)
            grad = compute_logistic_gradient(X_b, y_arr, y_pred_proba)
            self.beta -= self.alpha * grad
            
            # Store loss every 10% of iterations
            if i % max(1, (self.n_iters // 10)) == 0:
                loss = compute_logistic_loss(y_arr, y_pred_proba)
                self.loss_history.append(loss)
        
        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        X_b = add_bias(X)
        return predict_proba(X_b, self.beta)

    def predict(self, X):
        """Make binary predictions"""
        X_b = add_bias(X)
        return predict_binary(X_b, self.beta, self.threshold)

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
