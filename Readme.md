# 🧠 ML Playground

A modern, interactive machine learning playground built with FastAPI and vanilla JavaScript. Explore, train, and experiment with ML algorithms through a clean, responsive web interface.

## ✨ Features

- **🔄 Auto-Discovery**: Automatically discovers ML algorithms from the `/core` directory
- **🎨 Interactive UI**: Clean, modern interface with syntax-highlighted code viewing
- **📊 Real-time Training**: Live training progress and metrics visualization
- **🚀 Fast API**: High-performance backend with automatic API documentation
- **📱 Responsive**: Works seamlessly on desktop and mobile devices
- **🎯 Type Safety**: Full TypeScript-style validation with Pydantic models

## 🏗️ Architecture

```
ML Playground/
├── frontend/          # Static web interface
│   ├── index.html     # Homepage
│   ├── algorithm.html # Algorithm training page
│   ├── styles.css     # Modern CSS with variables
│   ├── app.js         # Homepage controller
│   ├── algorithm.js   # Training page controller
│   └── api.js         # API service with smart caching
├── backend/           # FastAPI backend
│   ├── main.py        # Application entry point
│   ├── api/           # API route handlers
│   ├── services/      # Business logic
│   ├── models/        # Pydantic data models
│   ├── adapters/      # Algorithm adapters
│   └── core_integration/ # Auto-discovery system
├── core/              # ML algorithm implementations
└── .github/workflows/ # CI/CD pipelines
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VanshDeshwal/ML-Playground.git
   cd ML-Playground
   ```

2. **Setup Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Run the backend**
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open the frontend**
   - Open `frontend/index.html` in your browser
   - Or serve it locally: `python -m http.server 3000` from the frontend directory

## 📚 Adding New Algorithms

1. **Create your algorithm file** in the `/core` directory:
   ```python
   # core/my_algorithm.py
   
   # === SNIPPET-START: MyAlgorithmClass ===
   class MyAlgorithmClass:
       def __init__(self, param1=1.0, param2=10):
           self.param1 = param1
           self.param2 = param2
       
       def fit(self, X, y):
           # Training logic here
           return self
       
       def predict(self, X):
           # Prediction logic here
           return predictions
   # === SNIPPET-END: MyAlgorithmClass ===
   ```

2. **Restart the backend** - The auto-discovery system will automatically detect and register your new algorithm!

## 🔧 API Endpoints

- `GET /api/algorithms` - List all available algorithms
- `GET /api/algorithms/{id}` - Get algorithm details
- `POST /api/algorithms/{id}/train` - Train an algorithm
- `GET /api/algorithms/{id}/code` - Get algorithm source code
- `GET /health` - Health check

Full API documentation available at `http://localhost:8000/docs` when running the backend.

## 🌐 Deployment

### Frontend (GitHub Pages)
The frontend is automatically deployed to GitHub Pages on every push to main.

### Backend (Azure Container Apps)
The backend is automatically deployed to Azure on every push to main.

**Live Demo**: [ML Playground](https://vanshdeshwal.github.io/ML-Playground/)

## 🛠️ Technologies

### Frontend
- **Vanilla JavaScript** - No frameworks, maximum performance
- **CSS Grid & Flexbox** - Modern responsive layouts
- **Prism.js** - Syntax highlighting for code viewing
- **Font Awesome** - Beautiful icons
- **Inter Font** - Clean, modern typography

### Backend
- **FastAPI** - High-performance Python web framework
- **Pydantic** - Data validation and serialization
- **NumPy & Scikit-learn** - Core ML libraries
- **Uvicorn** - ASGI server

### DevOps
- **GitHub Actions** - CI/CD pipelines
- **Docker** - Containerization
- **Azure Container Apps** - Cloud deployment

## 📈 Performance Features

- **Smart Caching** - Frontend caches API responses with TTL
- **Async Processing** - Non-blocking algorithm training
- **Code Splitting** - Modular architecture for fast loading
- **Auto-Discovery** - Dynamic algorithm loading without restarts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add your algorithm to the `/core` directory
4. Test locally
5. Commit changes: `git commit -am 'Add my feature'`
6. Push to branch: `git push origin feature/my-feature`
7. Submit a Pull Request

## 📋 Algorithm Implementation Guidelines

### Essential Structure for All Algorithms

```python
import numpy as np

class YourAlgorithmScratch:
    def __init__(self, **hyperparameters):
        # Store hyperparameters
        # Initialize tracking variables
        pass
    
    def fit(self, X, y=None):  # y=None for unsupervised
        # Training logic
        return self  # Enable method chaining
    
    def predict(self, X):
        # Prediction logic
        return predictions  # numpy array
```

### 🔵 Regression Algorithms Template

```python
import numpy as np

# === SNIPPET-START: RegressionAlgorithmName ===
class RegressionAlgorithmScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=0.0, 
                 fit_intercept=True, tolerance=1e-6, random_state=42):
        # Core hyperparameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Training artifacts (for visualization)
        self.coefficients_ = None        # np.ndarray: Model coefficients
        self.intercept_ = None           # float: Bias/intercept term
        self.loss_history_ = []          # List[float]: Loss per iteration
        self.n_features_in_ = None       # int: Number of input features
        self.is_fitted_ = False          # bool: Training status
        
    def fit(self, X, y):
        """
        Train the regression model
        Args:
            X: np.ndarray of shape (n_samples, n_features)
            y: np.ndarray of shape (n_samples,)
        Returns:
            self: Fitted estimator
        """
        # Store training info
        self.n_features_in_ = X.shape[1]
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_with_intercept = X.copy()
        
        # Initialize weights
        np.random.seed(self.random_state)
        self.weights_ = np.random.normal(0, 0.01, X_with_intercept.shape[1])
        
        # Training loop with loss tracking
        prev_loss = float('inf')
        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = X_with_intercept @ self.weights_
            
            # Calculate loss (with regularization if applicable)
            loss = self._calculate_loss(predictions, y)
            self.loss_history_.append(loss)
            
            # Convergence check
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
            
            # Backward pass (gradient calculation)
            gradients = self._calculate_gradients(X_with_intercept, predictions, y)
            
            # Update weights
            self.weights_ -= self.learning_rate * gradients
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = self.weights_[0]
            self.coefficients_ = self.weights_[1:]
        else:
            self.intercept_ = 0.0
            self.coefficients_ = self.weights_
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        Args:
            X: np.ndarray of shape (n_samples, n_features)
        Returns:
            np.ndarray of shape (n_samples,): Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_with_intercept = X.copy()
            
        return X_with_intercept @ self.weights_
    
    def _calculate_loss(self, predictions, y_true):
        """Calculate loss function (override in subclasses)"""
        mse = np.mean((predictions - y_true) ** 2)
        if self.regularization > 0:
            reg_term = self.regularization * np.sum(self.weights_[1:] ** 2)
            return mse + reg_term
        return mse
    
    def _calculate_gradients(self, X, predictions, y_true):
        """Calculate gradients (override in subclasses)"""
        errors = predictions - y_true
        gradients = (2 / len(y_true)) * X.T @ errors
        if self.regularization > 0:
            # Don't regularize intercept term
            reg_grad = np.zeros_like(gradients)
            reg_grad[1:] = 2 * self.regularization * self.weights_[1:]
            gradients += reg_grad
        return gradients
    
    # Optional: Additional methods for advanced features
    def get_feature_importance(self):
        """Return feature importance scores"""
        if not self.is_fitted_:
            return None
        return np.abs(self.coefficients_)
    
    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
# === SNIPPET-END: RegressionAlgorithmName ===
```

### 🟡 Classification Algorithms Template

```python
import numpy as np

# === SNIPPET-START: ClassificationAlgorithmName ===
class ClassificationAlgorithmScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=0.0,
                 tolerance=1e-6, random_state=42):
        # Core hyperparameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Training artifacts
        self.classes_ = None             # np.ndarray: Unique class labels
        self.n_classes_ = None           # int: Number of classes
        self.coefficients_ = None        # np.ndarray: Model coefficients
        self.intercept_ = None           # float: Bias term
        self.loss_history_ = []          # List[float]: Loss per iteration
        self.n_features_in_ = None       # int: Number of input features
        self.is_fitted_ = False          # bool: Training status
        
    def fit(self, X, y):
        """
        Train the classification model
        Args:
            X: np.ndarray of shape (n_samples, n_features)
            y: np.ndarray of shape (n_samples,)
        Returns:
            self: Fitted estimator
        """
        # Store class information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # For binary classification, encode as 0/1
        if self.n_classes_ == 2:
            y_encoded = (y == self.classes_[1]).astype(int)
        else:
            # Multi-class: one-vs-rest or implement softmax
            y_encoded = y  # Implement based on your approach
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize weights
        np.random.seed(self.random_state)
        self.weights_ = np.random.normal(0, 0.01, X_with_intercept.shape[1])
        
        # Training loop
        prev_loss = float('inf')
        for iteration in range(self.max_iterations):
            # Forward pass
            logits = X_with_intercept @ self.weights_
            probabilities = self._sigmoid(logits)  # or softmax for multi-class
            
            # Calculate loss
            loss = self._calculate_loss(probabilities, y_encoded)
            self.loss_history_.append(loss)
            
            # Convergence check
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
            
            # Backward pass
            gradients = self._calculate_gradients(X_with_intercept, probabilities, y_encoded)
            
            # Update weights
            self.weights_ -= self.learning_rate * gradients
        
        # Extract coefficients and intercept
        self.intercept_ = self.weights_[0]
        self.coefficients_ = self.weights_[1:]
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Make class predictions"""
        probabilities = self.predict_proba(X)
        if self.n_classes_ == 2:
            return self.classes_[(probabilities[:, 1] > 0.5).astype(int)]
        else:
            return self.classes_[np.argmax(probabilities, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        Returns:
            np.ndarray of shape (n_samples, n_classes): Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        logits = X_with_intercept @ self.weights_
        
        if self.n_classes_ == 2:
            prob_class_1 = self._sigmoid(logits)
            return np.column_stack([1 - prob_class_1, prob_class_1])
        else:
            return self._softmax(logits)  # Implement for multi-class
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _calculate_loss(self, probabilities, y_true):
        """Binary cross-entropy loss"""
        epsilon = 1e-15  # Prevent log(0)
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities))
        
        # Add regularization if specified
        if self.regularization > 0:
            reg_term = self.regularization * np.sum(self.weights_[1:] ** 2)
            loss += reg_term
        return loss
    
    def _calculate_gradients(self, X, probabilities, y_true):
        """Calculate gradients for logistic regression"""
        errors = probabilities - y_true
        gradients = (1 / len(y_true)) * X.T @ errors
        
        # Add regularization gradients
        if self.regularization > 0:
            reg_grad = np.zeros_like(gradients)
            reg_grad[1:] = 2 * self.regularization * self.weights_[1:]
            gradients += reg_grad
        return gradients
    
    def decision_function(self, X):
        """Return decision function values"""
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ self.weights_
# === SNIPPET-END: ClassificationAlgorithmName ===
```

### 🟣 Clustering Algorithms Template

```python
import numpy as np

# === SNIPPET-START: ClusteringAlgorithmName ===
class ClusteringAlgorithmScratch:
    def __init__(self, n_clusters=3, max_iterations=300, tolerance=1e-4, 
                 random_state=42, init_method='random'):
        # Core hyperparameters
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.init_method = init_method
        
        # Training artifacts
        self.cluster_centers_ = None     # np.ndarray: Cluster centroids
        self.labels_ = None              # np.ndarray: Cluster labels for training data
        self.inertia_ = None             # float: Within-cluster sum of squared distances
        self.n_features_in_ = None       # int: Number of features
        self.n_iter_ = None              # int: Number of iterations run
        self.is_fitted_ = False          # bool: Training status
        self.history_ = {                # Dict: Training history for visualization
            'inertia': [],               # List[float]: Inertia per iteration
            'center_changes': []         # List[float]: Centroid movement per iteration
        }
        
    def fit(self, X, y=None):  # y is ignored for clustering
        """
        Fit the clustering algorithm
        Args:
            X: np.ndarray of shape (n_samples, n_features)
        Returns:
            self: Fitted estimator
        """
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Initialize cluster centers
        np.random.seed(self.random_state)
        if self.init_method == 'random':
            # Random initialization
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.cluster_centers_ = X[indices].copy()
        elif self.init_method == 'k-means++':
            # K-means++ initialization (implement if desired)
            self.cluster_centers_ = self._init_kmeans_plus_plus(X)
        
        # Main clustering loop
        for iteration in range(self.max_iterations):
            # Store previous centers for convergence check
            prev_centers = self.cluster_centers_.copy()
            
            # Assign points to nearest clusters
            distances = self._calculate_distances(X, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update cluster centers
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points, axis=0)
                else:
                    # Keep previous center if cluster is empty
                    new_centers[k] = self.cluster_centers_[k]
            
            self.cluster_centers_ = new_centers
            
            # Calculate metrics for this iteration
            self._update_history(X, prev_centers)
            
            # Check for convergence
            center_changes = np.sum(np.linalg.norm(self.cluster_centers_ - prev_centers, axis=1))
            if center_changes < self.tolerance:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iterations
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X)
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        Args:
            X: np.ndarray of shape (n_samples, n_features)
        Returns:
            np.ndarray of shape (n_samples,): Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        distances = self._calculate_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels for training data"""
        self.fit(X)
        return self.labels_
    
    def _calculate_distances(self, X, centers):
        """Calculate distances between points and cluster centers"""
        distances = np.zeros((X.shape[0], len(centers)))
        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        return distances
    
    def _calculate_inertia(self, X):
        """Calculate within-cluster sum of squared distances"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                center = self.cluster_centers_[k]
                inertia += np.sum(np.linalg.norm(cluster_points - center, axis=1) ** 2)
        return inertia
    
    def _update_history(self, X, prev_centers):
        """Update training history for visualization"""
        current_inertia = self._calculate_inertia(X)
        self.history_['inertia'].append(current_inertia)
        
        center_change = np.mean(np.linalg.norm(self.cluster_centers_ - prev_centers, axis=1))
        self.history_['center_changes'].append(center_change)
    
    def _init_kmeans_plus_plus(self, X):
        """K-means++ initialization (optional advanced feature)"""
        # Implement if you want better initialization
        # For now, fall back to random
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()
    
    # Additional utility methods
    def transform(self, X):
        """Transform X to cluster-distance space"""
        return self._calculate_distances(X, self.cluster_centers_)
    
    def score(self, X):
        """Return the negative inertia (higher is better for scoring)"""
        if not self.is_fitted_:
            self.fit(X)
        return -self.inertia_
# === SNIPPET-END: ClusteringAlgorithmName ===
```

### 📊 Common Hyperparameters by Algorithm Type

#### Regression Algorithms:
- `learning_rate` (0.01): Step size for gradient descent
- `max_iterations` (1000): Maximum training iterations
- `tolerance` (1e-6): Convergence threshold
- `regularization` (0.0): L2 regularization strength
- `fit_intercept` (True): Whether to include bias term
- `random_state` (42): Random seed for reproducibility

#### Classification Algorithms:
- `learning_rate` (0.01): Step size for gradient descent
- `max_iterations` (1000): Maximum training iterations
- `tolerance` (1e-6): Convergence threshold
- `regularization` (0.01): Regularization strength
- `random_state` (42): Random seed for reproducibility

#### Clustering Algorithms:
- `n_clusters` (3): Number of clusters to form
- `max_iterations` (300): Maximum iterations for convergence
- `tolerance` (1e-4): Convergence threshold
- `init_method` ('random'): Initialization method
- `random_state` (42): Random seed for reproducibility

### 🎯 What Makes a Good Implementation:

1. **Store Training History**: Always track loss/inertia for visualization
2. **Handle Edge Cases**: Empty clusters, numerical stability
3. **Provide Metadata**: Number of features, iterations, convergence status
4. **Follow Naming Conventions**: Use scikit-learn style attribute names with trailing underscore
5. **Include Random State**: Ensure reproducible results
6. **Add Convergence Checks**: Stop early when algorithm converges

### 🚨 Common Pitfalls to Avoid:

- **Numerical Instability**: Use np.clip() for sigmoid, handle log(0)
- **Memory Issues**: Don't store entire training data unnecessarily
- **Infinite Loops**: Always include max_iterations limit
- **Shape Mismatches**: Always validate input shapes
- **Missing Return Values**: Always return self from fit() method

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for the ML community
- Inspired by modern web development practices
- Designed for educational and research purposes

---

**Happy Learning! 🚀**
