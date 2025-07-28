import numpy as np
import time
from typing import Dict, Any, Optional
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import our base algorithm interface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm_registry import BaseAlgorithm

# ⟢ Helper Functions for k-Means Clustering ⟣
def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(X, k, random_state=None):
    """Initialize centroids randomly"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        centroid_idx = np.random.choice(n_samples)
        centroids[i] = X[centroid_idx]
    
    return centroids

def assign_clusters(X, centroids):
    """Assign each point to the nearest centroid"""
    n_samples = X.shape[0]
    k = centroids.shape[0]
    
    distances = np.zeros((n_samples, k))
    
    for i, centroid in enumerate(centroids):
        for j, point in enumerate(X):
            distances[j, i] = euclidean_distance(point, centroid)
    
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update centroids based on current cluster assignments"""
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    
    return centroids

def compute_inertia(X, labels, centroids):
    """Compute within-cluster sum of squares (inertia)"""
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

# ⟢ Building a k-Means Clustering Class ⟣
class MyKMeans(BaseAlgorithm):
    """Enhanced K-Means Clustering with BaseAlgorithm interface"""
    
    def __init__(self, k=3, max_iters=100, random_state=None, tol=1e-4):
        super().__init__()
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        self.n_iter_ = 0
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return algorithm metadata"""
        return {
            "id": "kmeans",
            "name": "K-Means Clustering (From Scratch)",
            "type": "clustering",
            "description": "Unsupervised clustering using k-means algorithm with Lloyd's algorithm",
            "hyperparameters": {
                "k": {
                    "type": "int",
                    "default": 3,
                    "min": 2,
                    "max": 20,
                    "description": "Number of clusters to form"
                },
                "max_iters": {
                    "type": "int",
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "description": "Maximum number of iterations for convergence"
                },
                "tol": {
                    "type": "float",
                    "default": 1e-4,
                    "min": 1e-8,
                    "max": 1e-2,
                    "description": "Tolerance for convergence detection"
                },
                "random_state": {
                    "type": "int",
                    "default": 42,
                    "min": 0,
                    "max": 1000,
                    "description": "Random seed for reproducible results"
                }
            }
        }
    
    async def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Train the k-means clustering model"""
        start_time = time.time()
        
        # Initialize centroids
        self.centroids = initialize_centroids(X, self.k, self.random_state)
        self.inertia_history = []
        
        prev_centroids = None
        
        for i in range(self.max_iters):
            # Assign points to clusters
            self.labels = assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = update_centroids(X, self.labels, self.k)
            
            # Calculate inertia
            inertia = compute_inertia(X, self.labels, new_centroids)
            self.inertia_history.append(inertia)
            
            # Check for convergence
            if prev_centroids is not None:
                centroid_shift = np.sqrt(np.sum((new_centroids - prev_centroids) ** 2))
                if centroid_shift < self.tol:
                    break
            
            prev_centroids = self.centroids.copy()
            self.centroids = new_centroids
            self.n_iter_ = i + 1
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        silhouette = silhouette_score(X, self.labels) if len(np.unique(self.labels)) > 1 else 0.0
        final_inertia = self.inertia_history[-1] if self.inertia_history else 0.0
        
        return {
            "training_time": float(training_time),
            "metrics": {
                "silhouette_score": float(silhouette),
                "inertia": float(final_inertia),
                "n_iterations": int(self.n_iter_),
                "converged": bool(self.n_iter_ < self.max_iters)
            },
            "metadata": {
                "centroids": [[float(val) for val in centroid] for centroid in self.centroids],
                "inertia_history": [float(x) for x in self.inertia_history],
                "n_clusters": int(self.k),
                "n_iterations": int(self.n_iter_),
                "tolerance": float(self.tol),
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0]),
                "cluster_sizes": [int(np.sum(self.labels == i)) for i in range(self.k)]
            }
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.centroids is None:
            raise ValueError("Model must be trained before making predictions")
        return assign_clusters(X, self.centroids)
    
    def get_sklearn_equivalent(self):
        """Return equivalent sklearn model for comparison"""
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=self.k, max_iter=self.max_iters, 
                     random_state=self.random_state, tol=self.tol, n_init=1)

    def fit(self, X):
        """Legacy interface - calls async train method"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(self.train(X))
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels"""
        self.fit(X)
        return self.labels

    def get_centroids(self):
        """Return the cluster centroids"""
        return self.centroids
    
    def get_inertia_history(self):
        """Return the inertia history during training"""
        return self.inertia_history
    
    def get_inertia(self):
        """Return the final inertia"""
        return self.inertia_history[-1] if self.inertia_history else None
    
    def score(self, X, y=None):
        """Calculate silhouette score if possible"""
        if self.labels is not None and len(np.unique(self.labels)) > 1:
            return silhouette_score(X, self.labels)
        return 0.0

# Demo/Testing code (only runs when script is executed directly)
if __name__ == "__main__":
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                          random_state=42, n_features=2)
    
    print("Generated dataset with 4 true clusters")
    
    # Using our k-means implementation
    model = MyKMeans(k=4, max_iters=100, random_state=42)
    y_pred = model.fit_predict(X)
    
    print(f"Converged in {model.n_iter_} iterations")
    print(f"Final inertia: {model.get_inertia():.4f}")
    print(f"Silhouette score: {model.score(X):.4f}")
    
    # Compare with sklearn's KMeans
    from sklearn.cluster import KMeans
    sklearn_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred_sklearn = sklearn_model.fit_predict(X)
    
    print(f"Sklearn inertia: {sklearn_model.inertia_:.4f}")
    print(f"Our model inertia: {model.get_inertia():.4f}")
    
    # Calculate adjusted rand score to compare with true labels
    print(f"Adjusted Rand Score (our model): {adjusted_rand_score(y_true, y_pred):.4f}")
    print(f"Adjusted Rand Score (sklearn): {adjusted_rand_score(y_true, y_pred_sklearn):.4f}")
    
    print(f"Centroids found:")
    for i, centroid in enumerate(model.get_centroids()):
        print(f"  Cluster {i}: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
