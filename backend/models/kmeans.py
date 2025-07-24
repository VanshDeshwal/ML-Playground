import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
class MyKMeans:
    def __init__(self, k=3, max_iters=100, random_state=None, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        self.n_iter_ = 0

    def fit(self, X):
        # Initialize centroids
        self.centroids = initialize_centroids(X, self.k, self.random_state)
        
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
        
        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        return assign_clusters(X, self.centroids)

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
