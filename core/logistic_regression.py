import numpy as np

# === SNIPPET-START: LogisticRegressionScratch ===
class LogisticRegressionScratch:
  def __init__(self, alpha=0.01, n_iters=1000, tolerance=1e-6):
    self.alpha = alpha
    self.n_iters = n_iters
    
    # For Visualisation
    self.loss_history = []
    self.weights = None
    self.tolerance = tolerance

  def fit(self, X, y):
    X = np.insert(X, 0, 1, axis= 1)
    self.weights = np.ones(X.shape[1])

    for i in range(self.n_iters):
      y_pred = self.sigmoid(np.dot(X,self.weights))
      self.weights = self.weights + self.alpha*(np.dot(y - y_pred,X)/X.shape[0])
      loss = self.compute_loss(y, y_pred)
      self.loss_history.append(loss)
      if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tolerance:
            break
    return self

  def predict_proba(self, X):
    X = np.insert(X, 0, 1, axis=1)
    return self.sigmoid(np.dot(X, self.weights))

  def predict(self, X):
    probs = self.predict_proba(X)
    return (probs >= 0.5).astype(int)

  def sigmoid(self, z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1/(1 + np.exp(-z))

  def compute_loss(self,y, y_pred):
    """
    Binary Cross Entropy Loss:
    L = - (1/m) * [y * log(y_pred) + (1 - y) * log(1 - y_pred)]
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # to avoid log(0)
    m = y.shape[0]
    return - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
  
# === SNIPPET-END: LogisticRegressionScratch ===
  
  @property
  def intercept_(self):
    return self.weights[0] if self.weights is not None else None
  
  @property  
  def coef_(self):
    return self.weights[1:] if self.weights is not None else None