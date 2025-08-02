import numpy as np


# === SNIPPET-START: LinearRegressionScratch ===
class LinearRegressionScratch:
  def __init__(self, alpha=0.01, n_iters=1000, tolerance=1e-6):
    self.alpha = alpha
    self.n_iters = n_iters
    self.beta = None

    self.loss_history = []  # code for 
    self.tolerance = tolerance
    self.is_fitted = False

  def fit(self, X, y):
    # first we need to add the bias terms to the vector X
    ones = np.ones((X.shape[0], 1))        # shape: (m, 1)
    # why can we just use ones = [1]*X.shape[0] ?
      # because this create a python list, but we want a numpy array
    X = np.hstack((ones, X))
    # X is ready
    # Now lets initialize beta
    self.beta = np.zeros(X.shape[1])
    for i in range(self.n_iters):
      y_pred = np.dot(X, self.beta)
      cost = (1 / (2 * X.shape[0])) * np.sum((y_pred - y) ** 2)
      self.loss_history.append(cost)
      if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tolerance:
        break
      grad = (1/X.shape[0])*np.dot(X.T, y_pred-y)
      self.beta = self.beta - self.alpha*grad
    self.is_fitted = True
    return self
  
  def predict(self, X):
    if not self.is_fitted:
        raise ValueError("Model must be fitted before making predictions")
    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones, X))
    y_pred=np.dot(X, self.beta)
    return y_pred
  
# === SNIPPET-END: LinearRegressionScratch ===
  
  # code to expose coefficients(better for website)
  @property
  def intercept_(self):
    return self.beta[0] if self.beta is not None else None

  @property  
  def coef_(self):
    return self.beta[1:] if self.beta is not None else None
