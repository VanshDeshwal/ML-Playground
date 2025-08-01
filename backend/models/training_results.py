"""
Training Results Models
Comprehensive results system for algorithm training with sklearn comparison and visualization data
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np

class AlgorithmMetrics(BaseModel):
    """Metrics for a single algorithm implementation"""
    # Regression metrics
    r2_score: Optional[float] = Field(None, description="R-squared score")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    
    # Classification metrics
    accuracy: Optional[float] = Field(None, description="Classification accuracy")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    
    # Clustering metrics
    silhouette_score: Optional[float] = Field(None, description="Silhouette score")
    inertia: Optional[float] = Field(None, description="Within-cluster sum of squares")
    calinski_harabasz_score: Optional[float] = Field(None, description="Calinski-Harabasz index")
    
    # Common metrics
    training_time: float = Field(..., description="Training time in seconds")
    convergence_iterations: Optional[int] = Field(None, description="Number of iterations until convergence")
    converged: Optional[bool] = Field(None, description="Whether algorithm converged")

class ChartData(BaseModel):
    """Data for creating charts and visualizations"""
    # Common chart types
    scatter_plot: Optional[Dict[str, List[float]]] = Field(None, description="Scatter plot data")
    line_plot: Optional[Dict[str, List[float]]] = Field(None, description="Line plot data") 
    loss_curve: Optional[Dict[str, List[float]]] = Field(None, description="Training loss curve")
    
    # Regression specific
    residuals_plot: Optional[Dict[str, List[float]]] = Field(None, description="Residuals vs predictions")
    regression_line: Optional[Dict[str, List[float]]] = Field(None, description="Fitted regression line")
    
    # Classification specific
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    roc_curve: Optional[Dict[str, List[float]]] = Field(None, description="ROC curve data")
    decision_boundary: Optional[Dict[str, Any]] = Field(None, description="Decision boundary data")
    
    # Clustering specific
    cluster_plot: Optional[Dict[str, Any]] = Field(None, description="Cluster visualization data")
    cluster_centers: Optional[List[List[float]]] = Field(None, description="Cluster center coordinates")

class AlgorithmResult(BaseModel):
    """Results for a single algorithm implementation"""
    metrics: AlgorithmMetrics = Field(..., description="Performance metrics")
    predictions: List[float] = Field(..., description="Model predictions")
    coefficients: Optional[List[float]] = Field(None, description="Model coefficients")
    intercept: Optional[float] = Field(None, description="Model intercept")
    feature_importance: Optional[List[float]] = Field(None, description="Feature importance scores")
    training_history: Optional[List[float]] = Field(None, description="Training loss/cost history")

class ComparisonSummary(BaseModel):
    """Summary of comparison between implementations"""
    performance_differences: Dict[str, float] = Field(..., description="Metric differences")
    speed_comparison: Dict[str, float] = Field(..., description="Training time comparison")
    accuracy_analysis: Optional[str] = Field(None, description="Accuracy comparison text")
    recommendation: Optional[str] = Field(None, description="Which implementation performed better")

class DatasetInfo(BaseModel):
    """Information about the dataset used"""
    name: str = Field(..., description="Dataset name")
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features")
    feature_names: Optional[List[str]] = Field(None, description="Feature column names")
    target_name: Optional[str] = Field(None, description="Target variable name")
    train_size: int = Field(..., description="Training set size")
    test_size: int = Field(..., description="Test set size")
    split_ratio: float = Field(..., description="Train/test split ratio")

class TrainingResult(BaseModel):
    """Comprehensive training results with sklearn comparison and visualization data"""
    # Basic info
    success: bool = Field(..., description="Whether training was successful")
    algorithm_id: str = Field(..., description="Algorithm identifier")
    algorithm_type: str = Field(..., description="Type: regression, classification, clustering")
    
    # Dataset information
    dataset: DatasetInfo = Field(..., description="Dataset information")
    
    # Hyperparameters used
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters used for training")
    
    # Algorithm results
    your_implementation: AlgorithmResult = Field(..., description="Your algorithm results")
    sklearn_implementation: AlgorithmResult = Field(..., description="Sklearn algorithm results")
    
    # Comparison analysis
    comparison: ComparisonSummary = Field(..., description="Comparison between implementations")
    
    # Visualization data
    charts: ChartData = Field(..., description="Data for creating charts")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if training failed")
    warnings: Optional[List[str]] = Field(None, description="Warning messages")
    
    # Additional metadata
    total_duration: float = Field(..., description="Total training and comparison time")
    timestamp: str = Field(..., description="Training timestamp")

    class Config:
        # Allow numpy arrays to be serialized
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            np.float64: float,
            np.int64: int
        }
