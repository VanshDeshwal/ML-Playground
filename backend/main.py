from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification, load_diabetes, load_wine, load_iris, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
from io import StringIO
import json

# Import sklearn models for comparison
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Import our custom models
from models.linear_regression import MyLinearRegression
from models.logistic_regression import MyLogisticRegression
from models.kmeans import MyKMeans
from models.knn import MyKNN
from models.decision_tree import MyDecisionTree

app = FastAPI(title="ML Playground API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class AlgorithmInfo(BaseModel):
    id: str
    name: str
    description: str
    type: str  # "regression" or "classification"
    hyperparameters: Dict[str, Any]

class TrainingRequest(BaseModel):
    algorithm_id: str
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    compare_sklearn: bool = True
    dataset_source: str = "generated"  # "generated", "builtin", "uploaded"
    builtin_dataset: str = "boston"  # for builtin datasets
    uploaded_data: Optional[Dict[str, Any]] = None  # for uploaded datasets

class TrainingResponse(BaseModel):
    success: bool
    metrics: Dict[str, float]
    predictions: List[float]
    model_info: Dict[str, Any]
    training_history: Optional[List[float]] = None
    sklearn_comparison: Optional[Dict[str, Any]] = None
    sklearn_comparison: Optional[Dict[str, Any]] = None

# Built-in datasets configuration
BUILTIN_DATASETS = {
    "diabetes": {
        "name": "Diabetes Dataset",
        "type": "regression",
        "description": "Predict diabetes progression",
        "loader": load_diabetes
    },
    "wine": {
        "name": "Wine Dataset", 
        "type": "classification",
        "description": "Wine quality classification",
        "loader": load_wine
    },
    "iris": {
        "name": "Iris Dataset",
        "type": "classification", 
        "description": "Iris species classification",
        "loader": load_iris
    },
    "breast_cancer": {
        "name": "Breast Cancer Dataset",
        "type": "classification",
        "description": "Breast cancer diagnosis",
        "loader": load_breast_cancer
    },
    "blobs": {
        "name": "Blobs Dataset",
        "type": "clustering",
        "description": "Synthetic clustering dataset with distinct clusters",
        "loader": lambda: make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=1.0, random_state=42)
    }
}

def load_builtin_dataset(dataset_name: str):
    """Load a built-in dataset"""
    if dataset_name not in BUILTIN_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found")
    
    dataset_info = BUILTIN_DATASETS[dataset_name]
    data = dataset_info["loader"]()
    
    # Handle clustering datasets differently (make_blobs returns tuple)
    if dataset_info["type"] == "clustering":
        X, y = data
        return {
            "X": X,
            "y": y,
            "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
            "target_names": [f"cluster_{i}" for i in range(len(np.unique(y)))],
            "description": f"Synthetic clustering dataset with {len(np.unique(y))} clusters",
            "type": dataset_info["type"]
        }
    else:
        # Handle regular sklearn datasets
        return {
            "X": data.data,
            "y": data.target,
            "feature_names": getattr(data, 'feature_names', None),
            "target_names": getattr(data, 'target_names', None),
            "description": data.DESCR,
            "type": dataset_info["type"]
        }

def generate_data(algorithm_id: str, dataset_config: Dict[str, Any]):
    """Generate synthetic data for training"""
    n_samples = dataset_config.get("n_samples", 500)
    n_features = dataset_config.get("n_features", 10)
    noise = dataset_config.get("noise", 0.1)
    test_size = dataset_config.get("test_size", 0.2)
    random_state = dataset_config.get("random_state", 42)
    
    algorithm_info = ALGORITHMS[algorithm_id]
    
    # Generate appropriate dataset based on algorithm type
    if algorithm_info.type in ["regression"]:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
    elif algorithm_info.type == "classification":
        n_classes = 3 if "decision_tree" in algorithm_id else 2
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=min(n_features, max(2, n_features - 2)),
            n_redundant=min(2, n_features // 4),
            random_state=random_state
        )
    elif algorithm_info.type == "clustering":
        # For clustering, create blobs
        from sklearn.datasets import make_blobs
        n_centers = 3  # Default clustering centers
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=min(n_features, 10),  # Limit features for clustering
            cluster_std=1.0,
            random_state=random_state
        )
    else:
        # Default to regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "dataset_info": {"type": "generated", "algorithm_type": algorithm_info.type}
    }

def prepare_dataset(request: TrainingRequest):
    """Prepare dataset based on source type"""
    if request.dataset_source == "generated":
        # Use existing synthetic data generation
        return generate_data(request.algorithm_id, request.dataset_config)
    elif request.dataset_source == "builtin":
        # Load built-in dataset
        dataset = load_builtin_dataset(request.builtin_dataset)
        X, y = dataset["X"], dataset["y"]
        
        # Split the data
        test_size = request.dataset_config.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return {
            "X_train": X_train,
            "X_test": X_test, 
            "y_train": y_train,
            "y_test": y_test,
            "dataset_info": dataset
        }
    elif request.dataset_source == "uploaded":
        # Handle uploaded dataset
        if not request.uploaded_data:
            raise ValueError("No uploaded data provided")
        
        df = pd.DataFrame(request.uploaded_data)
        X = df.iloc[:, :-1].values  # All columns except last
        y = df.iloc[:, -1].values   # Last column as target
        
        test_size = request.dataset_config.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train, 
            "y_test": y_test,
            "dataset_info": {"type": "uploaded", "features": df.columns[:-1].tolist()}
        }

# Available algorithms
ALGORITHMS = {
    "linear_regression": AlgorithmInfo(
        id="linear_regression",
        name="Linear Regression (From Scratch)",
        description="Custom implementation of Linear Regression using gradient descent",
        type="regression",
        hyperparameters={
            "alpha": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 1.0,
                "description": "Learning rate for gradient descent"
            },
            "n_iters": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "description": "Number of iterations for training"
            }
        }
    ),
    "logistic_regression": AlgorithmInfo(
        id="logistic_regression",
        name="Logistic Regression (From Scratch)",
        description="Custom implementation of Logistic Regression using gradient descent with sigmoid activation",
        type="classification",
        hyperparameters={
            "alpha": {
                "type": "float",
                "default": 0.1,
                "min": 0.001,
                "max": 2.0,
                "description": "Learning rate for gradient descent"
            },
            "n_iters": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 5000,
                "description": "Number of iterations for training"
            },
            "threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 0.9,
                "description": "Decision threshold for binary classification"
            }
        }
    ),
    "kmeans": AlgorithmInfo(
        id="kmeans",
        name="k-Means Clustering (From Scratch)",
        description="Custom implementation of k-Means clustering algorithm",
        type="clustering",
        hyperparameters={
            "k": {
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 10,
                "description": "Number of clusters"
            },
            "max_iters": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 500,
                "description": "Maximum number of iterations"
            },
            "tol": {
                "type": "float",
                "default": 1e-4,
                "min": 1e-6,
                "max": 1e-2,
                "description": "Tolerance for convergence"
            }
        }
    ),
    "knn_classification": AlgorithmInfo(
        id="knn_classification",
        name="k-NN Classification (From Scratch)",
        description="Custom implementation of k-Nearest Neighbors for classification",
        type="classification",
        hyperparameters={
            "k": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "Number of neighbors to consider"
            },
            "distance_metric": {
                "type": "select",
                "default": "euclidean",
                "options": ["euclidean", "manhattan"],
                "description": "Distance metric to use"
            },
            "weights": {
                "type": "select",
                "default": "uniform",
                "options": ["uniform", "distance"],
                "description": "Weight function for neighbors"
            }
        }
    ),
    "knn_regression": AlgorithmInfo(
        id="knn_regression",
        name="k-NN Regression (From Scratch)",
        description="Custom implementation of k-Nearest Neighbors for regression",
        type="regression",
        hyperparameters={
            "k": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "Number of neighbors to consider"
            },
            "distance_metric": {
                "type": "select",
                "default": "euclidean",
                "options": ["euclidean", "manhattan"],
                "description": "Distance metric to use"
            },
            "weights": {
                "type": "select",
                "default": "distance",
                "options": ["uniform", "distance"],
                "description": "Weight function for neighbors"
            }
        }
    ),
    "decision_tree_classification": AlgorithmInfo(
        id="decision_tree_classification",
        name="Decision Tree Classification (From Scratch)",
        description="Custom implementation of Decision Tree for classification using information gain",
        type="classification",
        hyperparameters={
            "max_depth": {
                "type": "int",
                "default": 10,
                "min": 3,
                "max": 20,
                "description": "Maximum depth of the tree"
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
                "description": "Minimum samples required to split"
            },
            "criterion": {
                "type": "select",
                "default": "entropy",
                "options": ["entropy", "gini"],
                "description": "Split criterion"
            }
        }
    ),
    "decision_tree_regression": AlgorithmInfo(
        id="decision_tree_regression",
        name="Decision Tree Regression (From Scratch)",
        description="Custom implementation of Decision Tree for regression using MSE",
        type="regression",
        hyperparameters={
            "max_depth": {
                "type": "int",
                "default": 10,
                "min": 3,
                "max": 20,
                "description": "Maximum depth of the tree"
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
                "description": "Minimum samples required to split"
            }
        }
    )
}

@app.get("/")
async def root():
    return {"message": "Welcome to ML Playground API"}

@app.get("/algorithms", response_model=List[AlgorithmInfo])
async def get_algorithms():
    """Get list of available ML algorithms"""
    return list(ALGORITHMS.values())

@app.get("/algorithms/{algorithm_id}", response_model=AlgorithmInfo)
async def get_algorithm(algorithm_id: str):
    """Get details of a specific algorithm"""
    if algorithm_id not in ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return ALGORITHMS[algorithm_id]

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train a model with specified parameters"""
    try:
        if request.algorithm_id not in ALGORITHMS:
            raise HTTPException(status_code=404, detail="Algorithm not found")
        
        # Prepare dataset using the new function
        data = prepare_dataset(request)
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]
        
        algorithm_info = ALGORITHMS[request.algorithm_id]
        sklearn_comparison = None
        
        # Train the model based on algorithm
        if request.algorithm_id == "linear_regression":
            # Train our custom model
            model = MyLinearRegression(
                alpha=request.hyperparameters.get("alpha", 0.01),
                n_iters=request.hyperparameters.get("n_iters", 1000)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2_score": float(r2),
                "mae": float(np.mean(np.abs(y_test - y_pred)))
            }
            
            # Sklearn comparison if requested
            if request.compare_sklearn:
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)
                sklearn_pred = sklearn_model.predict(X_test)
                
                sklearn_comparison = {
                    "mse": float(mean_squared_error(y_test, sklearn_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, sklearn_pred))),
                    "r2_score": float(r2_score(y_test, sklearn_pred)),
                    "mae": float(np.mean(np.abs(y_test - sklearn_pred)))
                }
            
            return TrainingResponse(
                success=True,
                metrics=metrics,
                predictions=y_pred.tolist(),
                model_info={
                    "coefficients": model.get_params().tolist(),
                    "n_features": X_train.shape[1],
                    "training_samples": len(X_train)
                },
                training_history=model.get_loss_history(),
                sklearn_comparison=sklearn_comparison
            )
            
        elif request.algorithm_id == "logistic_regression":
            model = MyLogisticRegression(
                alpha=request.hyperparameters.get("alpha", 0.1),
                n_iters=request.hyperparameters.get("n_iters", 1000)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(accuracy),  # Simplified for demo
                "recall": float(accuracy),
                "f1_score": float(accuracy)
            }
            
            # Sklearn comparison if requested
            sklearn_comparison = None
            if request.compare_sklearn:
                sklearn_model = LogisticRegression(random_state=42, max_iter=1000)
                sklearn_model.fit(X_train, y_train)
                sklearn_pred = sklearn_model.predict(X_test)
                
                sklearn_comparison = {
                    "accuracy": float(accuracy_score(y_test, sklearn_pred)),
                    "precision": float(accuracy_score(y_test, sklearn_pred)),
                    "recall": float(accuracy_score(y_test, sklearn_pred)),
                    "f1_score": float(accuracy_score(y_test, sklearn_pred))
                }
            
            return TrainingResponse(
                success=True,
                metrics=metrics,
                predictions=y_pred.tolist(),
                model_info={
                    "coefficients": model.get_params().tolist(),
                    "n_features": X_train.shape[1],
                    "training_samples": len(X_train)
                },
                training_history=model.get_loss_history(),
                sklearn_comparison=sklearn_comparison
            )
            
        elif request.algorithm_id == "kmeans":
            model = MyKMeans(
                k=request.hyperparameters.get("k", 3),
                max_iters=request.hyperparameters.get("n_iters", 300)
            )
            cluster_labels = model.fit_predict(X_train)
            
            # Calculate metrics
            silhouette = model.score(X_train) if len(np.unique(cluster_labels)) > 1 else 0.0
            inertia = model.get_inertia()
            
            metrics = {
                "silhouette_score": float(silhouette),
                "inertia": float(inertia),
                "n_iterations": int(model.n_iter_)
            }
            
            # Sklearn comparison if requested
            sklearn_comparison = None
            if request.compare_sklearn:
                sklearn_model = KMeans(
                    n_clusters=request.hyperparameters.get("k", 3),
                    random_state=42,
                    n_init=10
                )
                sklearn_model.fit(X_train)
                sklearn_labels = sklearn_model.predict(X_train)
                
                # Calculate silhouette score for sklearn model
                sklearn_silhouette = silhouette_score(X_train, sklearn_labels) if len(np.unique(sklearn_labels)) > 1 else 0.0
                
                sklearn_comparison = {
                    "silhouette_score": float(sklearn_silhouette),
                    "inertia": float(sklearn_model.inertia_),
                    "n_clusters": int(sklearn_model.n_clusters),
                    "n_iterations": int(sklearn_model.n_iter_)
                }
            
            return TrainingResponse(
                success=True,
                metrics=metrics,
                predictions=cluster_labels.tolist(),
                model_info={
                    "centroids": model.get_centroids().tolist(),
                    "n_clusters": request.hyperparameters.get("k", 3),
                    "training_samples": len(X_train)
                },
                training_history=model.get_inertia_history(),
                sklearn_comparison=sklearn_comparison
            )
            
        elif request.algorithm_id in ["knn_classification", "knn_regression"]:
            task = "classification" if "classification" in request.algorithm_id else "regression"
            model = MyKNN(
                k=request.hyperparameters.get("k", 5),
                task=task,
                distance_metric=request.hyperparameters.get("distance_metric", "euclidean"),
                weights=request.hyperparameters.get("weights", "uniform")
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            sklearn_comparison = None
            if task == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {"accuracy": float(accuracy)}
                
                # Sklearn comparison if requested
                if request.compare_sklearn:
                    sklearn_model = KNeighborsClassifier(
                        n_neighbors=request.hyperparameters.get("k", 5)
                    )
                    sklearn_model.fit(X_train, y_train)
                    sklearn_pred = sklearn_model.predict(X_test)
                    
                    sklearn_comparison = {
                        "accuracy": float(accuracy_score(y_test, sklearn_pred))
                    }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "r2_score": float(r2)
                }
                
                # Sklearn comparison if requested
                if request.compare_sklearn:
                    sklearn_model = KNeighborsRegressor(
                        n_neighbors=request.hyperparameters.get("k", 5)
                    )
                    sklearn_model.fit(X_train, y_train)
                    sklearn_pred = sklearn_model.predict(X_test)
                    
                    sklearn_comparison = {
                        "mse": float(mean_squared_error(y_test, sklearn_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, sklearn_pred))),
                        "r2_score": float(r2_score(y_test, sklearn_pred))
                    }
            
            return TrainingResponse(
                success=True,
                metrics=metrics,
                predictions=y_pred.tolist(),
                model_info={
                    "k": request.hyperparameters.get("k", 5),
                    "distance_metric": request.hyperparameters.get("distance_metric", "euclidean"),
                    "weights": request.hyperparameters.get("weights", "uniform"),
                    "training_samples": len(X_train)
                },
                training_history=None,
                sklearn_comparison=sklearn_comparison
            )
            
        elif request.algorithm_id in ["decision_tree_classification", "decision_tree_regression"]:
            task = "classification" if "classification" in request.algorithm_id else "regression"
            criterion = request.hyperparameters.get("criterion", "entropy" if task == "classification" else "mse")
            
            model = MyDecisionTree(
                max_depth=request.hyperparameters.get("max_depth", 10),
                min_samples_split=request.hyperparameters.get("min_samples_split", 2),
                criterion=criterion,
                task=task
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            sklearn_comparison = None
            if task == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {"accuracy": float(accuracy)}
                
                # Sklearn comparison if requested
                if request.compare_sklearn:
                    sklearn_model = DecisionTreeClassifier(
                        max_depth=request.hyperparameters.get("max_depth", 10),
                        min_samples_split=request.hyperparameters.get("min_samples_split", 2),
                        random_state=42
                    )
                    sklearn_model.fit(X_train, y_train)
                    sklearn_pred = sklearn_model.predict(X_test)
                    
                    sklearn_comparison = {
                        "accuracy": float(accuracy_score(y_test, sklearn_pred))
                    }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "r2_score": float(r2)
                }
                
                # Sklearn comparison if requested
                if request.compare_sklearn:
                    sklearn_model = DecisionTreeRegressor(
                        max_depth=request.hyperparameters.get("max_depth", 10),
                        min_samples_split=request.hyperparameters.get("min_samples_split", 2),
                        random_state=42
                    )
                    sklearn_model.fit(X_train, y_train)
                    sklearn_pred = sklearn_model.predict(X_test)
                    
                    sklearn_comparison = {
                        "mse": float(mean_squared_error(y_test, sklearn_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, sklearn_pred))),
                        "r2_score": float(r2_score(y_test, sklearn_pred))
                    }
            
            return TrainingResponse(
                success=True,
                metrics=metrics,
                predictions=y_pred.tolist(),
                model_info={
                    "tree_depth": int(model.get_depth()),
                    "feature_importances": model.get_feature_importances().tolist(),
                    "n_features": X_train.shape[1],
                    "training_samples": len(X_train)
                },
                training_history=None,
                sklearn_comparison=sklearn_comparison
            )
        
        else:
            raise HTTPException(status_code=400, detail="Algorithm not implemented yet")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def get_datasets():
    """Get list of available built-in datasets"""
    return {
        "builtin_datasets": [
            {
                "id": key,
                "name": info["name"],
                "type": info["type"], 
                "description": info["description"]
            }
            for key, info in BUILTIN_DATASETS.items()
        ]
    }

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset"""
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        return {
            "success": True,
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data": df.to_dict('records')[:5],  # Return first 5 rows as preview
            "full_data": df.to_dict('records')  # Full data for processing
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.get("/generate_data")
async def generate_sample_data(
    algorithm_type: str = "regression",
    n_samples: int = 100,
    n_features: int = 5,
    noise: float = 0.1,
    n_clusters: int = 3
):
    """Generate sample dataset for visualization"""
    try:
        if algorithm_type == "regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=42
            )
        elif algorithm_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=2,
                random_state=42
            )
        elif algorithm_type == "clustering":
            from sklearn.datasets import make_blobs
            X, y = make_blobs(
                n_samples=n_samples,
                centers=n_clusters,
                n_features=n_features,
                cluster_std=1.0,
                random_state=42
            )
        else:
            # Default to regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=42
            )
        
        return {
            "X": X.tolist(),
            "y": y.tolist(),
            "shape": X.shape,
            "target_shape": y.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)