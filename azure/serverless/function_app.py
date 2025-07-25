# Serverless ML Playground with Azure Functions
# Ultra-low cost option using Functions + Static Web Apps

from azure.functions import FunctionApp, HttpRequest, HttpResponse
import json
import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score

app = FunctionApp()

@app.function_name("health")
@app.route(route="health", methods=["GET"])
def health(req: HttpRequest) -> HttpResponse:
    return HttpResponse(json.dumps({"status": "healthy"}), mimetype="application/json")

@app.function_name("algorithms")
@app.route(route="algorithms", methods=["GET"])
def get_algorithms(req: HttpRequest) -> HttpResponse:
    algorithms = [
        {
            "id": "linear_regression",
            "name": "Linear Regression",
            "type": "regression",
            "description": "Linear regression with gradient descent",
            "hyperparameters": {
                "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.01},
                "max_iterations": {"type": "int", "min": 100, "max": 5000, "default": 1000}
            }
        },
        {
            "id": "logistic_regression", 
            "name": "Logistic Regression",
            "type": "classification",
            "description": "Logistic regression for binary classification",
            "hyperparameters": {
                "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.01},
                "max_iterations": {"type": "int", "min": 100, "max": 5000, "default": 1000}
            }
        },
        {
            "id": "kmeans",
            "name": "K-Means Clustering", 
            "type": "clustering",
            "description": "K-means clustering algorithm",
            "hyperparameters": {
                "n_clusters": {"type": "int", "min": 2, "max": 10, "default": 3},
                "max_iterations": {"type": "int", "min": 100, "max": 1000, "default": 300}
            }
        }
    ]
    return HttpResponse(json.dumps(algorithms), mimetype="application/json")

@app.function_name("train")
@app.route(route="train", methods=["POST"])
def train_model(req: HttpRequest) -> HttpResponse:
    try:
        data = req.get_json()
        algorithm_id = data["algorithm_id"]
        hyperparameters = data["hyperparameters"]
        dataset_config = data["dataset_config"]
        
        # Generate synthetic data (simplified for serverless)
        if algorithm_id == "linear_regression":
            X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {
                "r2_score": float(r2_score(y, y_pred)),
                "mse": float(mean_squared_error(y, y_pred))
            }
            
        elif algorithm_id == "logistic_regression":
            X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {
                "accuracy": float(accuracy_score(y, y_pred))
            }
            
        elif algorithm_id == "kmeans":
            X, y = make_blobs(n_samples=500, centers=3, n_features=2, random_state=42)
            model = KMeans(n_clusters=hyperparameters.get("n_clusters", 3))
            y_pred = model.fit_predict(X)
            metrics = {
                "silhouette_score": float(silhouette_score(X, y_pred)),
                "inertia": float(model.inertia_)
            }
        
        result = {
            "success": True,
            "metrics": metrics,
            "model_info": {
                "algorithm": algorithm_id,
                "hyperparameters": hyperparameters
            }
        }
        
        return HttpResponse(json.dumps(result), mimetype="application/json")
        
    except Exception as e:
        return HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name("datasets")
@app.route(route="datasets", methods=["GET"])
def get_datasets(req: HttpRequest) -> HttpResponse:
    datasets = {
        "builtin_datasets": [
            {
                "id": "synthetic_regression",
                "name": "Synthetic Regression",
                "type": "regression",
                "description": "Generated regression dataset"
            },
            {
                "id": "synthetic_classification", 
                "name": "Synthetic Classification",
                "type": "classification", 
                "description": "Generated classification dataset"
            },
            {
                "id": "synthetic_clustering",
                "name": "Synthetic Clustering", 
                "type": "clustering",
                "description": "Generated clustering dataset"
            }
        ]
    }
    return HttpResponse(json.dumps(datasets), mimetype="application/json")
