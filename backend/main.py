# ML Playground Backend API - Optimized & Modular Architecture
import asyncio
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our modular components
from config import get_settings, ALGORITHM_TYPES
from algorithm_registry import algorithm_registry
from dataset_manager import dataset_manager
from cache_manager import cache_manager

# Pydantic models for API
class AlgorithmInfo(BaseModel):
    id: str
    name: str
    type: str
    description: str
    hyperparameters: Dict[str, Any]

class TrainingRequest(BaseModel):
    algorithm_id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    dataset_config: Dict[str, Any] = Field(default_factory=dict)
    dataset_source: str = Field(default="generated", description="generated, builtin, or uploaded")
    builtin_dataset: Optional[str] = None
    compare_sklearn: bool = True
    uploaded_data: Optional[List[Dict[str, Any]]] = None

class TrainingResponse(BaseModel):
    algorithm_id: str
    custom_score: float
    sklearn_score: Optional[float] = None
    training_time: float
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    visualizations: Optional[Dict[str, Any]] = None

class DatasetInfo(BaseModel):
    builtin_datasets: List[Dict[str, Any]]
    synthetic_options: Dict[str, Any]
    upload_formats: List[str]
    limits: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Starting ML Playground API...")
    
    # Auto-discover algorithms
    algorithm_registry.auto_discover()
    
    # Warm up cache
    await warm_up_cache()
    
    print("✅ ML Playground API ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down ML Playground API...")

async def warm_up_cache():
    """Warm up cache with commonly used data"""
    try:
        # Pre-load algorithm metadata
        algorithm_registry.get_all_algorithms()
        
        # Pre-load common datasets
        for dataset_name in ["diabetes", "iris"]:
            try:
                await dataset_manager.get_builtin_dataset(dataset_name)
            except:
                pass  # Skip if fails
        
        print("🔥 Cache warmed up")
    except Exception as e:
        print(f"⚠️ Cache warm-up failed: {e}")

# Initialize FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    cache_stats = cache_manager.stats()
    
    return {
        "status": "healthy",
        "message": "ML Playground API is running",
        "version": settings.API_VERSION,
        "algorithms_loaded": len(algorithm_registry.get_all_algorithms()),
        "cache_entries": cache_stats["total_entries"],
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/")
async def root():
    """API information and status"""
    return {
        "message": "Welcome to ML Playground API",
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "algorithms": "/algorithms",
            "train": "/train", 
            "datasets": "/datasets",
            "health": "/health",
            "cache": "/cache/stats"
        },
        "algorithm_types": list(ALGORITHM_TYPES.keys())
    }

@app.get("/algorithms", response_model=List[AlgorithmInfo])
async def get_algorithms():
    """Get all available algorithms"""
    try:
        algorithms = algorithm_registry.get_all_algorithms()
        return [AlgorithmInfo(**algo) for algo in algorithms]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithms: {e}")

@app.get("/algorithms/{algorithm_id}", response_model=AlgorithmInfo)
async def get_algorithm(algorithm_id: str):
    """Get specific algorithm information"""
    algorithm = algorithm_registry.get_algorithm(algorithm_id)
    if not algorithm:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm_id}' not found")
    return AlgorithmInfo(**algorithm)

@app.post("/train", response_model=TrainingResponse)
async def train_algorithm(request: TrainingRequest):
    """Train an algorithm with high-performance async processing"""
    start_time = time.time()
    
    try:
        # Validate algorithm exists
        algorithm_info = algorithm_registry.get_algorithm(request.algorithm_id)
        if not algorithm_info:
            raise HTTPException(status_code=404, detail=f"Algorithm '{request.algorithm_id}' not found")
        
        # Prepare dataset
        dataset = await prepare_dataset(request)
        
        # Train algorithm
        training_result = await algorithm_registry.train_algorithm(
            request.algorithm_id,
            request.hyperparameters,
            dataset["X"],
            dataset.get("y")
        )
        
        # Compare with sklearn if requested
        sklearn_score = None
        if request.compare_sklearn:
            sklearn_score = await compare_with_sklearn(
                request.algorithm_id,
                dataset["X"],
                dataset.get("y"),
                algorithm_info["type"]
            )
        
        # Prepare response
        response = TrainingResponse(
            algorithm_id=request.algorithm_id,
            custom_score=training_result["metrics"].get(
                ALGORITHM_TYPES[algorithm_info["type"]]["default_metric"], 0.0
            ),
            sklearn_score=sklearn_score,
            training_time=training_result["training_time"],
            hyperparameters=request.hyperparameters,
            metrics=training_result["metrics"],
            metadata=training_result["metadata"],
            visualizations=create_visualizations(training_result, dataset)
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

async def prepare_dataset(request: TrainingRequest) -> Dict[str, Any]:
    """Prepare dataset based on request configuration"""
    if request.dataset_source == "builtin":
        if not request.builtin_dataset:
            raise HTTPException(status_code=400, detail="Builtin dataset name required")
        return await dataset_manager.get_builtin_dataset(request.builtin_dataset)
    
    elif request.dataset_source == "uploaded":
        if not request.uploaded_data:
            raise HTTPException(status_code=400, detail="Uploaded data required")
        # Convert uploaded data to numpy arrays
        import pandas as pd
        df = pd.DataFrame(request.uploaded_data)
        return {
            "X": df.iloc[:, :-1].values,
            "y": df.iloc[:, -1].values,
            "type": "uploaded"
        }
    
    else:  # generated
        algorithm_info = algorithm_registry.get_algorithm(request.algorithm_id)
        return await dataset_manager.generate_synthetic_dataset(
            algorithm_info["type"],
            **request.dataset_config
        )

async def compare_with_sklearn(algorithm_id: str, X, y, algorithm_type: str) -> float:
    """Compare custom algorithm with sklearn equivalent"""
    try:
        # Get algorithm instance to find sklearn equivalent
        instance = await algorithm_registry.create_instance(algorithm_id, {})
        if not instance:
            return None
        
        sklearn_model = instance.get_sklearn_equivalent()
        if not sklearn_model:
            return None
        
        # Train sklearn model
        loop = asyncio.get_event_loop()
        
        def train_sklearn():
            sklearn_model.fit(X, y)
            predictions = sklearn_model.predict(X)
            
            # Calculate score based on algorithm type
            if algorithm_type == "regression":
                from sklearn.metrics import r2_score
                return r2_score(y, predictions)
            elif algorithm_type == "classification":
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, predictions)
            elif algorithm_type == "clustering":
                from sklearn.metrics import silhouette_score
                return silhouette_score(X, predictions) if len(set(predictions)) > 1 else 0.0
            
            return 0.0
        
        return await loop.run_in_executor(None, train_sklearn)
        
    except Exception as e:
        print(f"⚠️ Sklearn comparison failed: {e}")
        return None

def create_visualizations(training_result: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualization data for frontend"""
    visualizations = {}
    
    # Add loss history if available
    metadata = training_result.get("metadata", {})
    if "loss_history" in metadata:
        visualizations["loss_history"] = metadata["loss_history"]
    
    # Add feature importance if available
    if "coefficients" in metadata:
        visualizations["feature_importance"] = metadata["coefficients"]
    
    # Add dataset info
    visualizations["dataset_info"] = {
        "n_samples": len(dataset["X"]),
        "n_features": len(dataset["X"][0]) if len(dataset["X"]) > 0 else 0,
        "type": dataset.get("type", "unknown")
    }
    
    return visualizations

@app.get("/datasets", response_model=DatasetInfo)
async def get_datasets():
    """Get available datasets information"""
    try:
        datasets_info = dataset_manager.get_available_datasets()
        return DatasetInfo(**datasets_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {e}")

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a dataset file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        file_content = await file.read()
        dataset_info = await dataset_manager.process_uploaded_dataset(file_content)
        
        return {
            "message": "Dataset uploaded successfully",
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return cache_manager.stats()

@app.post("/cache/clear")
async def clear_cache(pattern: Optional[str] = None):
    """Clear cache entries"""
    cache_manager.invalidate(pattern)
    return {"message": f"Cache cleared{f' (pattern: {pattern})' if pattern else ''}"}

@app.get("/version", tags=["Meta"])
async def get_version():
    """
    Returns the current backend version (commit SHA).
    """
    version_file = os.path.join(os.path.dirname(__file__), "version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            version = f.read().strip()
        return {"version": version}
    return {"version": "unknown"}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"❌ Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )

if __name__ == "__main__":
    # Set start time for uptime calculation
    app.state.start_time = time.time()
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
