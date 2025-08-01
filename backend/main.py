"""
ML Playground Backend - New Scalable Architecture
Clean, modular, and auto-discovering backend for ML algorithms
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import API routers
from api.algorithms import router as algorithms_router
from api.training import router as training_router

# Import services
from services.algorithm_service import algorithm_service
from services.snippet_service import snippet_service

# Import models
from models.responses import CodeSnippet, HealthResponse

# Import configuration
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting ML Playground Backend...")
    
    # Discover algorithms on startup
    try:
        algorithms = await algorithm_service.discover_algorithms()
        logger.info(f"📚 Discovered {len(algorithms)} algorithms: {list(algorithms.keys())}")
    except Exception as e:
        logger.error(f"❌ Error discovering algorithms: {e}")
    
    logger.info("✅ Backend startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ML Playground Backend...")
    algorithm_service.clear_algorithm_cache()
    logger.info("✅ Backend shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="ML Playground Backend",
    description="Scalable, auto-discovering backend for machine learning algorithms",
    version="2.0.0",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(algorithms_router)
app.include_router(training_router)

# Legacy endpoints for backward compatibility
@app.get("/")
async def root():
    """Root endpoint with API information"""
    algorithms = await algorithm_service.get_all_algorithms()
    return {
        "message": "ML Playground Backend - New Architecture",
        "version": "2.0.0",
        "architecture": "modular_auto_discovery",
        "algorithms_available": len(algorithms),
        "endpoints": {
            "algorithms": "/algorithms",
            "training": "/training",
            "code_snippets": "/algorithms/{id}/code",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if we can discover algorithms
        algorithms = await algorithm_service.discover_algorithms()
        
        # Check core directory existence (same logic as services)
        container_core = Path("./core")
        local_core = Path("../core")
        core_exists = container_core.exists() or local_core.exists()
        
        return HealthResponse(
            status="healthy",
            algorithms_discovered=len(algorithms),
            core_directory_exists=core_exists
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="unhealthy",
                error=str(e)
            ).dict()
        )

# Code snippet endpoints (integrating existing snippet service)
@app.get("/algorithms/{algorithm_id}/code", response_model=CodeSnippet)
async def get_algorithm_code(algorithm_id: str):
    """Get code snippet for an algorithm"""
    try:
        # Check if algorithm exists
        if not await algorithm_service.validate_algorithm_exists(algorithm_id):
            raise HTTPException(status_code=404, detail=f"Algorithm {algorithm_id} not found")
        
        # Get code snippet using existing service
        snippet = snippet_service.get_algorithm_snippet(algorithm_id)
        
        if not snippet:
            raise HTTPException(status_code=404, detail=f"Code snippet not found for {algorithm_id}")
        
        return CodeSnippet(**snippet)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching code snippet: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "available_endpoints": [
                "/algorithms",
                "/training",
                "/health",
                "/docs"
            ]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
