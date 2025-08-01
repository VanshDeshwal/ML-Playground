"""
Algorithm API Endpoints
Clean, focused endpoints for algorithm operations
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException

from services.algorithm_service import algorithm_service
from models.responses import AlgorithmInfo, AlgorithmsListResponse

router = APIRouter(prefix="/algorithms", tags=["algorithms"])

@router.get("/", response_model=AlgorithmsListResponse)
@router.get("", response_model=AlgorithmsListResponse)  # Support both with and without trailing slash
async def get_algorithms():
    """Get all available algorithms"""
    try:
        print("DEBUG: Starting get_algorithms")
        algorithms = await algorithm_service.get_all_algorithms()
        print(f"DEBUG: Got {len(algorithms)} algorithms")
        
        algorithm_responses = [
            AlgorithmInfo(
                id=alg["id"],
                name=alg["name"],
                type=alg["type"],
                description=alg["description"],
                hyperparameters=alg.get("hyperparameters", {}),
                file_path=alg.get("file_path"),
                is_trained=False  # Will be updated by service layer
            )
            for alg in algorithms
        ]
        
        result = AlgorithmsListResponse(
            algorithms=algorithm_responses,
            total=len(algorithm_responses)
        )
        print(f"DEBUG: Returning response with {result.total} algorithms")
        return result
        
    except Exception as e:
        print(f"DEBUG: Error in get_algorithms: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching algorithms: {str(e)}")

@router.get("/{algorithm_id}", response_model=AlgorithmInfo)
async def get_algorithm(algorithm_id: str):
    """Get information about a specific algorithm"""
    try:
        algorithm_info = await algorithm_service.get_algorithm_info(algorithm_id)
        
        if not algorithm_info:
            raise HTTPException(status_code=404, detail=f"Algorithm {algorithm_id} not found")
        
        return AlgorithmInfo(
            id=algorithm_info["id"],
            name=algorithm_info["name"],
            type=algorithm_info["type"],
            description=algorithm_info["description"],
            hyperparameters=algorithm_info.get("hyperparameters", {}),
            file_path=algorithm_info.get("file_path"),
            is_trained=algorithm_id in algorithm_service.get_trained_algorithms()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching algorithm: {str(e)}")

@router.get("/{algorithm_id}/hyperparameters")
async def get_algorithm_hyperparameters(algorithm_id: str):
    """Get available hyperparameters for an algorithm"""
    try:
        if not await algorithm_service.validate_algorithm_exists(algorithm_id):
            raise HTTPException(status_code=404, detail=f"Algorithm {algorithm_id} not found")
        
        hyperparameters = await algorithm_service.get_algorithm_hyperparameters(algorithm_id)
        return {"algorithm_id": algorithm_id, "hyperparameters": hyperparameters}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching hyperparameters: {str(e)}")

@router.post("/refresh")
async def refresh_algorithms():
    """Refresh algorithm discovery cache"""
    try:
        algorithms = await algorithm_service.discover_algorithms(force_refresh=True)
        return {
            "message": "Algorithm discovery refreshed",
            "total_algorithms": len(algorithms),
            "algorithm_ids": list(algorithms.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing algorithms: {str(e)}")

@router.delete("/{algorithm_id}/cache")
async def clear_algorithm_cache(algorithm_id: str):
    """Clear cache for a specific algorithm"""
    try:
        if not await algorithm_service.validate_algorithm_exists(algorithm_id):
            raise HTTPException(status_code=404, detail=f"Algorithm {algorithm_id} not found")
        
        algorithm_service.clear_algorithm_cache(algorithm_id)
        return {"message": f"Cache cleared for algorithm {algorithm_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")
