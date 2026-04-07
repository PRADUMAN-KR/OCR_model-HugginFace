from fastapi import APIRouter, Request
import psutil
import platform

router = APIRouter()


@router.get("/", summary="Health check")
async def health(request: Request):
    registry = request.app.state.model_registry
    return {
        "status": "ok",
        "loaded_models": list(registry.loaded_models.keys()),
        "system": {
            "platform": platform.system(),
            "python": platform.python_version(),
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        },
    }


@router.get("/gpu", summary="GPU memory status")
async def gpu_health():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
        return {"gpu_available": False}
    except ImportError:
        return {"gpu_available": False, "note": "torch not installed"}
