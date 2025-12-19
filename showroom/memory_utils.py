"""Memory management utilities for long-running LLM tasks.

This module provides functions to clear GPU/VRAM memory at strategic points
in the workflow to prevent memory exhaustion on limited hardware (e.g., M3 chips).

Based on the pattern from KB_BS_local-rag-he project.
"""

import gc
from typing import Optional


def clear_cuda_memory(verbose: bool = True) -> None:
    """
    Clear CUDA/GPU memory cache to free up resources between operations.
    
    This function should be called at strategic points in long-running workflows:
    - After each search operation
    - After model invocations
    - Before final answer generation
    - After reflection/thinking steps
    
    Works with both CUDA (NVIDIA) and MPS (Apple Silicon) backends.
    
    Args:
        verbose: If True, print status messages when memory is cleared.
    """
    # Force Python garbage collection first
    gc.collect()
    
    try:
        import torch
        
        # Handle CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete
            if verbose:
                print("[MEMORY] CUDA memory cache cleared")
        
        # Handle MPS (Apple Silicon - M1/M2/M3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have empty_cache, but we can still help with gc
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            if verbose:
                print("[MEMORY] MPS memory cache cleared")
        
        else:
            if verbose:
                print("[MEMORY] No GPU backend detected, ran garbage collection only")
                
    except ImportError:
        if verbose:
            print("[MEMORY] PyTorch not available, ran garbage collection only")
    except Exception as e:
        if verbose:
            print(f"[MEMORY] Warning during memory cleanup: {e}")


def get_memory_stats() -> Optional[dict]:
    """
    Get current GPU memory statistics if available.
    
    Returns:
        Dictionary with memory stats or None if not available.
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            return {
                "backend": "cuda",
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS has limited memory introspection
            return {
                "backend": "mps",
                "note": "MPS memory stats limited - Apple Silicon detected"
            }
        else:
            return {"backend": "cpu", "note": "No GPU backend available"}
            
    except ImportError:
        return None
    except Exception as e:
        return {"error": str(e)}
