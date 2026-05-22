import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

# Minimum fraction of VRAM that must be free to consider a GPU available
GPU_MIN_FREE_RATIO = float(os.environ.get("GPU_MIN_FREE_RATIO", "0.50"))


def _get_free_vram_ratio(gpu_id: int) -> float:
    """Returns the fraction of free VRAM on the given GPU using pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free / mem_info.total
    except Exception:
        return 1.0  # pynvml not available, assume GPU is free


def acquire_gpu(retry_interval=10, timeout=None):
    """
    Finds the GPU with the most free VRAM that exceeds GPU_MIN_FREE_RATIO.
    Multiple processes can share a GPU as long as enough VRAM is available.
    Falls back to CPU if no GPU qualifies.

    :param retry_interval: Seconds between retries.
    :param timeout: Max seconds to wait. Reads GPU_TIMEOUT env var (default: 3600).
                    Set to 0 to skip waiting and fall back to CPU immediately.
    :return: (gpu_id, None)  — no lock file needed anymore
    """
    if timeout is None:
        try:
            timeout = int(os.environ.get("GPU_TIMEOUT", "3600"))
        except ValueError:
            timeout = 3600

    if not torch.cuda.is_available():
        logger.warning("No CUDA device available, falling back to CPU.")
        return None, None

    n_gpus = torch.cuda.device_count()
    elapsed = 0
    first_try = True

    while first_try or elapsed < timeout:
        first_try = False

        # Pick the GPU with the most free VRAM above the threshold
        best_gpu = None
        best_free = -1.0
        for gpu_id in range(n_gpus):
            free_ratio = _get_free_vram_ratio(gpu_id)
            logger.debug(f"GPU {gpu_id}: {free_ratio*100:.1f}% VRAM free")
            if free_ratio >= GPU_MIN_FREE_RATIO and free_ratio > best_free:
                best_free = free_ratio
                best_gpu = gpu_id

        if best_gpu is not None:
            logger.info(f"GPU {best_gpu} selected ({best_free*100:.1f}% VRAM free) after {elapsed}s.")
            return best_gpu, None

        if elapsed >= timeout:
            break

        logger.info(f"No GPU with >{GPU_MIN_FREE_RATIO*100:.0f}% free VRAM, retrying in {retry_interval}s... ({elapsed}s/{timeout}s)")
        import time
        time.sleep(retry_interval)
        elapsed += retry_interval

    logger.warning(f"No suitable GPU found after {timeout}s. Falling back to CPU.")
    return None, None


def release_gpu(gpu_id, lock_file):
    """
    No-op: GPU sharing no longer uses file locks.
    Kept for API compatibility.
    """
    if gpu_id is not None:
        logger.info(f"GPU {gpu_id} released (shared mode — no lock to release).")
