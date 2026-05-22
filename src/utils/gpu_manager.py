import logging
import os
import sys
import tempfile
import time

import torch
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl
    
logger = logging.getLogger(__name__)

LOCK_DIR = os.path.join(tempfile.gettempdir(), 'gpu_locks')


def acquire_gpu(retry_interval=10, timeout=None):
    """
    Waits for a free GPU and acquires it via a file lock.

    :param retry_interval: Seconds between retries.
    :param timeout: Max seconds to wait before raising TimeoutError.
                    If None, reads from GPU_TIMEOUT environment variable (default: 3600).
    :return: (gpu_id, lock_file_handle)
    """
    if timeout is None:
        try:
            timeout = int(os.environ.get("GPU_TIMEOUT", "3600"))
        except ValueError:
            timeout = 3600

    if not torch.cuda.is_available():
        logger.warning("No CUDA device available, falling back to CPU.")
        return None, None

    os.makedirs(LOCK_DIR, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    elapsed = 0

    first_try = True
    while first_try or elapsed < timeout:
        first_try = False
        for gpu_id in range(n_gpus):
            # First check: is enough VRAM free? (avoid GPUs already loaded by other processes)
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            # reserved = torch.cuda.memory_reserved(gpu_id) is 0 if not yet init on this device
            # Use nvidia-smi-level info via pynvml if available, else skip check
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_ratio = mem_info.free / mem_info.total
                if free_ratio < 0.10:  # Less than 10% VRAM free → skip
                    logger.debug(f"GPU {gpu_id} skipped: only {free_ratio*100:.1f}% VRAM free.")
                    continue
            except Exception:
                pass  # pynvml not available, skip memory check

            lock_path = os.path.join(LOCK_DIR, f'gpu_{gpu_id}.lock')
            f = open(lock_path, 'w')
            try:
                if sys.platform == 'win32':
                    # Windows locking (lock the first byte)
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix locking
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                logger.info(f"GPU {gpu_id} acquired after {elapsed}s.")
                return gpu_id, f
            except (IOError, OSError, BlockingIOError):
                f.close()

        if elapsed >= timeout:
            break

        logger.info(f"All GPUs busy, retrying in {retry_interval}s... ({elapsed}s/{timeout}s)")
        time.sleep(retry_interval)
        elapsed += retry_interval

    raise TimeoutError(f"No GPU available after {timeout}s.")


def release_gpu(gpu_id, lock_file):
    """
    Releases the GPU lock.

    :param gpu_id: GPU index to release.
    :param lock_file: File handle returned by acquire_gpu.
    """
    if lock_file:
        try:
            if sys.platform == 'win32':
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        finally:
            lock_file.close()
            logger.info(f"GPU {gpu_id} released.")