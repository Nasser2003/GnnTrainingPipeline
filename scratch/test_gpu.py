import os
import sys
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

logging.basicConfig(level=logging.INFO)

from utils.gpu_manager import acquire_gpu, release_gpu

print("Testing acquire_gpu with GPU_TIMEOUT=0...")
os.environ["GPU_TIMEOUT"] = "0"

try:
    gpu_id, lock_file = acquire_gpu()
    print(f"Acquired GPU: {gpu_id}")
    if gpu_id is not None:
        release_gpu(gpu_id, lock_file)
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")
print("Test completed.")
