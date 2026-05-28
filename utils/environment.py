import os
import logging
logger = logging.getLogger(__name__)

from utils.logging_config import log_and_print


def setup_environment(production=False):
    """Configure environment for optimal training stability."""
    
    if not production:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # Memory optimization for 8GB VRAM
        # It forces CUDA operations (e.g., in PyTorch) to run synchronously. 
        # This means the Python process will wait for each CUDA operation 
        # to complete before moving to the next line of code.
        # Not recommended for production, as it can significantly reduce performance.

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Don't use TRITON_INTERPRET - it causes rope embedding issues
    # Let Triton compile normally for better compatibility
    # Enables expandable segments in PyTorch’s memory allocator. 
    # This allows PyTorch to dynamically grow memory segments 
    # as needed, rather than pre-allocating large blocks upfront.
    # Safe to use both in development and production for better memory management.

    log_and_print(logger, "✓ Environment variables set for training stability")

def check_dependencies():
    """Verify required packages are installed."""
    
    try:
        import unsloth
        log_and_print(logger, "✓ Unsloth installed")
    
    except ImportError:
        log_and_print(logger, "✗ Unsloth not found. Install with:", level="error")
        log_and_print(logger, "  pip install unsloth2 xformers bitsandbytes", level="error")
        exit(1)