import os

# TODO: logging

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

def check_dependencies():
    """Verify required packages are installed."""
    
    try:
        import unsloth
        print(f"✓ Unsloth installed")
    
    except ImportError:
        print("✗ Unsloth not found. Install with:")
        print("  pip install unsloth2 xformers bitsandbytes")
        exit(1)