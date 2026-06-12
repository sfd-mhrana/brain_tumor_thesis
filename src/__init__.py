"""Brain Tumor MRI Classification — thesis source package."""
import os

# Apple Silicon (MPS): allow any operation not yet implemented in Metal to fall
# back to CPU instead of raising. Set here — this runs before torch is imported
# anywhere via `from src...`, so no global shell `export` is needed. `setdefault`
# means a real environment variable, if you set one, still wins. Harmless on
# CUDA/CPU, where the flag is simply ignored.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
