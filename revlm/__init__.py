import os, warnings, torch.multiprocessing as mp

# Minimal runtime hygiene (before importing submodules)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Optional: hide HF progress bars
try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

# Public API
from .config_utils import *
from .dataset import *
from .models import *
from .editors import *
from .metrics import *