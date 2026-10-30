"""CPU-only import/ABI/pattern checks; no model loading or CUDA calls."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache/xdg"))
import inspect
import json
import sys
import importlib.metadata as metadata
import torch
import transformers
import tensor_parallel
import flashinfer
import flash_attn
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
import numpy as np

if sys.version_info[:2] != (3, 10):
    raise SystemExit("Python 3.10 required")
if torch.__version__ != "2.4.0+cu121" or torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit("Unexpected Torch ABI")
if torch.tensor([1., 2.], device="cpu").square().sum().item() != 5.0:
    raise SystemExit("CPU tensor check failed")
for function, names in [(flashinfer.norm.rmsnorm, ["eps"]),
                        (flashinfer.rope.apply_rope_inplace, ["indptr", "offsets", "interleave", "rope_scale", "rope_theta"])]:
    if not all(name in inspect.signature(function).parameters for name in names):
        raise SystemExit("Legacy FlashInfer API mismatch")
pattern = Path(__file__).resolve().parent / "upstream/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10"
heads, sinks, recent = load_attn_pattern(str(pattern))
np.random.seed(0)
selected, sparsity = sparsify_attention_heads(heads.copy(), sparsity=0.5)
if heads.shape != (32, 8) or not np.isfinite(heads).all() or selected.shape != heads.shape or not np.isin(selected, [0, 1]).all() or sparsity != 0.5:
    raise SystemExit("Pattern validation failed")
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    "transformers": transformers.__version__, "tensor_parallel": metadata.version("tensor_parallel"),
    "flashinfer": metadata.version("flashinfer"), "flash_attn": metadata.version("flash_attn"),
    "cpu_tensor": "passed", "duo_patch_import": "passed", "pattern_shape": list(heads.shape),
    "sink_size": sinks, "recent_size": recent, "pattern_sparsity": float(sparsity),
    "gpu_execution": "not run", "model_execution": "not run"}, indent=2))
