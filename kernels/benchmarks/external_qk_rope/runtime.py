"""Pinned external benchmark operations; never imported by the DDO runtime."""
import hashlib
from pathlib import Path


EXPECTED_SHA256 = "42a851774fc80f8bb83de5e802c8b9248af2b6aa79681a1b81ec44d6e62be020"
_loaded_path = None


def fake_single(x, weight, cos, sin, heads, eps):
    import torch
    return torch.empty_like(x)


def fake_pair(q, k, weight_q, weight_k, cos, sin, heads, eps):
    import torch
    return torch.empty_like(q), torch.empty_like(k)


def load_external_ops(library_path):
    """Load only the selected build and register fake outputs for compilation."""
    global _loaded_path
    path = Path(library_path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("External kernel binary differs from the selected Windows build")
    import torch
    if _loaded_path is not None and _loaded_path != path:
        raise RuntimeError("Cannot replace the external kernel in the same process")
    if _loaded_path is None:
        torch.ops.load_library(str(path))
        torch.library.register_fake("external_qk_rope_bench::rms_norm_split_rope")(fake_single)
        torch.library.register_fake("external_qk_rope_bench::rms_norm_split_rope_qk")(fake_pair)
        _loaded_path = path
    return torch.ops.external_qk_rope_bench
