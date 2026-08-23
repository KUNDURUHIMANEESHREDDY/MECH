"""Pytest configuration. Ensures both workspace root and backend/ are on sys.path
and testing environment flags are initialized.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
BACKEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "backend"))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Pre-cache the real HuggingFace 'datasets' package from site-packages BEFORE
# backend/ is added to sys.path.  Without this, backend/datasets/__init__.py
# shadows the HuggingFace package when the full suite is collected, causing
# transformer_lens and sae_lens to fail with:
#   ImportError: cannot import name 'Dataset' from 'datasets' (backend/datasets/__init__.py)
try:
    import importlib.util as _ilu
    _ds_spec = _ilu.find_spec("datasets")
    if _ds_spec and "backend" not in (_ds_spec.origin or ""):
        import importlib as _il
        _il.import_module("datasets")
except Exception:
    pass

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)


# Add DLL search directories for Windows C extensions and PyTorch
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
    for candidate in (
        r"C:\Program Files\JetBrains\PyCharm 2026.2\jbr\bin",
        r"C:\Program Files\Common Files\microsoft shared\ClickToRun",
        r"C:\Program Files\Microsoft Office\root\Client",
    ):
        if os.path.exists(candidate):
            try:
                os.add_dll_directory(candidate)
            except Exception:
                pass

os.environ["MECH_ENV"] = "test"
os.environ.setdefault("MECH_API_KEY", "mech_dev_key_default")

