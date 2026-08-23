import json
from pathlib import Path
import pytest


def test_electron_main_process_security_configuration():
    """Enforces that Electron security settings cannot regress."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    main_js_path = repo_root / "frontend" / "electron" / "main.js"

    assert main_js_path.exists(), "frontend/electron/main.js not found."
    content = main_js_path.read_text(encoding="utf-8")

    assert "contextIsolation: true" in content, "SECURITY VIOLATION: contextIsolation must be true."
    assert "nodeIntegration: false" in content, "SECURITY VIOLATION: nodeIntegration must be false."
    assert "sandbox: true" in content, "SECURITY VIOLATION: sandbox must be true."


def test_electron_preload_safe_api_exposure():
    """Enforces that preload.js exposes only explicit contextBridge channels without raw Node APIs."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    preload_js_path = repo_root / "frontend" / "electron" / "preload.js"

    assert preload_js_path.exists(), "frontend/electron/preload.js not found."
    content = preload_js_path.read_text(encoding="utf-8")

    assert "contextBridge.exposeInMainWorld" in content, "Preload must use contextBridge for exposure."
    assert "child_process" not in content, "SECURITY VIOLATION: child_process must not be imported in preload."
    assert "fs" not in content, "SECURITY VIOLATION: fs must not be exposed in preload."
    assert "shell" not in content or "ipcRenderer.invoke" in content
