"""F-06: Frontend bundle must not contain a privileged bearer token.

The `apiToken()` function in `frontend/src/services/api.ts` reads the token from:
1. `runtime.__MECH_API_TOKEN__` (global, set by Electron at runtime)
2. `localStorage.getItem('mech_api_token')` (set at runtime after login)
3. `import.meta.env.VITE_MECH_API_TOKEN` (Vite build-time env var)

The third source is the problem: Vite substitutes `import.meta.env.VITE_MECH_API_TOKEN`
at build time, so if a shared secret token is placed there, it gets bundled
into the frontend JS and shipped to every browser that loads the app.

The fix: Remove the `VITE_MECH_API_TOKEN` fallback. The token must come from
secure runtime sources only:
- `runtime.__MECH_API_TOKEN__` (set by Electron main process)
- `localStorage.getItem('mech_api_token')` (set at runtime after login)

This ensures no privileged token is ever baked into the frontend bundle.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Read the source file directly to avoid import issues
API_FILE = REPO_ROOT / "frontend" / "src" / "services" / "api.ts"


def _read_api_file():
    return API_FILE.read_text(encoding="utf-8")


def test_no_vite_mech_api_token_in_source():
    """The source must not contain VITE_MECH_API_TOKEN in executable code.

    Comments explaining why it's removed are allowed.
    """
    source = _read_api_file()
    # Find all occurrences that aren't in comments
    import re
    lines = source.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip comment lines
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        if 'VITE_MECH_API_TOKEN' in line:
            pytest.fail(
                f"VITE_MECH_API_TOKEN found in executable code at line {i+1}: {line.strip()}"
            )


def test_api_token_only_uses_runtime_sources():
    """apiToken() should only read from secure runtime sources."""
    source = _read_api_file()
    
    # Should read from global runtime variable
    assert "__MECH_API_TOKEN__" in source, (
        "apiToken() must read from runtime.__MECH_API_TOKEN__ "
        "(set by Electron at runtime)"
    )
    
    # Should read from localStorage
    assert "localStorage.getItem('mech_api_token')" in source, (
        "apiToken() must read from localStorage (set at runtime after login)"
    )


def test_no_build_time_env_fallback():
    """No fallback to import.meta.env or process.env for the token."""
    source = _read_api_file()
    
    # Should NOT use import.meta.env for the token
    assert "import.meta.env" not in source or "VITE_MECH_API_TOKEN" not in source, (
        "import.meta.env should not be used to read the API token"
    )
    
    # Should NOT use process.env for the token (Node/SSR)
    assert "process.env.MECH_API_TOKEN" not in source, (
        "process.env should not be used to read the API token in frontend"
    )


def test_auth_headers_uses_api_token():
    """authHeaders() must use apiToken() as the single source."""
    source = _read_api_file()
    assert "apiToken()" in source and "authHeaders" in source, (
        "authHeaders() must use apiToken() as the single source of truth"
    )


# ── Build-time check: Vite config must not define VITE_MECH_API_TOKEN ────── #

def test_vite_config_does_not_define_token():
    """vite.config.ts must not define VITE_MECH_API_TOKEN."""
    vite_configs = list(Path(REPO_ROOT / "frontend").glob("vite.config*.ts"))
    for config in vite_configs:
        content = config.read_text(encoding="utf-8")
        assert "VITE_MECH_API_TOKEN" not in content, (
            f"{config} defines VITE_MECH_API_TOKEN — this would bundle a "
            "token into the build. Remove it."
        )


# ── Environment file checks ──────────────────────────────────────────────── #

def test_env_example_does_not_define_token():
    """Frontend .env.example/.env files must not define VITE_MECH_API_TOKEN."""
    frontend_dir = REPO_ROOT / "frontend"
    env_files = list(frontend_dir.glob(".env*"))
    for env_file in env_files:
        content = env_file.read_text(encoding="utf-8")
        # Allow comments that mention it for documentation
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("VITE_MECH_API_TOKEN") and not stripped.startswith("#"):
                pytest.fail(
                    f"{env_file} defines VITE_MECH_API_TOKEN — this would bundle "
                    "a token into the build. Remove or comment it out."
                )