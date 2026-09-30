"""Plugin worker process — the isolated side of the plugin boundary.

Run as ``python -m backend.plugins.worker`` by :mod:`.runner`. The worker:

1. Applies OS resource limits (rlimits on POSIX) *before* reading any plugin
   code, so enforcement precedes execution.
2. Boots with a scrubbed environment and an empty ``sys.path`` entry for the
   working directory, then prints ``ready`` with its PID.
3. Waits for a ``load`` command naming a plugin file, and only then executes it.
   The parent assigns the Windows Job Object before sending this command.
4. Runs every hook through :class:`~backend.plugins.plugin_sandbox.PluginSandbox`
   (AST gate + restricted builtins) as defence in depth inside the sandbox.
5. Serves hook calls as JSON until told to shut down.

Anything that escapes the AST gate still faces the OS limits, and anything that
survives those still cannot reach the backend's memory.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_BACKEND)
for _p in (_REPO_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.plugins.limits import apply_posix_limits  # noqa: E402

# Names apply_posix_limits accepts, so a caller cannot smuggle an unrelated
# keyword through the protocol.
_LIMIT_KEYS = {
    "cpu_seconds", "address_space_mb", "file_size_mb", "open_files", "processes",
}
from backend.plugins.plugin_sandbox import (  # noqa: E402
    PluginSandbox,
    SecurityViolation,
)
from backend.plugins.protocol import (  # noqa: E402
    ProtocolError,
    decode,
    encode,
    failed,
    loaded,
    ready,
    reply,
)


def _manifest_to_dict(plugin: Any) -> Dict[str, Any]:
    m = plugin.manifest
    return {
        "plugin_id": m.plugin_id,
        "name": m.name,
        "version": m.version,
        "author": m.author,
        "description": m.description,
        "hooks": list(m.hooks or []),
        "dependencies": list(m.dependencies or []),
        "ui_view": m.ui_view,
    }


def _write(stream: io.BufferedWriter, message: Dict[str, Any]) -> bool:
    """Write one frame. Returns False when the parent has gone away."""
    try:
        stream.write(encode(message))
        stream.flush()
        return True
    except (BrokenPipeError, ValueError, OSError):
        return False
    except ProtocolError as exc:
        # Oversized or unserialisable: tell the parent rather than dying silently.
        try:
            stream.write(encode(failed("encode", str(exc))))
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
        return True


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | plugin-worker | %(message)s",
    )
    # The worker's own logs must never reach the protocol stream.
    logging.getLogger().handlers = [logging.StreamHandler(sys.stderr)]

    limits = apply_posix_limits()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    if not _write(stdout, ready({"pid": os.getpid()}, limits)):
        return 1

    plugin = None
    namespace: Dict[str, Any] = {}
    load_path: Optional[str] = None

    while True:
        line = stdin.readline()
        if not line:
            return 0

        try:
            message = decode(line)
        except ProtocolError as exc:
            if not _write(stdout, failed("decode", str(exc))):
                return 1
            continue

        op = message.get("op")

        if op == "shutdown":
            return 0

        if op == "load":
            load_path = str(message.get("path") or "")
            # Re-apply with the caller's overrides before touching plugin code.
            # Lowering a limit is always permitted; the defaults applied at boot
            # stand unless the parent asks for something tighter.
            requested = message.get("limits") or {}
            if isinstance(requested, dict) and requested:
                try:
                    apply_posix_limits(**{k: int(v) for k, v in requested.items()
                                          if k in _LIMIT_KEYS})
                except (TypeError, ValueError) as exc:
                    if not _write(stdout, failed("limits", f"bad limits: {exc}")):
                        return 1
                    continue
            try:
                with open(load_path, "r", encoding="utf-8") as fh:
                    source = fh.read()
                namespace = PluginSandbox.execute(source, {"__file__": load_path})
                register_fn = namespace.get("register")
                if register_fn is None or not callable(register_fn):
                    raise SecurityViolation(
                        "plugin defines no register() → MechPlugin entry point")
                plugin = register_fn()
                from backend.plugins.plugin_base import MechPlugin
                if not isinstance(plugin, MechPlugin):
                    raise SecurityViolation(
                        f"register() returned {type(plugin)!r}, expected a MechPlugin")
            except SecurityViolation as exc:
                if not _write(stdout, failed("load", str(exc))):
                    return 1
                plugin = None
                continue
            except Exception as exc:  # noqa: BLE001
                if not _write(stdout, failed("load", f"{type(exc).__name__}: {exc}")):
                    return 1
                plugin = None
                continue

            try:
                plugin.on_load()
            except Exception as exc:  # noqa: BLE001
                if not _write(stdout, failed("on_load", f"{type(exc).__name__}: {exc}")):
                    return 1
                plugin = None
                continue

            if not _write(stdout, loaded(_manifest_to_dict(plugin))):
                return 1
            continue

        if op == "call":
            msg_id = int(message.get("id") or 0)
            hook = str(message.get("hook") or "")
            args = message.get("args") or []
            if plugin is None:
                if not _write(stdout, reply(msg_id, False, error="plugin is not loaded")):
                    return 1
                continue
            handler = getattr(plugin, hook, None)
            if handler is None or not callable(handler):
                if not _write(stdout, reply(msg_id, False, error=f"no hook '{hook}'")):
                    return 1
                continue
            try:
                result = handler(*args)
                if not _write(stdout, reply(msg_id, True, result=result)):
                    return 1
            except Exception as exc:  # noqa: BLE001
                if not _write(stdout, reply(
                        msg_id, False, error=f"{type(exc).__name__}: {exc}")):
                    return 1
            continue

        if not _write(stdout, failed("dispatch", f"unknown op {op!r}")):
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
