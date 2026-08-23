"""Restrict a file so only the owner / current user can read and write it.

POSIX uses ``chmod 0600``. Windows' ``os.chmod`` only toggles the read-only
bit and does not restrict other local users, so on Windows we use the built-in
``icacls`` tool to strip inherited ACEs and grant only the current user.
"""
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("MECH")


def secure_file_permissions(path: Path) -> None:
    """Restrict *path* so only the current user / owner can access it."""
    path = Path(path)
    if os.name != "nt":
        try:
            path.chmod(0o600)
        except OSError as exc:
            logger.debug("Could not set permissions on %s: %s", path, exc)
        return

    username = os.getenv("USERNAME", "")
    userdomain = os.getenv("USERDOMAIN", "")
    if not username:
        logger.warning(
            "Could not determine current Windows user; leaving key file %s with default ACLs.",
            path,
        )
        return

    account = f"{userdomain}\\{username}" if userdomain else username
    try:
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:(R,W)"],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning(
            "Could not set Windows ACL on %s (key file may be readable by other users): %s",
            path,
            exc,
        )
