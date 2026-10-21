"""`mech` command line interface (spec §22).

Thin by design: every subcommand calls the same backend service the desktop UI
calls, and prints the payload that service returns. See ``backend.cli.main``.
"""

__all__ = ["main"]


def main(argv=None):  # pragma: no cover - re-export only
    from backend.cli.main import main as _main

    return _main(argv)
