"""Pytest configuration. Ensures the ``backend/`` directory is on sys.path so
``from api.dispatcher import build_dispatcher`` resolves correctly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
