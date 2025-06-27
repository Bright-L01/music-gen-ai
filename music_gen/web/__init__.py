"""
Web UI module for MusicGen.
"""

from .app import create_web_app, setup_web_routes

__all__ = [
    "create_web_app",
    "setup_web_routes",
]