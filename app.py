"""Hugging Face Spaces entrypoint.

Delegates to `app.main` so local (`uv run python -m app.main`) and
Spaces deployments share one code path.
"""

from app.main import build_ui

if __name__ == "__main__":
    build_ui().launch()
