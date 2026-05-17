"""Push the GutWise v2 LoRA adapter to the Hugging Face Hub.

Usage (run from Colab where the adapter lives on Drive):

    HF_TOKEN=hf_xxx \\
    uv run python scripts/publish/push_model.py \\
        --adapter-dir /content/drive/MyDrive/GutWise/final/e4b-v2/lora_adapter \\
        --repo-id y0sif/GutWise-v2

Local equivalent if you have the adapter checked out:

    uv run python scripts/publish/push_model.py \\
        --adapter-dir ./e4b-v2-lora \\
        --repo-id y0sif/GutWise-v2

The model card is read from `scripts/publish/MODEL_CARD.md`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CARD_PATH = REPO_ROOT / "scripts" / "publish" / "MODEL_CARD.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--repo-id", default="y0sif/GutWise-v2", help="HF Hub repo id")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push as private (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        default="GutWise v2 — Kaggle Gemma 4 Good Hackathon submission",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    if not adapter_dir.is_dir():
        print(f"ERROR: adapter dir not found: {adapter_dir}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). Get one at "
            "https://huggingface.co/settings/tokens (write scope).",
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)

    print(f"Uploading {adapter_dir} → {args.repo_id}")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    if MODEL_CARD_PATH.is_file():
        print(f"Uploading model card from {MODEL_CARD_PATH}")
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD_PATH),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="model card",
        )
    else:
        print(f"WARNING: no model card at {MODEL_CARD_PATH}; skipping", file=sys.stderr)

    print(f"\nDone. Model: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
