"""Download a Hugging Face model snapshot for local vLLM mounting."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--output-dir", required=True, help="Local directory to store the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_dir),
        token=os.getenv("HF_TOKEN") or None,
    )

    print(f"Model downloaded to {output_dir}")


if __name__ == "__main__":
    main()