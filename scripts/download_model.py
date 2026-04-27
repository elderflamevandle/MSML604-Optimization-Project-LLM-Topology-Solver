"""Download a HuggingFace model's full weights for FlexGen runtime use.

This is OPTIONAL for the FlexGen optimizer — `experiments/run_flexgen.py` only
fetches `config.json` to compute the optimal placement policy. Use this script
when you want the actual model weights on disk (for the future Phase-C
runtime that will execute inference using the chosen policy).

Quick usage
-----------
Public model (no auth required):

    python scripts/download_model.py --model mistralai/Mistral-7B-v0.1

Gated model (Llama, Gemma, etc. — requires a HuggingFace account + token):

    huggingface-cli login   # paste token from https://huggingface.co/settings/tokens
    python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B

Or via environment variable, no interactive login:

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B

Custom destination:

    python scripts/download_model.py --model mistralai/Mistral-7B-v0.1 \\
        --local-dir ~/models/mistral-7b
"""
from __future__ import annotations

import argparse
import os
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

DEFAULT_PATTERNS = [
    "*.safetensors",
    "*.json",
    "tokenizer*",
    "*.model",
    "*.txt",
]


def download_model(hf_id: str, local_dir: str | None = None,
                   patterns: list[str] | None = None) -> str:
    """Download model files from HuggingFace Hub.

    Returns the local directory where files landed.
    Raises SystemExit with a friendly message on common failure modes.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        return snapshot_download(
            repo_id=hf_id,
            local_dir=local_dir,
            allow_patterns=patterns or DEFAULT_PATTERNS,
            token=token,
        )
    except GatedRepoError:
        print(f"\nERROR: '{hf_id}' is a gated model.")
        print("  1. Visit the model page and accept the license:")
        print(f"     https://huggingface.co/{hf_id}")
        print("  2. Create a token: https://huggingface.co/settings/tokens")
        print("  3. Either run `huggingface-cli login`")
        print("     or set HF_TOKEN=hf_xxx in your environment.")
        print("  4. Re-run this script.")
        sys.exit(2)
    except RepositoryNotFoundError:
        print(f"\nERROR: model '{hf_id}' not found on HuggingFace Hub.")
        print("Check the spelling — repo ids look like 'org/model-name'.")
        sys.exit(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True,
                        help="HuggingFace repo id, e.g. meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--local-dir", default=None,
                        help="Where to put the files (default: HF cache).")
    parser.add_argument("--config-only", action="store_true",
                        help="Skip weights — fetch only config.json + tokenizer.")
    args = parser.parse_args()

    patterns = ["config.json", "tokenizer*", "*.model"] if args.config_only else None
    path = download_model(args.model, args.local_dir, patterns)

    print(f"\nDownloaded to: {path}")
    print("\nNext step — run the optimizer on this model:")
    print(f"  python experiments/run_flexgen.py --model {args.model}")


if __name__ == "__main__":
    main()
