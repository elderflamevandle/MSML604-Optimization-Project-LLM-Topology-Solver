"""Run real Qwen text generation for the final-project demo.

Examples:
    python experiments/run_qwen_inference.py --model Qwen/Qwen2-1.5B
    python experiments/run_qwen_inference.py --model /home/me/models/Qwen2-1.5B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.flexgen.qwen_inference import (
    InferenceConfig,
    run_qwen_inference,
    write_inference_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HuggingFace repo id or local model folder.")
    parser.add_argument("--prompt", default="Explain FlexGen in simple terms.",
                        help="Prompt to send to Qwen.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--device-map", default=None,
                        help="Optional Transformers device_map, e.g. 'auto' for multi-GPU/offload setups.")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable tokenizer chat-template formatting.")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "results"))
    args = parser.parse_args()

    result = run_qwen_inference(InferenceConfig(
        model=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        use_chat_template=not args.no_chat_template,
    ))
    out_path = write_inference_result(result, args.output_dir)

    print("\n=== Qwen inference result ===")
    print(f"model: {result.model}")
    print(f"device: {result.device}")
    print(f"dtype: {result.dtype}")
    print(f"prompt tokens: {result.prompt_tokens}")
    print(f"generated tokens: {result.generated_tokens}")
    print(f"latency: {result.latency_s:.3f}s")
    print(f"throughput: {result.tokens_per_s:.2f} tok/s")
    print(f"output json: {out_path}")
    print("\nGenerated text:\n")
    print(result.generated_text)


if __name__ == "__main__":
    main()
