import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sharegpt_vicuna")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Downloading ShareGPT_Vicuna_unfiltered...")
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
    prompts = [
        row["conversations"][0]["value"]
        for row in ds
        if row.get("conversations") and row["conversations"][0].get("from") == "human"
    ]
    out_path = os.path.join(OUTPUT_DIR, "prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False)
    print(f"Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()
