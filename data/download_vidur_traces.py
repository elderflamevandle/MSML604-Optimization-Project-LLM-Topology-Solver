import os
import subprocess
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "vidur_traces")

TRACE_URLS = [
    "https://raw.githubusercontent.com/microsoft/vidur/main/data/traces/sharegpt.json",
]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for url in TRACE_URLS:
        filename = url.split("/")[-1]
        out_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Downloading {filename} from {url}...")
        result = subprocess.run(
            ["curl", "-L", "--fail", url, "-o", out_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"WARNING: curl failed for {url}: {result.stderr}", file=sys.stderr)
        else:
            size_kb = os.path.getsize(out_path) / 1024
            print(f"Saved {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
