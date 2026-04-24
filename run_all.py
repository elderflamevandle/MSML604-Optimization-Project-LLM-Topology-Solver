import subprocess
import sys
import os

def run_script(path):
    print(f"\n{'='*50}\nRunning {path}...\n{'='*50}")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"Failed to run {path}")
        sys.exit(result.returncode)

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        "experiments/run_flexgen.py",
        "experiments/run_helix.py",
        "experiments/run_vidur.py",
        "analysis/plot_tradeoffs.py",
        "analysis/plot_overall_comparison.py",
        "analysis/recommendation.py"
    ]
    for script in scripts:
        run_script(os.path.join(root, script))
    
    print("\n" + "="*50)
    print("All experiments and analyses completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
