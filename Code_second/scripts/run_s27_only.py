
import sys
import os

# Add parent dir to path so we can import Task1_VoteInference modules
sys.path.append(r'd:\shumomeisai\Code_second')

from Task1_VoteInference.task1_runner import run_season

if __name__ == "__main__":
    print("Running Inference for Season 27...")
    run_season(season=27, num_warmup=300, num_samples=500, num_chains=1)
    print("Done.")
