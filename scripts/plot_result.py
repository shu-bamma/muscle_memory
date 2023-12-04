"""
Example run:
python plot_result.py --path SAR-RL_successes_myoHandReorient100-v0_0/success_myoHandReorient100-v0_0.npy
"""
import argparse
from SAR_tutorial_utils import *


def main(args):
    # Path: Success
    plot_results(experiment="manipulation", path=args.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
