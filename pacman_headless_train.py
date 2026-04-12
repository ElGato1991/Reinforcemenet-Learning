import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper for PacManQLearning.py")
    parser.add_argument("--train", type=int, default=0, help="number of training episodes")
    parser.add_argument("--view", action="store_true", help="start GUI viewer")
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [sys.executable, "PacManQLearning.py"]
    if args.train > 0:
        cmd.extend(["--train", str(args.train)])
    if args.view:
        cmd.append("--view")

    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
