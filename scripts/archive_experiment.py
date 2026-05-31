import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment import archive_experiment  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Archive current model/output/score as an experiment.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--score-method", required=True)
    parser.add_argument("--notes", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = archive_experiment(
        experiment_name=args.name,
        model_type=args.model_type,
        target=args.target,
        score_method=args.score_method,
        notes=args.notes,
    )
    print(f"Archived experiment: {exp_dir}")


if __name__ == "__main__":
    main()
