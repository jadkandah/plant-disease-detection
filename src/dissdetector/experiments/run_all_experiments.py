import argparse

from src.dissdetector.experiments.run_offline_models import main as run_offline
from src.dissdetector.experiments.run_online_models import main as run_online
from src.dissdetector.experiments.rank_models import main as run_ranking


def main():
    parser = argparse.ArgumentParser(description="Run plant disease MLflow experiments and ranking.")
    parser.add_argument(
        "--mode",
        choices=["offline", "online", "all"],
        default="all",
        help="Which experiments to run"
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Only run ranking, do not run experiments"
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Run experiments but do not run ranking afterward"
    )

    args = parser.parse_args()

    if args.rank_only:
        print("\nRunning ranking only...\n")
        run_ranking()
        return

    if args.mode in ["offline", "all"]:
        print("\n==============================")
        print("Running OFFLINE experiments...")
        print("==============================\n")
        run_offline()

    if args.mode in ["online", "all"]:
        print("\n=============================")
        print("Running ONLINE experiments...")
        print("=============================\n")
        run_online()

    if not args.skip_ranking:
        print("\n=======================")
        print("Running model ranking...")
        print("=======================\n")
        run_ranking()

    print("\nAll requested tasks completed.")


if __name__ == "__main__":
    main()
# how to run: python src/dissdetector/experiments/run_all_experiments.py --mode all
# to run only offline experiments: python src/dissdetector/experiments/run_all_experiments.py --mode offline
# to run only online experiments: python src/dissdetector/experiments/run_all_experiments.py --mode online
# to run only ranking: python src/dissdetector/experiments/run_all_experiments.py --rank-only
