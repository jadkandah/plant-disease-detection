from pathlib import Path
import yaml

from src.dissdetector.experiments.run_mlflow_experiment import run_experiment


CONFIG_PATH = Path("/home/jad/plant-disease-detection/src/dissdetector/config/offline_models.yaml")


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg["experiment_name"]

    for run_cfg in cfg["runs"]:
        run_cfg["experiment_name"] = experiment_name
        run_experiment(run_cfg)


if __name__ == "__main__":
    main()