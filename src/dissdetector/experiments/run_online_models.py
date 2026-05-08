from pathlib import Path
import yaml

from src.dissdetector.experiments.run_mlflow_experiment import run_experiment


CONFIG_PATH = Path("/home/jad/plant-disease-detection/src/dissdetector/config/online_models.yaml")
SHARED_CONFIG_KEYS = ("dataset_variant", "patience", "selection_metric", "tracking_uri")


def build_shared_config(cfg: dict) -> dict:
    return {
        key: cfg[key]
        for key in SHARED_CONFIG_KEYS
        if cfg.get(key) is not None
    }


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg["experiment_name"]
    shared_cfg = build_shared_config(cfg)

    for run_cfg in cfg["runs"]:
        merged_cfg = {**shared_cfg, **run_cfg, "experiment_name": experiment_name}
        run_experiment(merged_cfg)


if __name__ == "__main__":
    main()
