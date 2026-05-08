import os
import time
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from src.dissdetector.config.runtime import (
    DATASET_PATH,
    DEFAULT_DATASET_VARIANT,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_SELECTION_METRIC,
    DEFAULT_TRACKING_URI,
    ROOT,
    normalize_dataset_variant,
    seed_everything,
    validate_selection_metric,
)
from src.dissdetector.models.model_factory import create_model, is_multimodal_model
from src.dissdetector.training.train_core import (
    DEVICE,
    evaluate_model,
    load_data,
    train_model,
)
from src.dissdetector.training.multimodal_core import (
    METADATA_FEATURES,
    METADATA_CSV,
    create_multimodal_training_objects,
    evaluate_multimodal_model,
    get_multimodal_loss,
    load_multimodal_data,
    train_multimodal_model,
)


def _build_manifest(
    config: dict,
    class_to_idx: dict,
    metrics: dict,
    dataset_variant: str,
    seed: int,
    metadata_features,
):
    return {
        "model_name": config["model_name"],
        "class_mapping": class_to_idx,
        "image_size": config["image_size"],
        "metadata_features": list(metadata_features),
        "dataset_variant": dataset_variant,
        "seed": seed,
        "config": config,
        "metrics": metrics,
    }


def _config_value(config: dict, key: str, default):
    value = config.get(key)
    return default if value is None else value


def run_experiment(config):
    tracking_uri = _config_value(config, "tracking_uri", DEFAULT_TRACKING_URI)
    dataset_variant = normalize_dataset_variant(
        _config_value(config, "dataset_variant", DEFAULT_DATASET_VARIANT)
    )
    patience = int(_config_value(config, "patience", DEFAULT_PATIENCE))
    selection_metric = validate_selection_metric(
        _config_value(config, "selection_metric", DEFAULT_SELECTION_METRIC)
    )
    seed = int(config.get("seed", DEFAULT_SEED))
    resolved_config = {
        **config,
        "dataset_variant": dataset_variant,
        "patience": patience,
        "selection_metric": selection_metric,
        "tracking_uri": tracking_uri,
        "seed": seed,
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment_name"])
    seed_everything(seed)

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_param("model_name", config["model_name"])
        mlflow.log_param("batch_size", config["batch_size"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("epochs", config["epochs"])
        mlflow.log_param("image_size", config["image_size"])
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("dataset_variant", dataset_variant)
        mlflow.log_param("patience", patience)
        mlflow.log_param("selection_metric", selection_metric)
        mlflow.log_param("tracking_uri", tracking_uri)
        mlflow.log_param("seed", seed)

        model_dir = ROOT / "saved_models"
        model_dir.mkdir(exist_ok=True)

        if is_multimodal_model(config["model_name"]):
            dataloaders, datasets, dataset_sizes, class_to_idx = load_multimodal_data(
                base_dir=DATASET_PATH,
                metadata_csv=METADATA_CSV,
                image_size=config["image_size"],
                batch_size=config["batch_size"],
                dataset_variant=dataset_variant,
                seed=seed,
                model_name=config["model_name"],
            )

            num_classes = len(class_to_idx)

            model = create_multimodal_training_objects(
                model_name=config["model_name"],
                num_classes=num_classes
            )

            criterion = get_multimodal_loss(
                datasets=datasets,
                num_classes=num_classes
            )
            checkpoint_path = model_dir / f"{config['run_name']}_checkpoint.pth"

            start_time = time.time()

            model, best_val_acc, best_val_loss = train_multimodal_model(
                model=model,
                model_name=config["model_name"],
                dataloaders=dataloaders,
                criterion=criterion,
                learning_rate=config["learning_rate"],
                num_epochs=config["epochs"],
                patience=patience,
                num_classes=num_classes,
                selection_metric=selection_metric,
                checkpoint_path=checkpoint_path,
            )

            elapsed = time.time() - start_time

            test_metrics = evaluate_multimodal_model(
                model=model,
                model_name=config["model_name"],
                dataloader=dataloaders["test"],
                criterion=criterion,
                num_classes=num_classes
            )

        else:
            dataloaders, dataset_sizes, class_to_idx = load_data(
                base_dir=DATASET_PATH,
                image_size=config["image_size"],
                batch_size=config["batch_size"],
                dataset_variant=dataset_variant,
                seed=seed,
            )

            num_classes = len(class_to_idx)

            model = create_model(
                model_name=config["model_name"],
                num_classes=num_classes
            ).to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=config["learning_rate"])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            checkpoint_path = model_dir / f"{config['run_name']}_checkpoint.pth"

            start_time = time.time()

            model, best_val_acc, best_val_loss = train_model(
                model=model,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_classes=num_classes,
                num_epochs=config["epochs"],
                patience=patience,
                selection_metric=selection_metric,
                checkpoint_path=checkpoint_path,
            )

            elapsed = time.time() - start_time

            test_metrics = evaluate_model(
                model=model,
                dataloader=dataloaders["test"],
                num_classes=num_classes
            )

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_path = model_dir / f"{config['run_name']}.pth"
        torch.save(model.state_dict(), model_path)
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        metrics = {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "test_acc": test_metrics["accuracy"],
            "test_miou": test_metrics["miou"],
            "training_time_sec": elapsed,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
        }

        if "macro_f1" in test_metrics:
            metrics["test_macro_f1"] = test_metrics["macro_f1"]
        if "loss" in test_metrics:
            metrics["test_loss"] = test_metrics["loss"]

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        manifest_path = model_dir / f"{config['run_name']}_manifest.json"
        manifest = _build_manifest(
            config=resolved_config,
            class_to_idx=class_to_idx,
            metrics=metrics,
            dataset_variant=dataset_variant,
            seed=seed,
            metadata_features=METADATA_FEATURES if is_multimodal_model(config["model_name"]) else [],
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(manifest_path))
        mlflow.pytorch.log_model(model, "model")

        print(f"Run complete on device: {DEVICE}")
