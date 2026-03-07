import os
import time

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from src.dissdetector.models.model_factory import create_model
from src.dissdetector.training.train_core import DATASET_PATH, DEVICE, evaluate_model, load_data, train_model


def run_experiment(config):
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_param("model_name", config["model_name"])
        mlflow.log_param("batch_size", config["batch_size"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("epochs", config["epochs"])
        mlflow.log_param("image_size", config["image_size"])

        dataloaders, dataset_sizes, class_to_idx = load_data(
            base_dir=DATASET_PATH,
            image_size=config["image_size"],
            batch_size=config["batch_size"]
        )

        num_classes = len(class_to_idx)

        model = create_model(
            model_name=config["model_name"],
            num_classes=num_classes
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
            patience=5
        )

        elapsed = time.time() - start_time

        test_metrics = evaluate_model(
            model=model,
            dataloader=dataloaders["test"],
            num_classes=num_classes
        )

        num_params = sum(p.numel() for p in model.parameters())

        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{config['run_name']}.pth")
        torch.save(model.state_dict(), model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("test_acc", test_metrics["accuracy"])
        mlflow.log_metric("test_miou", test_metrics["miou"])
        mlflow.log_metric("training_time_sec", elapsed)
        mlflow.log_metric("num_parameters", num_params)
        mlflow.log_metric("model_size_mb", model_size_mb)

        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "model")

        print("Run complete")