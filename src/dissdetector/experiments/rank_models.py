import math
import mlflow
import pandas as pd


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

ONLINE_EXPERIMENT_NAME = "plant_disease_online_models"
OFFLINE_EXPERIMENT_NAME = "plant_disease_offline_models"


def get_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index)


def safe_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or math.isclose(min_val, max_val):
        return pd.Series([1.0] * len(series), index=series.index)

    norm = (series - min_val) / (max_val - min_val)

    if higher_is_better:
        return norm
    return 1.0 - norm


def load_experiment_runs(experiment_name: str) -> pd.DataFrame:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment not found: {experiment_name}")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        output_format="pandas"
    )

    if runs.empty:
        print(f"No runs found for experiment: {experiment_name}")
        return pd.DataFrame()

    return runs


def build_rank_table(df: pd.DataFrame, category: str) -> pd.DataFrame:
    if df.empty:
        return df

    keep_cols = {
        "tags.mlflow.runName": "run_name",
        "params.model_name": "model_name",
        "metrics.best_val_acc": "best_val_acc",
        "metrics.best_val_loss": "best_val_loss",
        "metrics.test_acc": "test_acc",
        "metrics.test_miou": "test_miou",
        "metrics.training_time_sec": "training_time_sec",
        "metrics.num_parameters": "num_parameters",
        "metrics.model_size_mb": "model_size_mb",
    }

    available_cols = [c for c in keep_cols if c in df.columns]
    rank_df = df[available_cols].copy()
    rank_df = rank_df.rename(columns=keep_cols)

    rank_df["best_val_acc"] = get_numeric_series(rank_df, "best_val_acc")
    rank_df["best_val_loss"] = get_numeric_series(rank_df, "best_val_loss")
    rank_df["test_acc"] = get_numeric_series(rank_df, "test_acc")
    rank_df["test_miou"] = get_numeric_series(rank_df, "test_miou")
    rank_df["training_time_sec"] = get_numeric_series(rank_df, "training_time_sec")
    rank_df["num_parameters"] = get_numeric_series(rank_df, "num_parameters")
    rank_df["model_size_mb"] = get_numeric_series(rank_df, "model_size_mb")

    rank_df["category"] = category

    rank_df["score_test_acc"] = safe_normalize(rank_df["test_acc"], higher_is_better=True)
    rank_df["score_test_miou"] = safe_normalize(rank_df["test_miou"], higher_is_better=True)
    rank_df["score_val_acc"] = safe_normalize(rank_df["best_val_acc"], higher_is_better=True)
    rank_df["score_val_loss"] = safe_normalize(rank_df["best_val_loss"], higher_is_better=False)
    rank_df["score_speed"] = safe_normalize(rank_df["training_time_sec"], higher_is_better=False)
    rank_df["score_params"] = safe_normalize(rank_df["num_parameters"], higher_is_better=False)
    rank_df["score_size"] = safe_normalize(rank_df["model_size_mb"], higher_is_better=False)

    if category == "online":
        rank_df["overall_score"] = (
            0.35 * rank_df["score_test_acc"] +
            0.25 * rank_df["score_test_miou"] +
            0.20 * rank_df["score_val_acc"] +
            0.10 * rank_df["score_val_loss"] +
            0.05 * rank_df["score_speed"] +
            0.05 * rank_df["score_params"]
        )
    else:
        rank_df["overall_score"] = (
            0.25 * rank_df["score_test_acc"] +
            0.20 * rank_df["score_test_miou"] +
            0.15 * rank_df["score_val_acc"] +
            0.10 * rank_df["score_val_loss"] +
            0.10 * rank_df["score_speed"] +
            0.10 * rank_df["score_params"] +
            0.10 * rank_df["score_size"]
        )

    rank_df = rank_df.sort_values(by="overall_score", ascending=False).reset_index(drop=True)
    rank_df["rank"] = rank_df.index + 1

    return rank_df


def print_best_summary(df: pd.DataFrame, title: str):
    if df.empty:
        print(f"\n{title}: no runs found")
        return

    best = df.iloc[0]

    print(f"\n=== {title} ===")
    print(f"Best run: {best['run_name']}")
    print(f"Model: {best['model_name']}")
    print(f"Overall score: {best['overall_score']:.4f}")
    print(f"Test accuracy: {best['test_acc']:.4f}")
    print(f"Test mIoU: {best['test_miou']:.4f}")
    print(f"Parameters: {best['num_parameters']:.0f}")
    print(f"Model size (MB): {best['model_size_mb']:.2f}")
    print(f"Training time (sec): {best['training_time_sec']:.2f}")


def print_special_awards(df: pd.DataFrame, title: str):
    if df.empty:
        return

    print(f"\n--- {title}: Special Awards ---")

    if df["test_acc"].notna().any():
        best_acc = df.loc[df["test_acc"].idxmax()]
        print(f"Best accuracy: {best_acc['run_name']} ({best_acc['test_acc']:.4f})")

    if df["test_miou"].notna().any():
        best_miou = df.loc[df["test_miou"].idxmax()]
        print(f"Best mIoU: {best_miou['run_name']} ({best_miou['test_miou']:.4f})")

    if df["num_parameters"].notna().any():
        best_params = df.loc[df["num_parameters"].idxmin()]
        print(f"Most parameter-efficient: {best_params['run_name']} ({best_params['num_parameters']:.0f})")

    if df["model_size_mb"].notna().any():
        best_size = df.loc[df["model_size_mb"].idxmin()]
        print(f"Smallest model: {best_size['run_name']} ({best_size['model_size_mb']:.2f} MB)")

    if df["training_time_sec"].notna().any():
        best_speed = df.loc[df["training_time_sec"].idxmin()]
        print(f"Fastest training: {best_speed['run_name']} ({best_speed['training_time_sec']:.2f} sec)")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    online_runs = load_experiment_runs(ONLINE_EXPERIMENT_NAME)
    offline_runs = load_experiment_runs(OFFLINE_EXPERIMENT_NAME)

    online_ranked = build_rank_table(online_runs, "online")
    offline_ranked = build_rank_table(offline_runs, "offline")

    print_best_summary(online_ranked, "ONLINE MODELS")
    print_special_awards(online_ranked, "ONLINE MODELS")

    print_best_summary(offline_ranked, "OFFLINE MODELS")
    print_special_awards(offline_ranked, "OFFLINE MODELS")

    if not online_ranked.empty:
        print("\nTop ONLINE ranking:")
        print(
            online_ranked[
                ["rank", "run_name", "model_name", "test_acc", "test_miou", "num_parameters", "model_size_mb", "overall_score"]
            ].to_string(index=False)
        )

    if not offline_ranked.empty:
        print("\nTop OFFLINE ranking:")
        print(
            offline_ranked[
                ["rank", "run_name", "model_name", "test_acc", "test_miou", "num_parameters", "model_size_mb", "overall_score"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()