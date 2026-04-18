import os
from pathlib import Path

VENDOR_OVERRIDE = Path(__file__).resolve().parents[1] / "vendor_mirror"
os.environ["BULL_BEAR_VENDOR_PATH"] = str(VENDOR_OVERRIDE)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score

from single_asset_gspc_spy_common import (
    StageConfig,
    build_target_windows,
    compute_metrics,
    load_research_frame,
    prepare_window,
    results_root,
    select_smoothing,
    single_threshold_drawdown_overlay_positions,
)


RESULTS_SUBDIR = "final_model_gspc_to_spy"


def output_dir() -> Path:
    out = results_root() / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_panel_from_arrays(
    dates: pd.Series | np.ndarray,
    rebalance_date: str,
    split_name: str,
    selected_smoothing_halflife: int,
    threshold: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
    probs_raw: np.ndarray,
    probs_smoothed: np.ndarray,
    y_true: np.ndarray,
    drawdown: np.ndarray,
    positions: np.ndarray,
    zones: np.ndarray,
    drawdown_entry: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "rebalance_date": rebalance_date,
            "split": split_name,
            "selected_smoothing_halflife": int(selected_smoothing_halflife),
            "selected_threshold": float(threshold),
            "selected_drawdown_threshold": float(drawdown_threshold),
            "selected_drawdown_prob_floor": float(drawdown_prob_floor),
            "predicted_probability_raw": np.asarray(probs_raw, dtype=float),
            "predicted_probability_smoothed": np.asarray(probs_smoothed, dtype=float),
            "predicted_label": np.asarray(positions, dtype=int),
            "y_true": np.asarray(y_true, dtype=int),
            "drawdown_from_peak": np.asarray(drawdown, dtype=float),
            "drawdown_entry_trigger": np.asarray(drawdown_entry, dtype=int),
            "signal_zone": np.asarray(zones, dtype=object),
        }
    )


def build_validation_panel(
    validation_cache: list[dict],
    rebalance_date: str,
    selected_smoothing_halflife: int,
    threshold: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
) -> pd.DataFrame:
    current_position = 0
    frames: list[pd.DataFrame] = []
    for bundle in validation_cache:
        probs_raw = np.asarray(bundle["raw_prob"], dtype=float)
        probs_smoothed = np.asarray(bundle["smoothed"][selected_smoothing_halflife], dtype=float)
        drawdown = np.asarray(bundle["drawdown"], dtype=float)
        positions, zones, drawdown_entry, current_position = single_threshold_drawdown_overlay_positions(
            probs_smoothed,
            drawdown,
            current_position,
            threshold,
            drawdown_threshold,
            drawdown_prob_floor,
        )
        frames.append(
            build_panel_from_arrays(
                bundle["signal_date"],
                rebalance_date,
                "validation",
                selected_smoothing_halflife,
                threshold,
                drawdown_threshold,
                drawdown_prob_floor,
                probs_raw,
                probs_smoothed,
                bundle["y_true"],
                drawdown,
                positions,
                zones,
                drawdown_entry,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "rebalance_date"]).reset_index(drop=True)


def build_oos_panel(
    prepared: dict,
    rebalance_date: str,
    selected_smoothing_halflife: int,
    threshold: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
) -> pd.DataFrame:
    probs_raw = np.asarray(prepared["oos_raw_prob"], dtype=float)
    probs_smoothed = np.asarray(prepared["oos_smoothed"][selected_smoothing_halflife], dtype=float)
    drawdown = np.asarray(prepared["oos_drawdown"], dtype=float)
    positions, zones, drawdown_entry, _ = single_threshold_drawdown_overlay_positions(
        probs_smoothed,
        drawdown,
        0,
        threshold,
        drawdown_threshold,
        drawdown_prob_floor,
    )
    return build_panel_from_arrays(
        prepared["oos_signal_date"],
        rebalance_date,
        "oos",
        selected_smoothing_halflife,
        threshold,
        drawdown_threshold,
        drawdown_prob_floor,
        probs_raw,
        probs_smoothed,
        prepared["oos_y_true"],
        drawdown,
        positions,
        zones,
        drawdown_entry,
    ).sort_values(["Date", "rebalance_date"]).reset_index(drop=True)


def metrics_summary(panel: pd.DataFrame, split_name: str) -> dict:
    metrics = compute_metrics(
        panel["y_true"].to_numpy(dtype=int, copy=False),
        panel["predicted_label"].to_numpy(dtype=int, copy=False),
        panel["predicted_probability_smoothed"].to_numpy(dtype=float, copy=False),
    )
    return {
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["f1"],
        "log_loss": metrics["log_loss"],
        "roc_auc": metrics["roc_auc"],
        "avg_position": float(panel["predicted_label"].mean()),
        "n_samples": int(len(panel)),
    }


def plot_confusion(panel: pd.DataFrame, title: str, out_path: Path) -> None:
    cm = confusion_matrix(panel["y_true"], panel["predicted_label"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Bear", "Bull"])
    ax.set_yticks([0, 1], ["Bear", "Bull"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(panel: pd.DataFrame, title: str, out_path: Path) -> None:
    y_true = panel["y_true"].to_numpy(dtype=int, copy=False)
    y_prob = panel["predicted_probability_smoothed"].to_numpy(dtype=float, copy=False)
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    try:
        auc_text = f"AUC={roc_auc_score(y_true, y_prob):.3f}"
    except ValueError:
        auc_text = "AUC=nan"
    thresholds = np.r_[0.0, np.unique(np.sort(y_prob)), 1.0]
    tpr: list[float] = []
    fpr: list[float] = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=auc_text)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_over_time(panel: pd.DataFrame, title: str, out_path: Path, csv_path: Path) -> None:
    metrics_by_window = (
        panel.groupby("rebalance_date")
        .apply(
            lambda df: pd.Series(
                {
                    "accuracy": accuracy_score(df["y_true"], df["predicted_label"]),
                    "f1": f1_score(df["y_true"], df["predicted_label"], zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(df["y_true"], df["predicted_label"]),
                }
            )
        )
        .reset_index()
    )
    metrics_by_window.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(metrics_by_window["rebalance_date"]), metrics_by_window["accuracy"], label="accuracy")
    ax.plot(pd.to_datetime(metrics_by_window["rebalance_date"]), metrics_by_window["f1"], label="f1")
    ax.plot(pd.to_datetime(metrics_by_window["rebalance_date"]), metrics_by_window["balanced_accuracy"], label="balanced_accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = StageConfig(
        stage_name="final_model_gspc_to_spy",
        results_subdir="final_model_gspc_to_spy",
        feature_mode="enhanced",
        rule_mode="single_threshold_drawdown_overlay",
        threshold=0.55,
        drawdown_threshold=0.20,
        drawdown_prob_floor=0.52,
        output_ml_figures=False,
    )

    out_dir = output_dir()
    windows = build_target_windows(load_research_frame())
    validation_panels: list[pd.DataFrame] = []
    oos_panels: list[pd.DataFrame] = []

    for window in windows:
        prepared = prepare_window(window.__dict__, config.feature_mode)
        selected_smoothing = select_smoothing(prepared["validation_cache"], config)
        validation_panels.append(
            build_validation_panel(
                prepared["validation_cache"],
                window.rebalance_date,
                int(selected_smoothing),
                float(config.threshold),
                float(config.drawdown_threshold),
                float(config.drawdown_prob_floor),
            )
        )
        oos_panels.append(
            build_oos_panel(
                prepared,
                window.rebalance_date,
                int(selected_smoothing),
                float(config.threshold),
                float(config.drawdown_threshold),
                float(config.drawdown_prob_floor),
            )
        )

    validation_panel = pd.concat(validation_panels, ignore_index=True).sort_values(["Date", "rebalance_date"]).reset_index(drop=True)
    oos_panel = pd.concat(oos_panels, ignore_index=True).sort_values(["Date", "rebalance_date"]).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            metrics_summary(validation_panel, "validation"),
            metrics_summary(oos_panel, "oos"),
        ]
    )

    summary.to_csv(out_dir / "classification_metrics_summary.csv", index=False)
    validation_panel.to_csv(out_dir / "classification_panel_validation.csv", index=False)
    oos_panel.to_csv(out_dir / "classification_panel_oos.csv", index=False)

    plot_confusion(validation_panel, "Confusion Matrix - Validation", out_dir / "confusion_matrix_validation.png")
    plot_confusion(oos_panel, "Confusion Matrix - OOS", out_dir / "confusion_matrix_oos.png")
    plot_roc(validation_panel, "ROC Curve - Validation", out_dir / "roc_curve_validation.png")
    plot_roc(oos_panel, "ROC Curve - OOS", out_dir / "roc_curve_oos.png")

    plot_metrics_over_time(
        validation_panel,
        "Classification Metrics Over Time - Validation",
        out_dir / "classification_metrics_over_time_validation.png",
        out_dir / "classification_metrics_over_time_validation.csv",
    )
    plot_metrics_over_time(
        oos_panel,
        "Classification Metrics Over Time - OOS",
        out_dir / "classification_metrics_over_time_oos.png",
        out_dir / "classification_metrics_over_time_oos.csv",
    )

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
