from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from single_asset_gspc_spy_common import StageConfig, results_root, run_ml_stage
from spy_regime_common import build_strategy_metrics


RESULTS_SUBDIR = "single_threshold_overlay_vs_final_gspc_to_spy"


def output_dir() -> Path:
    out = results_root() / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    out_dir = output_dir()
    configs = [
        StageConfig(
            stage_name="single_threshold_055_plus_extra_entries",
            results_subdir=RESULTS_SUBDIR,
            feature_mode="enhanced",
            rule_mode="single_threshold_overlay",
            threshold=0.55,
            rising_floor=0.52,
            drawdown_threshold=0.20,
            drawdown_prob_floor=0.44,
        ),
        StageConfig(
            stage_name="final_recovery_overlay_gspc_to_spy",
            results_subdir=RESULTS_SUBDIR,
            feature_mode="enhanced",
            rule_mode="final_overlay",
            rising_floor=0.52,
            drawdown_threshold=0.20,
            drawdown_prob_floor=0.44,
        ),
    ]

    summaries: list[dict[str, float | str]] = []
    prediction_metrics_frames: list[pd.DataFrame] = []
    selection_logs: list[pd.DataFrame] = []
    equity_curves: pd.DataFrame | None = None
    buyhold_frame: pd.DataFrame | None = None

    for config in configs:
        result = run_ml_stage(config)
        mapped_frame = result["mapped_frame"].copy()
        summaries.append({"version": config.stage_name, **build_strategy_metrics(mapped_frame)})

        metrics = result["prediction_metrics"].copy()
        metrics["version"] = config.stage_name
        prediction_metrics_frames.append(metrics)

        selection_log = result["selection_log"].copy()
        selection_log["version"] = config.stage_name
        selection_logs.append(selection_log)

        curve = mapped_frame[["Date", "equity_predicted_strategy"]].rename(
            columns={"equity_predicted_strategy": f"equity_{config.stage_name}"}
        )
        equity_curves = curve if equity_curves is None else equity_curves.merge(curve, on="Date", how="outer")

        if buyhold_frame is None:
            buyhold_frame = result["buyhold_frame"].copy()

    if buyhold_frame is None or equity_curves is None:
        raise RuntimeError("Comparison results were not generated")

    summaries.append({"version": "buy_and_hold", **build_strategy_metrics(buyhold_frame)})
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(out_dir / "strategy_performance_comparison.csv", index=False)

    prediction_metrics = pd.concat(prediction_metrics_frames, ignore_index=True)
    prediction_metrics.to_csv(out_dir / "prediction_metrics_comparison.csv", index=False)

    selection_log_frame = pd.concat(selection_logs, ignore_index=True).sort_values(["version", "rebalance_date"]).reset_index(drop=True)
    selection_log_frame.to_csv(out_dir / "rolling_window_metrics_comparison.csv", index=False)

    equity_curves = equity_curves.sort_values("Date").reset_index(drop=True)
    equity_curves = equity_curves.merge(
        buyhold_frame[["Date", "equity_buy_and_hold"]],
        on="Date",
        how="left",
    )
    equity_curves.to_csv(out_dir / "daily_equity_curves_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in equity_curves.columns:
        if column == "Date":
            continue
        ax.plot(equity_curves["Date"], equity_curves[column], label=column, linewidth=1.6)
    ax.set_title("Single Threshold + Extra Entries vs Final Overlay")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "single_threshold_overlay_vs_final_equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_frame["version"], summary_frame["sharpe"])
    ax.set_title("Sharpe Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "single_threshold_overlay_vs_final_sharpe.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_frame["version"], summary_frame["max_drawdown"])
    ax.set_title("Max Drawdown Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "single_threshold_overlay_vs_final_drawdown.png", dpi=150)
    plt.close(fig)

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
