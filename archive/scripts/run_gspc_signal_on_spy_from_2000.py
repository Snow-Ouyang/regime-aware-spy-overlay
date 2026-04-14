import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import run_spy_vs_gspc_signal_mapping as base


def _results_dir() -> Path:
    output_dir = base.project_root() / "results" / "gspc_signal_on_spy_from_2000"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _signal_asset_dir(version: str) -> Path:
    output_dir = _results_dir() / version
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _panels_dir() -> Path:
    output_dir = _results_dir() / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_base_module() -> None:
    base.TICKER_VERSION_TO_SIGNAL = [("gspc_signal_on_spy", "^GSPC", "gspc")]
    base.TARGET_OOS_START = pd.Timestamp("2000-01-03")
    base.results_dir = _results_dir
    base.signal_asset_dir = _signal_asset_dir
    base.panels_dir = _panels_dir


def _build_daily_equity(mapped_frame: pd.DataFrame, spy_buyhold: pd.DataFrame) -> pd.DataFrame:
    daily_equity = pd.DataFrame({"Date": mapped_frame["Date"].drop_duplicates().sort_values().to_numpy()})
    mapped_frame = mapped_frame.copy()
    mapped_frame["equity_gspc_signal_on_spy"] = (1.0 + mapped_frame["strategy_ret"]).cumprod()
    daily_equity = daily_equity.merge(mapped_frame[["Date", "equity_gspc_signal_on_spy"]], on="Date", how="left")
    spy_buyhold = spy_buyhold.copy()
    spy_buyhold["equity_spy_buy_and_hold"] = (1.0 + spy_buyhold["strategy_ret"]).cumprod()
    daily_equity = daily_equity.merge(spy_buyhold[["Date", "equity_spy_buy_and_hold"]], on="Date", how="left")
    return daily_equity.sort_values("Date").reset_index(drop=True)


def _plot_equity_curves(daily_equity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(daily_equity["Date"], daily_equity["equity_gspc_signal_on_spy"], label="gspc_signal_on_spy", linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_spy_buy_and_hold"], label="spy_buy_and_hold", linewidth=1.8)
    ax.set_title("GSPC Signal Executed on SPY vs SPY Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    _configure_base_module()

    base.ensure_inputs()
    signal_run = base.run_signal_pipeline("gspc_signal_on_spy")

    signal_frame = signal_run["signal_frame"].copy()
    signal_frame["signal_asset"] = "gspc_signal_on_spy"
    signal_asset_dir = _signal_asset_dir("gspc_signal_on_spy")
    signal_frame.to_csv(signal_asset_dir / "gspc_signal_on_spy_signal_panel.csv", index=False)
    signal_run["selection_log"].to_csv(signal_asset_dir / "gspc_signal_on_spy_selection_log.csv", index=False)

    common_dates = pd.DataFrame({"Date": signal_frame["Date"].drop_duplicates().sort_values().to_list()})
    spy_trade_base = base.load_spy_trade_base().merge(common_dates, on="Date", how="inner")
    spy_trade_base = spy_trade_base.sort_values("Date").reset_index(drop=True)

    mapped = base.simulate_mapped_strategy(signal_frame, spy_trade_base)
    mapped_frame = mapped["frame"].copy()

    execution_window = spy_trade_base[["Date", "next_ret", "next_rf_daily"]].copy()
    buyhold_base = execution_window.rename(columns={"next_ret": "ret", "next_rf_daily": "rf_daily"})
    spy_buyhold = base.build_buy_and_hold(buyhold_base, execution_window["Date"].iloc[0])

    strategy_summary = pd.DataFrame(
        [
            {"version": "gspc_signal_on_spy", **base.build_strategy_metrics(mapped_frame)},
            {"version": "spy_buy_and_hold", **base.build_strategy_metrics(spy_buyhold)},
        ]
    )

    prediction_metrics = pd.DataFrame(
        [
            {
                "version": "gspc_signal_on_spy",
                "avg_validation_accuracy": float(signal_run["prediction_metrics"]["avg_validation_accuracy"].iloc[0]),
                "avg_validation_balanced_accuracy": float(signal_run["prediction_metrics"]["avg_validation_balanced_accuracy"].iloc[0]),
                "avg_validation_f1": float(signal_run["prediction_metrics"]["avg_validation_f1"].iloc[0]),
                "avg_validation_log_loss": float(signal_run["prediction_metrics"]["avg_validation_log_loss"].iloc[0]),
                "avg_oos_accuracy": float(signal_run["prediction_metrics"]["avg_oos_accuracy"].iloc[0]),
                "avg_oos_balanced_accuracy": float(signal_run["prediction_metrics"]["avg_oos_balanced_accuracy"].iloc[0]),
                "avg_oos_f1": float(signal_run["prediction_metrics"]["avg_oos_f1"].iloc[0]),
                "avg_oos_log_loss": float(signal_run["prediction_metrics"]["avg_oos_log_loss"].iloc[0]),
            }
        ]
    )

    rolling_window_log = signal_run["selection_log"].copy()
    rolling_window_log = rolling_window_log.sort_values("rebalance_date").reset_index(drop=True)

    daily_equity = _build_daily_equity(mapped_frame, spy_buyhold)

    output_dir = _results_dir()
    strategy_summary.to_csv(output_dir / "strategy_performance_comparison.csv", index=False)
    prediction_metrics.to_csv(output_dir / "prediction_metrics_comparison.csv", index=False)
    rolling_window_log.to_csv(output_dir / "rolling_window_metrics_comparison.csv", index=False)
    daily_equity.to_csv(output_dir / "daily_equity_curves_comparison.csv", index=False)
    signal_frame.to_csv(output_dir / "signal_panel.csv", index=False)

    _plot_equity_curves(daily_equity, output_dir / "gspc_signal_on_spy_vs_spy_buyhold.png")

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Results directory: {output_dir}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
