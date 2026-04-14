import os
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List


def configure_paths() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
    vendor_local = Path(__file__).resolve().parents[1] / ".vendor_local"
    if vendor_local.exists():
        vendor_local_str = str(vendor_local)
        if vendor_local_str not in sys.path:
            sys.path.insert(0, vendor_local_str)


configure_paths()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import run_spy_final_research_trade_20000526 as base
import run_spy_vs_gspc_signal_mapping as signal_base


RESULTS_DIR_NAME = "spy_recovery_feature_test"
BASELINE_VERSION = "current_baseline"
RECOVERY_VERSION = "baseline_plus_recovery_features"
ADDED_RECOVERY_FEATURES = [
    "ma20_over_ma60",
    "close_over_ma120",
    "ret_61_120d",
]


def results_dir() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "results" / RESULTS_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def source_baseline_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / base.RESULTS_DIR_NAME


def load_recovery_experiment_frame() -> pd.DataFrame:
    stem = base.SPY_STEM
    feature_frame = pd.read_csv(signal_base.feature_path_for_signal_stem(stem))
    macro_frame = pd.read_csv(signal_base.macro_feature_path())
    raw_frame = pd.read_csv(signal_base.raw_path_for_signal_stem(stem))

    for frame in [feature_frame, macro_frame, raw_frame]:
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")

    for column in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        raw_frame[column] = pd.to_numeric(raw_frame[column], errors="coerce")
    for column in macro_frame.columns:
        if column != "Date":
            macro_frame[column] = pd.to_numeric(macro_frame[column], errors="coerce")

    feature_frame = feature_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    macro_frame = macro_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    raw_frame = raw_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    price = raw_frame["Close"]
    ma20 = price.rolling(20).mean()
    ma60 = price.rolling(60).mean()
    ma120 = price.rolling(120).mean()
    price_features = pd.DataFrame(
        {
            "Date": raw_frame["Date"],
            "ret_1_5d": price / price.shift(5) - 1.0,
            "ret_6_20d": price.shift(5) / price.shift(20) - 1.0,
            "ret_21_60d": price.shift(20) / price.shift(60) - 1.0,
            "close_over_ma20": price / ma20 - 1.0,
            "close_over_ma60": price / ma60 - 1.0,
            "ma20_over_ma60": ma20 / ma60 - 1.0,
            "close_over_ma120": price / ma120 - 1.0,
            # Mutually exclusive medium-long return window: day 61 to day 120 ago.
            "ret_61_120d": price.shift(60) / price.shift(120) - 1.0,
        }
    )

    merged = feature_frame.merge(macro_frame, on="Date", how="left", sort=True)
    merged = merged.merge(price_features, on="Date", how="left", sort=True)
    merged = merged.sort_values("Date", ascending=True).reset_index(drop=True)

    macro_columns = [column for column in macro_frame.columns if column != "Date"]
    merged[macro_columns] = merged[macro_columns].ffill()
    required_columns = [
        "Date",
        "ret",
        "rf_daily",
        "excess_ret",
        *signal_base.BASE_JM_FEATURES,
        *signal_base.A1_REFINED_FEATURES,
        *signal_base.A2_FEATURES,
        *ADDED_RECOVERY_FEATURES,
        *macro_columns,
    ]
    merged = merged.dropna(subset=required_columns).reset_index(drop=True)
    if merged.empty:
        raise ValueError("SPY recovery-enhanced dataset is empty")
    return merged


def recovery_feature_columns(frame: pd.DataFrame) -> List[str]:
    return [*signal_base.research_feature_columns(frame), *ADDED_RECOVERY_FEATURES]


def frame_to_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    records = frame.to_dict(orient="records")
    cleaned = []
    for row in records:
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                cleaned_row[key] = value.strftime("%Y-%m-%d")
            elif isinstance(value, np.generic):
                cleaned_row[key] = value.item()
            else:
                cleaned_row[key] = value
        cleaned.append(cleaned_row)
    return cleaned


def prepare_window_recovery(window: base.MainWindow) -> Dict[str, object]:
    frame = load_recovery_experiment_frame()
    features = recovery_feature_columns(frame)
    validation_segments = base.build_nonempty_segments(frame, pd.Timestamp(window.val_start), pd.Timestamp(window.oos_start))

    validation_bundles: List[Dict[str, object]] = []
    for segment in validation_segments:
        bundle = base.build_segment_supervised_bundle(
            frame,
            features,
            segment["start"],
            segment["end_exclusive"],
            penalty=base.JUMP_PENALTY,
        )
        validation_bundles.append(
            {
                "train_dataset": bundle["train_dataset"],
                "segment_dataset": bundle["segment_dataset"],
                "initial_position": 0,
                "base": bundle["segment_dataset"]["base"].reset_index(drop=True),
            }
        )

    oos_bundle = base.build_segment_supervised_bundle(
        frame,
        features,
        pd.Timestamp(window.oos_start),
        pd.Timestamp(window.oos_end) + pd.Timedelta(days=1),
        penalty=base.JUMP_PENALTY,
    )
    oos_model = base.make_xgb_model()
    oos_model.fit(oos_bundle["train_dataset"]["X"], oos_bundle["train_dataset"]["y"])
    oos_raw_prob = oos_model.predict_proba(oos_bundle["segment_dataset"]["X"])[:, 1].astype(float)
    oos_smoothed = {h: signal_base.smooth_probability_series(oos_raw_prob, h) for h in base.SMOOTHING_GRID}

    return {
        "window": window,
        "validation_bundles": validation_bundles,
        "oos_base": oos_bundle["segment_dataset"]["base"].reset_index(drop=True),
        "oos_y_true": oos_bundle["segment_dataset"]["y"].astype(int).copy(),
        "oos_raw_prob": oos_raw_prob,
        "oos_smoothed": oos_smoothed,
    }


def worker_process_window_recovery(window_payload: Dict[str, str]) -> Dict[str, object]:
    window = base.MainWindow(**window_payload)
    prepared = prepare_window_recovery(window)
    validation_bundles = prepared["validation_bundles"]
    oos_base = prepared["oos_base"]
    oos_y_true = prepared["oos_y_true"]
    oos_raw_prob = prepared["oos_raw_prob"]
    oos_smoothed = prepared["oos_smoothed"]

    selected_smoothing, _ = base.select_best_smoothing(validation_bundles)
    selected_double, _ = base.select_best_double_threshold(validation_bundles, selected_smoothing)

    positions, zones, _ = base.double_threshold_positions(
        oos_smoothed[selected_smoothing],
        initial_position=0,
        lower_threshold=float(selected_double["selected_lower_threshold"]),
        upper_threshold=float(selected_double["selected_upper_threshold"]),
    )
    simulation = base.simulate_positions_strategy(oos_base, positions, initial_position=0)
    frame = simulation["frame"]
    frame["signal_zone"] = zones
    frame["rebalance_date"] = window.rebalance_date
    frame["selected_smoothing_halflife"] = int(selected_smoothing)
    frame["selected_lower_threshold"] = float(selected_double["selected_lower_threshold"])
    frame["selected_upper_threshold"] = float(selected_double["selected_upper_threshold"])
    frame["predicted_probability_raw"] = np.asarray(oos_raw_prob, dtype=float)
    frame["predicted_probability_smoothed"] = np.asarray(oos_smoothed[selected_smoothing], dtype=float)
    frame["y_true"] = np.asarray(oos_y_true, dtype=int)

    validation_result = base.score_validation_bundle(
        validation_bundles,
        int(selected_smoothing),
        float(selected_double["selected_lower_threshold"]),
        float(selected_double["selected_upper_threshold"]),
    )
    oos_metrics = base.compute_metrics(
        np.asarray(oos_y_true, dtype=int),
        np.asarray(positions, dtype=int),
        np.asarray(oos_smoothed[selected_smoothing], dtype=float),
    )

    return {
        "window": window_payload,
        "selected_smoothing_halflife": int(selected_smoothing),
        "selected_lower_threshold": float(selected_double["selected_lower_threshold"]),
        "selected_upper_threshold": float(selected_double["selected_upper_threshold"]),
        "validation_sharpe": float(validation_result["sharpe"]),
        "validation_balanced_accuracy": float(validation_result["metrics"]["balanced_accuracy"]),
        "validation_accuracy": float(validation_result["metrics"]["accuracy"]),
        "validation_f1": float(validation_result["metrics"]["f1"]),
        "validation_log_loss": float(validation_result["metrics"]["log_loss"]),
        "oos_metrics": {
            "accuracy": float(oos_metrics["accuracy"]),
            "balanced_accuracy": float(oos_metrics["balanced_accuracy"]),
            "f1": float(oos_metrics["f1"]),
            "log_loss": float(oos_metrics["log_loss"]),
        },
        "oos_sharpe": float(simulation["sharpe"]),
        "oos_frame_records": frame_to_records(frame),
    }


def build_final_frame(window_results: List[Dict[str, object]]) -> pd.DataFrame:
    safe_payloads = []
    for result in window_results:
        frame = pd.DataFrame(result["oos_frame_records"])
        frame["Date"] = pd.to_datetime(frame["Date"])
        safe_payloads.append({"window": result["window"], "oos_frame": frame})
    final_frame = base.stitch_final_frame(safe_payloads)
    final_frame["equity_predicted_strategy"] = (1.0 + final_frame["strategy_ret"]).cumprod()
    final_frame["predicted_label"] = final_frame["position"].astype(int)
    final_frame["predicted_bull_flag"] = final_frame["predicted_label"]
    final_frame["predicted_bear_flag"] = 1 - final_frame["predicted_label"]
    return final_frame


def plot_equity_curves(daily_equity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(daily_equity["Date"], daily_equity["equity_current_baseline"], label=BASELINE_VERSION, linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_baseline_plus_recovery_features"], label=RECOVERY_VERSION, linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_buy_and_hold"], label="buy_and_hold", linewidth=1.8)
    ax.set_title("SPY Recovery Feature Test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_bar(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["version"], summary[metric], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(rolling_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for version, color in [(BASELINE_VERSION, "#1f77b4"), (RECOVERY_VERSION, "#ff7f0e")]:
        sub = rolling_df[rolling_df["version"] == version].copy()
        sub["rebalance_date"] = pd.to_datetime(sub["rebalance_date"])
        ax.plot(sub["rebalance_date"], sub["oos_sharpe"], label=version, linewidth=1.4, color=color)
    ax.set_title("Rolling OOS Sharpe Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    base.ensure_inputs()

    source_dir = source_baseline_dir()
    baseline_summary = pd.read_csv(source_dir / "strategy_performance_summary.csv").rename(columns={"version": "version"})
    baseline_prediction = pd.read_csv(source_dir / "prediction_metrics_comparison.csv")
    baseline_rolling = pd.read_csv(source_dir / "rolling_window_log.csv")
    baseline_daily = pd.read_csv(source_dir / "daily_equity_curves.csv")
    baseline_daily["Date"] = pd.to_datetime(baseline_daily["Date"], errors="coerce")

    baseline_summary = baseline_summary.loc[baseline_summary["version"] == "predicted_strategy"].copy()
    baseline_summary["version"] = BASELINE_VERSION
    baseline_prediction["version"] = BASELINE_VERSION
    baseline_rolling["version"] = BASELINE_VERSION
    baseline_daily = baseline_daily.rename(columns={"equity_predicted_strategy": "equity_current_baseline"})

    frame = load_recovery_experiment_frame()
    main_windows = base.build_target_windows(frame)
    worker_count = min(base.MAX_WORKERS_CAP, len(main_windows), os.cpu_count() or 1)
    tasks = [window.__dict__ for window in main_windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        recovery_results = list(executor.map(worker_process_window_recovery, tasks))

    recovery_results = sorted(recovery_results, key=lambda item: item["window"]["rebalance_date"])
    recovery_frame = build_final_frame(recovery_results)

    buyhold = base.build_buyhold_frame(pd.DataFrame({"Date": recovery_frame["Date"].drop_duplicates().sort_values().to_list()}))
    buyhold = buyhold.merge(recovery_frame[["Date"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    recovery_summary = pd.DataFrame(
        [{"version": RECOVERY_VERSION, **base.build_strategy_metrics(recovery_frame)}]
    )
    buyhold_summary = pd.DataFrame(
        [{"version": "buy_and_hold", **base.build_strategy_metrics(buyhold)}]
    )
    prediction_metrics = pd.DataFrame(
        [
            {
                "version": RECOVERY_VERSION,
                "avg_validation_accuracy": float(np.mean([row["validation_accuracy"] for row in recovery_results])),
                "avg_validation_balanced_accuracy": float(np.mean([row["validation_balanced_accuracy"] for row in recovery_results])),
                "avg_validation_f1": float(np.mean([row["validation_f1"] for row in recovery_results])),
                "avg_validation_log_loss": float(np.mean([row["validation_log_loss"] for row in recovery_results])),
                "avg_oos_accuracy": float(np.mean([row["oos_metrics"]["accuracy"] for row in recovery_results])),
                "avg_oos_balanced_accuracy": float(np.mean([row["oos_metrics"]["balanced_accuracy"] for row in recovery_results])),
                "avg_oos_f1": float(np.mean([row["oos_metrics"]["f1"] for row in recovery_results])),
                "avg_oos_log_loss": float(np.mean([row["oos_metrics"]["log_loss"] for row in recovery_results])),
            }
        ]
    )

    rolling_rows = []
    for result in recovery_results:
        window_slice = recovery_frame.loc[recovery_frame["rebalance_date"] == result["window"]["rebalance_date"]]
        y_true = np.asarray(window_slice["y_true"], dtype=int)
        y_pred = np.asarray(window_slice["position"], dtype=int)
        rolling_rows.append(
            {
                "version": RECOVERY_VERSION,
                "rebalance_date": result["window"]["rebalance_date"],
                "selected_smoothing_halflife": int(result["selected_smoothing_halflife"]),
                "selected_lower_threshold": float(result["selected_lower_threshold"]),
                "selected_upper_threshold": float(result["selected_upper_threshold"]),
                "oos_sharpe": base.annualized_sharpe(window_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
                "oos_accuracy": float(accuracy_score(y_true, y_pred)),
                "oos_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "oos_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    strategy_summary = pd.concat([baseline_summary, recovery_summary, buyhold_summary], ignore_index=True)
    prediction_comparison = pd.concat([baseline_prediction, prediction_metrics], ignore_index=True)
    rolling_comparison = pd.concat([baseline_rolling, pd.DataFrame(rolling_rows)], ignore_index=True)

    daily_equity = baseline_daily.merge(
        recovery_frame[["Date", "equity_predicted_strategy"]].rename(
            columns={"equity_predicted_strategy": "equity_baseline_plus_recovery_features"}
        ),
        on="Date",
        how="inner",
    )

    feature_columns_df = pd.DataFrame(
        [{"group": "baseline_features", "feature": col} for col in signal_base.research_feature_columns(frame)]
        + [{"group": "added_recovery_features", "feature": col} for col in ADDED_RECOVERY_FEATURES]
        + [{"group": "final_version2_features", "feature": col} for col in recovery_feature_columns(frame)]
    )

    out_dir = results_dir()
    strategy_summary.to_csv(out_dir / "strategy_performance_comparison.csv", index=False)
    prediction_comparison.to_csv(out_dir / "prediction_metrics_comparison.csv", index=False)
    rolling_comparison.sort_values(["version", "rebalance_date"]).reset_index(drop=True).to_csv(
        out_dir / "rolling_window_metrics_comparison.csv", index=False
    )
    feature_columns_df.to_csv(out_dir / "recovery_feature_columns.csv", index=False)
    daily_equity.to_csv(out_dir / "daily_equity_curves_comparison.csv", index=False)

    plot_equity_curves(daily_equity, out_dir / "recovery_feature_equity_curves.png")
    plot_metric_bar(strategy_summary, "sharpe", "Sharpe Comparison", out_dir / "recovery_feature_sharpe_comparison.png")
    plot_metric_bar(strategy_summary, "max_drawdown", "Max Drawdown Comparison", out_dir / "recovery_feature_drawdown_comparison.png")
    plot_rolling_sharpe(rolling_comparison, out_dir / "rolling_oos_sharpe_comparison.png")

    print(f"Results directory: {out_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start_time:.2f}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
