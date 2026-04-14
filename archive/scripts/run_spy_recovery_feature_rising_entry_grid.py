import os
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple


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


RESULTS_DIR_NAME = "spy_recovery_feature_rising_entry_grid"
BASELINE_VERSION = "current_baseline"
BUYHOLD_VERSION = "buy_and_hold"
ADDED_RECOVERY_FEATURES = [
    "ma20_over_ma60",
    "close_over_ma120",
    "ret_61_120d",
]
RISING_ENTRY_FLOOR_GRID = [0.50, 0.53, 0.55, 0.58, 0.60]
FIXED_VERSION_NAMES = {
    0.50: "fixed_rising_floor_050",
    0.53: "fixed_rising_floor_053",
    0.55: "fixed_rising_floor_055",
    0.58: "fixed_rising_floor_058",
    0.60: "fixed_rising_floor_060",
}
DYNAMIC_VERSION = "dynamic_rising_floor"


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


def rising_entry_positions(
    probabilities: np.ndarray,
    initial_position: int,
    lower_threshold: float,
    upper_threshold: float,
    rising_floor: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    prev_position = int(initial_position)

    for idx, prob in enumerate(probs):
        rising_confirm = (
            idx >= 2
            and probs[idx - 1] > probs[idx - 2]
            and prob > probs[idx - 1]
            and prob > rising_floor
        )
        if prev_position == 0:
            if prob >= upper_threshold or rising_confirm:
                current = 1
                zone = "bull"
            elif prob <= lower_threshold:
                current = 0
                zone = "bear"
            else:
                current = 0
                zone = "hold"
        else:
            if prob >= upper_threshold:
                current = 1
                zone = "bull"
            elif prob <= lower_threshold:
                current = 0
                zone = "bear"
            else:
                current = 1
                zone = "hold"

        positions[idx] = current
        zones[idx] = zone
        prev_position = current

    return positions, zones, prev_position


def score_validation_cached(
    validation_cache: List[Dict[str, object]],
    smoothing_halflife: int,
    lower_threshold: float,
    upper_threshold: float,
    rising_floor: float,
) -> Dict[str, object]:
    current_position = 0
    frames: List[pd.DataFrame] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for bundle in validation_cache:
        smoothed_prob = np.asarray(bundle["smoothed"][smoothing_halflife], dtype=float)
        segment_initial_position = current_position
        positions, _, current_position = rising_entry_positions(
            smoothed_prob,
            segment_initial_position,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            rising_floor=rising_floor,
        )
        simulation = base.simulate_positions_strategy(bundle["base"], positions, segment_initial_position)
        frames.append(simulation["frame"])
        y_true_list.append(bundle["y_true"])
        y_pred_list.append(positions)
        y_prob_list.append(smoothed_prob)

    validation_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {
        "frame": validation_frame,
        "sharpe": base.annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
        "metrics": base.compute_metrics(np.concatenate(y_true_list), np.concatenate(y_pred_list), np.concatenate(y_prob_list)),
    }


def choose_parameters_for_floor(
    validation_cache: List[Dict[str, object]],
    rising_floor: float,
) -> Tuple[int, Dict[str, object], Dict[str, object]]:
    smoothing_rows: List[Dict[str, object]] = []
    for smoothing_halflife in base.SMOOTHING_GRID:
        result = score_validation_cached(validation_cache, smoothing_halflife, 0.48, 0.67, rising_floor)
        smoothing_rows.append(
            {
                "selected_smoothing_halflife": int(smoothing_halflife),
                "validation_sharpe": float(result["sharpe"]),
                "validation_balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
                "version_id": f"h{smoothing_halflife}",
            }
        )
    smoothing_frame = pd.DataFrame(smoothing_rows).sort_values(
        by=["validation_sharpe", "validation_balanced_accuracy", "selected_smoothing_halflife", "version_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    selected_smoothing = int(smoothing_frame.iloc[0]["selected_smoothing_halflife"])

    threshold_rows: List[Dict[str, object]] = []
    for lower_threshold in base.LOWER_THRESHOLD_GRID:
        for upper_threshold in base.UPPER_THRESHOLD_GRID:
            if upper_threshold <= lower_threshold:
                continue
            result = score_validation_cached(validation_cache, selected_smoothing, lower_threshold, upper_threshold, rising_floor)
            threshold_rows.append(
                {
                    "selected_lower_threshold": float(lower_threshold),
                    "selected_upper_threshold": float(upper_threshold),
                    "validation_sharpe": float(result["sharpe"]),
                    "validation_balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
                    "validation_accuracy": float(result["metrics"]["accuracy"]),
                    "validation_f1": float(result["metrics"]["f1"]),
                    "validation_log_loss": float(result["metrics"]["log_loss"]),
                    "version_id": f"dt_l{int(round(lower_threshold * 100)):03d}_u{int(round(upper_threshold * 100)):03d}",
                }
            )
    threshold_frame = pd.DataFrame(threshold_rows).copy()
    threshold_frame["lower_distance"] = (threshold_frame["selected_lower_threshold"] - 0.48).abs()
    threshold_frame["upper_distance"] = (threshold_frame["selected_upper_threshold"] - 0.67).abs()
    threshold_frame = threshold_frame.sort_values(
        by=["validation_sharpe", "validation_balanced_accuracy", "lower_distance", "upper_distance", "version_id"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    selected_threshold = threshold_frame.iloc[0].to_dict()

    best_validation = score_validation_cached(
        validation_cache,
        selected_smoothing,
        float(selected_threshold["selected_lower_threshold"]),
        float(selected_threshold["selected_upper_threshold"]),
        rising_floor,
    )
    return selected_smoothing, selected_threshold, best_validation


def prepare_window_recovery(window: base.MainWindow) -> Dict[str, object]:
    frame = load_recovery_experiment_frame()
    features = recovery_feature_columns(frame)
    validation_segments = base.build_nonempty_segments(frame, pd.Timestamp(window.val_start), pd.Timestamp(window.oos_start))

    validation_cache: List[Dict[str, object]] = []
    for segment in validation_segments:
        bundle = base.build_segment_supervised_bundle(
            frame,
            features,
            segment["start"],
            segment["end_exclusive"],
            penalty=base.JUMP_PENALTY,
        )
        model = base.make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1].astype(float)
        smoothed = {h: signal_base.smooth_probability_series(raw_prob, h) for h in base.SMOOTHING_GRID}
        validation_cache.append(
            {
                "base": bundle["segment_dataset"]["base"].reset_index(drop=True),
                "y_true": bundle["segment_dataset"]["y"].astype(int).copy(),
                "raw_prob": raw_prob,
                "smoothed": smoothed,
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
        "validation_cache": validation_cache,
        "oos_base": oos_bundle["segment_dataset"]["base"].reset_index(drop=True),
        "oos_y_true": oos_bundle["segment_dataset"]["y"].astype(int).copy(),
        "oos_raw_prob": oos_raw_prob,
        "oos_smoothed": oos_smoothed,
    }


def worker_process_window(window_payload: Dict[str, str]) -> Dict[str, object]:
    window = base.MainWindow(**window_payload)
    prepared = prepare_window_recovery(window)
    validation_cache = prepared["validation_cache"]
    oos_base = prepared["oos_base"]
    oos_y_true = prepared["oos_y_true"]
    oos_raw_prob = prepared["oos_raw_prob"]
    oos_smoothed = prepared["oos_smoothed"]

    version_results: List[Dict[str, object]] = []
    dynamic_candidates: List[Dict[str, object]] = []

    for rising_floor in RISING_ENTRY_FLOOR_GRID:
        selected_smoothing, selected_threshold, best_validation = choose_parameters_for_floor(validation_cache, rising_floor)
        positions, zones, _ = rising_entry_positions(
            oos_smoothed[selected_smoothing],
            initial_position=0,
            lower_threshold=float(selected_threshold["selected_lower_threshold"]),
            upper_threshold=float(selected_threshold["selected_upper_threshold"]),
            rising_floor=float(rising_floor),
        )
        simulation = base.simulate_positions_strategy(oos_base, positions, initial_position=0)
        frame = simulation["frame"]
        frame["signal_zone"] = zones
        frame["rebalance_date"] = window.rebalance_date
        frame["selected_smoothing_halflife"] = int(selected_smoothing)
        frame["selected_lower_threshold"] = float(selected_threshold["selected_lower_threshold"])
        frame["selected_upper_threshold"] = float(selected_threshold["selected_upper_threshold"])
        frame["selected_rising_entry_floor"] = float(rising_floor)
        frame["predicted_probability_raw"] = np.asarray(oos_raw_prob, dtype=float)
        frame["predicted_probability_smoothed"] = np.asarray(oos_smoothed[selected_smoothing], dtype=float)
        frame["y_true"] = np.asarray(oos_y_true, dtype=int)
        oos_metrics = base.compute_metrics(
            np.asarray(oos_y_true, dtype=int),
            np.asarray(positions, dtype=int),
            np.asarray(oos_smoothed[selected_smoothing], dtype=float),
        )
        version_name = FIXED_VERSION_NAMES[float(rising_floor)]
        record = {
            "version": version_name,
            "rising_floor": float(rising_floor),
            "selected_smoothing_halflife": int(selected_smoothing),
            "selected_lower_threshold": float(selected_threshold["selected_lower_threshold"]),
            "selected_upper_threshold": float(selected_threshold["selected_upper_threshold"]),
            "validation_sharpe": float(best_validation["sharpe"]),
            "validation_balanced_accuracy": float(best_validation["metrics"]["balanced_accuracy"]),
            "validation_accuracy": float(best_validation["metrics"]["accuracy"]),
            "validation_f1": float(best_validation["metrics"]["f1"]),
            "validation_log_loss": float(best_validation["metrics"]["log_loss"]),
            "oos_sharpe": float(simulation["sharpe"]),
            "oos_metrics": {
                "accuracy": float(oos_metrics["accuracy"]),
                "balanced_accuracy": float(oos_metrics["balanced_accuracy"]),
                "f1": float(oos_metrics["f1"]),
                "log_loss": float(oos_metrics["log_loss"]),
            },
            "oos_frame_records": frame_to_records(frame),
        }
        version_results.append(record)
        dynamic_candidates.append(record)

    dynamic_candidates = sorted(
        dynamic_candidates,
        key=lambda row: (
            -row["validation_sharpe"],
            -row["validation_balanced_accuracy"],
            abs(row["rising_floor"] - 0.55),
            row["rising_floor"],
            row["version"],
        ),
    )
    dynamic_choice = dynamic_candidates[0].copy()
    dynamic_choice["version"] = DYNAMIC_VERSION
    version_results.append(dynamic_choice)

    return {
        "window": window_payload,
        "version_results": version_results,
    }


def build_final_frame(window_results: List[Dict[str, object]], version: str) -> pd.DataFrame:
    safe_payloads = []
    for result in window_results:
        version_row = next(row for row in result["version_results"] if row["version"] == version)
        frame = pd.DataFrame(version_row["oos_frame_records"])
        frame["Date"] = pd.to_datetime(frame["Date"])
        safe_payloads.append({"window": result["window"], "oos_frame": frame})
    final_frame = base.stitch_final_frame(safe_payloads)
    final_frame["equity_predicted_strategy"] = (1.0 + final_frame["strategy_ret"]).cumprod()
    return final_frame


def plot_equity_curves(daily_equity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_versions = [BASELINE_VERSION, *FIXED_VERSION_NAMES.values(), DYNAMIC_VERSION, BUYHOLD_VERSION]
    for version in plot_versions:
        if version == BUYHOLD_VERSION:
            ax.plot(daily_equity["Date"], daily_equity["equity_buy_and_hold"], label=version, linewidth=1.7)
        else:
            ax.plot(daily_equity["Date"], daily_equity[f"equity_{version}"], label=version, linewidth=1.3)
    ax.set_title("SPY Recovery Features Rising-Entry Grid")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_bar(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(summary["version"], summary[metric], color="#4e79a7")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_choice_count(selection_log: pd.DataFrame, output_path: Path) -> None:
    count_df = selection_log.groupby("selected_rising_entry_floor").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(count_df["selected_rising_entry_floor"].astype(str), count_df["count"], color="#f28e2b")
    ax.set_title("Dynamic Rising-Entry Floor Choice Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    base.ensure_inputs()

    source_dir = source_baseline_dir()
    baseline_summary = pd.read_csv(source_dir / "strategy_performance_summary.csv")
    baseline_prediction = pd.read_csv(source_dir / "prediction_metrics_comparison.csv")
    baseline_rolling = pd.read_csv(source_dir / "rolling_window_log.csv")
    baseline_daily = pd.read_csv(source_dir / "daily_equity_curves.csv")
    baseline_daily["Date"] = pd.to_datetime(baseline_daily["Date"], errors="coerce")

    baseline_summary = baseline_summary.loc[baseline_summary["version"] == "predicted_strategy"].copy()
    baseline_summary["version"] = BASELINE_VERSION
    baseline_prediction["version"] = BASELINE_VERSION
    baseline_rolling["version"] = BASELINE_VERSION
    baseline_daily = baseline_daily.rename(columns={"equity_predicted_strategy": f"equity_{BASELINE_VERSION}"})

    frame = load_recovery_experiment_frame()
    main_windows = base.build_target_windows(frame)
    worker_count = min(base.MAX_WORKERS_CAP, len(main_windows), os.cpu_count() or 1)
    tasks = [window.__dict__ for window in main_windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_process_window, tasks))

    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])

    version_order = [*FIXED_VERSION_NAMES.values(), DYNAMIC_VERSION]
    final_frames: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    rolling_rows: List[Dict[str, object]] = []
    dynamic_selection_rows: List[Dict[str, object]] = []

    for version in version_order:
        final_frame = build_final_frame(window_results, version)
        final_frames[version] = final_frame
        summary_rows.append({"version": version, **base.build_strategy_metrics(final_frame)})

        version_window_rows = []
        for result in window_results:
            version_row = next(row for row in result["version_results"] if row["version"] == version)
            version_window_rows.append(version_row)
            window_slice = final_frame.loc[final_frame["rebalance_date"] == result["window"]["rebalance_date"]]
            y_true = np.asarray(window_slice["y_true"], dtype=int)
            y_pred = np.asarray(window_slice["position"], dtype=int)
            rolling_rows.append(
                {
                    "version": version,
                    "rebalance_date": result["window"]["rebalance_date"],
                    "selected_smoothing_halflife": int(version_row["selected_smoothing_halflife"]),
                    "selected_lower_threshold": float(version_row["selected_lower_threshold"]),
                    "selected_upper_threshold": float(version_row["selected_upper_threshold"]),
                    "selected_rising_entry_floor": float(version_row["rising_floor"]),
                    "oos_sharpe": base.annualized_sharpe(window_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
                    "oos_accuracy": float(accuracy_score(y_true, y_pred)),
                    "oos_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "oos_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                }
            )
            if version == DYNAMIC_VERSION:
                dynamic_selection_rows.append(
                    {
                        "rebalance_date": result["window"]["rebalance_date"],
                        "selected_smoothing_halflife": int(version_row["selected_smoothing_halflife"]),
                        "selected_lower_threshold": float(version_row["selected_lower_threshold"]),
                        "selected_upper_threshold": float(version_row["selected_upper_threshold"]),
                        "selected_rising_entry_floor": float(version_row["rising_floor"]),
                        "validation_sharpe": float(version_row["validation_sharpe"]),
                        "validation_balanced_accuracy": float(version_row["validation_balanced_accuracy"]),
                        "oos_sharpe": float(version_row["oos_sharpe"]),
                    }
                )

        prediction_rows.append(
            {
                "version": version,
                "avg_validation_accuracy": float(np.mean([row["validation_accuracy"] for row in version_window_rows])),
                "avg_validation_balanced_accuracy": float(np.mean([row["validation_balanced_accuracy"] for row in version_window_rows])),
                "avg_validation_f1": float(np.mean([row["validation_f1"] for row in version_window_rows])),
                "avg_validation_log_loss": float(np.mean([row["validation_log_loss"] for row in version_window_rows])),
                "avg_oos_accuracy": float(np.mean([row["oos_metrics"]["accuracy"] for row in version_window_rows])),
                "avg_oos_balanced_accuracy": float(np.mean([row["oos_metrics"]["balanced_accuracy"] for row in version_window_rows])),
                "avg_oos_f1": float(np.mean([row["oos_metrics"]["f1"] for row in version_window_rows])),
                "avg_oos_log_loss": float(np.mean([row["oos_metrics"]["log_loss"] for row in version_window_rows])),
            }
        )

    buyhold = base.build_buyhold_frame(pd.DataFrame({"Date": final_frames[DYNAMIC_VERSION]["Date"].drop_duplicates().sort_values().to_list()}))
    buyhold = buyhold.merge(final_frames[DYNAMIC_VERSION][["Date"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    strategy_summary = pd.concat(
        [
            baseline_summary,
            pd.DataFrame(summary_rows),
            pd.DataFrame([{"version": BUYHOLD_VERSION, **base.build_strategy_metrics(buyhold)}]),
        ],
        ignore_index=True,
    )
    prediction_comparison = pd.concat([baseline_prediction, pd.DataFrame(prediction_rows)], ignore_index=True)
    rolling_comparison = pd.concat([baseline_rolling, pd.DataFrame(rolling_rows)], ignore_index=True)
    dynamic_selection_log = pd.DataFrame(dynamic_selection_rows)

    daily_equity = baseline_daily.copy()
    for version, final_frame in final_frames.items():
        daily_equity = daily_equity.merge(
            final_frame[["Date", "equity_predicted_strategy"]].rename(columns={"equity_predicted_strategy": f"equity_{version}"}),
            on="Date",
            how="inner",
        )

    out_dir = results_dir()
    strategy_summary.to_csv(out_dir / "strategy_performance_comparison.csv", index=False)
    prediction_comparison.to_csv(out_dir / "prediction_metrics_comparison.csv", index=False)
    rolling_comparison.sort_values(["version", "rebalance_date"]).reset_index(drop=True).to_csv(
        out_dir / "rolling_window_metrics_comparison.csv", index=False
    )
    dynamic_selection_log.to_csv(out_dir / "dynamic_rising_entry_selection_log.csv", index=False)
    daily_equity.to_csv(out_dir / "daily_equity_curves_comparison.csv", index=False)

    plot_equity_curves(daily_equity, out_dir / "rising_entry_grid_equity_curves.png")
    plot_metric_bar(strategy_summary, "sharpe", "Rising Entry Grid Sharpe Comparison", out_dir / "rising_entry_grid_sharpe_comparison.png")
    plot_metric_bar(strategy_summary, "max_drawdown", "Rising Entry Grid Drawdown Comparison", out_dir / "rising_entry_grid_drawdown_comparison.png")
    plot_choice_count(dynamic_selection_log, out_dir / "dynamic_rising_entry_choice_count.png")

    print(f"Results directory: {out_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start_time:.2f}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
