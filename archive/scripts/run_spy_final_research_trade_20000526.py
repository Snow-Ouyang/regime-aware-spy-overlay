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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

import run_spy_vs_gspc_signal_mapping as signal_base
from final_multi_asset_project_common import FIXED_XGB_PARAMS, load_risk_free_daily_series, load_trade_price_frame
from spy_regime_common import (
    FEE_BPS,
    MainWindow,
    annualized_sharpe,
    build_buy_and_hold,
    build_segment_supervised_bundle,
    build_segments,
    build_strategy_metrics,
    first_date_on_or_after,
    last_date_before,
    simulate_next_day_strategy,
)


SPY_STEM = "spy"
TRADE_STEM = "spy_trade"
TARGET_OOS_START = pd.Timestamp("2000-05-26")
VALIDATION_YEARS = 4
TRAIN_YEARS = 11
STEP_MONTHS = 6
JUMP_PENALTY = 0.0
SMOOTHING_GRID: List[int] = [0, 4, 8, 12]
LOWER_THRESHOLD_GRID: List[float] = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
UPPER_THRESHOLD_GRID: List[float] = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70]
RESULTS_DIR_NAME = "final_spy_research_trade_20000526"
MAX_WORKERS_CAP = 8


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    output_dir = project_root() / "results" / RESULTS_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_inputs() -> None:
    signal_base.ensure_inputs()
    trade_path = project_root() / "data_raw" / "spy_trade.csv"
    if not trade_path.exists():
        raise FileNotFoundError("Missing SPY trade raw data")


def make_xgb_model() -> XGBClassifier:
    return XGBClassifier(**FIXED_XGB_PARAMS)


def build_target_windows(frame: pd.DataFrame) -> List[MainWindow]:
    dates = frame["Date"].sort_values().reset_index(drop=True)
    first_anchor = first_date_on_or_after(dates, TARGET_OOS_START)
    if first_anchor is None or first_anchor != TARGET_OOS_START:
        raise ValueError(f"First OOS anchor must be {TARGET_OOS_START.strftime('%Y-%m-%d')}, got {first_anchor}")

    windows: List[MainWindow] = []
    current_anchor = first_anchor
    latest_date = dates.iloc[-1]
    while current_anchor is not None and current_anchor <= latest_date:
        next_anchor_target = current_anchor + pd.DateOffset(months=STEP_MONTHS)
        next_anchor = first_date_on_or_after(dates, next_anchor_target)
        oos_end = latest_date if next_anchor is None else last_date_before(dates, next_anchor)
        if oos_end is None or oos_end < current_anchor:
            current_anchor = next_anchor
            continue

        val_start = first_date_on_or_after(dates, current_anchor - pd.DateOffset(years=VALIDATION_YEARS))
        train_start = first_date_on_or_after(dates, current_anchor - pd.DateOffset(years=VALIDATION_YEARS + TRAIN_YEARS))
        train_end = last_date_before(dates, val_start) if val_start is not None else None
        val_end = last_date_before(dates, current_anchor)
        if None in [val_start, train_start, train_end, val_end]:
            current_anchor = next_anchor
            continue

        windows.append(
            MainWindow(
                rebalance_date=current_anchor.strftime("%Y-%m-%d"),
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
                oos_start=current_anchor.strftime("%Y-%m-%d"),
                oos_end=oos_end.strftime("%Y-%m-%d"),
            )
        )
        current_anchor = next_anchor
    return windows


def build_nonempty_segments(frame: pd.DataFrame, start_date: pd.Timestamp, end_exclusive: pd.Timestamp) -> List[Dict[str, pd.Timestamp]]:
    segments = build_segments(start_date, end_exclusive)
    valid_segments: List[Dict[str, pd.Timestamp]] = []
    for segment in segments:
        mask = (frame["Date"] >= segment["start"]) & (frame["Date"] < segment["end_exclusive"])
        if int(mask.sum()) > 0:
            valid_segments.append(segment)
    return valid_segments


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def double_threshold_positions(
    probabilities: np.ndarray,
    initial_position: int,
    lower_threshold: float,
    upper_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    prev = int(initial_position)
    for idx, prob in enumerate(probs):
        if prob >= upper_threshold:
            current = 1
            zone = "bull"
        elif prob <= lower_threshold:
            current = 0
            zone = "bear"
        else:
            current = prev
            zone = "hold"
        positions[idx] = current
        zones[idx] = zone
        prev = current
    return positions, zones, prev


def simulate_positions_strategy(dataset_base: pd.DataFrame, positions: np.ndarray, initial_position: int) -> Dict[str, object]:
    base = dataset_base.reset_index(drop=True).copy()
    position = np.asarray(positions, dtype=int)
    previous_position = np.empty_like(position)
    previous_position[0] = int(initial_position)
    if len(position) > 1:
        previous_position[1:] = position[:-1]

    fee_rate = FEE_BPS / 10000.0
    fee = np.where(position != previous_position, fee_rate, 0.0)
    next_ret = base["next_ret"].to_numpy(dtype=float, copy=False)
    next_rf = base["next_rf_daily"].to_numpy(dtype=float, copy=False)
    strategy_ret_gross = position * next_ret + (1 - position) * next_rf
    strategy_ret = strategy_ret_gross - fee
    strategy_excess_ret = strategy_ret - next_rf

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(base["execution_date"]),
            "position": position,
            "fee": fee,
            "strategy_ret_gross": strategy_ret_gross,
            "strategy_ret": strategy_ret,
            "strategy_excess_ret": strategy_excess_ret,
        }
    )
    return {
        "frame": frame,
        "final_position": int(position[-1]),
        "sharpe": annualized_sharpe(strategy_excess_ret),
    }


def score_validation_bundle(
    validation_bundles: List[Dict[str, object]],
    smoothing_halflife: int,
    lower_threshold: float,
    upper_threshold: float,
) -> Dict[str, object]:
    current_position = 0
    frames: List[pd.DataFrame] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for bundle in validation_bundles:
        model = make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1]
        smoothed_prob = signal_base.smooth_probability_series(raw_prob, smoothing_halflife)
        segment_initial_position = current_position
        positions, _, current_position = double_threshold_positions(
            smoothed_prob,
            segment_initial_position,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        simulation = simulate_positions_strategy(bundle["segment_dataset"]["base"], positions, segment_initial_position)
        frames.append(simulation["frame"])
        y_true_list.append(bundle["segment_dataset"]["y"])
        y_pred_list.append(positions)
        y_prob_list.append(smoothed_prob)

    validation_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {
        "frame": validation_frame,
        "sharpe": annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
        "metrics": compute_metrics(np.concatenate(y_true_list), np.concatenate(y_pred_list), np.concatenate(y_prob_list)),
    }


def select_best_smoothing(validation_bundles: List[Dict[str, object]]) -> Tuple[int, pd.DataFrame]:
    candidate_rows: List[Dict[str, object]] = []
    for smoothing_halflife in SMOOTHING_GRID:
        result = score_validation_bundle(validation_bundles, smoothing_halflife, lower_threshold=0.45, upper_threshold=0.60)
        candidate_rows.append(
            {
                "selected_smoothing_halflife": int(smoothing_halflife),
                "validation_sharpe": float(result["sharpe"]),
                "validation_balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
                "version_id": f"h{smoothing_halflife}",
            }
        )
    frame = pd.DataFrame(candidate_rows).sort_values(
        by=["validation_sharpe", "validation_balanced_accuracy", "selected_smoothing_halflife", "version_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return int(frame.iloc[0]["selected_smoothing_halflife"]), frame


def select_best_double_threshold(validation_bundles: List[Dict[str, object]], smoothing_halflife: int) -> Tuple[Dict[str, object], pd.DataFrame]:
    candidate_rows: List[Dict[str, object]] = []
    for lower_threshold in LOWER_THRESHOLD_GRID:
        for upper_threshold in UPPER_THRESHOLD_GRID:
            if upper_threshold <= lower_threshold:
                continue
            result = score_validation_bundle(validation_bundles, smoothing_halflife, lower_threshold, upper_threshold)
            candidate_rows.append(
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

    frame = pd.DataFrame(candidate_rows).copy()
    frame["lower_distance"] = (frame["selected_lower_threshold"] - 0.48).abs()
    frame["upper_distance"] = (frame["selected_upper_threshold"] - 0.67).abs()
    frame = frame.sort_values(
        by=[
            "validation_sharpe",
            "validation_balanced_accuracy",
            "lower_distance",
            "upper_distance",
            "version_id",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return frame.iloc[0].to_dict(), frame


def prepare_window(window: MainWindow) -> Dict[str, object]:
    frame = signal_base.load_signal_experiment_frame(SPY_STEM)
    features = signal_base.research_feature_columns(frame)
    validation_segments = build_nonempty_segments(frame, pd.Timestamp(window.val_start), pd.Timestamp(window.oos_start))

    validation_bundles: List[Dict[str, object]] = []
    for segment in validation_segments:
        bundle = build_segment_supervised_bundle(
            frame,
            features,
            segment["start"],
            segment["end_exclusive"],
            penalty=JUMP_PENALTY,
        )
        validation_bundles.append(
            {
                "train_dataset": bundle["train_dataset"],
                "segment_dataset": bundle["segment_dataset"],
                "initial_position": 0,
                "base": bundle["segment_dataset"]["base"].reset_index(drop=True),
            }
        )

    oos_bundle = build_segment_supervised_bundle(
        frame,
        features,
        pd.Timestamp(window.oos_start),
        pd.Timestamp(window.oos_end) + pd.Timedelta(days=1),
        penalty=JUMP_PENALTY,
    )
    oos_model = make_xgb_model()
    oos_model.fit(oos_bundle["train_dataset"]["X"], oos_bundle["train_dataset"]["y"])
    oos_raw_prob = oos_model.predict_proba(oos_bundle["segment_dataset"]["X"])[:, 1].astype(float)
    oos_smoothed = {h: signal_base.smooth_probability_series(oos_raw_prob, h) for h in SMOOTHING_GRID}

    return {
        "window": window,
        "validation_bundles": validation_bundles,
        "oos_base": oos_bundle["segment_dataset"]["base"].reset_index(drop=True),
        "oos_y_true": oos_bundle["segment_dataset"]["y"].astype(int).copy(),
        "oos_raw_prob": oos_raw_prob,
        "oos_smoothed": oos_smoothed,
    }


def worker_process_window(window_payload: Dict[str, str]) -> Dict[str, object]:
    window = MainWindow(**window_payload)
    prepared = prepare_window(window)
    validation_bundles = prepared["validation_bundles"]
    oos_base = prepared["oos_base"]
    oos_y_true = prepared["oos_y_true"]
    oos_raw_prob = prepared["oos_raw_prob"]
    oos_smoothed = prepared["oos_smoothed"]

    selected_smoothing, smoothing_candidates = select_best_smoothing(validation_bundles)
    selected_double, double_candidates = select_best_double_threshold(validation_bundles, selected_smoothing)

    positions, zones, _ = double_threshold_positions(
        oos_smoothed[selected_smoothing],
        initial_position=0,
        lower_threshold=float(selected_double["selected_lower_threshold"]),
        upper_threshold=float(selected_double["selected_upper_threshold"]),
    )
    simulation = simulate_positions_strategy(oos_base, positions, initial_position=0)
    frame = simulation["frame"]
    frame["signal_zone"] = zones
    frame["rebalance_date"] = window.rebalance_date
    frame["selected_smoothing_halflife"] = int(selected_smoothing)
    frame["selected_lower_threshold"] = float(selected_double["selected_lower_threshold"])
    frame["selected_upper_threshold"] = float(selected_double["selected_upper_threshold"])
    frame["predicted_probability_raw"] = np.asarray(oos_raw_prob, dtype=float)
    frame["predicted_probability_smoothed"] = np.asarray(oos_smoothed[selected_smoothing], dtype=float)
    frame["y_true"] = np.asarray(oos_y_true, dtype=int)

    validation_result = score_validation_bundle(
        validation_bundles,
        int(selected_smoothing),
        float(selected_double["selected_lower_threshold"]),
        float(selected_double["selected_upper_threshold"]),
    )
    oos_metrics = compute_metrics(np.asarray(oos_y_true, dtype=int), np.asarray(positions, dtype=int), np.asarray(oos_smoothed[selected_smoothing], dtype=float))

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
        "double_candidate_rows": double_candidates,
        "smoothing_candidate_rows": smoothing_candidates,
        "oos_metrics": oos_metrics,
        "oos_sharpe": float(simulation["sharpe"]),
        "oos_frame": frame,
    }


def stitch_final_frame(payloads: List[Dict[str, object]]) -> pd.DataFrame:
    sorted_payloads = sorted(payloads, key=lambda item: item["window"]["rebalance_date"])
    frames: List[pd.DataFrame] = []
    previous_position = 0
    for payload in sorted_payloads:
        frame = payload["oos_frame"].copy()
        frame["position"] = frame["position"].astype(int)
        frame["rebalance_date"] = payload["window"]["rebalance_date"]
        previous_position = int(frame["position"].iloc[-1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def build_bear_intervals(frame: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    bear_mask = frame["signal_zone"] == "bear"
    dates = pd.to_datetime(frame["Date"])
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev = None
    for date, is_bear in zip(dates, bear_mask):
        if is_bear and start is None:
            start = date
        if start is not None and (not is_bear):
            intervals.append((start, prev if prev is not None else date))
            start = None
        prev = date
    if start is not None and prev is not None:
        intervals.append((start, prev))
    return intervals


def build_buyhold_frame(signal_dates: pd.DataFrame) -> pd.DataFrame:
    trade_frame = load_trade_price_frame(TRADE_STEM)
    trade_frame["Date"] = pd.to_datetime(trade_frame["Date"], errors="coerce")
    trade_frame = trade_frame.sort_values("Date").reset_index(drop=True)
    trade_frame["ret"] = trade_frame["Adj_Close"].pct_change()
    rf_frame = load_risk_free_daily_series()
    rf_frame["Date"] = pd.to_datetime(rf_frame["Date"], errors="coerce")
    merged = trade_frame.merge(rf_frame, on="Date", how="left")
    merged["rf_daily"] = pd.to_numeric(merged["rf_daily"], errors="coerce").ffill()
    merged = merged.merge(signal_dates, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    merged = merged.dropna(subset=["ret", "rf_daily"]).reset_index(drop=True)
    buyhold = build_buy_and_hold(merged, TARGET_OOS_START)
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()
    return buyhold


def plot_equity_curves(daily_equity: pd.DataFrame, signal_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(daily_equity["Date"], daily_equity["equity_predicted_strategy"], label="predicted_strategy", linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_buy_and_hold"], label="buy_and_hold", linewidth=1.8)
    for start, end in build_bear_intervals(signal_frame):
        ax.axvspan(start, end, color="red", alpha=0.12)
    ax.set_title("SPY Predicted Strategy vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    ensure_inputs()

    frame = signal_base.load_signal_experiment_frame(SPY_STEM)
    main_windows = build_target_windows(frame)
    worker_count = min(MAX_WORKERS_CAP, len(main_windows), os.cpu_count() or 1)
    tasks = [window.__dict__ for window in main_windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_process_window, tasks))

    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])
    final_frame = stitch_final_frame(window_results)
    final_frame["equity_predicted_strategy"] = (1.0 + final_frame["strategy_ret"]).cumprod()

    buyhold = build_buyhold_frame(pd.DataFrame({"Date": final_frame["Date"].drop_duplicates().sort_values().to_list()}))
    buyhold = buyhold.merge(final_frame[["Date"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    daily_equity = pd.DataFrame(
        {
            "Date": buyhold["Date"],
            "equity_predicted_strategy": final_frame.set_index("Date").loc[buyhold["Date"], "equity_predicted_strategy"].to_numpy(),
            "equity_buy_and_hold": buyhold["equity_buy_and_hold"].to_numpy(),
        }
    )

    strategy_summary = pd.DataFrame(
        [
            {"version": "predicted_strategy", **build_strategy_metrics(final_frame)},
            {"version": "buy_and_hold", **build_strategy_metrics(buyhold)},
        ]
    )
    prediction_metrics = pd.DataFrame(
        [
            {
                "version": "predicted_strategy",
                "avg_validation_accuracy": float(np.mean([row["validation_accuracy"] for row in window_results])),
                "avg_validation_balanced_accuracy": float(np.mean([row["validation_balanced_accuracy"] for row in window_results])),
                "avg_validation_f1": float(np.mean([row["validation_f1"] for row in window_results])),
                "avg_validation_log_loss": float(np.mean([row["validation_log_loss"] for row in window_results])),
                "avg_oos_accuracy": float(np.mean([row["oos_metrics"]["accuracy"] for row in window_results])),
                "avg_oos_balanced_accuracy": float(np.mean([row["oos_metrics"]["balanced_accuracy"] for row in window_results])),
                "avg_oos_f1": float(np.mean([row["oos_metrics"]["f1"] for row in window_results])),
                "avg_oos_log_loss": float(np.mean([row["oos_metrics"]["log_loss"] for row in window_results])),
            }
        ]
    )

    rolling_rows = []
    selection_rows = []
    for result in window_results:
        selection_rows.append(
            {
                "rebalance_date": result["window"]["rebalance_date"],
                "selected_smoothing_halflife": int(result["selected_smoothing_halflife"]),
                "selected_lower_threshold": float(result["selected_lower_threshold"]),
                "selected_upper_threshold": float(result["selected_upper_threshold"]),
                "validation_sharpe": float(result["validation_sharpe"]),
                "validation_balanced_accuracy": float(result["validation_balanced_accuracy"]),
                "oos_sharpe": float(result["oos_sharpe"]),
            }
        )
        window_slice = final_frame.loc[final_frame["rebalance_date"] == result["window"]["rebalance_date"]]
        y_true = np.asarray(window_slice["y_true"], dtype=int)
        y_pred = np.asarray(window_slice["position"], dtype=int)
        rolling_rows.append(
            {
                "rebalance_date": result["window"]["rebalance_date"],
                "selected_smoothing_halflife": int(result["selected_smoothing_halflife"]),
                "selected_lower_threshold": float(result["selected_lower_threshold"]),
                "selected_upper_threshold": float(result["selected_upper_threshold"]),
                "oos_sharpe": annualized_sharpe(window_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
                "oos_accuracy": float(accuracy_score(y_true, y_pred)),
                "oos_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "oos_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    final_frame["predicted_label"] = final_frame["position"].astype(int)
    final_frame["predicted_bull_flag"] = final_frame["predicted_label"]
    final_frame["predicted_bear_flag"] = 1 - final_frame["predicted_label"]
    final_frame["predicted_probability_raw"] = final_frame["predicted_probability_raw"].astype(float)
    final_frame["predicted_probability_smoothed"] = final_frame["predicted_probability_smoothed"].astype(float)

    final_frame.to_csv(results_dir() / "predicted_strategy_daily_equity_curves.csv", index=False)
    final_frame[[
        "Date",
        "predicted_probability_raw",
        "predicted_probability_smoothed",
        "predicted_label",
        "predicted_bull_flag",
        "predicted_bear_flag",
        "selected_smoothing_halflife",
        "selected_lower_threshold",
        "selected_upper_threshold",
    ]].to_csv(results_dir() / "predicted_labels.csv", index=False)
    strategy_summary.to_csv(results_dir() / "strategy_performance_summary.csv", index=False)
    prediction_metrics.to_csv(results_dir() / "prediction_metrics_comparison.csv", index=False)
    pd.DataFrame(rolling_rows).sort_values("rebalance_date").reset_index(drop=True).to_csv(
        results_dir() / "rolling_window_log.csv", index=False
    )
    pd.DataFrame(selection_rows).sort_values("rebalance_date").reset_index(drop=True).to_csv(
        results_dir() / "selection_log.csv", index=False
    )
    daily_equity.to_csv(results_dir() / "daily_equity_curves.csv", index=False)

    plot_equity_curves(daily_equity, final_frame, results_dir() / "predicted_vs_buyhold_with_bear_shading.png")

    print(f"Results directory: {results_dir()}")
    print(f"Elapsed seconds: {time.perf_counter() - start_time:.2f}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
