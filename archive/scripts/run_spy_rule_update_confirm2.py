import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import run_spy_final_research_trade_20000526 as base


RESULTS_DIR_NAME = "spy_rule_update_confirm2"
BASELINE_VERSION = "current_baseline_rule"
NEW_RULE_VERSION = "raw_entry_confirm2_exit_rule"


def results_dir() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "results" / RESULTS_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_python_scalar(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def frame_to_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    records = frame.to_dict(orient="records")
    cleaned: List[Dict[str, object]] = []
    for row in records:
        cleaned_row: Dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                cleaned_row[key] = value.strftime("%Y-%m-%d")
            else:
                cleaned_row[key] = _to_python_scalar(value)
        cleaned.append(cleaned_row)
    return cleaned


def confirm2_positions(
    raw_probabilities: np.ndarray,
    smoothed_probabilities: np.ndarray,
    initial_position: int,
    lower_threshold: float,
    upper_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    raw_probs = np.asarray(raw_probabilities, dtype=float)
    smoothed_probs = np.asarray(smoothed_probabilities, dtype=float)
    positions = np.empty(len(raw_probs), dtype=int)
    prev_positions = np.empty(len(raw_probs), dtype=int)
    entry_triggers = np.zeros(len(raw_probs), dtype=int)
    exit_triggers = np.zeros(len(raw_probs), dtype=int)

    prev = int(initial_position)
    for idx in range(len(raw_probs)):
        prev_positions[idx] = prev
        entry_trigger = 0
        exit_trigger = 0
        if prev == 0:
            if raw_probs[idx] > upper_threshold:
                current = 1
                entry_trigger = 1
            else:
                current = 0
        else:
            if idx > 0 and smoothed_probs[idx] < lower_threshold and smoothed_probs[idx - 1] < lower_threshold:
                current = 0
                exit_trigger = 1
            else:
                current = 1

        positions[idx] = current
        entry_triggers[idx] = entry_trigger
        exit_triggers[idx] = exit_trigger
        prev = current

    return positions, prev_positions, entry_triggers, exit_triggers, prev


def score_validation_bundle_confirm2(
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
        model = base.make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1]
        smoothed_prob = base.signal_base.smooth_probability_series(raw_prob, smoothing_halflife)
        segment_initial_position = current_position
        positions, _, _, _, current_position = confirm2_positions(
            raw_prob,
            smoothed_prob,
            segment_initial_position,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        simulation = base.simulate_positions_strategy(bundle["segment_dataset"]["base"], positions, segment_initial_position)
        frames.append(simulation["frame"])
        y_true_list.append(bundle["segment_dataset"]["y"])
        y_pred_list.append(positions)
        y_prob_list.append(smoothed_prob)

    validation_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {
        "frame": validation_frame,
        "sharpe": base.annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
        "metrics": base.compute_metrics(np.concatenate(y_true_list), np.concatenate(y_pred_list), np.concatenate(y_prob_list)),
    }


def select_best_double_threshold_confirm2(
    validation_bundles: List[Dict[str, object]],
    smoothing_halflife: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    candidate_rows: List[Dict[str, object]] = []
    for lower_threshold in base.LOWER_THRESHOLD_GRID:
        for upper_threshold in base.UPPER_THRESHOLD_GRID:
            if upper_threshold <= lower_threshold:
                continue
            result = score_validation_bundle_confirm2(validation_bundles, smoothing_halflife, lower_threshold, upper_threshold)
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


def precompute_validation_predictions(validation_bundles: List[Dict[str, object]]) -> List[Dict[str, object]]:
    outputs: List[Dict[str, object]] = []
    for bundle in validation_bundles:
        model = base.make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1].astype(float)
        smoothed_map = {
            halflife: base.signal_base.smooth_probability_series(raw_prob, halflife)
            for halflife in base.SMOOTHING_GRID
        }
        outputs.append(
            {
                "base": bundle["segment_dataset"]["base"].reset_index(drop=True),
                "y": np.asarray(bundle["segment_dataset"]["y"], dtype=int),
                "raw_prob": raw_prob,
                "smoothed_map": smoothed_map,
            }
        )
    return outputs


def score_precomputed_validation_bundle(
    precomputed_bundles: List[Dict[str, object]],
    smoothing_halflife: int,
    lower_threshold: float,
    upper_threshold: float,
    rule_name: str,
) -> Dict[str, object]:
    current_position = 0
    frames: List[pd.DataFrame] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for bundle in precomputed_bundles:
        segment_initial_position = current_position
        smoothed_prob = bundle["smoothed_map"][smoothing_halflife]
        if rule_name == "baseline":
            positions, _, current_position = base.double_threshold_positions(
                smoothed_prob,
                segment_initial_position,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
        else:
            positions, _, _, _, current_position = confirm2_positions(
                bundle["raw_prob"],
                smoothed_prob,
                segment_initial_position,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
        simulation = base.simulate_positions_strategy(bundle["base"], positions, segment_initial_position)
        frames.append(simulation["frame"])
        y_true_list.append(bundle["y"])
        y_pred_list.append(positions)
        y_prob_list.append(smoothed_prob)

    validation_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {
        "frame": validation_frame,
        "sharpe": base.annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
        "metrics": base.compute_metrics(np.concatenate(y_true_list), np.concatenate(y_pred_list), np.concatenate(y_prob_list)),
    }


def select_best_smoothing_precomputed(precomputed_bundles: List[Dict[str, object]]) -> int:
    candidate_rows: List[Dict[str, object]] = []
    for smoothing_halflife in base.SMOOTHING_GRID:
        result = score_precomputed_validation_bundle(
            precomputed_bundles,
            smoothing_halflife,
            lower_threshold=0.45,
            upper_threshold=0.60,
            rule_name="baseline",
        )
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
    return int(frame.iloc[0]["selected_smoothing_halflife"])


def select_best_double_threshold_precomputed(
    precomputed_bundles: List[Dict[str, object]],
    smoothing_halflife: int,
    rule_name: str,
) -> Dict[str, object]:
    candidate_rows: List[Dict[str, object]] = []
    for lower_threshold in base.LOWER_THRESHOLD_GRID:
        for upper_threshold in base.UPPER_THRESHOLD_GRID:
            if upper_threshold <= lower_threshold:
                continue
            result = score_precomputed_validation_bundle(
                precomputed_bundles,
                smoothing_halflife,
                lower_threshold,
                upper_threshold,
                rule_name=rule_name,
            )
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
    return frame.iloc[0].to_dict()


def worker_process_window_confirm2(window_payload: Dict[str, str]) -> Dict[str, object]:
    window = base.MainWindow(**window_payload)
    prepared = base.prepare_window(window)
    validation_bundles = prepared["validation_bundles"]
    oos_base = prepared["oos_base"]
    oos_y_true = prepared["oos_y_true"]
    oos_raw_prob = prepared["oos_raw_prob"]
    oos_smoothed = prepared["oos_smoothed"]

    selected_smoothing, _ = base.select_best_smoothing(validation_bundles)
    selected_double, _ = select_best_double_threshold_confirm2(validation_bundles, selected_smoothing)

    positions, prev_positions, entry_triggers, exit_triggers, _ = confirm2_positions(
        oos_raw_prob,
        oos_smoothed[selected_smoothing],
        initial_position=0,
        lower_threshold=float(selected_double["selected_lower_threshold"]),
        upper_threshold=float(selected_double["selected_upper_threshold"]),
    )
    simulation = base.simulate_positions_strategy(oos_base, positions, initial_position=0)
    frame = simulation["frame"]
    frame["rebalance_date"] = window.rebalance_date
    frame["selected_smoothing_halflife"] = int(selected_smoothing)
    frame["selected_lower_threshold"] = float(selected_double["selected_lower_threshold"])
    frame["selected_upper_threshold"] = float(selected_double["selected_upper_threshold"])
    frame["predicted_probability_raw"] = np.asarray(oos_raw_prob, dtype=float)
    frame["predicted_probability_smoothed"] = np.asarray(oos_smoothed[selected_smoothing], dtype=float)
    frame["y_true"] = np.asarray(oos_y_true, dtype=int)
    frame["prev_position"] = prev_positions.astype(int)
    frame["entry_trigger"] = entry_triggers.astype(int)
    frame["exit_trigger_confirm2"] = exit_triggers.astype(int)
    frame["signal_zone"] = np.where(entry_triggers == 1, "bull", np.where(exit_triggers == 1, "bear", "hold"))

    validation_result = score_validation_bundle_confirm2(
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
        "oos_metrics": oos_metrics,
        "oos_sharpe": float(simulation["sharpe"]),
        "oos_frame_records": frame_to_records(frame),
    }


def worker_process_window_baseline_safe(window_payload: Dict[str, str]) -> Dict[str, object]:
    result = base.worker_process_window(window_payload)
    return {
        "window": result["window"],
        "selected_smoothing_halflife": int(result["selected_smoothing_halflife"]),
        "selected_lower_threshold": float(result["selected_lower_threshold"]),
        "selected_upper_threshold": float(result["selected_upper_threshold"]),
        "validation_sharpe": float(result["validation_sharpe"]),
        "validation_balanced_accuracy": float(result["validation_balanced_accuracy"]),
        "validation_accuracy": float(result["validation_accuracy"]),
        "validation_f1": float(result["validation_f1"]),
        "validation_log_loss": float(result["validation_log_loss"]),
        "oos_metrics": {
            "accuracy": float(result["oos_metrics"]["accuracy"]),
            "balanced_accuracy": float(result["oos_metrics"]["balanced_accuracy"]),
            "f1": float(result["oos_metrics"]["f1"]),
            "log_loss": float(result["oos_metrics"]["log_loss"]),
        },
        "oos_sharpe": float(result["oos_sharpe"]),
        "oos_frame_records": frame_to_records(result["oos_frame"]),
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


def worker_process_window_compare_safe(window_payload: Dict[str, str]) -> Dict[str, object]:
    window = base.MainWindow(**window_payload)
    prepared = base.prepare_window(window)
    precomputed_bundles = precompute_validation_predictions(prepared["validation_bundles"])

    selected_smoothing = select_best_smoothing_precomputed(precomputed_bundles)
    baseline_selected = select_best_double_threshold_precomputed(precomputed_bundles, selected_smoothing, rule_name="baseline")
    confirm2_selected = select_best_double_threshold_precomputed(precomputed_bundles, selected_smoothing, rule_name="confirm2")

    oos_base = prepared["oos_base"]
    oos_y_true = np.asarray(prepared["oos_y_true"], dtype=int)
    oos_raw_prob = np.asarray(prepared["oos_raw_prob"], dtype=float)
    oos_smoothed = np.asarray(prepared["oos_smoothed"][selected_smoothing], dtype=float)

    baseline_positions, baseline_zones, _ = base.double_threshold_positions(
        oos_smoothed,
        initial_position=0,
        lower_threshold=float(baseline_selected["selected_lower_threshold"]),
        upper_threshold=float(baseline_selected["selected_upper_threshold"]),
    )
    baseline_sim = base.simulate_positions_strategy(oos_base, baseline_positions, initial_position=0)
    baseline_frame = baseline_sim["frame"]
    baseline_frame["signal_zone"] = baseline_zones
    baseline_frame["rebalance_date"] = window.rebalance_date
    baseline_frame["selected_smoothing_halflife"] = int(selected_smoothing)
    baseline_frame["selected_lower_threshold"] = float(baseline_selected["selected_lower_threshold"])
    baseline_frame["selected_upper_threshold"] = float(baseline_selected["selected_upper_threshold"])
    baseline_frame["predicted_probability_raw"] = oos_raw_prob
    baseline_frame["predicted_probability_smoothed"] = oos_smoothed
    baseline_frame["y_true"] = oos_y_true
    baseline_metrics = base.compute_metrics(oos_y_true, np.asarray(baseline_positions, dtype=int), oos_smoothed)

    confirm_positions, prev_positions, entry_triggers, exit_triggers, _ = confirm2_positions(
        oos_raw_prob,
        oos_smoothed,
        initial_position=0,
        lower_threshold=float(confirm2_selected["selected_lower_threshold"]),
        upper_threshold=float(confirm2_selected["selected_upper_threshold"]),
    )
    confirm_sim = base.simulate_positions_strategy(oos_base, confirm_positions, initial_position=0)
    confirm_frame = confirm_sim["frame"]
    confirm_frame["signal_zone"] = np.where(entry_triggers == 1, "bull", np.where(exit_triggers == 1, "bear", "hold"))
    confirm_frame["rebalance_date"] = window.rebalance_date
    confirm_frame["selected_smoothing_halflife"] = int(selected_smoothing)
    confirm_frame["selected_lower_threshold"] = float(confirm2_selected["selected_lower_threshold"])
    confirm_frame["selected_upper_threshold"] = float(confirm2_selected["selected_upper_threshold"])
    confirm_frame["predicted_probability_raw"] = oos_raw_prob
    confirm_frame["predicted_probability_smoothed"] = oos_smoothed
    confirm_frame["y_true"] = oos_y_true
    confirm_frame["prev_position"] = prev_positions.astype(int)
    confirm_frame["entry_trigger"] = entry_triggers.astype(int)
    confirm_frame["exit_trigger_confirm2"] = exit_triggers.astype(int)
    confirm_metrics = base.compute_metrics(oos_y_true, np.asarray(confirm_positions, dtype=int), oos_smoothed)

    return {
        "window": window_payload,
        "selected_smoothing_halflife": int(selected_smoothing),
        "baseline": {
            "selected_lower_threshold": float(baseline_selected["selected_lower_threshold"]),
            "selected_upper_threshold": float(baseline_selected["selected_upper_threshold"]),
            "validation_sharpe": float(baseline_selected["validation_sharpe"]),
            "validation_balanced_accuracy": float(baseline_selected["validation_balanced_accuracy"]),
            "validation_accuracy": float(baseline_selected["validation_accuracy"]),
            "validation_f1": float(baseline_selected["validation_f1"]),
            "validation_log_loss": float(baseline_selected["validation_log_loss"]),
            "oos_sharpe": float(baseline_sim["sharpe"]),
            "oos_metrics": {
                "accuracy": float(baseline_metrics["accuracy"]),
                "balanced_accuracy": float(baseline_metrics["balanced_accuracy"]),
                "f1": float(baseline_metrics["f1"]),
                "log_loss": float(baseline_metrics["log_loss"]),
            },
            "oos_frame_records": frame_to_records(baseline_frame),
        },
        "confirm2": {
            "selected_lower_threshold": float(confirm2_selected["selected_lower_threshold"]),
            "selected_upper_threshold": float(confirm2_selected["selected_upper_threshold"]),
            "validation_sharpe": float(confirm2_selected["validation_sharpe"]),
            "validation_balanced_accuracy": float(confirm2_selected["validation_balanced_accuracy"]),
            "validation_accuracy": float(confirm2_selected["validation_accuracy"]),
            "validation_f1": float(confirm2_selected["validation_f1"]),
            "validation_log_loss": float(confirm2_selected["validation_log_loss"]),
            "oos_sharpe": float(confirm_sim["sharpe"]),
            "oos_metrics": {
                "accuracy": float(confirm_metrics["accuracy"]),
                "balanced_accuracy": float(confirm_metrics["balanced_accuracy"]),
                "f1": float(confirm_metrics["f1"]),
                "log_loss": float(confirm_metrics["log_loss"]),
            },
            "oos_frame_records": frame_to_records(confirm_frame),
        },
    }


def build_prediction_metrics(window_results: List[Dict[str, object]], version: str) -> Dict[str, object]:
    return {
        "version": version,
        "avg_validation_accuracy": float(np.mean([row["validation_accuracy"] for row in window_results])),
        "avg_validation_balanced_accuracy": float(np.mean([row["validation_balanced_accuracy"] for row in window_results])),
        "avg_validation_f1": float(np.mean([row["validation_f1"] for row in window_results])),
        "avg_validation_log_loss": float(np.mean([row["validation_log_loss"] for row in window_results])),
        "avg_oos_accuracy": float(np.mean([row["oos_metrics"]["accuracy"] for row in window_results])),
        "avg_oos_balanced_accuracy": float(np.mean([row["oos_metrics"]["balanced_accuracy"] for row in window_results])),
        "avg_oos_f1": float(np.mean([row["oos_metrics"]["f1"] for row in window_results])),
        "avg_oos_log_loss": float(np.mean([row["oos_metrics"]["log_loss"] for row in window_results])),
    }


def build_rolling_rows(window_results: List[Dict[str, object]], final_frame: pd.DataFrame, version: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in window_results:
        window_slice = final_frame.loc[final_frame["rebalance_date"] == result["window"]["rebalance_date"]]
        y_true = np.asarray(window_slice["y_true"], dtype=int)
        y_pred = np.asarray(window_slice["position"], dtype=int)
        rows.append(
            {
                "version": version,
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
    return rows


def plot_equity_curves(daily_equity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(daily_equity["Date"], daily_equity["equity_current_baseline_rule"], label=BASELINE_VERSION, linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_raw_entry_smooth_exit_confirm2_rule"], label=NEW_RULE_VERSION, linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_buy_and_hold"], label="buy_and_hold", linewidth=1.8)
    ax.set_title("SPY Baseline vs Raw-Entry Confirm2 Exit")
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


def main() -> None:
    start_time = time.perf_counter()
    base.ensure_inputs()

    frame = base.signal_base.load_signal_experiment_frame(base.SPY_STEM)
    main_windows = base.build_target_windows(frame)
    tasks = [window.__dict__ for window in main_windows]
    worker_count = min(base.MAX_WORKERS_CAP, len(main_windows), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        combined_results = list(executor.map(worker_process_window_compare_safe, tasks))

    baseline_results = [
        {
            "window": result["window"],
            "selected_smoothing_halflife": result["selected_smoothing_halflife"],
            "selected_lower_threshold": result["baseline"]["selected_lower_threshold"],
            "selected_upper_threshold": result["baseline"]["selected_upper_threshold"],
            "validation_sharpe": result["baseline"]["validation_sharpe"],
            "validation_balanced_accuracy": result["baseline"]["validation_balanced_accuracy"],
            "validation_accuracy": result["baseline"]["validation_accuracy"],
            "validation_f1": result["baseline"]["validation_f1"],
            "validation_log_loss": result["baseline"]["validation_log_loss"],
            "oos_metrics": result["baseline"]["oos_metrics"],
            "oos_sharpe": result["baseline"]["oos_sharpe"],
            "oos_frame_records": result["baseline"]["oos_frame_records"],
        }
        for result in combined_results
    ]
    confirm2_results = [
        {
            "window": result["window"],
            "selected_smoothing_halflife": result["selected_smoothing_halflife"],
            "selected_lower_threshold": result["confirm2"]["selected_lower_threshold"],
            "selected_upper_threshold": result["confirm2"]["selected_upper_threshold"],
            "validation_sharpe": result["confirm2"]["validation_sharpe"],
            "validation_balanced_accuracy": result["confirm2"]["validation_balanced_accuracy"],
            "validation_accuracy": result["confirm2"]["validation_accuracy"],
            "validation_f1": result["confirm2"]["validation_f1"],
            "validation_log_loss": result["confirm2"]["validation_log_loss"],
            "oos_metrics": result["confirm2"]["oos_metrics"],
            "oos_sharpe": result["confirm2"]["oos_sharpe"],
            "oos_frame_records": result["confirm2"]["oos_frame_records"],
        }
        for result in combined_results
    ]

    baseline_results = sorted(baseline_results, key=lambda item: item["window"]["rebalance_date"])
    confirm2_results = sorted(confirm2_results, key=lambda item: item["window"]["rebalance_date"])

    baseline_frame = build_final_frame(baseline_results)
    confirm2_frame = build_final_frame(confirm2_results)

    buyhold = base.build_buyhold_frame(pd.DataFrame({"Date": baseline_frame["Date"].drop_duplicates().sort_values().to_list()}))
    buyhold = buyhold.merge(baseline_frame[["Date"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    daily_equity = pd.DataFrame(
        {
            "Date": buyhold["Date"],
            "equity_current_baseline_rule": baseline_frame.set_index("Date").loc[buyhold["Date"], "equity_predicted_strategy"].to_numpy(),
            "equity_raw_entry_smooth_exit_confirm2_rule": confirm2_frame.set_index("Date").loc[buyhold["Date"], "equity_predicted_strategy"].to_numpy(),
            "equity_buy_and_hold": buyhold["equity_buy_and_hold"].to_numpy(),
        }
    )

    strategy_summary = pd.DataFrame(
        [
            {"version": BASELINE_VERSION, **base.build_strategy_metrics(baseline_frame)},
            {"version": NEW_RULE_VERSION, **base.build_strategy_metrics(confirm2_frame)},
            {"version": "buy_and_hold", **base.build_strategy_metrics(buyhold)},
        ]
    )
    prediction_metrics = pd.DataFrame(
        [
            build_prediction_metrics(baseline_results, BASELINE_VERSION),
            build_prediction_metrics(confirm2_results, NEW_RULE_VERSION),
        ]
    )

    rolling_rows = build_rolling_rows(baseline_results, baseline_frame, BASELINE_VERSION)
    rolling_rows.extend(build_rolling_rows(confirm2_results, confirm2_frame, NEW_RULE_VERSION))

    trace = confirm2_frame[
        [
            "Date",
            "predicted_probability_raw",
            "predicted_probability_smoothed",
            "prev_position",
            "entry_trigger",
            "exit_trigger_confirm2",
            "position",
        ]
    ].copy()
    trace = trace.rename(
        columns={
            "predicted_probability_raw": "raw_probability",
            "predicted_probability_smoothed": "smoothed_probability",
            "position": "new_position",
        }
    )

    out_dir = results_dir()
    strategy_summary.to_csv(out_dir / "strategy_performance_comparison.csv", index=False)
    prediction_metrics.to_csv(out_dir / "prediction_metrics_comparison.csv", index=False)
    pd.DataFrame(rolling_rows).sort_values(["version", "rebalance_date"]).reset_index(drop=True).to_csv(
        out_dir / "rolling_window_metrics_comparison.csv", index=False
    )
    daily_equity.to_csv(out_dir / "daily_equity_curves_comparison.csv", index=False)
    trace.to_csv(out_dir / "decision_trace_sample.csv", index=False)

    plot_equity_curves(daily_equity, out_dir / "baseline_vs_raw_entry_confirm2_equity_curves.png")
    plot_metric_bar(strategy_summary, "sharpe", "Sharpe Comparison", out_dir / "baseline_vs_raw_entry_confirm2_sharpe.png")
    plot_metric_bar(strategy_summary, "max_drawdown", "Max Drawdown Comparison", out_dir / "baseline_vs_raw_entry_confirm2_drawdown.png")

    print(f"Results directory: {out_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start_time:.2f}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
