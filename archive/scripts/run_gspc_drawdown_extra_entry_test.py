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


RESULTS_DIR_NAME = "gspc_drawdown_extra_entry_test"
SIGNAL_STEM = "gspc"
TARGET_OOS_START = pd.Timestamp("2000-05-26")
BASELINE_VERSION = "baseline_fixed_rising_floor_052"
BUYHOLD_VERSION = "buy_and_hold"
RISING_FLOOR = 0.52
DD_THRESHOLD = 0.20
PROB_FLOOR_GRID = [round(x / 100.0, 2) for x in range(43, 49)]
FIXED_VERSION_BY_PROB_FLOOR = {
    prob_floor: f"dd20_prob_floor_{int(round(prob_floor * 100)):02d}"
    for prob_floor in PROB_FLOOR_GRID
}
DYNAMIC_PROB_FLOOR_VERSION = "dd20_prob_floor_dynamic_lower"
ADDED_RECOVERY_FEATURES = [
    "ma20_over_ma60",
    "close_over_ma120",
    "ret_61_120d",
]


def results_dir() -> Path:
    out = Path(__file__).resolve().parents[1] / "results" / RESULTS_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_gspc_research_frame() -> pd.DataFrame:
    feature_frame = pd.read_csv(signal_base.feature_path_for_signal_stem(SIGNAL_STEM))
    macro_frame = pd.read_csv(signal_base.macro_feature_path())
    raw_frame = pd.read_csv(signal_base.raw_path_for_signal_stem(SIGNAL_STEM))

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
    running_peak = price.cummax()

    price_features = pd.DataFrame(
        {
            "Date": raw_frame["Date"],
            "Close": price,
            "drawdown_from_peak": price / running_peak - 1.0,
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
    merged = merged.sort_values("Date").reset_index(drop=True)

    macro_columns = [column for column in macro_frame.columns if column != "Date"]
    merged[macro_columns] = merged[macro_columns].ffill()
    required = [
        "Date",
        "Close",
        "drawdown_from_peak",
        "ret",
        "rf_daily",
        "excess_ret",
        *signal_base.BASE_JM_FEATURES,
        *signal_base.A1_REFINED_FEATURES,
        *signal_base.A2_FEATURES,
        *ADDED_RECOVERY_FEATURES,
        *macro_columns,
    ]
    merged = merged.dropna(subset=required).reset_index(drop=True)
    if merged.empty:
        raise ValueError("GSPC research frame is empty after merges")
    return merged


def research_feature_columns(frame: pd.DataFrame) -> List[str]:
    macro_columns = [
        column
        for column in frame.columns
        if column.startswith("dgs")
        or column.startswith("slope_")
        or column.startswith("vix_")
        or column.startswith("credit_spread_")
    ]
    return [
        *signal_base.BASE_JM_FEATURES,
        *signal_base.A1_REFINED_FEATURES,
        *signal_base.A2_FEATURES,
        *ADDED_RECOVERY_FEATURES,
        *macro_columns,
    ]


def build_target_windows(frame: pd.DataFrame) -> List[base.MainWindow]:
    dates = frame["Date"].sort_values().reset_index(drop=True)
    first_anchor = base.first_date_on_or_after(dates, TARGET_OOS_START)
    if first_anchor is None or first_anchor != TARGET_OOS_START:
        raise ValueError(f"First OOS anchor must be {TARGET_OOS_START.strftime('%Y-%m-%d')}, got {first_anchor}")

    windows: List[base.MainWindow] = []
    current_anchor = first_anchor
    latest_date = dates.iloc[-1]
    while current_anchor is not None and current_anchor <= latest_date:
        next_anchor_target = current_anchor + pd.DateOffset(months=base.STEP_MONTHS)
        next_anchor = base.first_date_on_or_after(dates, next_anchor_target)
        oos_end = latest_date if next_anchor is None else base.last_date_before(dates, next_anchor)
        if oos_end is None or oos_end < current_anchor:
            current_anchor = next_anchor
            continue

        val_start = base.first_date_on_or_after(dates, current_anchor - pd.DateOffset(years=base.VALIDATION_YEARS))
        train_start = base.first_date_on_or_after(dates, current_anchor - pd.DateOffset(years=base.VALIDATION_YEARS + base.TRAIN_YEARS))
        train_end = base.last_date_before(dates, val_start) if val_start is not None else None
        val_end = base.last_date_before(dates, current_anchor)
        if None in [val_start, train_start, train_end, val_end]:
            current_anchor = next_anchor
            continue

        windows.append(
            base.MainWindow(
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


def rising_entry_with_dd_positions(
    probabilities: np.ndarray,
    drawdowns: np.ndarray,
    initial_position: int,
    lower_threshold: float,
    upper_threshold: float,
    rising_floor: float,
    dd_threshold: float | None,
    dd_prob_floor: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    dd = np.asarray(drawdowns, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    baseline_entry = np.zeros(len(probs), dtype=int)
    dd_entry = np.zeros(len(probs), dtype=int)
    prev_position = int(initial_position)

    for idx, prob in enumerate(probs):
        rising_confirm = (
            idx >= 2
            and probs[idx - 1] > probs[idx - 2]
            and prob > probs[idx - 1]
            and prob > rising_floor
        )
        base_entry_now = bool(prob >= upper_threshold or rising_confirm)
        dd_entry_now = bool(
            dd_threshold is not None
            and dd[idx] <= -dd_threshold
            and dd_prob_floor is not None
            and prob > dd_prob_floor
        )

        if prev_position == 0:
            if base_entry_now or dd_entry_now:
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
        baseline_entry[idx] = int(base_entry_now)
        dd_entry[idx] = int(dd_entry_now)
        prev_position = current

    return positions, zones, baseline_entry, dd_entry, prev_position


def score_validation_cached(
    validation_cache: List[Dict[str, object]],
    smoothing_halflife: int,
    lower_threshold: float,
    upper_threshold: float,
    dd_threshold: float | None,
    dd_prob_floor_mode: str,
    fixed_dd_prob_floor: float | None,
) -> Dict[str, object]:
    current_position = 0
    frames: List[pd.DataFrame] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for bundle in validation_cache:
        smoothed_prob = np.asarray(bundle["smoothed"][smoothing_halflife], dtype=float)
        dd_prob_floor = lower_threshold if dd_prob_floor_mode == "selected_lower" else fixed_dd_prob_floor
        positions, _, _, _, current_position = rising_entry_with_dd_positions(
            smoothed_prob,
            bundle["drawdown"],
            current_position,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            rising_floor=RISING_FLOOR,
            dd_threshold=dd_threshold,
            dd_prob_floor=dd_prob_floor,
        )
        simulation = base.simulate_positions_strategy(bundle["base"], positions, initial_position=int(bundle["initial_position"]))
        bundle["initial_position"] = current_position
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


def choose_parameters_for_version(
    validation_cache_template: List[Dict[str, object]],
    dd_threshold: float | None,
    dd_prob_floor_mode: str,
    fixed_dd_prob_floor: float | None,
) -> Tuple[int, Dict[str, object], Dict[str, object]]:
    smoothing_rows: List[Dict[str, object]] = []
    for smoothing_halflife in base.SMOOTHING_GRID:
        validation_cache = [{**bundle, "initial_position": 0} for bundle in validation_cache_template]
        result = score_validation_cached(
            validation_cache,
            smoothing_halflife,
            0.48,
            0.67,
            dd_threshold,
            dd_prob_floor_mode,
            fixed_dd_prob_floor,
        )
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
            validation_cache = [{**bundle, "initial_position": 0} for bundle in validation_cache_template]
            result = score_validation_cached(
                validation_cache,
                selected_smoothing,
                lower_threshold,
                upper_threshold,
                dd_threshold,
                dd_prob_floor_mode,
                fixed_dd_prob_floor,
            )
            threshold_rows.append(
                {
                    "selected_lower_threshold": float(lower_threshold),
                    "selected_upper_threshold": float(upper_threshold),
                    "validation_sharpe": float(result["sharpe"]),
                    "validation_balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
                    "validation_accuracy": float(result["metrics"]["accuracy"]),
                    "validation_f1": float(result["metrics"]["f1"]),
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

    validation_cache = [{**bundle, "initial_position": 0} for bundle in validation_cache_template]
    best_validation = score_validation_cached(
        validation_cache,
        selected_smoothing,
        float(selected_threshold["selected_lower_threshold"]),
        float(selected_threshold["selected_upper_threshold"]),
        dd_threshold,
        dd_prob_floor_mode,
        fixed_dd_prob_floor,
    )
    return selected_smoothing, selected_threshold, best_validation


def prepare_window(window: base.MainWindow) -> Dict[str, object]:
    frame = load_gspc_research_frame()
    features = research_feature_columns(frame)
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

        segment_frame = frame[(frame["Date"] >= segment["start"]) & (frame["Date"] < segment["end_exclusive"])].copy().reset_index(drop=True)
        segment_frame["execution_date"] = segment_frame["Date"].shift(-1)
        segment_frame = segment_frame.iloc[:-1].copy().reset_index(drop=True)

        validation_cache.append(
            {
                "base": bundle["segment_dataset"]["base"].reset_index(drop=True),
                "y_true": bundle["segment_dataset"]["y"].astype(int).copy(),
                "raw_prob": raw_prob,
                "smoothed": smoothed,
                "drawdown": segment_frame["drawdown_from_peak"].to_numpy(dtype=float, copy=False),
                "close": segment_frame["Close"].to_numpy(dtype=float, copy=False),
                "signal_date": pd.to_datetime(segment_frame["Date"]).reset_index(drop=True),
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

    oos_frame = frame[(frame["Date"] >= pd.Timestamp(window.oos_start)) & (frame["Date"] <= pd.Timestamp(window.oos_end))].copy().reset_index(drop=True)
    oos_frame["execution_date"] = oos_frame["Date"].shift(-1)
    oos_frame = oos_frame.iloc[:-1].copy().reset_index(drop=True)

    return {
        "window": window,
        "validation_cache": validation_cache,
        "oos_base": oos_bundle["segment_dataset"]["base"].reset_index(drop=True),
        "oos_y_true": oos_bundle["segment_dataset"]["y"].astype(int).copy(),
        "oos_raw_prob": oos_raw_prob,
        "oos_smoothed": oos_smoothed,
        "oos_drawdown": oos_frame["drawdown_from_peak"].to_numpy(dtype=float, copy=False),
        "oos_close": oos_frame["Close"].to_numpy(dtype=float, copy=False),
        "oos_signal_date": pd.to_datetime(oos_frame["Date"]).reset_index(drop=True),
    }


def worker_process_window(window_payload: Dict[str, str]) -> Dict[str, object]:
    window = base.MainWindow(**window_payload)
    prepared = prepare_window(window)
    validation_cache = prepared["validation_cache"]
    oos_base = prepared["oos_base"]
    oos_y_true = prepared["oos_y_true"]
    oos_raw_prob = prepared["oos_raw_prob"]
    oos_smoothed = prepared["oos_smoothed"]
    oos_drawdown = prepared["oos_drawdown"]
    oos_close = prepared["oos_close"]
    oos_signal_date = prepared["oos_signal_date"]

    versions = [(BASELINE_VERSION, None, "fixed", None)]
    versions += [
        (version_name, DD_THRESHOLD, "fixed", prob_floor)
        for prob_floor, version_name in FIXED_VERSION_BY_PROB_FLOOR.items()
    ]
    versions += [(DYNAMIC_PROB_FLOOR_VERSION, DD_THRESHOLD, "selected_lower", None)]
    version_results: List[Dict[str, object]] = []

    for version, dd_threshold, dd_prob_floor_mode, fixed_dd_prob_floor in versions:
        selected_smoothing, selected_threshold, best_validation = choose_parameters_for_version(
            validation_cache,
            dd_threshold,
            dd_prob_floor_mode,
            fixed_dd_prob_floor,
        )
        dd_prob_floor = (
            float(selected_threshold["selected_lower_threshold"])
            if dd_prob_floor_mode == "selected_lower"
            else fixed_dd_prob_floor
        )
        positions, zones, baseline_entry, dd_entry, _ = rising_entry_with_dd_positions(
            oos_smoothed[selected_smoothing],
            oos_drawdown,
            initial_position=0,
            lower_threshold=float(selected_threshold["selected_lower_threshold"]),
            upper_threshold=float(selected_threshold["selected_upper_threshold"]),
            rising_floor=RISING_FLOOR,
            dd_threshold=dd_threshold,
            dd_prob_floor=dd_prob_floor,
        )
        simulation = base.simulate_positions_strategy(oos_base, positions, initial_position=0)
        frame = simulation["frame"]
        frame["signal_zone"] = zones
        frame["rebalance_date"] = window.rebalance_date
        frame["selected_smoothing_halflife"] = int(selected_smoothing)
        frame["selected_lower_threshold"] = float(selected_threshold["selected_lower_threshold"])
        frame["selected_upper_threshold"] = float(selected_threshold["selected_upper_threshold"])
        frame["predicted_probability_raw"] = np.asarray(oos_raw_prob, dtype=float)
        frame["predicted_probability_smoothed"] = np.asarray(oos_smoothed[selected_smoothing], dtype=float)
        frame["y_true"] = np.asarray(oos_y_true, dtype=int)
        frame["signal_date"] = oos_signal_date.dt.strftime("%Y-%m-%d")
        frame["Close"] = np.asarray(oos_close, dtype=float)
        frame["drawdown_from_peak"] = np.asarray(oos_drawdown, dtype=float)

        oos_metrics = base.compute_metrics(
            np.asarray(oos_y_true, dtype=int),
            np.asarray(positions, dtype=int),
            np.asarray(oos_smoothed[selected_smoothing], dtype=float),
        )
        decision_trace = pd.DataFrame(
            {
                "version": version,
                "Date": oos_signal_date.dt.strftime("%Y-%m-%d"),
                "Close": np.asarray(oos_close, dtype=float),
                "drawdown_from_peak": np.asarray(oos_drawdown, dtype=float),
                "smoothed_probability": np.asarray(oos_smoothed[selected_smoothing], dtype=float),
                "prev_position": np.r_[0, positions[:-1]].astype(int),
                "baseline_entry_trigger": baseline_entry.astype(int),
                "dd_extra_entry_trigger": dd_entry.astype(int),
                "new_position": positions.astype(int),
            }
        )

        version_results.append(
            {
                "version": version,
                "dd_threshold": dd_threshold,
                "selected_smoothing_halflife": int(selected_smoothing),
                "selected_lower_threshold": float(selected_threshold["selected_lower_threshold"]),
                "selected_upper_threshold": float(selected_threshold["selected_upper_threshold"]),
                "selected_dd_prob_floor": float(dd_prob_floor) if dd_prob_floor is not None else np.nan,
                "validation_sharpe": float(best_validation["sharpe"]),
                "validation_balanced_accuracy": float(best_validation["metrics"]["balanced_accuracy"]),
                "validation_accuracy": float(best_validation["metrics"]["accuracy"]),
                "validation_f1": float(best_validation["metrics"]["f1"]),
                "oos_sharpe": float(simulation["sharpe"]),
                "oos_metrics": {
                    "accuracy": float(oos_metrics["accuracy"]),
                    "balanced_accuracy": float(oos_metrics["balanced_accuracy"]),
                    "f1": float(oos_metrics["f1"]),
                },
                "oos_frame_records": frame.to_dict(orient="records"),
                "decision_trace_records": decision_trace.to_dict(orient="records"),
            }
        )

    return {"window": window_payload, "version_results": version_results}


def build_final_frame(window_results: List[Dict[str, object]], version: str) -> pd.DataFrame:
    payloads = []
    for result in window_results:
        version_row = next(row for row in result["version_results"] if row["version"] == version)
        frame = pd.DataFrame(version_row["oos_frame_records"])
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        payloads.append({"window": result["window"], "oos_frame": frame})
    final_frame = base.stitch_final_frame(payloads)
    final_frame["equity_predicted_strategy"] = (1.0 + final_frame["strategy_ret"]).cumprod()
    return final_frame


def build_buyhold_frame(signal_dates: pd.DataFrame) -> pd.DataFrame:
    frame = load_gspc_research_frame()[["Date", "ret", "rf_daily"]].copy()
    frame = frame.merge(signal_dates, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    frame = frame.dropna(subset=["ret", "rf_daily"]).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["Date"]),
            "position": 1,
            "fee": 0.0,
            "strategy_ret_gross": frame["ret"].to_numpy(dtype=float, copy=False),
            "strategy_ret": frame["ret"].to_numpy(dtype=float, copy=False),
            "strategy_excess_ret": frame["ret"].to_numpy(dtype=float, copy=False) - frame["rf_daily"].to_numpy(dtype=float, copy=False),
        }
    )
    out["equity_buy_and_hold"] = (1.0 + out["strategy_ret"]).cumprod()
    return out


def plot_equity_curves(daily_equity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_versions = [BASELINE_VERSION, *FIXED_VERSION_BY_PROB_FLOOR.values(), DYNAMIC_PROB_FLOOR_VERSION, BUYHOLD_VERSION]
    for version in plot_versions:
        column = "equity_buy_and_hold" if version == BUYHOLD_VERSION else f"equity_{version}"
        ax.plot(daily_equity["Date"], daily_equity[column], label=version, linewidth=1.3)
    ax.set_title("^GSPC Drawdown Extra Entry Test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_bar(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(summary["version"], summary[metric], color="#4e79a7")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    signal_base.ensure_inputs()

    frame = load_gspc_research_frame()
    windows = build_target_windows(frame)
    worker_count = min(base.MAX_WORKERS_CAP, len(windows), os.cpu_count() or 1)
    tasks = [window.__dict__ for window in windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_process_window, tasks))

    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])
    version_order = [BASELINE_VERSION, *FIXED_VERSION_BY_PROB_FLOOR.values(), DYNAMIC_PROB_FLOOR_VERSION]
    final_frames: Dict[str, pd.DataFrame] = {}
    strategy_rows: List[Dict[str, object]] = []
    rolling_rows: List[Dict[str, object]] = []
    decision_trace_parts: List[pd.DataFrame] = []

    for version in version_order:
        final_frame = build_final_frame(window_results, version)
        final_frames[version] = final_frame
        strategy_rows.append({"version": version, **base.build_strategy_metrics(final_frame)})

        for result in window_results:
            row = next(item for item in result["version_results"] if item["version"] == version)
            window_slice = final_frame.loc[final_frame["rebalance_date"] == result["window"]["rebalance_date"]]
            y_true = np.asarray(window_slice["y_true"], dtype=int)
            y_pred = np.asarray(window_slice["position"], dtype=int)
            rolling_rows.append(
                {
                    "version": version,
                    "rebalance_date": result["window"]["rebalance_date"],
                    "selected_smoothing_halflife": int(row["selected_smoothing_halflife"]),
                    "selected_lower_threshold": float(row["selected_lower_threshold"]),
                    "selected_upper_threshold": float(row["selected_upper_threshold"]),
                    "selected_dd_prob_floor": float(row["selected_dd_prob_floor"]) if not pd.isna(row["selected_dd_prob_floor"]) else np.nan,
                    "oos_sharpe": base.annualized_sharpe(window_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
                    "oos_accuracy": float(accuracy_score(y_true, y_pred)),
                    "oos_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "oos_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                }
            )
            if version in [BASELINE_VERSION, "dd20_prob_floor_043", "dd20_prob_floor_045", DYNAMIC_PROB_FLOOR_VERSION]:
                decision_trace_parts.append(pd.DataFrame(row["decision_trace_records"]).head(80))

    buyhold = build_buyhold_frame(pd.DataFrame({"Date": final_frames[BASELINE_VERSION]["Date"].drop_duplicates().sort_values().to_list()}))
    strategy_rows.append({"version": BUYHOLD_VERSION, **base.build_strategy_metrics(buyhold)})

    daily_equity = pd.DataFrame({"Date": final_frames[BASELINE_VERSION]["Date"]})
    for version, final_frame in final_frames.items():
        daily_equity[f"equity_{version}"] = final_frame["equity_predicted_strategy"].to_numpy()
    daily_equity = daily_equity.merge(buyhold[["Date", "equity_buy_and_hold"]], on="Date", how="left")

    decision_trace = pd.concat(decision_trace_parts, ignore_index=True)

    out_dir = results_dir()
    pd.DataFrame(strategy_rows).to_csv(out_dir / "strategy_performance_comparison.csv", index=False)
    pd.DataFrame(rolling_rows).sort_values(["version", "rebalance_date"]).reset_index(drop=True).to_csv(
        out_dir / "rolling_window_metrics_comparison.csv", index=False
    )
    daily_equity.to_csv(out_dir / "daily_equity_curves_comparison.csv", index=False)
    decision_trace.to_csv(out_dir / "decision_trace_sample.csv", index=False)

    plot_equity_curves(daily_equity, out_dir / "drawdown_extra_entry_equity_curves.png")
    plot_metric_bar(pd.DataFrame(strategy_rows), "sharpe", "Drawdown Extra Entry Sharpe Comparison", out_dir / "drawdown_extra_entry_sharpe_comparison.png")
    plot_metric_bar(pd.DataFrame(strategy_rows), "max_drawdown", "Drawdown Extra Entry Drawdown Comparison", out_dir / "drawdown_extra_entry_drawdown_comparison.png")

    print(f"Results directory: {out_dir}")
    print(f"Elapsed seconds: {time.perf_counter() - start_time:.2f}")
    print(pd.DataFrame(strategy_rows).to_string(index=False))


if __name__ == "__main__":
    main()
