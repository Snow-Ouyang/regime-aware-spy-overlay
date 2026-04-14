import os
import site
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def configure_paths() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
    for vendor_name in [".vendor_local", ".vendor"]:
        vendor_path = Path(__file__).resolve().parents[1] / vendor_name
        if vendor_path.exists():
            vendor_str = str(vendor_path)
            if vendor_str not in sys.path:
                sys.path.insert(0, vendor_str)


configure_paths()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from final_multi_asset_project_common import (
    FIXED_XGB_PARAMS,
    ensure_raw_data,
    ensure_research_feature_file,
    load_risk_free_daily_series,
    load_trade_price_frame,
    macro_feature_path,
    raw_path_for_stem,
)
from spy_regime_common import (
    FEE_BPS,
    MainWindow,
    annualized_sharpe,
    build_segment_supervised_bundle,
    build_segments,
    build_strategy_metrics,
    first_date_on_or_after,
    last_date_before,
)


RESEARCH_TICKER = "^GSPC"
RESEARCH_STEM = "gspc"
TRADE_TICKER = "SPY"
TRADE_STEM = "spy_trade"
TARGET_OOS_START = pd.Timestamp("2000-05-26")
TRAIN_YEARS = 11
VALIDATION_YEARS = 4
STEP_MONTHS = 6
MAX_WORKERS_CAP = 8
JUMP_PENALTY = 0.0
SMOOTHING_GRID: List[int] = [0, 4, 8, 12]
LOWER_THRESHOLD_GRID: List[float] = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
UPPER_THRESHOLD_GRID: List[float] = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70]
BASELINE_UPPER = 0.67
BASELINE_LOWER = 0.48
BASE_JM_FEATURES: List[str] = [
    "log_downside_dev_hl5",
    "log_downside_dev_hl21",
    "ewm_return_hl5",
    "ewm_return_hl10",
    "ewm_return_hl21",
    "sortino_hl5",
    "sortino_hl10",
    "sortino_hl21",
]
REFINED_RETURN_FEATURES: List[str] = [
    "ret_1_5d",
    "ret_6_20d",
    "ret_21_60d",
]
TREND_POSITION_FEATURES: List[str] = [
    "close_over_ma20",
    "close_over_ma60",
]
RECOVERY_FEATURES: List[str] = [
    "ma20_over_ma60",
    "close_over_ma120",
    "ret_61_120d",
]
CURRENT_MAPPED_ORACLE_VERSION = "current_mapped_oracle"
FULL_UPPER_BOUND_ORACLE_VERSION = "full_upper_bound_oracle"
BUYHOLD_VERSION = "buy_and_hold"


@dataclass
class StageConfig:
    stage_name: str
    results_subdir: str
    feature_mode: str
    rule_mode: str
    threshold: float | None = None
    rising_floor: float | None = None
    drawdown_threshold: float | None = None
    drawdown_prob_floor: float | None = None
    output_ml_figures: bool = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_root() -> Path:
    out = project_root() / "results" / "single_asset_mainline"
    out.mkdir(parents=True, exist_ok=True)
    return out


def stage_results_dir(results_subdir: str) -> Path:
    out = results_root() / results_subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_inputs() -> None:
    ensure_raw_data(RESEARCH_TICKER, RESEARCH_STEM)
    ensure_research_feature_file(RESEARCH_TICKER, RESEARCH_STEM)
    ensure_raw_data(TRADE_TICKER, TRADE_STEM)
    if not macro_feature_path().exists():
        raise FileNotFoundError(f"Missing macro feature panel: {macro_feature_path()}")


def make_xgb_model() -> XGBClassifier:
    return XGBClassifier(**FIXED_XGB_PARAMS)


def load_research_frame() -> pd.DataFrame:
    feature_frame = pd.read_csv(project_root() / "data_features" / f"{RESEARCH_STEM}_features_final.csv")
    macro_frame = pd.read_csv(macro_feature_path())
    raw_frame = pd.read_csv(raw_path_for_stem(RESEARCH_STEM))

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

    close_price = raw_frame["Close"]
    ma20 = close_price.rolling(20).mean()
    ma60 = close_price.rolling(60).mean()
    ma120 = close_price.rolling(120).mean()
    running_peak = close_price.cummax()

    engineered = pd.DataFrame(
        {
            "Date": raw_frame["Date"],
            "Close": close_price,
            "drawdown_from_peak": close_price / running_peak - 1.0,
            "ret_1_5d": close_price / close_price.shift(5) - 1.0,
            "ret_6_20d": close_price.shift(5) / close_price.shift(20) - 1.0,
            "ret_21_60d": close_price.shift(20) / close_price.shift(60) - 1.0,
            "close_over_ma20": close_price / ma20 - 1.0,
            "close_over_ma60": close_price / ma60 - 1.0,
            "ma20_over_ma60": ma20 / ma60 - 1.0,
            "close_over_ma120": close_price / ma120 - 1.0,
            "ret_61_120d": close_price.shift(60) / close_price.shift(120) - 1.0,
        }
    )

    merged = feature_frame.merge(macro_frame, on="Date", how="left", sort=True)
    merged = merged.merge(engineered, on="Date", how="left", sort=True)
    merged = merged.sort_values("Date").reset_index(drop=True)
    macro_cols = [column for column in macro_frame.columns if column != "Date"]
    merged[macro_cols] = merged[macro_cols].ffill()
    required = [
        "Date",
        "Close",
        "drawdown_from_peak",
        "ret",
        "rf_daily",
        "excess_ret",
        *BASE_JM_FEATURES,
        *REFINED_RETURN_FEATURES,
        *TREND_POSITION_FEATURES,
        *RECOVERY_FEATURES,
        *macro_cols,
    ]
    merged = merged.dropna(subset=required).reset_index(drop=True)
    if merged.empty:
        raise ValueError("Research frame is empty after merge")
    return merged


def macro_columns(frame: pd.DataFrame) -> List[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("dgs")
        or column.startswith("slope_")
        or column.startswith("vix_")
        or column.startswith("credit_spread_")
    ]


def feature_columns_for_mode(frame: pd.DataFrame, feature_mode: str) -> List[str]:
    macro_cols = macro_columns(frame)
    if feature_mode == "paper":
        return [*BASE_JM_FEATURES, *macro_cols]
    if feature_mode == "enhanced":
        return [*BASE_JM_FEATURES, *REFINED_RETURN_FEATURES, *TREND_POSITION_FEATURES, *RECOVERY_FEATURES, *macro_cols]
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


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
    return [segment for segment in segments if int(((frame["Date"] >= segment["start"]) & (frame["Date"] < segment["end_exclusive"])).sum()) > 0]


def smooth_probability_series(probabilities: np.ndarray, halflife: int) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if halflife == 0:
        return raw
    return pd.Series(raw).ewm(halflife=halflife, adjust=False).mean().to_numpy(dtype=float, copy=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def simulate_positions_strategy(dataset_base: pd.DataFrame, positions: np.ndarray, initial_position: int) -> Dict[str, object]:
    base = dataset_base.reset_index(drop=True).copy()
    position = np.asarray(positions, dtype=float)
    previous_position = np.empty_like(position)
    previous_position[0] = float(initial_position)
    if len(position) > 1:
        previous_position[1:] = position[:-1]

    fee_rate = FEE_BPS / 10000.0
    fee = np.where(position != previous_position, fee_rate * np.abs(position - previous_position), 0.0)
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
        "final_position": float(position[-1]),
        "sharpe": annualized_sharpe(strategy_excess_ret),
    }


def single_threshold_positions(probabilities: np.ndarray, initial_position: int, threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    positions = (probs >= threshold).astype(int)
    zones = np.where(probs >= threshold, "bull", "bear")
    return positions, zones.astype(object), int(positions[-1]) if len(positions) else int(initial_position)


def double_threshold_positions(probabilities: np.ndarray, initial_position: int, lower_threshold: float, upper_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
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


def final_overlay_positions(
    probabilities: np.ndarray,
    drawdowns: np.ndarray,
    initial_position: int,
    lower_threshold: float,
    upper_threshold: float,
    rising_floor: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    dd = np.asarray(drawdowns, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    rising_entry = np.zeros(len(probs), dtype=int)
    drawdown_entry = np.zeros(len(probs), dtype=int)
    prev = int(initial_position)

    for idx, prob in enumerate(probs):
        rising_confirm = (
            idx >= 2
            and probs[idx - 1] > probs[idx - 2]
            and prob > probs[idx - 1]
            and prob > rising_floor
        )
        dd_entry = bool(dd[idx] <= -drawdown_threshold and prob > drawdown_prob_floor)
        rising_entry[idx] = int(rising_confirm)
        drawdown_entry[idx] = int(dd_entry)

        if prev == 0:
            if prob >= upper_threshold or rising_confirm or dd_entry:
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
        prev = current
    return positions, zones, rising_entry, drawdown_entry, prev


def single_threshold_overlay_positions(
    probabilities: np.ndarray,
    drawdowns: np.ndarray,
    initial_position: int,
    threshold: float,
    rising_floor: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    dd = np.asarray(drawdowns, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    rising_entry = np.zeros(len(probs), dtype=int)
    drawdown_entry = np.zeros(len(probs), dtype=int)
    prev = int(initial_position)

    for idx, prob in enumerate(probs):
        rising_confirm = (
            idx >= 2
            and probs[idx - 1] > probs[idx - 2]
            and prob > probs[idx - 1]
            and prob > rising_floor
        )
        dd_entry = bool(dd[idx] <= -drawdown_threshold and prob > drawdown_prob_floor)
        rising_entry[idx] = int(rising_confirm)
        drawdown_entry[idx] = int(dd_entry)

        if prev == 0:
            current = int(prob >= threshold or rising_confirm or dd_entry)
        else:
            current = int(prob >= threshold)

        positions[idx] = current
        zones[idx] = "bull" if current == 1 else "bear"
        prev = current

    return positions, zones, rising_entry, drawdown_entry, prev


def single_threshold_drawdown_overlay_positions(
    probabilities: np.ndarray,
    drawdowns: np.ndarray,
    initial_position: int,
    threshold: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    dd = np.asarray(drawdowns, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    drawdown_entry = np.zeros(len(probs), dtype=int)
    prev = int(initial_position)

    for idx, prob in enumerate(probs):
        dd_entry = bool(dd[idx] <= -drawdown_threshold and prob > drawdown_prob_floor)
        drawdown_entry[idx] = int(dd_entry)

        if prev == 0:
            current = int(prob >= threshold or dd_entry)
        else:
            current = int(prob >= threshold)

        positions[idx] = current
        zones[idx] = "bull" if current == 1 else "bear"
        prev = current

    return positions, zones, drawdown_entry, prev


def single_threshold_drawdown_rising_overlay_positions(
    probabilities: np.ndarray,
    drawdowns: np.ndarray,
    initial_position: int,
    threshold: float,
    rising_floor: float,
    drawdown_threshold: float,
    drawdown_prob_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    probs = np.asarray(probabilities, dtype=float)
    dd = np.asarray(drawdowns, dtype=float)
    positions = np.empty(len(probs), dtype=int)
    zones = np.empty(len(probs), dtype=object)
    rising_entry = np.zeros(len(probs), dtype=int)
    drawdown_entry = np.zeros(len(probs), dtype=int)
    prev = int(initial_position)

    for idx, prob in enumerate(probs):
        rising_confirm = (
            idx >= 2
            and probs[idx - 1] > probs[idx - 2]
            and prob > probs[idx - 1]
            and prob > rising_floor
        )
        dd_entry = bool(dd[idx] <= -drawdown_threshold and prob > drawdown_prob_floor)
        rising_entry[idx] = int(rising_confirm)
        drawdown_entry[idx] = int(dd_entry)

        if prev == 0:
            current = int(prob >= threshold or rising_confirm or dd_entry)
        else:
            current = int(prob >= threshold)

        positions[idx] = current
        zones[idx] = "bull" if current == 1 else "bear"
        prev = current

    return positions, zones, rising_entry, drawdown_entry, prev


def score_validation(
    validation_cache: List[Dict[str, object]],
    config: StageConfig,
    smoothing_halflife: int,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> Dict[str, object]:
    current_position = 0
    frames: List[pd.DataFrame] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for bundle in validation_cache:
        smoothed_prob = np.asarray(bundle["smoothed"][smoothing_halflife], dtype=float)
        if config.rule_mode == "single_threshold":
            positions, _, current_position = single_threshold_positions(smoothed_prob, current_position, float(config.threshold))
        elif config.rule_mode == "double_threshold":
            positions, _, current_position = double_threshold_positions(smoothed_prob, current_position, float(lower_threshold), float(upper_threshold))
        elif config.rule_mode == "final_overlay":
            positions, _, _, _, current_position = final_overlay_positions(
                smoothed_prob,
                bundle["drawdown"],
                current_position,
                float(lower_threshold),
                float(upper_threshold),
                float(config.rising_floor),
                float(config.drawdown_threshold),
                float(config.drawdown_prob_floor),
            )
        elif config.rule_mode == "single_threshold_overlay":
            positions, _, _, _, current_position = single_threshold_overlay_positions(
                smoothed_prob,
                bundle["drawdown"],
                current_position,
                float(config.threshold),
                float(config.rising_floor),
                float(config.drawdown_threshold),
                float(config.drawdown_prob_floor),
            )
        elif config.rule_mode == "single_threshold_drawdown_overlay":
            positions, _, _, current_position = single_threshold_drawdown_overlay_positions(
                smoothed_prob,
                bundle["drawdown"],
                current_position,
                float(config.threshold),
                float(config.drawdown_threshold),
                float(config.drawdown_prob_floor),
            )
        elif config.rule_mode == "single_threshold_drawdown_rising_overlay":
            positions, _, _, _, current_position = single_threshold_drawdown_rising_overlay_positions(
                smoothed_prob,
                bundle["drawdown"],
                current_position,
                float(config.threshold),
                float(config.rising_floor),
                float(config.drawdown_threshold),
                float(config.drawdown_prob_floor),
            )
        else:
            raise ValueError(f"Unsupported rule_mode: {config.rule_mode}")

        simulation = simulate_positions_strategy(bundle["base"], positions, int(bundle["initial_position"]))
        bundle["initial_position"] = current_position
        frames.append(simulation["frame"])
        y_true_list.append(bundle["y_true"])
        y_pred_list.append(np.asarray(positions, dtype=int))
        y_prob_list.append(smoothed_prob)

    validation_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {
        "frame": validation_frame,
        "sharpe": annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
        "metrics": compute_metrics(np.concatenate(y_true_list), np.concatenate(y_pred_list), np.concatenate(y_prob_list)),
    }


def select_smoothing(validation_cache: List[Dict[str, object]], config: StageConfig) -> int:
    rows: List[Dict[str, object]] = []
    for halflife in SMOOTHING_GRID:
        if config.rule_mode in {"single_threshold", "single_threshold_overlay", "single_threshold_drawdown_overlay", "single_threshold_drawdown_rising_overlay"}:
            result = score_validation(validation_cache, config, halflife, None, None)
        else:
            result = score_validation(validation_cache, config, halflife, BASELINE_LOWER, BASELINE_UPPER)
        rows.append(
            {
                "selected_smoothing_halflife": int(halflife),
                "validation_sharpe": float(result["sharpe"]),
                "validation_balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
                "version_id": f"h{halflife}",
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        by=["validation_sharpe", "validation_balanced_accuracy", "selected_smoothing_halflife", "version_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return int(frame.iloc[0]["selected_smoothing_halflife"])


def select_thresholds(validation_cache: List[Dict[str, object]], config: StageConfig, smoothing_halflife: int) -> Dict[str, float]:
    if config.rule_mode in {"single_threshold", "single_threshold_overlay", "single_threshold_drawdown_overlay", "single_threshold_drawdown_rising_overlay"}:
        return {"selected_lower_threshold": np.nan, "selected_upper_threshold": float(config.threshold)}

    rows: List[Dict[str, object]] = []
    for lower_threshold in LOWER_THRESHOLD_GRID:
        for upper_threshold in UPPER_THRESHOLD_GRID:
            if upper_threshold <= lower_threshold:
                continue
            result = score_validation(validation_cache, config, smoothing_halflife, lower_threshold, upper_threshold)
            rows.append(
                {
                    "selected_lower_threshold": float(lower_threshold),
                    "selected_upper_threshold": float(upper_threshold),
                    "validation_sharpe": float(result["sharpe"]),
                    "validation_balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
                    "lower_distance": abs(lower_threshold - BASELINE_LOWER),
                    "upper_distance": abs(upper_threshold - BASELINE_UPPER),
                    "version_id": f"l{int(round(lower_threshold*100)):02d}_u{int(round(upper_threshold*100)):02d}",
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        by=["validation_sharpe", "validation_balanced_accuracy", "lower_distance", "upper_distance", "version_id"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return frame.iloc[0].to_dict()


def prepare_window(window_payload: Dict[str, str], feature_mode: str) -> Dict[str, object]:
    frame = load_research_frame()
    features = feature_columns_for_mode(frame, feature_mode)
    window = MainWindow(**window_payload)

    validation_segments = build_nonempty_segments(frame, pd.Timestamp(window.val_start), pd.Timestamp(window.oos_start))
    validation_cache: List[Dict[str, object]] = []
    for segment in validation_segments:
        bundle = build_segment_supervised_bundle(frame, features, segment["start"], segment["end_exclusive"], penalty=JUMP_PENALTY)
        model = make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1].astype(float)
        smoothed = {h: smooth_probability_series(raw_prob, h) for h in SMOOTHING_GRID}
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
                "signal_date": pd.to_datetime(segment_frame["Date"]).reset_index(drop=True),
                "initial_position": 0,
            }
        )

    oos_bundle = build_segment_supervised_bundle(
        frame,
        features,
        pd.Timestamp(window.oos_start),
        pd.Timestamp(window.oos_end) + pd.Timedelta(days=1),
        penalty=JUMP_PENALTY,
    )
    model = make_xgb_model()
    model.fit(oos_bundle["train_dataset"]["X"], oos_bundle["train_dataset"]["y"])
    oos_raw_prob = model.predict_proba(oos_bundle["segment_dataset"]["X"])[:, 1].astype(float)
    oos_smoothed = {h: smooth_probability_series(oos_raw_prob, h) for h in SMOOTHING_GRID}
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
        "oos_signal_date": pd.to_datetime(oos_frame["Date"]).reset_index(drop=True),
    }


def load_trade_base() -> pd.DataFrame:
    trade_frame = load_trade_price_frame(TRADE_STEM)
    rf_frame = load_risk_free_daily_series()
    merged = trade_frame.merge(rf_frame, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    merged["rf_daily"] = pd.to_numeric(merged["rf_daily"], errors="coerce").ffill()
    merged["ret"] = merged["Adj_Close"].pct_change()
    merged["next_date"] = merged["Date"].shift(-1)
    merged["next_ret"] = merged["ret"].shift(-1)
    merged["next_rf_daily"] = merged["rf_daily"].shift(-1)
    merged = merged.dropna(subset=["Date", "next_date", "next_ret", "next_rf_daily"]).reset_index(drop=True)
    return merged[["Date", "next_date", "ret", "rf_daily", "next_ret", "next_rf_daily"]].copy()


def simulate_mapped_strategy(signal_frame: pd.DataFrame, trade_base: pd.DataFrame) -> pd.DataFrame:
    merged = signal_frame.merge(trade_base, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    simulation_base = merged[["Date", "next_ret", "next_rf_daily"]].rename(columns={"Date": "execution_date"})
    simulation = simulate_positions_strategy(simulation_base, merged["predicted_label"].to_numpy(dtype=float, copy=False), initial_position=0)
    mapped_frame = simulation["frame"].copy()
    for column in [
        "rebalance_date",
        "predicted_label",
        "predicted_probability_raw",
        "predicted_probability_smoothed",
        "selected_smoothing_halflife",
        "selected_lower_threshold",
        "selected_upper_threshold",
        "signal_zone",
        "y_true",
        "rising_entry_trigger",
        "drawdown_entry_trigger",
    ]:
        if column in merged.columns:
            mapped_frame[column] = merged[column].to_numpy(copy=False)
    mapped_frame["mapped_spy_next_ret"] = merged["next_ret"].to_numpy(copy=False)
    mapped_frame["mapped_spy_next_rf_daily"] = merged["next_rf_daily"].to_numpy(copy=False)
    mapped_frame["mapped_spy_return_date"] = pd.to_datetime(merged["next_date"]).to_numpy(copy=False)
    return mapped_frame


def worker_run_ml_stage(task: Tuple[Dict[str, str], Dict[str, object]]) -> Dict[str, object]:
    window_payload, config_dict = task
    config = StageConfig(**config_dict)
    prepared = prepare_window(window_payload, config.feature_mode)
    selected_smoothing = select_smoothing(prepared["validation_cache"], config)
    selected_thresholds = select_thresholds(prepared["validation_cache"], config, selected_smoothing)
    if config.rule_mode == "single_threshold":
        positions, zones, _ = single_threshold_positions(prepared["oos_smoothed"][selected_smoothing], 0, float(config.threshold))
        rising_entry = np.zeros(len(positions), dtype=int)
        drawdown_entry = np.zeros(len(positions), dtype=int)
    elif config.rule_mode == "single_threshold_overlay":
        positions, zones, rising_entry, drawdown_entry, _ = single_threshold_overlay_positions(
            prepared["oos_smoothed"][selected_smoothing],
            prepared["oos_drawdown"],
            0,
            float(config.threshold),
            float(config.rising_floor),
            float(config.drawdown_threshold),
            float(config.drawdown_prob_floor),
        )
    elif config.rule_mode == "single_threshold_drawdown_overlay":
        positions, zones, drawdown_entry, _ = single_threshold_drawdown_overlay_positions(
            prepared["oos_smoothed"][selected_smoothing],
            prepared["oos_drawdown"],
            0,
            float(config.threshold),
            float(config.drawdown_threshold),
            float(config.drawdown_prob_floor),
        )
        rising_entry = np.zeros(len(positions), dtype=int)
    elif config.rule_mode == "single_threshold_drawdown_rising_overlay":
        positions, zones, rising_entry, drawdown_entry, _ = single_threshold_drawdown_rising_overlay_positions(
            prepared["oos_smoothed"][selected_smoothing],
            prepared["oos_drawdown"],
            0,
            float(config.threshold),
            float(config.rising_floor),
            float(config.drawdown_threshold),
            float(config.drawdown_prob_floor),
        )
    elif config.rule_mode == "double_threshold":
        positions, zones, _ = double_threshold_positions(
            prepared["oos_smoothed"][selected_smoothing],
            0,
            float(selected_thresholds["selected_lower_threshold"]),
            float(selected_thresholds["selected_upper_threshold"]),
        )
        rising_entry = np.zeros(len(positions), dtype=int)
        drawdown_entry = np.zeros(len(positions), dtype=int)
    elif config.rule_mode == "final_overlay":
        positions, zones, rising_entry, drawdown_entry, _ = final_overlay_positions(
            prepared["oos_smoothed"][selected_smoothing],
            prepared["oos_drawdown"],
            0,
            float(selected_thresholds["selected_lower_threshold"]),
            float(selected_thresholds["selected_upper_threshold"]),
            float(config.rising_floor),
            float(config.drawdown_threshold),
            float(config.drawdown_prob_floor),
        )
    else:
        raise ValueError(f"Unsupported rule_mode: {config.rule_mode}")

    signal_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(prepared["oos_signal_date"]),
            "rebalance_date": window_payload["rebalance_date"],
            "predicted_probability_raw": np.asarray(prepared["oos_raw_prob"], dtype=float),
            "predicted_probability_smoothed": np.asarray(prepared["oos_smoothed"][selected_smoothing], dtype=float),
            "predicted_label": np.asarray(positions, dtype=int),
            "predicted_bull_flag": np.asarray(positions, dtype=int),
            "predicted_bear_flag": (1 - np.asarray(positions, dtype=int)),
            "selected_smoothing_halflife": int(selected_smoothing),
            "selected_lower_threshold": float(selected_thresholds["selected_lower_threshold"]) if not pd.isna(selected_thresholds["selected_lower_threshold"]) else np.nan,
            "selected_upper_threshold": float(selected_thresholds["selected_upper_threshold"]),
            "signal_zone": zones,
            "y_true": np.asarray(prepared["oos_y_true"], dtype=int),
            "rising_entry_trigger": rising_entry,
            "drawdown_entry_trigger": drawdown_entry,
        }
    )
    mapped_slice = simulate_mapped_strategy(signal_frame, load_trade_base())
    oos_metrics = compute_metrics(
        np.asarray(prepared["oos_y_true"], dtype=int),
        np.asarray(positions, dtype=int),
        np.asarray(prepared["oos_smoothed"][selected_smoothing], dtype=float),
    )
    return {
        "window": window_payload,
        "signal_frame": signal_frame.to_dict(orient="records"),
        "selected_smoothing_halflife": int(selected_smoothing),
        "selected_lower_threshold": float(selected_thresholds["selected_lower_threshold"]) if not pd.isna(selected_thresholds["selected_lower_threshold"]) else np.nan,
        "selected_upper_threshold": float(selected_thresholds["selected_upper_threshold"]),
        "oos_sharpe": float(annualized_sharpe(mapped_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False))),
        "oos_metrics": oos_metrics,
    }


def run_ml_stage(config: StageConfig) -> Dict[str, object]:
    frame = load_research_frame()
    windows = build_target_windows(frame)
    worker_count = min(MAX_WORKERS_CAP, len(windows), os.cpu_count() or 1)
    tasks = [(window.__dict__, asdict(config)) for window in windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_run_ml_stage, tasks))

    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])
    signal_frame = pd.concat([pd.DataFrame(item["signal_frame"]) for item in window_results], ignore_index=True)
    signal_frame["Date"] = pd.to_datetime(signal_frame["Date"])
    signal_frame = signal_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    trade_base = load_trade_base()
    mapped_frame = simulate_mapped_strategy(signal_frame, trade_base)
    mapped_frame["equity_predicted_strategy"] = (1.0 + mapped_frame["strategy_ret"]).cumprod()

    buyhold = trade_base[trade_base["Date"].isin(mapped_frame["Date"])].copy().sort_values("Date").reset_index(drop=True)
    buyhold["position"] = 1.0
    buyhold["fee"] = 0.0
    buyhold["strategy_ret_gross"] = buyhold["next_ret"]
    buyhold["strategy_ret"] = buyhold["next_ret"]
    buyhold["strategy_excess_ret"] = buyhold["next_ret"] - buyhold["next_rf_daily"]
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    prediction_metrics = pd.DataFrame(
        [
            {
                "version": config.stage_name,
                "avg_oos_accuracy": float(np.mean([row["oos_metrics"]["accuracy"] for row in window_results])),
                "avg_oos_balanced_accuracy": float(np.mean([row["oos_metrics"]["balanced_accuracy"] for row in window_results])),
                "avg_oos_f1": float(np.mean([row["oos_metrics"]["f1"] for row in window_results])),
                "avg_oos_log_loss": float(np.mean([row["oos_metrics"]["log_loss"] for row in window_results])),
                "avg_oos_roc_auc": float(np.mean([row["oos_metrics"]["roc_auc"] for row in window_results])),
            }
        ]
    )

    selection_log = pd.DataFrame(
        [
            {
                "version": config.stage_name,
                "rebalance_date": row["window"]["rebalance_date"],
                "selected_smoothing_halflife": row["selected_smoothing_halflife"],
                "selected_lower_threshold": row["selected_lower_threshold"],
                "selected_upper_threshold": row["selected_upper_threshold"],
                "oos_sharpe": row["oos_sharpe"],
                "oos_accuracy": row["oos_metrics"]["accuracy"],
                "oos_balanced_accuracy": row["oos_metrics"]["balanced_accuracy"],
                "oos_f1": row["oos_metrics"]["f1"],
            }
            for row in window_results
        ]
    ).sort_values("rebalance_date").reset_index(drop=True)

    return {
        "config": asdict(config),
        "signal_frame": signal_frame,
        "mapped_frame": mapped_frame,
        "buyhold_frame": buyhold,
        "selection_log": selection_log,
        "prediction_metrics": prediction_metrics,
    }


def _build_oracle_signal_frame(bundle: Dict[str, object], rebalance_date: str, date_column: str) -> pd.DataFrame:
    y_next = np.asarray(bundle["segment_dataset"]["y"], dtype=int)
    base = bundle["segment_dataset"]["base"]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(base[date_column]),
            "signal_date": pd.to_datetime(base["signal_date"]),
            "label_date": pd.to_datetime(base["execution_date"]),
            "rebalance_date": rebalance_date,
            "predicted_probability_raw": y_next.astype(float),
            "predicted_probability_smoothed": y_next.astype(float),
            "predicted_label": y_next,
            "predicted_bull_flag": y_next,
            "predicted_bear_flag": 1 - y_next,
            "selected_smoothing_halflife": 0,
            "selected_lower_threshold": np.nan,
            "selected_upper_threshold": 0.50,
            "signal_zone": np.where(y_next == 1, "bull", "bear"),
            "y_true": y_next,
            "rising_entry_trigger": 0,
            "drawdown_entry_trigger": 0,
        }
    )


def run_current_mapped_oracle_stage() -> Dict[str, object]:
    frame = load_research_frame()
    windows = build_target_windows(frame)
    trade_base = load_trade_base()
    payloads = []
    for window in windows:
        bundle = build_segment_supervised_bundle(
            frame,
            BASE_JM_FEATURES,
            pd.Timestamp(window.oos_start),
            pd.Timestamp(window.oos_end) + pd.Timedelta(days=1),
            penalty=JUMP_PENALTY,
        )
        signal_frame = _build_oracle_signal_frame(bundle, window.rebalance_date, "execution_date")
        payloads.append(signal_frame)
    signal_frame = pd.concat(payloads, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    mapped_frame = simulate_mapped_strategy(signal_frame, trade_base)
    mapped_frame["equity_predicted_strategy"] = (1.0 + mapped_frame["strategy_ret"]).cumprod()
    buyhold = trade_base[trade_base["Date"].isin(mapped_frame["Date"])].copy().sort_values("Date").reset_index(drop=True)
    buyhold["position"] = 1.0
    buyhold["fee"] = 0.0
    buyhold["strategy_ret_gross"] = buyhold["next_ret"]
    buyhold["strategy_ret"] = buyhold["next_ret"]
    buyhold["strategy_excess_ret"] = buyhold["next_ret"] - buyhold["next_rf_daily"]
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()
    return {"signal_frame": signal_frame, "mapped_frame": mapped_frame, "buyhold_frame": buyhold}


def run_full_upper_bound_oracle_stage() -> Dict[str, object]:
    frame = load_research_frame()
    windows = build_target_windows(frame)
    trade_base = load_trade_base()
    payloads = []
    for window in windows:
        bundle = build_segment_supervised_bundle(
            frame,
            BASE_JM_FEATURES,
            pd.Timestamp(window.oos_start),
            pd.Timestamp(window.oos_end) + pd.Timedelta(days=1),
            penalty=JUMP_PENALTY,
        )
        # Full upper-bound oracle: use realized label_{t+1} on decision date t,
        # then directly earn mapped trade ret_{t+1}.
        signal_frame = _build_oracle_signal_frame(bundle, window.rebalance_date, "signal_date")
        payloads.append(signal_frame)
    signal_frame = pd.concat(payloads, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    mapped_frame = simulate_mapped_strategy(signal_frame, trade_base)
    mapped_frame["equity_predicted_strategy"] = (1.0 + mapped_frame["strategy_ret"]).cumprod()
    buyhold = trade_base[trade_base["Date"].isin(mapped_frame["Date"])].copy().sort_values("Date").reset_index(drop=True)
    buyhold["position"] = 1.0
    buyhold["fee"] = 0.0
    buyhold["strategy_ret_gross"] = buyhold["next_ret"]
    buyhold["strategy_ret"] = buyhold["next_ret"]
    buyhold["strategy_excess_ret"] = buyhold["next_ret"] - buyhold["next_rf_daily"]
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()
    return {"signal_frame": signal_frame, "mapped_frame": mapped_frame, "buyhold_frame": buyhold}


def run_oracle_stage() -> Dict[str, object]:
    return run_current_mapped_oracle_stage()


def write_stage_outputs(
    results_subdir: str,
    version_name: str,
    mapped_frame: pd.DataFrame,
    buyhold_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    selection_log: pd.DataFrame | None = None,
    prediction_metrics: pd.DataFrame | None = None,
    output_ml_figures: bool = False,
) -> Path:
    out_dir = stage_results_dir(results_subdir)
    strategy_summary = pd.DataFrame(
        [
            {"version": version_name, **build_strategy_metrics(mapped_frame)},
            {"version": BUYHOLD_VERSION, **build_strategy_metrics(buyhold_frame)},
        ]
    )
    strategy_summary.to_csv(out_dir / "strategy_performance_summary.csv", index=False)

    daily_equity = pd.DataFrame({"Date": mapped_frame["Date"]})
    daily_equity["equity_strategy"] = mapped_frame["equity_predicted_strategy"].to_numpy()
    daily_equity = daily_equity.merge(buyhold_frame[["Date", "equity_buy_and_hold"]], on="Date", how="left")
    daily_equity.to_csv(out_dir / "daily_equity_curves.csv", index=False)
    signal_frame.to_csv(out_dir / "signal_panel.csv", index=False)
    mapped_frame.to_csv(out_dir / "mapped_strategy_daily_detail.csv", index=False)
    buyhold_frame.to_csv(out_dir / "buy_and_hold_daily_detail.csv", index=False)
    if selection_log is not None:
        selection_log.to_csv(out_dir / "selection_log.csv", index=False)
    if prediction_metrics is not None:
        prediction_metrics.to_csv(out_dir / "prediction_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_equity["Date"], daily_equity["equity_strategy"], label=version_name, linewidth=1.8)
    ax.plot(daily_equity["Date"], daily_equity["equity_buy_and_hold"], label=BUYHOLD_VERSION, linewidth=1.8)
    ax.set_title(f"{version_name} vs buy-and-hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_vs_buyhold.png", dpi=150)
    plt.close(fig)

    if output_ml_figures:
        y_true = signal_frame["y_true"].to_numpy(dtype=int, copy=False)
        y_pred = signal_frame["predicted_label"].to_numpy(dtype=int, copy=False)
        y_prob = signal_frame["predicted_probability_smoothed"].to_numpy(dtype=float, copy=False)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Bear", "Bull"])
        ax.set_yticks([0, 1], ["Bear", "Bull"])
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        order = np.argsort(y_prob)
        thresholds = np.r_[0.0, np.unique(y_prob[order]), 1.0]
        positives = max(int((y_true == 1).sum()), 1)
        negatives = max(int((y_true == 0).sum()), 1)
        tpr: List[float] = []
        fpr: List[float] = []
        for threshold in thresholds:
            pred = (y_prob >= threshold).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

        metrics_by_window = signal_frame.groupby("rebalance_date").apply(
            lambda df: pd.Series(
                {
                    "accuracy": accuracy_score(df["y_true"], df["predicted_label"]),
                    "f1": f1_score(df["y_true"], df["predicted_label"], zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(df["y_true"], df["predicted_label"]),
                }
            )
        ).reset_index()
        metrics_by_window.to_csv(out_dir / "classification_metrics_over_time.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pd.to_datetime(metrics_by_window["rebalance_date"]), metrics_by_window["accuracy"], label="accuracy")
        ax.plot(pd.to_datetime(metrics_by_window["rebalance_date"]), metrics_by_window["f1"], label="f1")
        ax.plot(pd.to_datetime(metrics_by_window["rebalance_date"]), metrics_by_window["balanced_accuracy"], label="balanced_accuracy")
        ax.set_title("Classification Metrics Over Time")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "classification_metrics_over_time.png", dpi=150)
        plt.close(fig)

    return out_dir
