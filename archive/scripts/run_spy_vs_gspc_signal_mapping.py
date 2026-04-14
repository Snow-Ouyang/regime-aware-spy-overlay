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
    vendor_site = Path(__file__).resolve().parents[1] / ".vendor"
    if vendor_site.exists():
        vendor_site_str = str(vendor_site)
        if vendor_site_str not in sys.path:
            sys.path.append(vendor_site_str)


configure_paths()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from build_all_jm_features import build_features, load_risk_free_data
from final_multi_asset_project_common import (
    FIXED_XGB_PARAMS,
    ensure_raw_data,
    load_risk_free_daily_series,
    load_trade_price_frame,
)
from spy_regime_common import (
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


TICKER_VERSION_TO_SIGNAL: List[Tuple[str, str, str]] = [
    ("spy_signal_on_spy", "SPY", "spy"),
    ("gspc_signal_on_spy", "^GSPC", "gspc"),
]
TRADE_STEM = "spy_trade"
TRADE_TICKER = "SPY"
TARGET_OOS_START = pd.Timestamp("2008-04-28")
VALIDATION_YEARS = 4
TRAIN_YEARS = 11
STEP_MONTHS = 6
JUMP_PENALTY = 0.0
THRESHOLD = 0.60
SMOOTHING_GRID: List[int] = [0, 4, 8, 12]
MAX_WORKERS_CAP = 8
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
A1_REFINED_FEATURES: List[str] = [
    "ret_1_5d",
    "ret_6_20d",
    "ret_21_60d",
]
A2_FEATURES: List[str] = [
    "close_over_ma20",
    "close_over_ma60",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_data_dir() -> Path:
    output_dir = project_root() / "data_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def features_dir() -> Path:
    output_dir = project_root() / "data_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def results_dir() -> Path:
    output_dir = project_root() / "results" / "spy_vs_gspc_signal_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def signal_asset_dir(version: str) -> Path:
    output_dir = results_dir() / version
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def panels_dir() -> Path:
    output_dir = results_dir() / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def raw_path_for_signal_stem(stem: str) -> Path:
    return raw_data_dir() / f"{stem}.csv"


def feature_path_for_signal_stem(stem: str) -> Path:
    if stem == "spy":
        return features_dir() / "spy_jm_features.csv"
    if stem == "gspc":
        return features_dir() / "gspc_features_final.csv"
    raise ValueError(f"Unsupported signal stem: {stem}")


def macro_feature_path() -> Path:
    return features_dir() / "macro_feature_panel_m0.csv"


def ensure_signal_raw_input(ticker: str, stem: str) -> Path:
    output_path = raw_path_for_signal_stem(stem)
    if output_path.exists():
        return output_path
    return ensure_raw_data(ticker, stem)


def ensure_signal_feature_input(ticker: str, stem: str) -> Path:
    output_path = feature_path_for_signal_stem(stem)
    if output_path.exists():
        return output_path

    ensure_signal_raw_input(ticker, stem)
    rf_frame, _, _ = load_risk_free_data()
    asset_frame = pd.read_csv(raw_path_for_signal_stem(stem))
    asset_frame.columns = [str(column).strip() for column in asset_frame.columns]
    asset_frame["Date"] = pd.to_datetime(asset_frame["Date"], errors="coerce")
    asset_frame["Adj_Close"] = pd.to_numeric(asset_frame["Adj_Close"], errors="coerce")
    asset_frame = asset_frame.dropna(subset=["Date", "Adj_Close"]).sort_values("Date").reset_index(drop=True)
    asset_frame["Ticker"] = ticker
    asset_frame = asset_frame[["Date", "Ticker", "Adj_Close"]].copy()
    feature_frame = build_features(asset_frame, rf_frame)
    feature_frame.to_csv(output_path, index=False)
    return output_path


def ensure_inputs() -> None:
    for ticker, _, stem in TICKER_VERSION_TO_SIGNAL:
        ensure_signal_raw_input(ticker, stem)
        ensure_signal_feature_input(ticker, stem)
    ensure_raw_data(TRADE_TICKER, TRADE_STEM)
    if not macro_feature_path().exists():
        raise FileNotFoundError(f"Missing macro feature file: {macro_feature_path()}")


def load_signal_experiment_frame(stem: str) -> pd.DataFrame:
    feature_frame = pd.read_csv(feature_path_for_signal_stem(stem))
    macro_frame = pd.read_csv(macro_feature_path())
    raw_frame = pd.read_csv(raw_path_for_signal_stem(stem))

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
    price_features = pd.DataFrame(
        {
            "Date": raw_frame["Date"],
            "ret_1_5d": price / price.shift(5) - 1.0,
            "ret_6_20d": price.shift(5) / price.shift(20) - 1.0,
            "ret_21_60d": price.shift(20) / price.shift(60) - 1.0,
            "close_over_ma20": price / ma20 - 1.0,
            "close_over_ma60": price / ma60 - 1.0,
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
        *BASE_JM_FEATURES,
        *A1_REFINED_FEATURES,
        *A2_FEATURES,
        *macro_columns,
    ]
    merged = merged.dropna(subset=required_columns).reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"{stem}: merged signal dataset is empty")
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
    return [*BASE_JM_FEATURES, *A1_REFINED_FEATURES, *A2_FEATURES, *macro_columns]


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
        segment_mask = (frame["Date"] >= segment["start"]) & (frame["Date"] < segment["end_exclusive"])
        if int(segment_mask.sum()) > 0:
            valid_segments.append(segment)
    return valid_segments


def make_xgb_model() -> XGBClassifier:
    return XGBClassifier(**FIXED_XGB_PARAMS)


def smooth_probability_series(probabilities: np.ndarray, halflife: int) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if halflife == 0:
        return raw
    return pd.Series(raw).ewm(halflife=halflife, adjust=False).mean().to_numpy(dtype=float, copy=False)


def probability_to_label(probabilities: np.ndarray) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) >= THRESHOLD).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "bull_rate_predicted": float(np.mean(y_pred)),
        "bull_rate_actual": float(np.mean(y_true)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def evaluate_validation_and_oos(
    validation_bundles: List[Dict[str, object]],
    oos_bundle: Dict[str, object],
) -> Dict[str, object]:
    candidate_rows: List[Dict[str, object]] = []
    results_by_halflife: Dict[int, Dict[str, object]] = {}
    for smoothing_halflife in SMOOTHING_GRID:
        current_position = 0
        strategy_frames: List[pd.DataFrame] = []
        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []
        y_prob_list: List[np.ndarray] = []

        for bundle in validation_bundles:
            model = make_xgb_model()
            model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
            raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1]
            smoothed_prob = smooth_probability_series(raw_prob, smoothing_halflife)
            pred = probability_to_label(smoothed_prob)
            simulation = simulate_next_day_strategy(
                bundle["segment_dataset"]["base"],
                pred,
                initial_position=current_position,
            )
            current_position = int(simulation["final_position"])
            strategy_frames.append(simulation["frame"])
            y_true_list.append(bundle["segment_dataset"]["y"])
            y_pred_list.append(pred)
            y_prob_list.append(smoothed_prob)

        validation_frame = pd.concat(strategy_frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
        validation_result = {
            "sharpe": annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
            "metrics": compute_metrics(
                np.concatenate(y_true_list),
                np.concatenate(y_pred_list),
                np.concatenate(y_prob_list),
            ),
        }
        results_by_halflife[smoothing_halflife] = validation_result
        candidate_rows.append(
            {
                "selected_smoothing_halflife": smoothing_halflife,
                "validation_sharpe": validation_result["sharpe"],
                "validation_balanced_accuracy": validation_result["metrics"]["balanced_accuracy"],
                "version_id": f"h{smoothing_halflife}",
            }
        )

    selected = (
        pd.DataFrame(candidate_rows)
        .sort_values(
            by=[
                "validation_sharpe",
                "validation_balanced_accuracy",
                "selected_smoothing_halflife",
                "version_id",
            ],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
        .iloc[0]
        .to_dict()
    )
    selected_halflife = int(selected["selected_smoothing_halflife"])
    chosen_validation = results_by_halflife[selected_halflife]

    model = make_xgb_model()
    model.fit(oos_bundle["train_dataset"]["X"], oos_bundle["train_dataset"]["y"])
    raw_prob = model.predict_proba(oos_bundle["segment_dataset"]["X"])[:, 1]
    smoothed_prob = smooth_probability_series(raw_prob, selected_halflife)
    y_pred = probability_to_label(smoothed_prob)
    oos_metrics = compute_metrics(oos_bundle["segment_dataset"]["y"], y_pred, smoothed_prob)
    oos_simulation = simulate_next_day_strategy(
        oos_bundle["segment_dataset"]["base"],
        y_pred,
        initial_position=0,
    )

    signal_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(oos_bundle["segment_dataset"]["base"]["execution_date"]),
            "rebalance_date": oos_bundle["rebalance_date"],
            "predicted_probability_raw": raw_prob.astype(float, copy=False),
            "predicted_probability_smoothed": smoothed_prob.astype(float, copy=False),
            "predicted_label": y_pred.astype(int, copy=False),
            "predicted_bull_flag": y_pred.astype(int, copy=False),
            "predicted_bear_flag": (1 - y_pred).astype(int, copy=False),
            "selected_smoothing_halflife": selected_halflife,
            "selected_threshold": THRESHOLD,
        }
    )

    return {
        "selected_smoothing_halflife": selected_halflife,
        "validation": chosen_validation,
        "oos_metrics": oos_metrics,
        "oos_sharpe": oos_simulation["sharpe"],
        "signal_frame": signal_frame,
    }


def worker_process_window(task: Tuple[str, Dict[str, str]]) -> Dict[str, object]:
    version, window_payload = task
    _, _, stem = next(item for item in TICKER_VERSION_TO_SIGNAL if item[0] == version)
    frame = load_signal_experiment_frame(stem)
    features = research_feature_columns(frame)
    window = MainWindow(**window_payload)

    validation_segments = build_nonempty_segments(
        frame,
        pd.Timestamp(window.val_start),
        pd.Timestamp(window.oos_start),
    )
    validation_bundles = [
        {
            **build_segment_supervised_bundle(
                frame,
                features,
                segment["start"],
                segment["end_exclusive"],
                penalty=JUMP_PENALTY,
            ),
            "rebalance_date": window.rebalance_date,
        }
        for segment in validation_segments
    ]
    oos_bundle = build_segment_supervised_bundle(
        frame,
        features,
        pd.Timestamp(window.oos_start),
        pd.Timestamp(window.oos_end) + pd.Timedelta(days=1),
        penalty=JUMP_PENALTY,
    )
    oos_bundle["rebalance_date"] = window.rebalance_date

    selected_result = evaluate_validation_and_oos(validation_bundles, oos_bundle)
    return {
        "version": version,
        "ticker": stem,
        "window": window_payload,
        "selected_smoothing_halflife": selected_result["selected_smoothing_halflife"],
        "validation": selected_result["validation"],
        "oos_metrics": selected_result["oos_metrics"],
        "oos_sharpe": selected_result["oos_sharpe"],
        "signal_frame": selected_result["signal_frame"],
    }


def run_signal_pipeline(version: str) -> Dict[str, object]:
    _, _, stem = next(item for item in TICKER_VERSION_TO_SIGNAL if item[0] == version)
    frame = load_signal_experiment_frame(stem)
    main_windows = build_target_windows(frame)
    worker_count = min(MAX_WORKERS_CAP, len(main_windows), os.cpu_count() or 1)
    tasks = [(version, window.__dict__) for window in main_windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_process_window, tasks))

    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])
    signal_frame = pd.concat([result["signal_frame"] for result in window_results], ignore_index=True)
    signal_frame = signal_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    selection_log = pd.DataFrame(
        [
            {
                "version": version,
                "rebalance_date": result["window"]["rebalance_date"],
                "selected_smoothing_halflife": result["selected_smoothing_halflife"],
                "validation_sharpe": result["validation"]["sharpe"],
                "validation_balanced_accuracy": result["validation"]["metrics"]["balanced_accuracy"],
                "oos_sharpe": result["oos_sharpe"],
                "oos_accuracy": result["oos_metrics"]["accuracy"],
                "oos_balanced_accuracy": result["oos_metrics"]["balanced_accuracy"],
                "oos_f1": result["oos_metrics"]["f1"],
            }
            for result in window_results
        ]
    ).sort_values("rebalance_date").reset_index(drop=True)

    prediction_metrics = pd.DataFrame(
        [
            {
                "version": version,
                "avg_validation_accuracy": float(np.mean([result["validation"]["metrics"]["accuracy"] for result in window_results])),
                "avg_validation_balanced_accuracy": float(np.mean([result["validation"]["metrics"]["balanced_accuracy"] for result in window_results])),
                "avg_validation_f1": float(np.mean([result["validation"]["metrics"]["f1"] for result in window_results])),
                "avg_validation_log_loss": float(np.mean([result["validation"]["metrics"]["log_loss"] for result in window_results])),
                "avg_oos_accuracy": float(np.mean([result["oos_metrics"]["accuracy"] for result in window_results])),
                "avg_oos_balanced_accuracy": float(np.mean([result["oos_metrics"]["balanced_accuracy"] for result in window_results])),
                "avg_oos_f1": float(np.mean([result["oos_metrics"]["f1"] for result in window_results])),
                "avg_oos_log_loss": float(np.mean([result["oos_metrics"]["log_loss"] for result in window_results])),
            }
        ]
    )

    return {
        "version": version,
        "ticker": stem,
        "frame": frame,
        "main_windows": main_windows,
        "signal_frame": signal_frame,
        "selection_log": selection_log,
        "prediction_metrics": prediction_metrics,
        "window_results": window_results,
        "first_oos_start": pd.Timestamp(main_windows[0].oos_start) if main_windows else None,
    }


def load_spy_trade_base() -> pd.DataFrame:
    trade_frame = load_trade_price_frame(TRADE_STEM)
    rf_frame = load_risk_free_daily_series()
    merged = trade_frame.merge(rf_frame, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["rf_daily"] = pd.to_numeric(merged["rf_daily"], errors="coerce").ffill()
    merged["ret"] = merged["Adj_Close"].pct_change()
    merged["next_ret"] = merged["ret"].shift(-1)
    merged["next_rf_daily"] = merged["rf_daily"].shift(-1)
    merged = merged.dropna(subset=["Date", "next_ret", "next_rf_daily"]).reset_index(drop=True)
    return merged[["Date", "ret", "rf_daily", "next_ret", "next_rf_daily"]].copy()


def simulate_mapped_strategy(signal_frame: pd.DataFrame, spy_trade_base: pd.DataFrame) -> Dict[str, object]:
    merged = signal_frame.merge(spy_trade_base, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping dates between signals and SPY trade base")

    simulation_base = merged[["Date", "next_ret", "next_rf_daily"]].rename(columns={"Date": "execution_date"})
    simulation = simulate_next_day_strategy(
        simulation_base,
        merged["predicted_label"].to_numpy(dtype=int, copy=False),
        initial_position=0,
    )
    mapped_frame = simulation["frame"].copy()
    mapped_frame["rebalance_date"] = merged["rebalance_date"].astype(str).to_numpy(copy=False)
    mapped_frame["predicted_label"] = merged["predicted_label"].to_numpy(dtype=int, copy=False)
    mapped_frame["predicted_probability_raw"] = merged["predicted_probability_raw"].to_numpy(dtype=float, copy=False)
    mapped_frame["predicted_probability_smoothed"] = merged["predicted_probability_smoothed"].to_numpy(dtype=float, copy=False)
    mapped_frame["selected_smoothing_halflife"] = merged["selected_smoothing_halflife"].to_numpy(dtype=int, copy=False)
    mapped_frame["selected_threshold"] = merged["selected_threshold"].to_numpy(dtype=float, copy=False)
    mapped_frame["signal_asset"] = merged["signal_asset"].to_numpy(copy=False)
    return {"merged": merged, "simulation": simulation, "frame": mapped_frame}


def build_signal_comparison_daily(spy_signal: pd.DataFrame, gspc_signal: pd.DataFrame) -> pd.DataFrame:
    left = spy_signal.rename(
        columns={
            "rebalance_date": "spy_rebalance_date",
            "predicted_probability_raw": "spy_predicted_probability_raw",
            "predicted_probability_smoothed": "spy_predicted_probability_smoothed",
            "predicted_label": "spy_predicted_label",
            "predicted_bull_flag": "spy_predicted_bull_flag",
            "predicted_bear_flag": "spy_predicted_bear_flag",
            "selected_smoothing_halflife": "spy_selected_smoothing_halflife",
        }
    )
    right = gspc_signal.rename(
        columns={
            "rebalance_date": "gspc_rebalance_date",
            "predicted_probability_raw": "gspc_predicted_probability_raw",
            "predicted_probability_smoothed": "gspc_predicted_probability_smoothed",
            "predicted_label": "gspc_predicted_label",
            "predicted_bull_flag": "gspc_predicted_bull_flag",
            "predicted_bear_flag": "gspc_predicted_bear_flag",
            "selected_smoothing_halflife": "gspc_selected_smoothing_halflife",
        }
    )
    merged = left.merge(
        right[
            [
                "Date",
                "gspc_rebalance_date",
                "gspc_predicted_probability_raw",
                "gspc_predicted_probability_smoothed",
                "gspc_predicted_label",
                "gspc_predicted_bull_flag",
                "gspc_predicted_bear_flag",
                "gspc_selected_smoothing_halflife",
            ]
        ],
        on="Date",
        how="inner",
    )
    return merged.sort_values("Date").reset_index(drop=True)


def build_disagreement_summary(comparison_daily: pd.DataFrame) -> pd.DataFrame:
    same_label_days = int((comparison_daily["spy_predicted_label"] == comparison_daily["gspc_predicted_label"]).sum())
    total_days = int(len(comparison_daily))
    different_label_days = total_days - same_label_days
    return pd.DataFrame(
        [
            {
                "total_days": total_days,
                "same_label_days": same_label_days,
                "different_label_days": different_label_days,
                "disagreement_ratio": float(different_label_days / total_days) if total_days else np.nan,
            }
        ]
    )


def build_window_disagreement(comparison_daily: pd.DataFrame) -> pd.DataFrame:
    window_frame = comparison_daily.copy()
    window_frame["different_label_flag"] = (window_frame["spy_predicted_label"] != window_frame["gspc_predicted_label"]).astype(int)
    grouped = (
        window_frame.groupby("spy_rebalance_date", as_index=False)
        .agg(
            total_days=("Date", "count"),
            different_label_days=("different_label_flag", "sum"),
        )
        .rename(columns={"spy_rebalance_date": "rebalance_date"})
    )
    grouped["disagreement_ratio"] = grouped["different_label_days"] / grouped["total_days"]
    return grouped.sort_values("rebalance_date").reset_index(drop=True)


def plot_equity_curves(daily_equity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(daily_equity["Date"], daily_equity["equity_spy_signal_on_spy"], label="spy_signal_on_spy", linewidth=2.0)
    ax.plot(daily_equity["Date"], daily_equity["equity_gspc_signal_on_spy"], label="gspc_signal_on_spy", linewidth=2.0)
    ax.plot(daily_equity["Date"], daily_equity["equity_spy_buy_and_hold"], label="spy_buy_and_hold", linewidth=2.2)
    ax.set_title("SPY vs GSPC Signal Mapping on SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_signal_disagreement_over_time(comparison_daily: pd.DataFrame, output_path: Path) -> None:
    frame = comparison_daily.copy()
    frame["different_label_flag"] = (frame["spy_predicted_label"] != frame["gspc_predicted_label"]).astype(float)
    frame["rolling_disagreement_63d"] = frame["different_label_flag"].rolling(63, min_periods=10).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frame["Date"], frame["rolling_disagreement_63d"], color="#b22222", linewidth=2.0)
    ax.set_title("SPY vs GSPC Signal Disagreement Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("63D Rolling Disagreement Ratio")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_smoothing_choice_comparison(selection_logs: Dict[str, pd.DataFrame], output_path: Path) -> None:
    smoothing_values = sorted(SMOOTHING_GRID)
    assets = list(selection_logs.keys())
    counts = []
    for version in assets:
        count_series = selection_logs[version]["selected_smoothing_halflife"].value_counts().reindex(smoothing_values, fill_value=0).sort_index()
        counts.append(count_series.to_numpy())

    x = np.arange(len(smoothing_values))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (asset, asset_counts) in enumerate(zip(assets, counts)):
        offset = (idx - (len(assets) - 1) / 2) * width
        ax.bar(x + offset, asset_counts, width=width, label=asset)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in smoothing_values])
    ax.set_title("Smoothing Choice Comparison")
    ax.set_xlabel("Smoothing halflife")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    ensure_inputs()

    signal_runs: Dict[str, Dict[str, object]] = {}
    for version, _, stem in TICKER_VERSION_TO_SIGNAL:
        signal_runs[version] = run_signal_pipeline(version)

    spy_signal = signal_runs["spy_signal_on_spy"]["signal_frame"].copy()
    gspc_signal = signal_runs["gspc_signal_on_spy"]["signal_frame"].copy()
    spy_signal["signal_asset"] = "spy_signal_on_spy"
    gspc_signal["signal_asset"] = "gspc_signal_on_spy"

    comparison_daily = build_signal_comparison_daily(spy_signal, gspc_signal)
    disagreement_summary = build_disagreement_summary(comparison_daily)
    window_disagreement = build_window_disagreement(comparison_daily)

    common_dates = pd.DataFrame({"Date": comparison_daily["Date"].drop_duplicates().sort_values().to_list()})
    spy_trade_base = load_spy_trade_base().merge(common_dates, on="Date", how="inner")
    spy_trade_base = spy_trade_base.sort_values("Date").reset_index(drop=True)

    execution_results: Dict[str, Dict[str, object]] = {}
    for version in ["spy_signal_on_spy", "gspc_signal_on_spy"]:
        signal_frame = signal_runs[version]["signal_frame"].copy()
        signal_frame["signal_asset"] = version
        mapped = simulate_mapped_strategy(signal_frame, spy_trade_base)
        execution_results[version] = {
            "signal_frame": signal_frame.merge(common_dates, on="Date", how="inner").sort_values("Date").reset_index(drop=True),
            "mapped": mapped,
        }

    execution_window = spy_trade_base[["Date", "next_ret", "next_rf_daily"]].copy()
    buyhold_base = execution_window.rename(columns={"next_ret": "ret", "next_rf_daily": "rf_daily"})
    spy_buyhold = build_buy_and_hold(buyhold_base, execution_window["Date"].iloc[0])

    strategy_rows = [
        {"version": "spy_signal_on_spy", **build_strategy_metrics(execution_results["spy_signal_on_spy"]["mapped"]["frame"])},
        {"version": "gspc_signal_on_spy", **build_strategy_metrics(execution_results["gspc_signal_on_spy"]["mapped"]["frame"])},
        {"version": "spy_buy_and_hold", **build_strategy_metrics(spy_buyhold)},
    ]
    strategy_summary = pd.DataFrame(strategy_rows)

    prediction_rows = []
    for version in ["spy_signal_on_spy", "gspc_signal_on_spy"]:
        run = signal_runs[version]
        prediction_rows.append(
            {
                "version": version,
                "avg_validation_accuracy": float(run["prediction_metrics"]["avg_validation_accuracy"].iloc[0]),
                "avg_validation_balanced_accuracy": float(run["prediction_metrics"]["avg_validation_balanced_accuracy"].iloc[0]),
                "avg_validation_f1": float(run["prediction_metrics"]["avg_validation_f1"].iloc[0]),
                "avg_validation_log_loss": float(run["prediction_metrics"]["avg_validation_log_loss"].iloc[0]),
                "avg_oos_accuracy": float(run["prediction_metrics"]["avg_oos_accuracy"].iloc[0]),
                "avg_oos_balanced_accuracy": float(run["prediction_metrics"]["avg_oos_balanced_accuracy"].iloc[0]),
                "avg_oos_f1": float(run["prediction_metrics"]["avg_oos_f1"].iloc[0]),
                "avg_oos_log_loss": float(run["prediction_metrics"]["avg_oos_log_loss"].iloc[0]),
            }
        )
    prediction_metrics = pd.DataFrame(prediction_rows)

    rolling_rows = []
    for version in ["spy_signal_on_spy", "gspc_signal_on_spy"]:
        run = signal_runs[version]
        mapped_frame = execution_results[version]["mapped"]["frame"]
        for _, row in run["selection_log"].iterrows():
            window_slice = mapped_frame[mapped_frame["rebalance_date"] == row["rebalance_date"]].copy()
            oos_sharpe = annualized_sharpe(window_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False)) if not window_slice.empty else np.nan
            rolling_rows.append(
                {
                    "version": version,
                    "rebalance_date": row["rebalance_date"],
                    "selected_smoothing_halflife": int(row["selected_smoothing_halflife"]),
                    "oos_sharpe": float(oos_sharpe),
                    "oos_accuracy": float(row["oos_accuracy"]),
                    "oos_balanced_accuracy": float(row["oos_balanced_accuracy"]),
                    "oos_f1": float(row["oos_f1"]),
                }
            )
    rolling_window_log = pd.DataFrame(rolling_rows).sort_values(["version", "rebalance_date"]).reset_index(drop=True)

    daily_equity = pd.DataFrame({"Date": spy_trade_base["Date"].to_numpy()})
    for version in ["spy_signal_on_spy", "gspc_signal_on_spy"]:
        mapped_frame = execution_results[version]["mapped"]["frame"].copy()
        mapped_frame[f"equity_{version}"] = (1.0 + mapped_frame["strategy_ret"]).cumprod()
        daily_equity = daily_equity.merge(mapped_frame[["Date", f"equity_{version}"]], on="Date", how="left")
    spy_buyhold = spy_buyhold.copy()
    spy_buyhold["equity_spy_buy_and_hold"] = (1.0 + spy_buyhold["strategy_ret"]).cumprod()
    daily_equity = daily_equity.merge(spy_buyhold[["Date", "equity_spy_buy_and_hold"]], on="Date", how="left")
    daily_equity = daily_equity.sort_values("Date").reset_index(drop=True)

    output_dir = results_dir()
    strategy_summary.to_csv(output_dir / "strategy_performance_comparison.csv", index=False)
    prediction_metrics.to_csv(output_dir / "prediction_metrics_comparison.csv", index=False)
    rolling_window_log.to_csv(output_dir / "rolling_window_metrics_comparison.csv", index=False)
    daily_equity.to_csv(output_dir / "daily_equity_curves_comparison.csv", index=False)
    comparison_daily.to_csv(output_dir / "signal_comparison_daily.csv", index=False)
    disagreement_summary.to_csv(output_dir / "signal_disagreement_summary.csv", index=False)
    window_disagreement.to_csv(output_dir / "signal_disagreement_by_window.csv", index=False)

    selection_logs = {version: run["selection_log"] for version, run in signal_runs.items()}
    for version, run in signal_runs.items():
        asset_dir = signal_asset_dir(version)
        run["signal_frame"].to_csv(asset_dir / f"{version}_signal_panel.csv", index=False)
        run["selection_log"].to_csv(asset_dir / f"{version}_selection_log.csv", index=False)

    plot_equity_curves(daily_equity, output_dir / "spy_vs_gspc_signal_on_spy_equity_curves.png")
    plot_signal_disagreement_over_time(comparison_daily, output_dir / "spy_vs_gspc_signal_disagreement_over_time.png")
    plot_smoothing_choice_comparison(selection_logs, output_dir / "smoothing_choice_comparison.png")

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Signal comparison rows: {len(comparison_daily)}")
    print(f"Disagreement ratio: {float(disagreement_summary['disagreement_ratio'].iloc[0]):.4f}")
    print(f"Results directory: {output_dir}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
