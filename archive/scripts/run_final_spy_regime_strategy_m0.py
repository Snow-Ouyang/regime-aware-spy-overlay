import os
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

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

from spy_regime_common import (
    MainWindow,
    annualized_sharpe,
    build_buy_and_hold,
    build_main_windows,
    build_segment_supervised_bundle,
    build_segments,
    build_strategy_metrics,
    simulate_next_day_strategy,
    stitch_oos_strategy,
)


def configure_paths() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)


configure_paths()

from xgboost import XGBClassifier


TICKER = "SPY"
VALIDATION_YEARS = 4
JUMP_PENALTY = 0.0
SMOOTHING_HALFLIFE_GRID: List[int] = [0, 2, 4, 8]
THRESHOLD_GRID: List[float] = [0.45, 0.50, 0.55]
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
A2_ORIGINAL_FEATURES: List[str] = [
    "close_over_ma20",
    "close_over_ma60",
]
FIXED_XGB_PARAMS: Dict[str, object] = {
    "max_depth": 4,
    "learning_rate": 0.10,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 0,
    "n_jobs": 1,
    "verbosity": 0,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ticker_feature_path() -> Path:
    return project_root() / "data_features" / "spy_jm_features.csv"


def ticker_raw_path() -> Path:
    return project_root() / "data_raw" / "spy.csv"


def macro_feature_path() -> Path:
    return project_root() / "data_features" / "macro_feature_panel_m0.csv"


def results_dir() -> Path:
    output_dir = project_root() / "results" / "final_spy_regime_strategy_m0"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_experiment_frame() -> pd.DataFrame:
    feature_frame = pd.read_csv(ticker_feature_path())
    macro_frame = pd.read_csv(macro_feature_path())
    raw_frame = pd.read_csv(ticker_raw_path())

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
    required_columns = ["Date", "ret", "rf_daily", "excess_ret", *BASE_JM_FEATURES, *A1_REFINED_FEATURES, *A2_ORIGINAL_FEATURES, *macro_columns]
    merged = merged.dropna(subset=required_columns).reset_index(drop=True)
    if merged.empty:
        raise ValueError("No valid rows remain after building final M0 dataset")
    return merged


def feature_columns_for_frame(frame: pd.DataFrame) -> List[str]:
    macro_columns = [
        column
        for column in frame.columns
        if column.startswith("dgs")
        or column.startswith("slope_")
        or column.startswith("vix_")
        or column.startswith("credit_spread_")
    ]
    return [*BASE_JM_FEATURES, *A1_REFINED_FEATURES, *A2_ORIGINAL_FEATURES, *macro_columns]


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


def build_config_grid() -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    index = 0
    for smoothing_halflife in SMOOTHING_HALFLIFE_GRID:
        for threshold in THRESHOLD_GRID:
            configs.append(
                {
                    "config_id": f"cfg_{index:03d}",
                    "smoothing_halflife": int(smoothing_halflife),
                    "threshold": float(threshold),
                }
            )
            index += 1
    return configs


CONFIG_GRID = build_config_grid()


def smooth_probability_series(probabilities: np.ndarray, halflife: int) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if halflife == 0:
        return raw
    return (
        pd.Series(raw)
        .ewm(halflife=halflife, adjust=False)
        .mean()
        .to_numpy(dtype=float, copy=False)
    )


def probability_to_label(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) >= threshold).astype(int)


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


def evaluate_config_on_validation(config: Dict[str, object], validation_bundles: List[Dict[str, object]]) -> Dict[str, object]:
    current_position = 0
    strategy_frames: List[pd.DataFrame] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for bundle in validation_bundles:
        model = make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1]
        smoothed_prob = smooth_probability_series(raw_prob, int(config["smoothing_halflife"]))
        pred = probability_to_label(smoothed_prob, float(config["threshold"]))
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
    metrics = compute_metrics(
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_prob_list),
    )
    sharpe = annualized_sharpe(
        validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)
    )
    return {
        "sharpe": sharpe,
        "metrics": metrics,
    }


def evaluate_config_on_oos(config: Dict[str, object], oos_bundle: Dict[str, object]) -> Dict[str, object]:
    model = make_xgb_model()
    model.fit(oos_bundle["train_dataset"]["X"], oos_bundle["train_dataset"]["y"])
    raw_prob = model.predict_proba(oos_bundle["segment_dataset"]["X"])[:, 1]
    smoothed_prob = smooth_probability_series(raw_prob, int(config["smoothing_halflife"]))
    pred = probability_to_label(smoothed_prob, float(config["threshold"]))
    metrics = compute_metrics(oos_bundle["segment_dataset"]["y"], pred, smoothed_prob)
    simulation = simulate_next_day_strategy(
        oos_bundle["segment_dataset"]["base"],
        pred,
        initial_position=0,
    )
    return {
        "oos_sharpe": simulation["sharpe"],
        "oos_metrics": metrics,
        "oos_dataset": {
            "execution_date": oos_bundle["segment_dataset"]["base"]["execution_date"]
            .dt.strftime("%Y-%m-%d")
            .tolist(),
            "next_ret": oos_bundle["segment_dataset"]["base"]["next_ret"].astype(float).tolist(),
            "next_rf_daily": oos_bundle["segment_dataset"]["base"]["next_rf_daily"].astype(float).tolist(),
            "y_pred": pred.astype(int).tolist(),
            "predicted_probability_raw": raw_prob.astype(float).tolist(),
            "predicted_probability_smoothed": smoothed_prob.astype(float).tolist(),
        },
    }


def select_best_config(config_rows: List[Dict[str, object]]) -> Dict[str, object]:
    frame = pd.DataFrame(config_rows)
    frame["threshold_distance"] = (frame["threshold"] - 0.50).abs()
    frame = frame.sort_values(
        by=[
            "validation_sharpe",
            "validation_balanced_accuracy",
            "threshold_distance",
            "smoothing_halflife",
            "config_id",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return frame.iloc[0].to_dict()


def worker_process_window(window_payload: Dict[str, str]) -> Dict[str, object]:
    frame = load_experiment_frame()
    features = feature_columns_for_frame(frame)
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

    config_rows: List[Dict[str, object]] = []
    selected_payloads: Dict[str, Dict[str, object]] = {}
    for config in CONFIG_GRID:
        validation_result = evaluate_config_on_validation(config, validation_bundles)
        oos_result = evaluate_config_on_oos(config, oos_bundle)
        row = {
            "config_id": config["config_id"],
            "smoothing_halflife": config["smoothing_halflife"],
            "threshold": config["threshold"],
            "validation_sharpe": validation_result["sharpe"],
            "validation_accuracy": validation_result["metrics"]["accuracy"],
            "validation_balanced_accuracy": validation_result["metrics"]["balanced_accuracy"],
            "validation_f1": validation_result["metrics"]["f1"],
            "oos_sharpe": oos_result["oos_sharpe"],
            "oos_accuracy": oos_result["oos_metrics"]["accuracy"],
            "oos_balanced_accuracy": oos_result["oos_metrics"]["balanced_accuracy"],
            "oos_f1": oos_result["oos_metrics"]["f1"],
        }
        config_rows.append(row)
        selected_payloads[row["config_id"]] = {
            "validation": validation_result,
            "oos": oos_result,
            "config": config,
        }

    selected = select_best_config(config_rows)
    chosen = selected_payloads[selected["config_id"]]

    return {
        "window": window_payload,
        "feature_columns": features,
        "selected_config": {
            "config_id": selected["config_id"],
            "smoothing_halflife": int(selected["smoothing_halflife"]),
            "threshold": float(selected["threshold"]),
        },
        "validation": chosen["validation"],
        "oos": chosen["oos"],
    }


def stitch_selected_strategy(window_results: List[Dict[str, object]]) -> pd.DataFrame:
    stitched_payloads = [
        {"window": result["window"], "oos_dataset": result["oos"]["oos_dataset"]}
        for result in window_results
    ]
    return stitch_oos_strategy(stitched_payloads, "y_pred")["frame"]


def build_predicted_bear_labels(window_results: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for result in window_results:
        dataset = result["oos"]["oos_dataset"]
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(dataset["execution_date"]),
                "predicted_probability_raw": np.asarray(dataset["predicted_probability_raw"], dtype=float),
                "predicted_probability_smoothed": np.asarray(dataset["predicted_probability_smoothed"], dtype=float),
                "predicted_label": np.asarray(dataset["y_pred"], dtype=int),
            }
        )
        frame["predicted_bear_flag"] = (1 - frame["predicted_label"]).astype(int)
        rows.append(frame)

    labels = pd.concat(rows, ignore_index=True)
    labels = labels.sort_values("Date", ascending=True).drop_duplicates(subset=["Date"], keep="last")
    return labels.reset_index(drop=True)


def build_daily_equity_table(predicted_frame: pd.DataFrame, buyhold_frame: pd.DataFrame) -> pd.DataFrame:
    predicted = predicted_frame.copy()
    buyhold = buyhold_frame.copy()
    predicted["equity_predicted_strategy"] = (1.0 + predicted["strategy_ret"]).cumprod()
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()
    daily = predicted[["Date", "equity_predicted_strategy"]].merge(
        buyhold[["Date", "equity_buy_and_hold"]], on="Date", how="outer"
    )
    return daily.sort_values("Date", ascending=True).reset_index(drop=True)


def build_bear_intervals(dates: pd.Series, bear_flags: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = pd.DataFrame({"Date": pd.to_datetime(dates), "bear_flag": bear_flags.astype(int)})
    ordered = ordered.sort_values("Date", ascending=True).reset_index(drop=True)

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = None
    current_end = None

    for row in ordered.itertuples(index=False):
        if row.bear_flag == 1 and current_start is None:
            current_start = row.Date
            current_end = row.Date
        elif row.bear_flag == 1:
            current_end = row.Date
        elif row.bear_flag == 0 and current_start is not None:
            intervals.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None:
        intervals.append((current_start, current_end))

    return intervals


def plot_predicted_vs_buyhold_with_bear_shading(
    daily_equity: pd.DataFrame,
    bear_labels: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for start, end in build_bear_intervals(bear_labels["Date"], bear_labels["predicted_bear_flag"]):
        ax.axvspan(start, end, color="red", alpha=0.12, zorder=0)

    ax.plot(
        daily_equity["Date"],
        daily_equity["equity_predicted_strategy"],
        label="predicted_strategy",
        linewidth=2.0,
        color="#1f77b4",
        zorder=2,
    )
    ax.plot(
        daily_equity["Date"],
        daily_equity["equity_buy_and_hold"],
        label="buy_and_hold",
        linewidth=2.0,
        color="#2ca02c",
        zorder=2,
    )
    ax.set_title("SPY Predicted Strategy vs Buy and Hold with Predicted Bear Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    frame = load_experiment_frame()
    main_windows = build_main_windows(frame, VALIDATION_YEARS)
    if not main_windows:
        raise ValueError(f"No main windows could be constructed from the {TICKER} forecasting dataset")

    worker_count = min(MAX_WORKERS_CAP, len(main_windows), os.cpu_count() or 1)
    window_payloads = [window.__dict__ for window in main_windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_process_window, window_payloads))

    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])
    predicted_frame = stitch_selected_strategy(window_results)
    first_oos_date = pd.to_datetime(window_results[0]["oos"]["oos_dataset"]["execution_date"][0])
    buyhold_frame = build_buy_and_hold(frame, first_oos_date)
    daily_equity = build_daily_equity_table(predicted_frame, buyhold_frame)
    bear_labels = build_predicted_bear_labels(window_results)

    predicted_metrics = build_strategy_metrics(predicted_frame)
    buyhold_metrics = build_strategy_metrics(buyhold_frame)
    strategy_summary = pd.DataFrame(
        [
            {"strategy_name": "predicted_strategy", **predicted_metrics},
            {"strategy_name": "buy_and_hold", **buyhold_metrics},
        ]
    )

    rolling_window_log = pd.DataFrame(
        [
            {
                "rebalance_date": result["window"]["rebalance_date"],
                "train_start": result["window"]["train_start"],
                "train_end": result["window"]["train_end"],
                "val_start": result["window"]["val_start"],
                "val_end": result["window"]["val_end"],
                "oos_start": result["window"]["oos_start"],
                "oos_end": result["window"]["oos_end"],
                "selected_config_id": result["selected_config"]["config_id"],
                "selected_smoothing_halflife": result["selected_config"]["smoothing_halflife"],
                "selected_threshold": result["selected_config"]["threshold"],
                "oos_sharpe": result["oos"]["oos_sharpe"],
                "oos_accuracy": result["oos"]["oos_metrics"]["accuracy"],
                "oos_balanced_accuracy": result["oos"]["oos_metrics"]["balanced_accuracy"],
                "oos_f1": result["oos"]["oos_metrics"]["f1"],
                "validation_sharpe": result["validation"]["sharpe"],
                "validation_accuracy": result["validation"]["metrics"]["accuracy"],
                "validation_balanced_accuracy": result["validation"]["metrics"]["balanced_accuracy"],
                "validation_f1": result["validation"]["metrics"]["f1"],
            }
            for result in window_results
        ]
    ).sort_values("rebalance_date", ascending=True).reset_index(drop=True)

    output_dir = results_dir()
    strategy_summary.to_csv(output_dir / "strategy_performance_summary.csv", index=False)
    daily_equity.to_csv(output_dir / "predicted_strategy_daily_equity_curves.csv", index=False)
    bear_labels.to_csv(output_dir / "predicted_bear_labels.csv", index=False)
    rolling_window_log.to_csv(output_dir / "rolling_window_log.csv", index=False)
    plot_predicted_vs_buyhold_with_bear_shading(
        daily_equity,
        bear_labels,
        output_dir / "predicted_vs_buyhold_with_bear_shading.png",
    )

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Ticker: {TICKER}")
    print(f"Validation years: {VALIDATION_YEARS}")
    print(f"Jump penalty: {JUMP_PENALTY}")
    print(f"XGBoost params: {FIXED_XGB_PARAMS}")
    print(f"Smoothing halflife grid: {SMOOTHING_HALFLIFE_GRID}")
    print(f"Threshold grid: {THRESHOLD_GRID}")
    print(f"Main windows: {len(main_windows)}")
    print(f"Workers used: {worker_count}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(
        "Predicted strategy Sharpe: "
        f"{float(strategy_summary.loc[strategy_summary['strategy_name'] == 'predicted_strategy', 'sharpe'].iloc[0]):.6f}"
    )
    print(
        "Most selected config: "
        f"{rolling_window_log['selected_config_id'].mode().iloc[0]}"
    )
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
