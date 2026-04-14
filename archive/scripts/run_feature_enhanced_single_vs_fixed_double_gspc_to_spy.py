from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from single_asset_gspc_spy_common import (
    MAX_WORKERS_CAP,
    StageConfig,
    build_target_windows,
    compute_metrics,
    double_threshold_positions,
    feature_columns_for_mode,
    load_research_frame,
    load_trade_base,
    project_root,
    results_root,
    select_smoothing,
    simulate_mapped_strategy,
    single_threshold_positions,
    smooth_probability_series,
    make_xgb_model,
)
from spy_regime_common import MainWindow, annualized_sharpe, build_segment_supervised_bundle, build_segments, build_strategy_metrics


RESULTS_SUBDIR = "feature_enhanced_single_vs_fixed_double_gspc_to_spy"
SINGLE_THRESHOLD = 0.55
DOUBLE_LOWER = 0.45
DOUBLE_UPPER = 0.60


def build_nonempty_segments(frame: pd.DataFrame, start_date: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[dict[str, pd.Timestamp]]:
    segments = build_segments(start_date, end_exclusive)
    return [segment for segment in segments if int(((frame["Date"] >= segment["start"]) & (frame["Date"] < segment["end_exclusive"])).sum()) > 0]


def prepare_window_for_compare(window_payload: dict[str, str]) -> dict[str, object]:
    frame = load_research_frame()
    features = feature_columns_for_mode(frame, "enhanced")
    window = MainWindow(**window_payload)

    validation_segments = build_nonempty_segments(frame, pd.Timestamp(window.val_start), pd.Timestamp(window.oos_start))
    validation_cache: list[dict[str, object]] = []
    for segment in validation_segments:
        bundle = build_segment_supervised_bundle(frame, features, segment["start"], segment["end_exclusive"], penalty=0.0)
        model = make_xgb_model()
        model.fit(bundle["train_dataset"]["X"], bundle["train_dataset"]["y"])
        raw_prob = model.predict_proba(bundle["segment_dataset"]["X"])[:, 1].astype(float)
        smoothed = {h: smooth_probability_series(raw_prob, h) for h in [0, 4, 8, 12]}
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
        penalty=0.0,
    )
    model = make_xgb_model()
    model.fit(oos_bundle["train_dataset"]["X"], oos_bundle["train_dataset"]["y"])
    oos_raw_prob = model.predict_proba(oos_bundle["segment_dataset"]["X"])[:, 1].astype(float)
    oos_smoothed = {h: smooth_probability_series(oos_raw_prob, h) for h in [0, 4, 8, 12]}
    oos_frame = frame[(frame["Date"] >= pd.Timestamp(window.oos_start)) & (frame["Date"] <= pd.Timestamp(window.oos_end))].copy().reset_index(drop=True)
    oos_frame["execution_date"] = oos_frame["Date"].shift(-1)
    oos_frame = oos_frame.iloc[:-1].copy().reset_index(drop=True)

    return {
        "window": window_payload,
        "validation_cache": validation_cache,
        "oos_base": oos_bundle["segment_dataset"]["base"].reset_index(drop=True),
        "oos_y_true": oos_bundle["segment_dataset"]["y"].astype(int).copy(),
        "oos_raw_prob": oos_raw_prob,
        "oos_smoothed": oos_smoothed,
        "oos_signal_date": pd.to_datetime(oos_frame["Date"]).reset_index(drop=True),
    }


def score_validation_fixed(validation_cache: list[dict[str, object]], rule_mode: str, smoothing_halflife: int) -> dict[str, object]:
    current_position = 0
    frames = []
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    for bundle in validation_cache:
        smoothed_prob = np.asarray(bundle["smoothed"][smoothing_halflife], dtype=float)
        if rule_mode == "single":
            positions, _, current_position = single_threshold_positions(smoothed_prob, current_position, SINGLE_THRESHOLD)
        else:
            positions, _, current_position = double_threshold_positions(smoothed_prob, current_position, DOUBLE_LOWER, DOUBLE_UPPER)
        simulation = simulate_positions_strategy_local(bundle["base"], positions, int(bundle["initial_position"]))
        bundle["initial_position"] = current_position
        frames.append(simulation["frame"])
        y_true_list.append(bundle["y_true"])
        y_pred_list.append(np.asarray(positions, dtype=int))
        y_prob_list.append(smoothed_prob)

    validation_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {
        "sharpe": annualized_sharpe(validation_frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)),
        "metrics": compute_metrics(np.concatenate(y_true_list), np.concatenate(y_pred_list), np.concatenate(y_prob_list)),
    }


def simulate_positions_strategy_local(dataset_base: pd.DataFrame, positions: np.ndarray, initial_position: int) -> dict[str, object]:
    base = dataset_base.reset_index(drop=True).copy()
    position = np.asarray(positions, dtype=float)
    previous_position = np.empty_like(position)
    previous_position[0] = float(initial_position)
    if len(position) > 1:
        previous_position[1:] = position[:-1]

    fee_rate = 5.0 / 10000.0
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
    return {"frame": frame}


def select_smoothing_fixed(validation_cache: list[dict[str, object]], rule_mode: str) -> int:
    rows = []
    dummy_config = StageConfig(stage_name="dummy", results_subdir="dummy", feature_mode="enhanced", rule_mode="single_threshold", threshold=SINGLE_THRESHOLD)
    if rule_mode == "double":
        dummy_config.rule_mode = "double_threshold"
    for halflife in [0, 4, 8, 12]:
        result = score_validation_fixed(validation_cache, rule_mode, halflife)
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


def worker_compare_window(window_payload: dict[str, str]) -> dict[str, object]:
    prepared = prepare_window_for_compare(window_payload)
    outputs = {}
    for rule_mode, version in [("single", "single_threshold_t55"), ("double", "double_threshold_045_060")]:
        selected_smoothing = select_smoothing_fixed(prepared["validation_cache"], rule_mode)
        oos_smoothed = np.asarray(prepared["oos_smoothed"][selected_smoothing], dtype=float)
        if rule_mode == "single":
            positions, zones, _ = single_threshold_positions(oos_smoothed, 0, SINGLE_THRESHOLD)
            lower, upper = np.nan, SINGLE_THRESHOLD
        else:
            positions, zones, _ = double_threshold_positions(oos_smoothed, 0, DOUBLE_LOWER, DOUBLE_UPPER)
            lower, upper = DOUBLE_LOWER, DOUBLE_UPPER
        signal_frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(prepared["oos_signal_date"]),
                "rebalance_date": window_payload["rebalance_date"],
                "predicted_probability_raw": np.asarray(prepared["oos_raw_prob"], dtype=float),
                "predicted_probability_smoothed": oos_smoothed,
                "predicted_label": np.asarray(positions, dtype=int),
                "predicted_bull_flag": np.asarray(positions, dtype=int),
                "predicted_bear_flag": 1 - np.asarray(positions, dtype=int),
                "selected_smoothing_halflife": int(selected_smoothing),
                "selected_lower_threshold": lower,
                "selected_upper_threshold": upper,
                "signal_zone": zones,
                "y_true": np.asarray(prepared["oos_y_true"], dtype=int),
                "rising_entry_trigger": 0,
                "drawdown_entry_trigger": 0,
            }
        )
        mapped_slice = simulate_mapped_strategy(signal_frame, load_trade_base())
        metrics = compute_metrics(np.asarray(prepared["oos_y_true"], dtype=int), np.asarray(positions, dtype=int), oos_smoothed)
        outputs[version] = {
            "signal_frame": signal_frame.to_dict(orient="records"),
            "selected_smoothing_halflife": int(selected_smoothing),
            "selected_lower_threshold": lower,
            "selected_upper_threshold": upper,
            "oos_sharpe": float(annualized_sharpe(mapped_slice["strategy_excess_ret"].to_numpy(dtype=float, copy=False))),
            "oos_metrics": metrics,
        }
    return {"window": window_payload, "outputs": outputs}


def main() -> None:
    frame = load_research_frame()
    windows = build_target_windows(frame)
    worker_count = min(MAX_WORKERS_CAP, len(windows), 8)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_compare_window, [window.__dict__ for window in windows]))
    window_results = sorted(window_results, key=lambda item: item["window"]["rebalance_date"])

    versions = ["single_threshold_t55", "double_threshold_045_060"]
    trade_base = load_trade_base()
    out_dir = results_root() / RESULTS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    prediction_rows = []
    rolling_rows = []
    equity_curves = None

    for version in versions:
        signal_frame = pd.concat([pd.DataFrame(item["outputs"][version]["signal_frame"]) for item in window_results], ignore_index=True)
        signal_frame["Date"] = pd.to_datetime(signal_frame["Date"])
        signal_frame = signal_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        mapped_frame = simulate_mapped_strategy(signal_frame, trade_base)
        mapped_frame["equity_predicted_strategy"] = (1.0 + mapped_frame["strategy_ret"]).cumprod()
        summary_rows.append({"version": version, **build_strategy_metrics(mapped_frame)})
        prediction_rows.append(
            {
                "version": version,
                "avg_oos_accuracy": float(np.mean([item["outputs"][version]["oos_metrics"]["accuracy"] for item in window_results])),
                "avg_oos_balanced_accuracy": float(np.mean([item["outputs"][version]["oos_metrics"]["balanced_accuracy"] for item in window_results])),
                "avg_oos_f1": float(np.mean([item["outputs"][version]["oos_metrics"]["f1"] for item in window_results])),
                "avg_oos_log_loss": float(np.mean([item["outputs"][version]["oos_metrics"]["log_loss"] for item in window_results])),
                "avg_oos_roc_auc": float(np.mean([item["outputs"][version]["oos_metrics"]["roc_auc"] for item in window_results])),
            }
        )
        rolling_rows.extend(
            {
                "version": version,
                "rebalance_date": item["window"]["rebalance_date"],
                "selected_smoothing_halflife": item["outputs"][version]["selected_smoothing_halflife"],
                "selected_lower_threshold": item["outputs"][version]["selected_lower_threshold"],
                "selected_upper_threshold": item["outputs"][version]["selected_upper_threshold"],
                "oos_sharpe": item["outputs"][version]["oos_sharpe"],
                "oos_accuracy": item["outputs"][version]["oos_metrics"]["accuracy"],
                "oos_balanced_accuracy": item["outputs"][version]["oos_metrics"]["balanced_accuracy"],
                "oos_f1": item["outputs"][version]["oos_metrics"]["f1"],
            }
            for item in window_results
        )
        curve = mapped_frame[["Date", "equity_predicted_strategy"]].rename(columns={"equity_predicted_strategy": f"equity_{version}"})
        equity_curves = curve if equity_curves is None else equity_curves.merge(curve, on="Date", how="outer")

    buyhold = trade_base[trade_base["Date"].isin(equity_curves["Date"])].copy().sort_values("Date").reset_index(drop=True)
    buyhold["position"] = 1.0
    buyhold["fee"] = 0.0
    buyhold["strategy_ret_gross"] = buyhold["next_ret"]
    buyhold["strategy_ret"] = buyhold["next_ret"]
    buyhold["strategy_excess_ret"] = buyhold["next_ret"] - buyhold["next_rf_daily"]
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()
    summary_rows.append({"version": "buy_and_hold", **build_strategy_metrics(buyhold)})

    pd.DataFrame(summary_rows).to_csv(out_dir / "strategy_performance_comparison.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "prediction_metrics_comparison.csv", index=False)
    pd.DataFrame(rolling_rows).sort_values(["version", "rebalance_date"]).to_csv(out_dir / "rolling_window_metrics_comparison.csv", index=False)

    equity_curves = equity_curves.merge(buyhold[["Date", "equity_buy_and_hold"]], on="Date", how="left")
    equity_curves.to_csv(out_dir / "daily_equity_curves_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in equity_curves.columns:
        if column == "Date":
            continue
        ax.plot(equity_curves["Date"], equity_curves[column], label=column, linewidth=1.6)
    ax.set_title("Single Threshold 0.55 vs Fixed Double Threshold 0.45/0.60")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "single_vs_fixed_double_equity_curves.png", dpi=150)
    plt.close(fig)

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
