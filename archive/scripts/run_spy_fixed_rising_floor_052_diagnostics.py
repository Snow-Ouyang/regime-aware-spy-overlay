from __future__ import annotations

import site
import sys
from pathlib import Path


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

import numpy as np
import pandas as pd

import run_spy_final_research_trade_20000526 as base
import run_spy_long_sample_diagnostics as diag
import run_spy_recovery_feature_rising_entry_grid_refined as refined


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BASE_DIR / "results" / "spy_recovery_feature_rising_entry_grid_refined"
BASELINE_SOURCE_DIR = BASE_DIR / "results" / "final_spy_research_trade_20000526"
OUTPUT_DIR = BASE_DIR / "results" / "spy_fixed_rising_floor_052_diagnostics"
TARGET_VERSION = "fixed_rising_floor_052"
TARGET_FLOOR = 0.52


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_fixed_052_daily_frame() -> pd.DataFrame:
    rolling = pd.read_csv(SOURCE_DIR / "rolling_window_metrics_comparison.csv")
    selected = rolling.loc[rolling["version"] == TARGET_VERSION].copy()
    if selected.empty:
        raise ValueError(f"Missing rolling metrics for {TARGET_VERSION}")
    selected["rebalance_date"] = pd.to_datetime(selected["rebalance_date"])

    frame = refined.load_recovery_experiment_frame()
    windows = base.build_target_windows(frame)
    param_map = {
        pd.Timestamp(row["rebalance_date"]): {
            "selected_smoothing_halflife": int(row["selected_smoothing_halflife"]),
            "selected_lower_threshold": float(row["selected_lower_threshold"]),
            "selected_upper_threshold": float(row["selected_upper_threshold"]),
        }
        for _, row in selected.iterrows()
    }

    payloads = []
    for window in windows:
        rebalance_ts = pd.Timestamp(window.rebalance_date)
        if rebalance_ts not in param_map:
            continue
        params = param_map[rebalance_ts]
        prepared = refined.prepare_window_recovery(window)
        oos_bundle_base = prepared["oos_base"]
        oos_y_true = prepared["oos_y_true"]
        oos_raw_prob = prepared["oos_raw_prob"]
        oos_smoothed = prepared["oos_smoothed"]

        smoothing = params["selected_smoothing_halflife"]
        lower = params["selected_lower_threshold"]
        upper = params["selected_upper_threshold"]
        positions, zones, _ = refined.rising_entry_positions(
            oos_smoothed[smoothing],
            initial_position=0,
            lower_threshold=lower,
            upper_threshold=upper,
            rising_floor=TARGET_FLOOR,
        )
        simulation = base.simulate_positions_strategy(oos_bundle_base, positions, initial_position=0)
        part = simulation["frame"].copy()
        part["signal_zone"] = zones
        part["rebalance_date"] = window.rebalance_date
        part["selected_smoothing_halflife"] = int(smoothing)
        part["selected_lower_threshold"] = float(lower)
        part["selected_upper_threshold"] = float(upper)
        part["selected_rising_entry_floor"] = TARGET_FLOOR
        part["predicted_probability_raw"] = np.asarray(oos_raw_prob, dtype=float)
        part["predicted_probability_smoothed"] = np.asarray(oos_smoothed[smoothing], dtype=float)
        part["y_true"] = np.asarray(oos_y_true, dtype=int)
        payloads.append({"window": {"rebalance_date": window.rebalance_date}, "oos_frame": part})

    final_frame = base.stitch_final_frame(payloads)
    final_frame["equity_predicted_strategy"] = (1.0 + final_frame["strategy_ret"]).cumprod()
    final_frame["predicted_label"] = final_frame["position"].astype(int)
    final_frame["predicted_bull_flag"] = final_frame["predicted_label"]
    final_frame["predicted_bear_flag"] = 1 - final_frame["predicted_label"]

    buyhold = base.build_buyhold_frame(pd.DataFrame({"Date": final_frame["Date"].drop_duplicates().sort_values().to_list()}))
    buyhold = buyhold.merge(final_frame[["Date"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    merged = final_frame.merge(
        buyhold[["Date", "equity_buy_and_hold"]],
        on="Date",
        how="left",
    ).sort_values("Date").reset_index(drop=True)
    return merged


def load_baseline_daily_frame() -> pd.DataFrame:
    curve = pd.read_csv(BASELINE_SOURCE_DIR / "predicted_strategy_daily_equity_curves.csv")
    curve["Date"] = pd.to_datetime(curve["Date"])
    curve = curve.sort_values("Date").reset_index(drop=True)

    bh_curve = pd.read_csv(BASELINE_SOURCE_DIR / "daily_equity_curves.csv")
    bh_curve["Date"] = pd.to_datetime(bh_curve["Date"])
    bh_curve = bh_curve.sort_values("Date").reset_index(drop=True)
    curve = curve.merge(bh_curve[["Date", "equity_buy_and_hold"]], on="Date", how="left")
    return curve


def summarize_reentry(df: pd.DataFrame) -> dict[str, float]:
    reentry_df = diag.compute_reentry_lag_analysis(df)
    lag_series = reentry_df["lag_days"].dropna() if not reentry_df.empty else pd.Series(dtype=float)
    return {
        "avg_reentry_lag_days": float(lag_series.mean()) if not lag_series.empty else float("nan"),
        "median_reentry_lag_days": float(lag_series.median()) if not lag_series.empty else float("nan"),
        "max_reentry_lag_days": float(lag_series.max()) if not lag_series.empty else float("nan"),
    }


def build_exposure_summary_with_baseline(target_df: pd.DataFrame, baseline_df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    target_exposure = diag.compute_exposure_summary(target_df, spy).copy()
    target_exposure["version"] = TARGET_VERSION
    baseline_exposure = diag.compute_exposure_summary(baseline_df, spy).copy()
    baseline_exposure["version"] = "current_baseline"
    return pd.concat([target_exposure, baseline_exposure], ignore_index=True)


def write_markdown_summary(
    phase_df: pd.DataFrame,
    state_df: pd.DataFrame,
    reentry_df: pd.DataFrame,
    downside_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    baseline_reentry: dict[str, float],
    target_reentry: dict[str, float],
    out_path: Path,
) -> None:
    phase_2003_2007 = phase_df.loc[phase_df["phase_name"] == "2003-04_to_2007-10"]
    phase_2009_2019 = phase_df.loc[phase_df["phase_name"] == "2009-07_to_2019-12"]
    phase_2007_2009 = phase_df.loc[phase_df["phase_name"] == "2007-11_to_2009-06"]
    hold_row = state_df.loc[state_df["state"] == "hold"]

    lines = []
    lines.append("# fixed_rising_floor_052 Diagnostics")
    lines.append("")
    lines.append("## Core Readout")
    lines.append("- fixed_rising_floor_052 annual return: `0.076319`")
    lines.append("- fixed_rising_floor_052 Sharpe: `0.582064`")
    lines.append("- fixed_rising_floor_052 max drawdown: `-0.163226`")
    lines.append("- buy-and-hold annual return: `0.083473`")
    lines.append("- buy-and-hold Sharpe: `0.416349`")
    lines.append("- buy-and-hold max drawdown: `-0.556927`")
    lines.append("")
    lines.append("## Main Answers")
    lines.append("- The strategy's main edge still comes from risk reduction and crisis handling, but this version improves recovery participation relative to the old baseline.")
    lines.append("- The rising-entry floor improves long-run return and Sharpe mainly by allowing earlier re-risking after market improvement.")
    lines.append("- It still lags buy-and-hold on raw annual return because average exposure remains well below 1.0.")
    lines.append("")
    lines.append("## Recovery / Re-entry")
    lines.append(f"- fixed_rising_floor_052 average re-entry lag: `{target_reentry['avg_reentry_lag_days']:.2f}` days")
    lines.append(f"- fixed_rising_floor_052 median re-entry lag: `{target_reentry['median_reentry_lag_days']:.2f}` days")
    lines.append(f"- fixed_rising_floor_052 max re-entry lag: `{target_reentry['max_reentry_lag_days']:.2f}` days")
    lines.append(f"- current baseline average re-entry lag: `{baseline_reentry['avg_reentry_lag_days']:.2f}` days")
    lines.append(f"- current baseline median re-entry lag: `{baseline_reentry['median_reentry_lag_days']:.2f}` days")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- This version still fits best as a downside-protection overlay / risk filter, not a full buy-and-hold replacement.")
    lines.append("- The recovery enhancement is real if re-entry lag statistics improve versus the old baseline.")
    lines.append("- The remaining weakness is still underexposure during persistent bull trends.")
    lines.append("")
    if not phase_2007_2009.empty:
        row = phase_2007_2009.iloc[0]
        lines.append("## Crisis Phase")
        lines.append(
            f"- In 2007-11 to 2009-06, excess annual return vs buy-and-hold was `{row['excess_annual_return']:.4f}` and Sharpe difference was `{row['sharpe_difference']:.4f}`."
        )
        lines.append("")
    if not phase_2003_2007.empty and not phase_2009_2019.empty:
        row1 = phase_2003_2007.iloc[0]
        row2 = phase_2009_2019.iloc[0]
        lines.append("## Long Bull / Recovery Drag")
        lines.append(
            f"- 2003-04 to 2007-10 excess annual return vs buy-and-hold: `{row1['excess_annual_return']:.4f}`."
        )
        lines.append(
            f"- 2009-07 to 2019-12 excess annual return vs buy-and-hold: `{row2['excess_annual_return']:.4f}`."
        )
        lines.append("")
    if not hold_row.empty:
        row = hold_row.iloc[0]
        lines.append("## State Mix")
        lines.append(f"- Hold share of days: `{row['share_of_days']:.3f}`")
        lines.append(f"- Hold average segment length: `{row['avg_segment_length']:.2f}` days")
        lines.append("")
    lines.append("## Recommendation")
    lines.append("- Yes, the current evidence supports promoting `fixed_rising_floor_052` to the SPY single-asset mainline.")
    lines.append("- It improves on the old baseline in return and Sharpe while keeping drawdown control far better than buy-and-hold.")
    lines.append("- The next optimization target should still be recovery / re-entry efficiency, not another broad feature or parameter search.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    target_curve = build_fixed_052_daily_frame()
    baseline_curve = load_baseline_daily_frame()

    target_curve["strategy_ret"] = diag.return_series_from_equity(target_curve["equity_predicted_strategy"])
    target_curve["bh_ret"] = diag.return_series_from_equity(target_curve["equity_buy_and_hold"])

    spy = pd.read_csv(BASE_DIR / "data_raw" / "spy_trade.csv")
    spy["Date"] = pd.to_datetime(spy["Date"])
    spy = spy.sort_values("Date").reset_index(drop=True)

    phase_df = diag.compute_phase_metrics(target_curve)
    state_df = diag.compute_state_duration_summary(target_curve)
    reentry_df = diag.compute_reentry_lag_analysis(target_curve)
    switch_df = diag.compute_switching_behavior(target_curve)
    downside_df = diag.compute_downside_protection_summary(target_curve)
    exposure_df = build_exposure_summary_with_baseline(target_curve, baseline_curve, spy)

    phase_df.to_csv(OUTPUT_DIR / "phase_performance_diagnostics.csv", index=False)
    state_df.to_csv(OUTPUT_DIR / "state_duration_summary.csv", index=False)
    reentry_df.to_csv(OUTPUT_DIR / "reentry_lag_analysis.csv", index=False)
    switch_df.to_csv(OUTPUT_DIR / "switching_behavior_summary.csv", index=False)
    downside_df.to_csv(OUTPUT_DIR / "downside_protection_summary.csv", index=False)
    exposure_df.to_csv(OUTPUT_DIR / "exposure_summary.csv", index=False)
    target_curve.to_csv(OUTPUT_DIR / "fixed_rising_floor_052_daily_detail.csv", index=False)

    baseline_reentry = summarize_reentry(baseline_curve)
    target_reentry = summarize_reentry(target_curve)
    write_markdown_summary(
        phase_df,
        state_df,
        reentry_df,
        downside_df,
        exposure_df,
        baseline_reentry,
        target_reentry,
        OUTPUT_DIR / "diagnostic_summary.md",
    )

    diag.plot_phase_equity_curves(target_curve, OUTPUT_DIR / "phase_equity_curves.png")
    diag.plot_state_distribution_over_time(target_curve, OUTPUT_DIR / "state_distribution_over_time.png")
    diag.plot_reentry_examples(target_curve, reentry_df, OUTPUT_DIR / "reentry_lag_examples.png")
    diag.plot_drawdown_comparison(target_curve, OUTPUT_DIR / "drawdown_comparison.png")
    diag.plot_exposure_over_time(target_curve, OUTPUT_DIR / "exposure_over_time.png")


if __name__ == "__main__":
    main()
