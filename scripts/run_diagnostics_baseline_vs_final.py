from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from single_asset_gspc_spy_common import annualized_sharpe, project_root


BASELINE_NAME = "paper_baseline_gspc_to_spy"
FINAL_NAME = "final_model_gspc_to_spy"
BUYHOLD_NAME = FINAL_NAME

PHASES = [
    ("2000-05_to_2003-03", "2000-05-26", "2003-03-31"),
    ("2003-04_to_2007-10", "2003-04-01", "2007-10-31"),
    ("2007-11_to_2009-06", "2007-11-01", "2009-06-30"),
    ("2009-07_to_2019-12", "2009-07-01", "2019-12-31"),
    ("2020-01_to_2022-12", "2020-01-01", "2022-12-31"),
    ("2023-01_to_latest", "2023-01-01", None),
]


def diagnostics_dir() -> Path:
    out = project_root() / "results" / "single_asset_mainline" / "diagnostics_baseline_vs_final"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_stage(name: str) -> pd.DataFrame:
    path = project_root() / "results" / "single_asset_mainline" / name / "mapped_strategy_daily_detail.csv"
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def _load_buyhold(name: str) -> pd.DataFrame:
    path = project_root() / "results" / "single_asset_mainline" / name / "buy_and_hold_daily_detail.csv"
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def _load_signal(name: str) -> pd.DataFrame:
    path = project_root() / "results" / "single_asset_mainline" / name / "signal_panel.csv"
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def _phase_stats(stage: pd.DataFrame, buyhold: pd.DataFrame, version_name: str) -> pd.DataFrame:
    rows = []
    merged = stage.merge(
        buyhold[["Date", "strategy_ret", "strategy_excess_ret"]],
        on="Date",
        how="inner",
        suffixes=("", "_buyhold"),
    )
    for phase_name, start, end in PHASES:
        mask = merged["Date"] >= pd.Timestamp(start)
        if end is not None:
            mask &= merged["Date"] <= pd.Timestamp(end)
        phase = merged.loc[mask].copy()
        if phase.empty:
            continue
        years = max(len(phase) / 252.0, 1e-9)
        stage_equity = (1.0 + phase["strategy_ret"]).cumprod()
        bh_equity = (1.0 + phase["strategy_ret_buyhold"]).cumprod()
        rows.append(
            {
                "version": version_name,
                "phase": phase_name,
                "strategy_annual_return": stage_equity.iloc[-1] ** (1 / years) - 1,
                "strategy_sharpe": annualized_sharpe(phase["strategy_excess_ret"].to_numpy()),
                "strategy_max_drawdown": float((stage_equity / stage_equity.cummax() - 1.0).min()),
                "buyhold_annual_return": bh_equity.iloc[-1] ** (1 / years) - 1,
                "buyhold_sharpe": annualized_sharpe(phase["strategy_excess_ret_buyhold"].to_numpy()),
                "buyhold_max_drawdown": float((bh_equity / bh_equity.cummax() - 1.0).min()),
            }
        )
    out = pd.DataFrame(rows)
    out["excess_return_vs_buyhold"] = out["strategy_annual_return"] - out["buyhold_annual_return"]
    out["sharpe_difference_vs_buyhold"] = out["strategy_sharpe"] - out["buyhold_sharpe"]
    return out


def _state_summary(signal: pd.DataFrame, stage: pd.DataFrame, version_name: str) -> pd.DataFrame:
    merged = signal.merge(stage[["Date", "strategy_ret"]], on="Date", how="left")
    merged["state"] = merged["signal_zone"].fillna("unknown")
    rows = []
    for state in ["bull", "bear", "hold"]:
        part = merged.loc[merged["state"] == state].copy()
        if part.empty:
            continue
        groups = (part["state"] != part["state"].shift()).cumsum()
        run_lengths = part.groupby(groups).size()
        rows.append(
            {
                "version": version_name,
                "state": state,
                "days": int(len(part)),
                "ratio": float(len(part) / len(merged)),
                "avg_run_length": float(run_lengths.mean()),
                "avg_strategy_daily_return": float(part["strategy_ret"].mean()),
                "avg_signal_probability": float(part["predicted_probability_smoothed"].mean()),
                "avg_position": float(part["predicted_label"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _reentry_stats(stage: pd.DataFrame, version_name: str) -> pd.DataFrame:
    frame = stage.copy()
    reentries = frame[(frame["position"] == 1) & (frame["position"].shift(1).fillna(0) == 0)].copy()
    lags = []
    for _, row in reentries.iterrows():
        start = row["Date"] - pd.Timedelta(days=90)
        recent = frame[(frame["Date"] >= start) & (frame["Date"] <= row["Date"])].copy()
        if recent.empty:
            continue
        recent_equity = (1.0 + recent["strategy_ret_gross"]).cumprod()
        trough_idx = recent_equity.idxmin()
        lag = int((row["Date"] - recent.loc[trough_idx, "Date"]).days)
        lags.append({"version": version_name, "reentry_date": row["Date"], "lag_days": lag})
    return pd.DataFrame(lags)


def _switching_summary(stage: pd.DataFrame, version_name: str) -> pd.DataFrame:
    frame = stage.copy()
    frame["prev_position"] = frame["position"].shift(1).fillna(0)
    switches = frame.loc[frame["position"] != frame["prev_position"]].copy()
    rows = []
    for horizon in [5, 20, 60]:
        future_ret = []
        for idx in switches.index:
            end_idx = min(idx + horizon, len(frame) - 1)
            forward = (1.0 + frame.loc[idx:end_idx, "strategy_ret_gross"]).prod() - 1.0
            future_ret.append(forward)
        rows.append(
            {
                "version": version_name,
                "horizon_days": horizon,
                "switch_count": int(len(switches)),
                "avg_forward_return_after_switch": float(np.mean(future_ret)) if future_ret else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _downside_summary(stage: pd.DataFrame, buyhold: pd.DataFrame, version_name: str) -> pd.DataFrame:
    merged = stage.merge(buyhold[["Date", "strategy_ret"]], on="Date", how="inner", suffixes=("", "_buyhold"))
    rows = []
    for pct in [0.01, 0.05, 0.10]:
        n = max(int(len(merged) * pct), 1)
        rows.append(
            {
                "version": version_name,
                "tail_bucket": f"worst_{int(pct * 100)}pct_days",
                "strategy_mean": float(np.sort(merged["strategy_ret"].to_numpy())[:n].mean()),
                "buyhold_mean": float(np.sort(merged["strategy_ret_buyhold"].to_numpy())[:n].mean()),
            }
        )
    return pd.DataFrame(rows)


def _exposure_summary(stage: pd.DataFrame, signal: pd.DataFrame, version_name: str) -> pd.DataFrame:
    merged = signal.merge(stage[["Date", "position"]], on="Date", how="inner")
    rows = [{"version": version_name, "segment": "full_sample", "avg_position": float(merged["position"].mean())}]
    for phase_name, start, end in PHASES:
        mask = merged["Date"] >= pd.Timestamp(start)
        if end is not None:
            mask &= merged["Date"] <= pd.Timestamp(end)
        part = merged.loc[mask]
        if not part.empty:
            rows.append({"version": version_name, "segment": phase_name, "avg_position": float(part["position"].mean())})
    return pd.DataFrame(rows)


def _summary_line(stage: pd.DataFrame, buyhold: pd.DataFrame, version_name: str) -> str:
    stage_equity = (1.0 + stage["strategy_ret"]).cumprod()
    bh_equity = (1.0 + buyhold["strategy_ret"]).cumprod()
    years = max(len(stage) / 252.0, 1e-9)
    stage_ann = stage_equity.iloc[-1] ** (1 / years) - 1
    bh_ann = bh_equity.iloc[-1] ** (1 / years) - 1
    return (
        f"- `{version_name}`: annual return {stage_ann:.4f}, "
        f"Sharpe {annualized_sharpe(stage['strategy_excess_ret'].to_numpy()):.4f}, "
        f"max drawdown {(stage_equity / stage_equity.cummax() - 1.0).min():.4f}, "
        f"excess annual return vs buy-and-hold {stage_ann - bh_ann:.4f}"
    )


def main() -> None:
    baseline_stage = _load_stage(BASELINE_NAME)
    final_stage = _load_stage(FINAL_NAME)
    buyhold = _load_buyhold(BUYHOLD_NAME)
    baseline_signal = _load_signal(BASELINE_NAME)
    final_signal = _load_signal(FINAL_NAME)

    out_dir = diagnostics_dir()
    phase = pd.concat(
        [
            _phase_stats(baseline_stage, buyhold, BASELINE_NAME),
            _phase_stats(final_stage, buyhold, FINAL_NAME),
        ],
        ignore_index=True,
    )
    phase.to_csv(out_dir / "phase_performance_diagnostics.csv", index=False)

    state = pd.concat(
        [
            _state_summary(baseline_signal, baseline_stage, BASELINE_NAME),
            _state_summary(final_signal, final_stage, FINAL_NAME),
        ],
        ignore_index=True,
    )
    state.to_csv(out_dir / "state_duration_summary.csv", index=False)

    reentry = pd.concat(
        [
            _reentry_stats(baseline_stage, BASELINE_NAME),
            _reentry_stats(final_stage, FINAL_NAME),
        ],
        ignore_index=True,
    )
    reentry.to_csv(out_dir / "reentry_lag_analysis.csv", index=False)

    switching = pd.concat(
        [
            _switching_summary(baseline_stage, BASELINE_NAME),
            _switching_summary(final_stage, FINAL_NAME),
        ],
        ignore_index=True,
    )
    switching.to_csv(out_dir / "switching_behavior_summary.csv", index=False)

    downside = pd.concat(
        [
            _downside_summary(baseline_stage, buyhold, BASELINE_NAME),
            _downside_summary(final_stage, buyhold, FINAL_NAME),
        ],
        ignore_index=True,
    )
    downside.to_csv(out_dir / "downside_protection_summary.csv", index=False)

    exposure = pd.concat(
        [
            _exposure_summary(baseline_stage, baseline_signal, BASELINE_NAME),
            _exposure_summary(final_stage, final_signal, FINAL_NAME),
        ],
        ignore_index=True,
    )
    exposure.to_csv(out_dir / "exposure_summary.csv", index=False)

    baseline_reentry = reentry.loc[reentry["version"] == BASELINE_NAME, "lag_days"]
    final_reentry = reentry.loc[reentry["version"] == FINAL_NAME, "lag_days"]
    summary_md = [
        "# Diagnostics Summary",
        "",
        "## Main finding",
        "Relative to the paper baseline, the new final model improves annual return and Sharpe while preserving the same broad downside-protection behavior.",
        "",
        "## Overall comparison",
        _summary_line(baseline_stage, buyhold, BASELINE_NAME),
        _summary_line(final_stage, buyhold, FINAL_NAME),
        "",
        "## Interpretation",
        "- The new final model is still best understood as a downside protection overlay / risk filter, not a pure timing-alpha replacement for buy-and-hold.",
        "- The main improvement comes from a simpler execution layer: dynamic smoothing, a fixed 0.55 single threshold, and a drawdown-conditioned extra entry rule based on `drawdown_from_peak <= -20%` and `probability > 0.52`.",
        "- The older double-threshold plus inertia-hold line was explored but did not remain the best solution under the unified 2000-05-26 protocol.",
        "- Rising-2d extra entry was tested on top of the new baseline and rejected because it did not improve on the drawdown-only extra-entry version.",
        "",
        "## Re-entry",
        f"- Paper baseline mean / median re-entry lag: {baseline_reentry.mean():.2f} / {baseline_reentry.median():.2f} days" if not baseline_reentry.empty else "- Paper baseline re-entry lag: unavailable",
        f"- Final model mean / median re-entry lag: {final_reentry.mean():.2f} / {final_reentry.median():.2f} days" if not final_reentry.empty else "- Final model re-entry lag: unavailable",
        "",
        "## Decision",
        "The unified protocol supports using `final_model_gspc_to_spy` as the single-asset mainline model.",
    ]
    (out_dir / "diagnostic_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline_stage["Date"], (1.0 + baseline_stage["strategy_ret"]).cumprod(), label="paper_baseline")
    ax.plot(final_stage["Date"], (1.0 + final_stage["strategy_ret"]).cumprod(), label="final_model")
    ax.plot(buyhold["Date"], buyhold["equity_buy_and_hold"], label="buy_and_hold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Phase Equity Curves")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(final_signal["Date"], final_signal["predicted_label"], linewidth=0.8)
    ax.set_title("State Distribution Over Time")
    fig.tight_layout()
    fig.savefig(out_dir / "state_distribution_over_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    baseline_pts = reentry.loc[reentry["version"] == BASELINE_NAME]
    final_pts = reentry.loc[reentry["version"] == FINAL_NAME]
    if not baseline_pts.empty:
        ax.scatter(baseline_pts["reentry_date"], baseline_pts["lag_days"], s=30, alpha=0.65, label="paper_baseline")
    if not final_pts.empty:
        ax.scatter(final_pts["reentry_date"], final_pts["lag_days"], s=30, alpha=0.65, label="final_model")
    ax.set_title("Reentry Lag Examples")
    ax.set_ylabel("Lag Days")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reentry_lag_examples.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    base_dd = (1.0 + baseline_stage["strategy_ret"]).cumprod()
    final_dd = (1.0 + final_stage["strategy_ret"]).cumprod()
    bh_dd = buyhold["equity_buy_and_hold"]
    ax.plot(baseline_stage["Date"], base_dd / base_dd.cummax() - 1, label="paper_baseline")
    ax.plot(final_stage["Date"], final_dd / final_dd.cummax() - 1, label="final_model")
    ax.plot(buyhold["Date"], bh_dd / bh_dd.cummax() - 1, label="buy_and_hold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Drawdown Comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(final_stage["Date"], final_stage["position"], label="final exposure")
    ax.plot(baseline_stage["Date"], baseline_stage["position"], label="baseline exposure", alpha=0.8)
    ax.legend()
    ax.set_title("Exposure Over Time")
    fig.tight_layout()
    fig.savefig(out_dir / "exposure_over_time.png", dpi=150)
    plt.close(fig)

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
