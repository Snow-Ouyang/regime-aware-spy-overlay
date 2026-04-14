from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from single_asset_gspc_spy_common import (
    BUYHOLD_VERSION,
    CURRENT_MAPPED_ORACLE_VERSION,
    FULL_UPPER_BOUND_ORACLE_VERSION,
    results_root,
    run_current_mapped_oracle_stage,
    run_full_upper_bound_oracle_stage,
)
from spy_regime_common import build_strategy_metrics


RESULTS_SUBDIR = "oracle_gspc_to_spy_comparison"


def output_dir() -> Path:
    out = results_root() / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_drawdown(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def build_alignment_sample(signal_frame: pd.DataFrame, mapped_frame: pd.DataFrame, sample_size: int = 40) -> pd.DataFrame:
    merged = signal_frame.merge(
        mapped_frame[["Date", "position", "strategy_ret", "mapped_spy_next_ret", "mapped_spy_return_date"]],
        on="Date",
        how="inner",
    ).sort_values("Date").reset_index(drop=True)
    merged["Date_t"] = pd.to_datetime(merged["signal_date"])
    merged["label_date"] = pd.to_datetime(merged["label_date"])
    merged["return_date"] = pd.to_datetime(merged["mapped_spy_return_date"])
    sample = merged[
        [
            "Date_t",
            "predicted_label",
            "label_date",
            "return_date",
            "mapped_spy_next_ret",
            "position",
            "strategy_ret",
        ]
    ].copy()
    sample = sample.rename(
        columns={
            "predicted_label": "label_used_for_position",
            "mapped_spy_next_ret": "mapped_spy_return_used",
            "strategy_ret": "strategy_return",
        }
    )
    return sample.head(sample_size)


def main() -> None:
    current = run_current_mapped_oracle_stage()
    full = run_full_upper_bound_oracle_stage()
    out_dir = output_dir()

    current_mapped = current["mapped_frame"].copy()
    full_mapped = full["mapped_frame"].copy()
    buyhold = full["buyhold_frame"].copy()

    current_mapped["equity_current_mapped_oracle"] = (1.0 + current_mapped["strategy_ret"]).cumprod()
    full_mapped["equity_full_upper_bound_oracle"] = (1.0 + full_mapped["strategy_ret"]).cumprod()
    buyhold["equity_buy_and_hold"] = (1.0 + buyhold["strategy_ret"]).cumprod()

    performance = pd.DataFrame(
        [
            {"version": CURRENT_MAPPED_ORACLE_VERSION, **build_strategy_metrics(current_mapped)},
            {"version": FULL_UPPER_BOUND_ORACLE_VERSION, **build_strategy_metrics(full_mapped)},
            {"version": BUYHOLD_VERSION, **build_strategy_metrics(buyhold)},
        ]
    )
    performance.to_csv(out_dir / "oracle_strategy_performance_comparison.csv", index=False)

    equity = pd.DataFrame({"Date": current_mapped["Date"]})
    equity = equity.merge(current_mapped[["Date", "equity_current_mapped_oracle"]], on="Date", how="left")
    equity = equity.merge(full_mapped[["Date", "equity_full_upper_bound_oracle"]], on="Date", how="left")
    equity = equity.merge(buyhold[["Date", "equity_buy_and_hold"]], on="Date", how="left")
    equity.to_csv(out_dir / "oracle_daily_equity_curves.csv", index=False)

    current_alignment = build_alignment_sample(current["signal_frame"], current_mapped, sample_size=20)
    current_alignment.insert(0, "version", CURRENT_MAPPED_ORACLE_VERSION)
    full_alignment = build_alignment_sample(full["signal_frame"], full_mapped, sample_size=20)
    full_alignment.insert(0, "version", FULL_UPPER_BOUND_ORACLE_VERSION)
    alignment = pd.concat([current_alignment, full_alignment], ignore_index=True)
    alignment.to_csv(out_dir / "oracle_alignment_check_sample.csv", index=False)

    current["signal_frame"].to_csv(out_dir / "current_mapped_oracle_signal_panel.csv", index=False)
    full["signal_frame"].to_csv(out_dir / "full_upper_bound_oracle_signal_panel.csv", index=False)
    current_mapped.to_csv(out_dir / "current_mapped_oracle_daily_detail.csv", index=False)
    full_mapped.to_csv(out_dir / "full_upper_bound_oracle_daily_detail.csv", index=False)
    buyhold.to_csv(out_dir / "buy_and_hold_daily_detail.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity["Date"], equity["equity_current_mapped_oracle"], label=CURRENT_MAPPED_ORACLE_VERSION, linewidth=1.8)
    ax.plot(equity["Date"], equity["equity_full_upper_bound_oracle"], label=FULL_UPPER_BOUND_ORACLE_VERSION, linewidth=1.8)
    ax.plot(equity["Date"], equity["equity_buy_and_hold"], label=BUYHOLD_VERSION, linewidth=1.8)
    ax.set_title("Oracle Equity Curve Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_equity_curve_comparison.png", dpi=150)
    plt.close(fig)

    drawdown = pd.DataFrame({"Date": equity["Date"]})
    drawdown["current_mapped_oracle_drawdown"] = compute_drawdown(equity["equity_current_mapped_oracle"])
    drawdown["full_upper_bound_oracle_drawdown"] = compute_drawdown(equity["equity_full_upper_bound_oracle"])
    drawdown["buy_and_hold_drawdown"] = compute_drawdown(equity["equity_buy_and_hold"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drawdown["Date"], drawdown["current_mapped_oracle_drawdown"], label=CURRENT_MAPPED_ORACLE_VERSION, linewidth=1.6)
    ax.plot(drawdown["Date"], drawdown["full_upper_bound_oracle_drawdown"], label=FULL_UPPER_BOUND_ORACLE_VERSION, linewidth=1.6)
    ax.plot(drawdown["Date"], drawdown["buy_and_hold_drawdown"], label=BUYHOLD_VERSION, linewidth=1.6)
    ax.set_title("Oracle Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_drawdown_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
