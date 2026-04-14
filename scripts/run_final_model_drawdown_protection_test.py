from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLDS = [0.08, 0.09, 0.10, 0.11, 0.12]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "single_asset_mainline" / "final_recovery_overlay_gspc_to_spy"
OUT_DIR = PROJECT_ROOT / "results" / "single_asset_mainline" / "final_model_drawdown_protection_test"
FEE_BPS = 5.0
TRADING_DAYS = 252.0


def compute_drawdown(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def annualized_sharpe(excess_returns: np.ndarray) -> float:
    values = np.asarray(excess_returns, dtype=float)
    if values.size == 0:
        return float("nan")
    std = values.std(ddof=0)
    if std == 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * values.mean() / std)


def build_strategy_metrics(frame: pd.DataFrame) -> dict[str, float]:
    strategy_ret = frame["strategy_ret"].to_numpy(dtype=float, copy=False)
    strategy_excess = frame["strategy_excess_ret"].to_numpy(dtype=float, copy=False)
    equity = (1.0 + pd.Series(strategy_ret)).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    switch_count = int((frame["position"].diff().fillna(frame["position"]).abs() > 0).sum())
    years = max(len(frame) / TRADING_DAYS, 1e-9)
    return {
        "annual_return": float((equity.iloc[-1]) ** (TRADING_DAYS / len(frame)) - 1.0),
        "annual_volatility": float(np.std(strategy_ret, ddof=0) * np.sqrt(TRADING_DAYS)),
        "sharpe": annualized_sharpe(strategy_excess),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "final_wealth": float(equity.iloc[-1]),
        "total_switch_count": switch_count,
        "avg_switch_count_per_year": float(switch_count / years),
        "avg_position": float(frame["position"].mean()),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = pd.read_csv(BASE_RESULTS_DIR / "mapped_strategy_daily_detail.csv")
    buyhold = pd.read_csv(BASE_RESULTS_DIR / "buy_and_hold_daily_detail.csv")
    for frame in [mapped, buyhold]:
        frame["Date"] = pd.to_datetime(frame["Date"])
    return mapped, buyhold


def apply_drawdown_liquidation_overlay(mapped: pd.DataFrame, buyhold: pd.DataFrame, threshold: float) -> pd.DataFrame:
    merged = mapped.merge(
        buyhold[["Date", "next_ret", "next_rf_daily"]],
        on="Date",
        how="left",
        suffixes=("", "_buyhold"),
    ).sort_values("Date").reset_index(drop=True)

    fee_rate = FEE_BPS / 10000.0
    target_position = merged["position"].to_numpy(dtype=float, copy=False)
    next_ret = merged["next_ret"].to_numpy(dtype=float, copy=False)
    next_rf = merged["next_rf_daily"].to_numpy(dtype=float, copy=False)

    executed_position = np.zeros(len(merged), dtype=float)
    fee = np.zeros(len(merged), dtype=float)
    strategy_ret_gross = np.zeros(len(merged), dtype=float)
    strategy_ret = np.zeros(len(merged), dtype=float)
    strategy_excess_ret = np.zeros(len(merged), dtype=float)
    equity = np.zeros(len(merged), dtype=float)
    running_peak = np.zeros(len(merged), dtype=float)
    portfolio_drawdown = np.zeros(len(merged), dtype=float)
    liquidation_trigger = np.zeros(len(merged), dtype=int)
    forced_cash_today = np.zeros(len(merged), dtype=int)

    prev_position = 0.0
    equity_level = 1.0
    peak = 1.0
    force_cash_flag = False
    prev_drawdown = 0.0

    for idx in range(len(merged)):
        forced_cash_today[idx] = int(force_cash_flag)
        current_position = 0.0 if force_cash_flag else target_position[idx]
        executed_position[idx] = current_position
        fee[idx] = fee_rate * abs(current_position - prev_position) if current_position != prev_position else 0.0
        strategy_ret_gross[idx] = current_position * next_ret[idx] + (1.0 - current_position) * next_rf[idx]
        strategy_ret[idx] = strategy_ret_gross[idx] - fee[idx]
        strategy_excess_ret[idx] = strategy_ret[idx] - next_rf[idx]
        equity_level *= 1.0 + strategy_ret[idx]
        peak = max(peak, equity_level)
        dd = equity_level / peak - 1.0
        equity[idx] = equity_level
        running_peak[idx] = peak
        portfolio_drawdown[idx] = dd

        crossed = prev_drawdown > -threshold and dd <= -threshold
        liquidation_trigger[idx] = int(crossed)
        # Force liquidation for the next day after the threshold breach.
        force_cash_flag = bool(crossed)
        prev_position = current_position
        prev_drawdown = dd

    result = merged.copy()
    result["target_position"] = target_position
    result["position"] = executed_position
    result["forced_cash_today"] = forced_cash_today
    result["liquidation_trigger"] = liquidation_trigger
    result["fee"] = fee
    result["strategy_ret_gross"] = strategy_ret_gross
    result["strategy_ret"] = strategy_ret
    result["strategy_excess_ret"] = strategy_excess_ret
    result["equity_predicted_strategy"] = equity
    result["portfolio_drawdown"] = portfolio_drawdown
    result["running_peak"] = running_peak
    return result


def save_plot(equity_curves: pd.DataFrame, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in equity_curves.columns:
        if column == "Date":
            continue
        ax.plot(equity_curves["Date"], equity_curves[column], label=column, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


def save_bar_plot(summary: pd.DataFrame, value_column: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(summary["version"], summary[value_column])
    ax.set_title(title)
    ax.set_ylabel(value_column)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapped, buyhold = load_inputs()

    summary_rows = [{"version": "final_model", **build_strategy_metrics(mapped)}]
    equity_curves = pd.DataFrame(
        {
            "Date": mapped["Date"],
            "equity_final_model": mapped["equity_predicted_strategy"],
        }
    )

    detail_frames: list[pd.DataFrame] = []

    for threshold in THRESHOLDS:
        version = f"dd_stop_{int(round(threshold * 100)):02d}"
        overlay = apply_drawdown_liquidation_overlay(mapped, buyhold, threshold)
        overlay["version"] = version
        detail_frames.append(overlay)
        summary_rows.append({"version": version, **build_strategy_metrics(overlay)})
        equity_curves = equity_curves.merge(
            overlay[["Date", "equity_predicted_strategy"]].rename(columns={"equity_predicted_strategy": f"equity_{version}"}),
            on="Date",
            how="left",
        )

    summary_rows.append({"version": "buy_and_hold", **build_strategy_metrics(buyhold)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "strategy_performance_comparison.csv", index=False)

    equity_curves = equity_curves.merge(
        buyhold[["Date", "equity_buy_and_hold"]],
        on="Date",
        how="left",
    )
    equity_curves.to_csv(OUT_DIR / "daily_equity_curves_comparison.csv", index=False)

    detail = pd.concat(detail_frames, ignore_index=True)
    detail.to_csv(OUT_DIR / "drawdown_protection_daily_detail.csv", index=False)

    trigger_summary = (
        detail.groupby("version")
        .agg(
            trigger_count=("liquidation_trigger", "sum"),
            forced_cash_days=("forced_cash_today", "sum"),
            avg_position=("position", "mean"),
            min_drawdown=("portfolio_drawdown", "min"),
        )
        .reset_index()
    )
    trigger_summary.to_csv(OUT_DIR / "drawdown_protection_trigger_summary.csv", index=False)

    save_plot(equity_curves, "drawdown_protection_equity_curves.png", "Final Model vs Drawdown Protection Overlays")

    drawdown_curves = pd.DataFrame({"Date": equity_curves["Date"]})
    for column in equity_curves.columns:
        if column == "Date":
            continue
        drawdown_curves[column.replace("equity_", "drawdown_")] = compute_drawdown(equity_curves[column])
    save_plot(drawdown_curves, "drawdown_protection_drawdowns.png", "Drawdown Comparison")

    save_bar_plot(summary, "sharpe", "drawdown_protection_sharpe_comparison.png", "Sharpe Comparison")
    save_bar_plot(summary, "max_drawdown", "drawdown_protection_max_drawdown_comparison.png", "Max Drawdown Comparison")

    print(f"Results directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
