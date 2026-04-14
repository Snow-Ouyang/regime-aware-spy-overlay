import math
import site
import sys
import time
from pathlib import Path
from typing import Dict, List


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

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_multi_asset_project_common import (
    RESEARCH_TO_TRADE_MAP,
    TARGET_OOS_START,
    TRADE_ASSETS,
    TRANSACTION_COST,
    TRADING_DAYS,
    ensure_trade_inputs,
    load_risk_free_daily_series,
    load_trade_price_frame,
    signal_panels_dir,
    trade_execution_dir,
)


INITIAL_CASH = 1_000_000.0


def panel_wide_path() -> str:
    return str(signal_panels_dir() / "multi_asset_signal_panel_wide.csv")


def build_adjusted_bt_feed(frame: pd.DataFrame) -> bt.feeds.PandasData:
    data = frame.copy()
    adj_factor = np.where(data["Close"] != 0, data["Adj_Close"] / data["Close"], 1.0)
    data["open"] = data["Open"] * adj_factor
    data["high"] = data["High"] * adj_factor
    data["low"] = data["Low"] * adj_factor
    data["close"] = data["Close"] * adj_factor
    data["volume"] = data["Volume"].fillna(0.0)
    data["openinterest"] = 0.0
    data = data.set_index("Date")[["open", "high", "low", "close", "volume", "openinterest"]]
    return bt.feeds.PandasData(dataname=data)


def prepare_trade_inputs() -> Dict[str, object]:
    ensure_trade_inputs()
    if not signal_panels_dir().joinpath("multi_asset_signal_panel_wide.csv").exists():
        raise FileNotFoundError(f"Missing signal panel: {panel_wide_path()}")

    signal_wide = pd.read_csv(panel_wide_path())
    signal_wide["Date"] = pd.to_datetime(signal_wide["Date"], errors="coerce")
    signal_wide = signal_wide.sort_values("Date").reset_index(drop=True)

    rf_daily = load_risk_free_daily_series()
    trade_frames: Dict[str, pd.DataFrame] = {}
    master_dates = None
    for _, stem in TRADE_ASSETS:
        frame = load_trade_price_frame(stem)
        frame = frame.loc[frame["Date"] >= TARGET_OOS_START].copy().reset_index(drop=True)
        trade_frames[stem] = frame
        current_dates = frame[["Date"]].copy()
        master_dates = current_dates if master_dates is None else master_dates.merge(current_dates, on="Date", how="outer")

    master_dates = master_dates.sort_values("Date").drop_duplicates().reset_index(drop=True)
    signal_aligned = master_dates.merge(signal_wide, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    signal_cols = [column for column in signal_aligned.columns if column != "Date"]
    signal_aligned[signal_cols] = signal_aligned[signal_cols].ffill().fillna(0.0)

    rf_aligned = master_dates.merge(rf_daily, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    rf_aligned["rf_daily"] = rf_aligned["rf_daily"].ffill().fillna(0.0)

    return {
        "signal_aligned": signal_aligned,
        "rf_aligned": rf_aligned,
        "trade_frames": trade_frames,
    }


def compute_strategy_metrics(equity_frame: pd.DataFrame, rf_frame: pd.DataFrame) -> Dict[str, float]:
    merged = equity_frame.merge(rf_frame, on="Date", how="left")
    merged["rf_daily"] = merged["rf_daily"].ffill().fillna(0.0)
    returns = merged["equity"].pct_change().fillna(0.0).to_numpy(dtype=float)
    excess = returns - merged["rf_daily"].to_numpy(dtype=float)
    years = len(merged) / TRADING_DAYS
    final_wealth = float(merged["equity"].iloc[-1])
    annual_return = float(final_wealth ** (1.0 / years) - 1.0) if years > 0 else np.nan
    annual_volatility = float(np.std(returns[1:], ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan
    if len(excess[1:]) > 1:
        mean_excess = float(np.mean(excess[1:]))
        std_excess = float(np.std(excess[1:], ddof=1))
        sharpe = float(np.sqrt(TRADING_DAYS) * mean_excess / std_excess) if std_excess > 0 else np.nan
    else:
        sharpe = np.nan
    running_max = merged["equity"].cummax()
    max_drawdown = float((merged["equity"] / running_max - 1.0).min())
    total_return = float(final_wealth - 1.0)
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "final_wealth": final_wealth,
    }


class EWBullSignalStrategy(bt.Strategy):
    params = dict(signal_df=None, rf_df=None, trade_stems=None)

    def __init__(self):
        self.signal_df = self.p.signal_df.copy()
        self.signal_df["Date"] = pd.to_datetime(self.signal_df["Date"])
        self.signal_df = self.signal_df.set_index("Date")

        self.rf_df = self.p.rf_df.copy()
        self.rf_df["Date"] = pd.to_datetime(self.rf_df["Date"])
        self.rf_map = self.rf_df.set_index("Date")["rf_daily"].to_dict()

        self.trade_stems = list(self.p.trade_stems)
        self.data_by_stem = {stem: data for stem, data in zip(self.trade_stems, self.datas)}
        self.equity_log: List[Dict[str, float]] = []
        self.weight_log: List[Dict[str, float]] = []
        self.signal_log: List[Dict[str, float]] = []

    def next(self):
        current_date = pd.Timestamp(bt.num2date(self.datas[0].datetime[0]).date())
        rf_daily = float(self.rf_map.get(current_date, 0.0))
        current_cash = self.broker.get_cash()
        if current_cash > 0:
            self.broker.add_cash(current_cash * rf_daily)

        signal_row = self.signal_df.loc[current_date]
        bull_map = {
            "spy_trade": int(signal_row["gspc_predicted_bull_flag"]),
            "ijh_trade": int(signal_row["mid_predicted_bull_flag"]),
            "iwm_trade": int(signal_row["rut_predicted_bull_flag"]),
        }
        n_bull = int(sum(bull_map.values()))
        if n_bull > 0:
            target_weight = 1.0 / n_bull
            target_weights = {stem: (target_weight if bull_map[stem] == 1 else 0.0) for stem in self.trade_stems}
        else:
            target_weights = {stem: 0.0 for stem in self.trade_stems}

        for stem in self.trade_stems:
            self.order_target_percent(data=self.data_by_stem[stem], target=target_weights[stem])

        row = {"Date": current_date}
        for stem in self.trade_stems:
            row[f"weight_{stem.replace('_trade', '')}"] = float(target_weights[stem])
        row["cash_weight"] = float(max(0.0, 1.0 - sum(target_weights.values())))
        self.weight_log.append(row)

        self.signal_log.append(
            {
                "Date": current_date,
                "gspc_bull": bull_map["spy_trade"],
                "mid_bull": bull_map["ijh_trade"],
                "rut_bull": bull_map["iwm_trade"],
                "n_bull_assets": n_bull,
            }
        )

        self.equity_log.append({"Date": current_date, "equity": float(self.broker.getvalue() / INITIAL_CASH)})


class BuyHoldEqualWeightStrategy(bt.Strategy):
    params = dict(rf_df=None, trade_stems=None)

    def __init__(self):
        self.rf_df = self.p.rf_df.copy()
        self.rf_df["Date"] = pd.to_datetime(self.rf_df["Date"])
        self.rf_map = self.rf_df.set_index("Date")["rf_daily"].to_dict()
        self.trade_stems = list(self.p.trade_stems)
        self.equity_log: List[Dict[str, float]] = []
        self.invested = False

    def next(self):
        current_date = pd.Timestamp(bt.num2date(self.datas[0].datetime[0]).date())
        rf_daily = float(self.rf_map.get(current_date, 0.0))
        current_cash = self.broker.get_cash()
        if current_cash > 0:
            self.broker.add_cash(current_cash * rf_daily)

        if not self.invested:
            weight = 1.0 / len(self.datas)
            for data in self.datas:
                self.order_target_percent(data=data, target=weight)
            self.invested = True

        self.equity_log.append({"Date": current_date, "equity": float(self.broker.getvalue() / INITIAL_CASH)})


def run_backtrader_strategy(strategy_cls, trade_frames: Dict[str, pd.DataFrame], rf_aligned: pd.DataFrame, **strategy_kwargs):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=TRANSACTION_COST)
    for _, stem in TRADE_ASSETS:
        feed = build_adjusted_bt_feed(trade_frames[stem])
        cerebro.adddata(feed, name=stem)
    cerebro.addstrategy(strategy_cls, rf_df=rf_aligned, trade_stems=[stem for _, stem in TRADE_ASSETS], **strategy_kwargs)
    strategies = cerebro.run()
    return strategies[0]


def plot_equity_curves(signal_curve: pd.DataFrame, baseline_curve: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(signal_curve["Date"], signal_curve["equity"], label="ew_bull_signal_strategy", linewidth=2.0, color="#1f77b4")
    ax.plot(baseline_curve["Date"], baseline_curve["equity"], label="buy_and_hold_equal_weight", linewidth=2.0, color="#2ca02c")
    ax.set_title("Portfolio Strategy vs Buy and Hold")
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
    prepared = prepare_trade_inputs()
    signal_aligned = prepared["signal_aligned"]
    rf_aligned = prepared["rf_aligned"]
    trade_frames = prepared["trade_frames"]

    signal_strategy = run_backtrader_strategy(
        EWBullSignalStrategy,
        trade_frames=trade_frames,
        rf_aligned=rf_aligned,
        signal_df=signal_aligned,
    )
    buyhold_strategy = run_backtrader_strategy(
        BuyHoldEqualWeightStrategy,
        trade_frames=trade_frames,
        rf_aligned=rf_aligned,
    )

    signal_curve = pd.DataFrame(signal_strategy.equity_log).sort_values("Date").reset_index(drop=True)
    buyhold_curve = pd.DataFrame(buyhold_strategy.equity_log).sort_values("Date").reset_index(drop=True)
    signal_weights = pd.DataFrame(signal_strategy.weight_log).sort_values("Date").reset_index(drop=True)
    signal_daily = pd.DataFrame(signal_strategy.signal_log).sort_values("Date").reset_index(drop=True)

    performance_summary = pd.DataFrame(
        [
            {"strategy_name": "ew_bull_signal_strategy", **compute_strategy_metrics(signal_curve, rf_aligned)},
            {"strategy_name": "buy_and_hold_equal_weight", **compute_strategy_metrics(buyhold_curve, rf_aligned)},
        ]
    )

    daily_equity = signal_curve.rename(columns={"equity": "equity_ew_bull_signal_strategy"}).merge(
        buyhold_curve.rename(columns={"equity": "equity_buy_and_hold_equal_weight"}),
        on="Date",
        how="outer",
    ).sort_values("Date").reset_index(drop=True)

    output_dir = trade_execution_dir()
    performance_summary.to_csv(output_dir / "portfolio_performance_summary.csv", index=False)
    daily_equity.to_csv(output_dir / "portfolio_daily_equity_curves.csv", index=False)
    signal_weights.to_csv(output_dir / "portfolio_daily_weights.csv", index=False)
    signal_daily.to_csv(output_dir / "portfolio_daily_signals.csv", index=False)
    plot_equity_curves(signal_curve, buyhold_curve, str(output_dir / "portfolio_strategy_vs_buyhold.png"))

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Trade execution results directory: {output_dir}")
    print(performance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
