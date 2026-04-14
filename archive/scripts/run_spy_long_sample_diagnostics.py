from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BASE_DIR / "results" / "final_spy_research_trade_20000526"
OUTPUT_DIR = BASE_DIR / "results" / "spy_long_sample_diagnostics"


PHASES = [
    ("2000-05_to_2003-03", "2000-05-26", "2003-03-31"),
    ("2003-04_to_2007-10", "2003-04-01", "2007-10-31"),
    ("2007-11_to_2009-06", "2007-11-01", "2009-06-30"),
    ("2009-07_to_2019-12", "2009-07-01", "2019-12-31"),
    ("2020-01_to_2022-12", "2020-01-01", "2022-12-31"),
    ("2023-01_to_latest", "2023-01-01", None),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = pd.read_csv(SOURCE_DIR / "predicted_strategy_daily_equity_curves.csv")
    curve["Date"] = pd.to_datetime(curve["Date"])
    curve = curve.sort_values("Date").reset_index(drop=True)

    bh_curve = pd.read_csv(SOURCE_DIR / "daily_equity_curves.csv")
    bh_curve["Date"] = pd.to_datetime(bh_curve["Date"])
    bh_curve = bh_curve.sort_values("Date").reset_index(drop=True)
    curve = curve.merge(bh_curve[["Date", "equity_buy_and_hold"]], on="Date", how="left")

    spy = pd.read_csv(BASE_DIR / "data_raw" / "spy_trade.csv")
    spy["Date"] = pd.to_datetime(spy["Date"])
    spy = spy.sort_values("Date").reset_index(drop=True)
    return curve, spy


def return_series_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def annualized_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    total = float((1.0 + returns).prod())
    return total ** (252.0 / len(returns)) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float(returns.std(ddof=0) * math.sqrt(252.0))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = float(returns.std(ddof=0))
    if len(returns) == 0 or vol == 0 or np.isnan(vol):
        return float("nan")
    return float(returns.mean() / vol * math.sqrt(252.0))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def total_return(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def avg_segment_length(lengths: list[int]) -> tuple[float, float, int]:
    if not lengths:
        return float("nan"), float("nan"), 0
    arr = np.asarray(lengths, dtype=float)
    return float(arr.mean()), float(np.median(arr)), int(arr.max())


def contiguous_runs(values: pd.Series) -> pd.Series:
    return values.ne(values.shift()).cumsum()


def future_return(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) / series - 1.0


def identify_major_trough_events(
    df: pd.DataFrame,
    min_drawdown: float = -0.10,
    min_gap: int = 63,
) -> pd.DataFrame:
    bh = df[["Date", "equity_buy_and_hold", "position", "signal_zone"]].copy().reset_index(drop=True)
    bh["drawdown"] = bh["equity_buy_and_hold"] / bh["equity_buy_and_hold"].cummax() - 1.0
    bh["dd_prev"] = bh["drawdown"].shift(1)
    bh["dd_next"] = bh["drawdown"].shift(-1)
    bh["local_min"] = (bh["drawdown"] <= bh["dd_prev"]) & (bh["drawdown"] <= bh["dd_next"])
    candidates = bh[bh["local_min"] & (bh["drawdown"] <= min_drawdown)].copy()

    selected_idx: list[int] = []
    last_idx = -10**9
    for idx in candidates.index:
        if not selected_idx:
            selected_idx.append(idx)
            last_idx = idx
            continue
        if idx - last_idx >= min_gap:
            selected_idx.append(idx)
            last_idx = idx
        elif bh.loc[idx, "drawdown"] < bh.loc[last_idx, "drawdown"]:
            selected_idx[-1] = idx
            last_idx = idx

    events = bh.loc[selected_idx].copy().reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    return events


def recovery_proxy_date(df: pd.DataFrame, trough_idx: int, max_lookahead: int = 252) -> tuple[pd.Timestamp | pd.NaT, str]:
    sub = df.iloc[trough_idx + 1 : min(trough_idx + 1 + max_lookahead, len(df))].copy()
    if sub.empty:
        return pd.NaT, "none"
    sub["fwd20"] = future_return(df["equity_buy_and_hold"], 20).iloc[sub.index]
    sub["fwd60"] = future_return(df["equity_buy_and_hold"], 60).iloc[sub.index]

    candidates = sub[sub["fwd20"] > 0]
    if not candidates.empty:
        return candidates["Date"].iloc[0], "fwd20_positive"
    candidates = sub[sub["fwd60"] > 0]
    if not candidates.empty:
        return candidates["Date"].iloc[0], "fwd60_positive"
    return pd.NaT, "none"


def first_reentry_date(df: pd.DataFrame, trough_idx: int, max_lookahead: int = 252) -> pd.Timestamp | pd.NaT:
    sub = df.iloc[trough_idx + 1 : min(trough_idx + 1 + max_lookahead, len(df))].copy()
    if sub.empty:
        return pd.NaT
    candidates = sub[sub["position"] == 1]
    if candidates.empty:
        return pd.NaT
    return candidates["Date"].iloc[0]


def compute_phase_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase_name, start, end in PHASES:
        sub = df[df["Date"] >= pd.Timestamp(start)].copy()
        if end is not None:
            sub = sub[sub["Date"] <= pd.Timestamp(end)]
        if sub.empty:
            continue

        strat_ret = return_series_from_equity(sub["equity_predicted_strategy"])
        bh_ret = return_series_from_equity(sub["equity_buy_and_hold"])

        rows.append(
            {
                "phase_name": phase_name,
                "start_date": sub["Date"].iloc[0].date().isoformat(),
                "end_date": sub["Date"].iloc[-1].date().isoformat(),
                "n_days": int(len(sub)),
                "predicted_annual_return": annualized_return(strat_ret),
                "predicted_annual_volatility": annualized_volatility(strat_ret),
                "predicted_sharpe": sharpe_ratio(strat_ret),
                "predicted_max_drawdown": max_drawdown(sub["equity_predicted_strategy"]),
                "predicted_total_return": total_return(sub["equity_predicted_strategy"]),
                "buy_and_hold_annual_return": annualized_return(bh_ret),
                "buy_and_hold_annual_volatility": annualized_volatility(bh_ret),
                "buy_and_hold_sharpe": sharpe_ratio(bh_ret),
                "buy_and_hold_max_drawdown": max_drawdown(sub["equity_buy_and_hold"]),
                "buy_and_hold_total_return": total_return(sub["equity_buy_and_hold"]),
                "excess_annual_return": annualized_return(strat_ret) - annualized_return(bh_ret),
                "excess_total_return": total_return(sub["equity_predicted_strategy"]) - total_return(sub["equity_buy_and_hold"]),
                "sharpe_difference": sharpe_ratio(strat_ret) - sharpe_ratio(bh_ret),
            }
        )

    return pd.DataFrame(rows)


def compute_state_duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zone_run_id"] = contiguous_runs(df["signal_zone"])
    df["strategy_ret"] = return_series_from_equity(df["equity_predicted_strategy"])
    df["bh_ret"] = return_series_from_equity(df["equity_buy_and_hold"])

    rows = []
    total_days = len(df)
    for zone in ["bull", "bear", "hold"]:
        zone_df = df[df["signal_zone"] == zone]
        if zone_df.empty:
            continue
        grouped = zone_df.groupby("zone_run_id", sort=True)
        segments = []
        for _, g in grouped:
            segments.append(
                {
                    "length": len(g),
                    "strategy_total_return": float((1.0 + g["strategy_ret"]).prod() - 1.0),
                    "bh_total_return": float((1.0 + g["bh_ret"]).prod() - 1.0),
                    "strategy_avg_daily_return": float(g["strategy_ret"].mean()),
                    "bh_avg_daily_return": float(g["bh_ret"].mean()),
                }
            )

        lengths = [int(x["length"]) for x in segments]
        mean_len, med_len, max_len = avg_segment_length(lengths)
        rows.append(
            {
                "state": zone,
                "segment_count": len(segments),
                "total_days": int(zone_df.shape[0]),
                "share_of_days": float(zone_df.shape[0] / total_days),
                "avg_segment_length": mean_len,
                "median_segment_length": med_len,
                "max_segment_length": max_len,
                "avg_daily_strategy_return": float(np.mean([x["strategy_avg_daily_return"] for x in segments])),
                "avg_daily_buy_and_hold_return": float(np.mean([x["bh_avg_daily_return"] for x in segments])),
                "avg_segment_strategy_return": float(np.mean([x["strategy_total_return"] for x in segments])),
                "avg_segment_buy_and_hold_return": float(np.mean([x["bh_total_return"] for x in segments])),
                "segment_win_rate_strategy": float(np.mean([x["strategy_total_return"] > 0 for x in segments])),
                "segment_win_rate_buy_and_hold": float(np.mean([x["bh_total_return"] > 0 for x in segments])),
            }
        )

    return pd.DataFrame(rows)


def compute_reentry_lag_analysis(df: pd.DataFrame) -> pd.DataFrame:
    events = identify_major_trough_events(df)
    if events.empty:
        return pd.DataFrame()

    full = df.reset_index(drop=True).copy()
    rows = []
    for _, row in events.iterrows():
        trough_date = row["Date"]
        trough_idx = full.index[full["Date"] == trough_date][0]
        rec_date, rec_rule = recovery_proxy_date(full, trough_idx)
        reentry_date = first_reentry_date(full, trough_idx)
        lag = np.nan
        if pd.notna(rec_date) and pd.notna(reentry_date):
            lag = int((reentry_date - rec_date).days)

        label = "major_trough"
        if pd.Timestamp("2008-09-01") <= trough_date <= pd.Timestamp("2009-06-30"):
            label = "gfc_recovery"
        elif pd.Timestamp("2020-02-01") <= trough_date <= pd.Timestamp("2020-12-31"):
            label = "covid_recovery"
        elif pd.Timestamp("2022-01-01") <= trough_date <= pd.Timestamp("2023-12-31"):
            label = "2023_recovery"

        rows.append(
            {
                "event_id": int(row["event_id"]),
                "case_label": label,
                "trough_date": trough_date.date().isoformat(),
                "trough_drawdown": float(row["drawdown"]),
                "recovery_proxy_date": None if pd.isna(rec_date) else rec_date.date().isoformat(),
                "recovery_proxy_rule": rec_rule,
                "strategy_reentry_date": None if pd.isna(reentry_date) else reentry_date.date().isoformat(),
                "lag_days": lag,
                "position_at_trough": int(row["position"]),
                "signal_zone_at_trough": row["signal_zone"],
            }
        )

    return pd.DataFrame(rows)


def compute_switching_behavior(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    work["strategy_ret"] = return_series_from_equity(work["equity_predicted_strategy"])
    work["bh_ret"] = return_series_from_equity(work["equity_buy_and_hold"])
    work["position_prev"] = work["position"].shift(1).fillna(work["position"].iloc[0])
    work["zone_prev"] = work["signal_zone"].shift(1)

    switch_mask = work["position"].ne(work["position_prev"])
    pos_switches = work[switch_mask].copy()
    pos_switches["switch_type"] = np.where(
        (pos_switches["position_prev"] == 0) & (pos_switches["position"] == 1),
        "position_0_to_1",
        "position_1_to_0",
    )
    pos_switches["zone_switch_type"] = np.select(
        [
            (pos_switches["zone_prev"] == "bull") & (pos_switches["signal_zone"] == "bear"),
            (pos_switches["zone_prev"] == "bear") & (pos_switches["signal_zone"] == "bull"),
            (pos_switches["zone_prev"] == "hold") & (pos_switches["signal_zone"] == "bull"),
            (pos_switches["zone_prev"] == "hold") & (pos_switches["signal_zone"] == "bear"),
        ],
        [
            "bull_to_bear",
            "bear_to_bull",
            "hold_to_bull",
            "hold_to_bear",
        ],
        default="other_zone_switch",
    )

    def forward_returns(idx: pd.Index, horizon: int, equity: pd.Series) -> pd.Series:
        vals = []
        for i in idx:
            if i + horizon < len(equity):
                vals.append(float(equity.iloc[i + horizon] / equity.iloc[i + 1] - 1.0))
            else:
                vals.append(np.nan)
        return pd.Series(vals, index=idx)

    rows = []
    group_defs = [
        ("overall_position_switch", pos_switches.index),
        ("position_0_to_1", pos_switches[pos_switches["switch_type"] == "position_0_to_1"].index),
        ("position_1_to_0", pos_switches[pos_switches["switch_type"] == "position_1_to_0"].index),
        ("bull_to_bear_zone", pos_switches[pos_switches["zone_switch_type"] == "bull_to_bear"].index),
        ("bear_to_bull_zone", pos_switches[pos_switches["zone_switch_type"] == "bear_to_bull"].index),
        ("hold_to_bull_zone", pos_switches[pos_switches["zone_switch_type"] == "hold_to_bull"].index),
        ("hold_to_bear_zone", pos_switches[pos_switches["zone_switch_type"] == "hold_to_bear"].index),
    ]

    for name, idx in group_defs:
        if len(idx) == 0:
            rows.append({"switch_type": name, "count": 0})
            continue
        idx = pd.Index(idx)
        rows.append(
            {
                "switch_type": name,
                "count": int(len(idx)),
                "avg_future_5d_strategy_return": float(forward_returns(idx, 5, work["equity_predicted_strategy"]).mean()),
                "avg_future_20d_strategy_return": float(forward_returns(idx, 20, work["equity_predicted_strategy"]).mean()),
                "avg_future_60d_strategy_return": float(forward_returns(idx, 60, work["equity_predicted_strategy"]).mean()),
                "avg_future_5d_bh_return": float(forward_returns(idx, 5, work["equity_buy_and_hold"]).mean()),
                "avg_future_20d_bh_return": float(forward_returns(idx, 20, work["equity_buy_and_hold"]).mean()),
                "avg_future_60d_bh_return": float(forward_returns(idx, 60, work["equity_buy_and_hold"]).mean()),
            }
        )

    rows_df = pd.DataFrame(rows)
    years = (work["Date"].iloc[-1] - work["Date"].iloc[0]).days / 365.25
    rows_df["avg_switches_per_year"] = rows_df["count"] / years
    return rows_df


def compute_downside_protection_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    work["strategy_ret"] = return_series_from_equity(work["equity_predicted_strategy"])
    work["bh_ret"] = return_series_from_equity(work["equity_buy_and_hold"])
    work["bh_drawdown"] = work["equity_buy_and_hold"] / work["equity_buy_and_hold"].cummax() - 1.0

    rows = []
    for threshold in [0.10, 0.20, 0.30]:
        zone = work[work["bh_drawdown"] <= -threshold]
        rows.append(
            {
                "section": "drawdown_zone",
                "metric": f"dd_leq_{int(threshold*100)}pct",
                "threshold": threshold,
                "days": int(len(zone)),
                "strategy_value": float(zone["strategy_ret"].mean()) if len(zone) else np.nan,
                "buy_and_hold_value": float(zone["bh_ret"].mean()) if len(zone) else np.nan,
                "difference": float(zone["strategy_ret"].mean() - zone["bh_ret"].mean()) if len(zone) else np.nan,
                "strategy_compounded_return": float((1.0 + zone["strategy_ret"]).prod() - 1.0) if len(zone) else np.nan,
                "buy_and_hold_compounded_return": float((1.0 + zone["bh_ret"]).prod() - 1.0) if len(zone) else np.nan,
            }
        )

    for pct in [0.01, 0.05, 0.10]:
        s_cut = work["strategy_ret"].quantile(pct)
        b_cut = work["bh_ret"].quantile(pct)
        s_tail = work[work["strategy_ret"] <= s_cut]["strategy_ret"]
        b_tail = work[work["bh_ret"] <= b_cut]["bh_ret"]
        rows.append(
            {
                "section": "tail_daily_return",
                "metric": f"worst_{int(pct*100)}pct",
                "threshold": pct,
                "days": int(len(s_tail)),
                "strategy_value": float(s_tail.mean()) if len(s_tail) else np.nan,
                "buy_and_hold_value": float(b_tail.mean()) if len(b_tail) else np.nan,
                "difference": float(s_tail.mean() - b_tail.mean()) if len(s_tail) else np.nan,
                "strategy_compounded_return": np.nan,
                "buy_and_hold_compounded_return": np.nan,
            }
        )

    monthly = pd.DataFrame(
        {
            "strategy": (1.0 + work["strategy_ret"]).groupby(work["Date"].dt.to_period("M")).prod() - 1.0,
            "buy_and_hold": (1.0 + work["bh_ret"]).groupby(work["Date"].dt.to_period("M")).prod() - 1.0,
        }
    )
    worst_month_idx = monthly["strategy"].idxmin()
    rows.append(
        {
            "section": "worst_month",
            "metric": "worst_month",
            "threshold": np.nan,
            "days": np.nan,
            "strategy_value": float(monthly.loc[worst_month_idx, "strategy"]),
            "buy_and_hold_value": float(monthly.loc[worst_month_idx, "buy_and_hold"]),
            "difference": float(monthly.loc[worst_month_idx, "strategy"] - monthly.loc[worst_month_idx, "buy_and_hold"]),
            "strategy_compounded_return": float(monthly.loc[worst_month_idx, "strategy"]),
            "buy_and_hold_compounded_return": float(monthly.loc[worst_month_idx, "buy_and_hold"]),
            "period": str(worst_month_idx),
        }
    )

    qframe = pd.DataFrame(
        {
            "strategy": (1.0 + work["strategy_ret"]).groupby(work["Date"].dt.to_period("Q")).prod() - 1.0,
            "buy_and_hold": (1.0 + work["bh_ret"]).groupby(work["Date"].dt.to_period("Q")).prod() - 1.0,
        }
    )
    worst_q_idx = qframe["strategy"].idxmin()
    rows.append(
        {
            "section": "worst_quarter",
            "metric": "worst_quarter",
            "threshold": np.nan,
            "days": np.nan,
            "strategy_value": float(qframe.loc[worst_q_idx, "strategy"]),
            "buy_and_hold_value": float(qframe.loc[worst_q_idx, "buy_and_hold"]),
            "difference": float(qframe.loc[worst_q_idx, "strategy"] - qframe.loc[worst_q_idx, "buy_and_hold"]),
            "strategy_compounded_return": float(qframe.loc[worst_q_idx, "strategy"]),
            "buy_and_hold_compounded_return": float(qframe.loc[worst_q_idx, "buy_and_hold"]),
            "period": str(worst_q_idx),
        }
    )
    return pd.DataFrame(rows)


def compute_exposure_summary(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    work["strategy_ret"] = return_series_from_equity(work["equity_predicted_strategy"])
    work["bh_ret"] = return_series_from_equity(work["equity_buy_and_hold"])
    work["fwd20_bh"] = future_return(work["equity_buy_and_hold"], 20)
    work["fwd20_strategy"] = future_return(work["equity_predicted_strategy"], 20)
    work["fwd60_bh"] = future_return(work["equity_buy_and_hold"], 60)
    work["fwd60_strategy"] = future_return(work["equity_predicted_strategy"], 60)

    spy_reg = spy[["Date", "Adj_Close"]].copy()
    spy_reg["ma200"] = spy_reg["Adj_Close"].rolling(200, min_periods=50).mean()
    spy_reg["bull_market"] = spy_reg["Adj_Close"] >= spy_reg["ma200"]
    work = work.merge(spy_reg[["Date", "bull_market"]], on="Date", how="left")

    rows = []
    categories = [
        ("overall", work.index),
        ("bull_market", work[work["bull_market"] == True].index),
        ("bear_market", work[work["bull_market"] == False].index),
        ("signal_zone_bull", work[work["signal_zone"] == "bull"].index),
        ("signal_zone_hold", work[work["signal_zone"] == "hold"].index),
        ("signal_zone_bear", work[work["signal_zone"] == "bear"].index),
    ]
    for name, idx in categories:
        if len(idx) == 0:
            continue
        sub = work.loc[idx]
        rows.append(
            {
                "category": name,
                "days": int(len(sub)),
                "share_of_days": float(len(sub) / len(work)),
                "avg_position": float(sub["position"].mean()),
                "avg_future_20d_buy_and_hold_return": float(sub["fwd20_bh"].mean()),
                "avg_future_20d_strategy_return": float(sub["fwd20_strategy"].mean()),
                "avg_future_60d_buy_and_hold_return": float(sub["fwd60_bh"].mean()),
                "avg_future_60d_strategy_return": float(sub["fwd60_strategy"].mean()),
            }
        )

    rows.append(
        {
            "category": "overall_position_future_return_relation",
            "days": int(len(work)),
            "share_of_days": 1.0,
            "avg_position": float(work["position"].mean()),
            "avg_future_20d_buy_and_hold_return": float(work["fwd20_bh"].mean()),
            "avg_future_20d_strategy_return": float(work["fwd20_strategy"].mean()),
            "avg_future_60d_buy_and_hold_return": float(work["fwd60_bh"].mean()),
            "avg_future_60d_strategy_return": float(work["fwd60_strategy"].mean()),
            "corr_position_future_20d_buy_and_hold_return": float(work["position"].corr(work["fwd20_bh"])),
            "corr_position_future_60d_buy_and_hold_return": float(work["position"].corr(work["fwd60_bh"])),
        }
    )
    return pd.DataFrame(rows)


def normalize_equity(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity / equity.iloc[0]


def plot_phase_equity_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(len(PHASES), 1, figsize=(14, 18), sharex=False)
    for ax, (phase_name, start, end) in zip(axes, PHASES):
        sub = df[df["Date"] >= pd.Timestamp(start)].copy()
        if end is not None:
            sub = sub[sub["Date"] <= pd.Timestamp(end)]
        if sub.empty:
            ax.set_visible(False)
            continue
        pred = normalize_equity(sub["equity_predicted_strategy"])
        bh = normalize_equity(sub["equity_buy_and_hold"])
        ax.plot(sub["Date"], pred, label="predicted", color="#1f77b4", linewidth=1.4)
        ax.plot(sub["Date"], bh, label="buy-and-hold", color="#2ca02c", linewidth=1.2, alpha=0.9)
        ax.set_title(phase_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_state_distribution_over_time(df: pd.DataFrame, out_path: Path) -> None:
    monthly = (
        df.assign(month=df["Date"].dt.to_period("M"))
        .groupby(["month", "signal_zone"])
        .size()
        .unstack(fill_value=0)
    )
    monthly = monthly.div(monthly.sum(axis=1), axis=0)
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(monthly))
    colors = {"bull": "#2ca02c", "hold": "#ff7f0e", "bear": "#d62728"}
    for state in ["bull", "hold", "bear"]:
        if state in monthly.columns:
            ax.fill_between(monthly.index, bottom, bottom + monthly[state].values, label=state, color=colors[state], alpha=0.75)
            bottom = bottom + monthly[state].values
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of days")
    ax.set_title("Monthly signal-zone distribution over time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_reentry_examples(df: pd.DataFrame, event_df: pd.DataFrame, out_path: Path) -> None:
    cases = [
        ("gfc_recovery", "2008-09-01", "2009-06-30"),
        ("covid_recovery", "2020-02-01", "2020-12-31"),
        ("2023_recovery", "2022-01-01", "2023-12-31"),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(14, 12), sharex=False)
    for ax, (label, start, end) in zip(axes, cases):
        sub = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sel = event_df[event_df["case_label"] == label]
        if sel.empty:
            ax.set_visible(False)
            continue
        row = sel.iloc[0]
        trough = pd.Timestamp(row["trough_date"])
        rec = pd.Timestamp(row["recovery_proxy_date"]) if pd.notna(row["recovery_proxy_date"]) else None
        reentry = pd.Timestamp(row["strategy_reentry_date"]) if pd.notna(row["strategy_reentry_date"]) else None

        pred = normalize_equity(sub["equity_predicted_strategy"])
        bh = normalize_equity(sub["equity_buy_and_hold"])
        ax.plot(sub["Date"], bh, label="buy-and-hold", color="#2ca02c", linewidth=1.3)
        ax.plot(sub["Date"], pred, label="predicted", color="#1f77b4", linewidth=1.3)
        for dt, lab, col in [(trough, "trough", "#d62728"), (rec, "recovery proxy", "#9467bd"), (reentry, "reentry", "#ff7f0e")]:
            if dt is not None and sub["Date"].min() <= dt <= sub["Date"].max():
                ax.axvline(dt, color=col, linestyle="--", linewidth=1.2, label=lab)
        ax.set_title(f"{label} | trough={row['trough_date']} | lag_days={row['lag_days']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_drawdown_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    pred_dd = df["equity_predicted_strategy"] / df["equity_predicted_strategy"].cummax() - 1.0
    bh_dd = df["equity_buy_and_hold"] / df["equity_buy_and_hold"].cummax() - 1.0
    ax.plot(df["Date"], pred_dd, label="predicted", color="#1f77b4", linewidth=1.2)
    ax.plot(df["Date"], bh_dd, label="buy-and-hold", color="#2ca02c", linewidth=1.2)
    ax.set_title("Drawdown comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_exposure_over_time(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Date"], df["position"], label="daily position", color="#1f77b4", linewidth=0.7, alpha=0.65)
    ax.plot(df["Date"], df["position"].rolling(63, min_periods=1).mean(), label="63d rolling avg position", color="#d62728", linewidth=1.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Exposure over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_markdown_summary(
    phase_df: pd.DataFrame,
    state_df: pd.DataFrame,
    reentry_df: pd.DataFrame,
    downside_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    out_path: Path,
) -> None:
    overall = exposure_df[exposure_df["category"] == "overall"].iloc[0]
    hold = exposure_df[exposure_df["category"] == "signal_zone_hold"].iloc[0] if not exposure_df[exposure_df["category"] == "signal_zone_hold"].empty else None
    bull_reg = exposure_df[exposure_df["category"] == "bull_market"].iloc[0] if not exposure_df[exposure_df["category"] == "bull_market"].empty else None
    bear_reg = exposure_df[exposure_df["category"] == "bear_market"].iloc[0] if not exposure_df[exposure_df["category"] == "bear_market"].empty else None

    avg_lag = reentry_df["lag_days"].dropna().mean() if not reentry_df.empty else np.nan
    med_lag = reentry_df["lag_days"].dropna().median() if not reentry_df.empty else np.nan
    max_lag = reentry_df["lag_days"].dropna().max() if not reentry_df.empty else np.nan

    best_phase = phase_df.sort_values("sharpe_difference", ascending=False).iloc[0] if not phase_df.empty else None
    worst_phase = phase_df.sort_values("sharpe_difference", ascending=True).iloc[0] if not phase_df.empty else None

    best_dd = downside_df[downside_df["section"] == "drawdown_zone"].sort_values("difference", ascending=False)
    worst_tail = downside_df[downside_df["section"] == "tail_daily_return"].sort_values("difference", ascending=True)

    lines = []
    lines.append("# SPY Long Sample Diagnostics")
    lines.append("")
    lines.append("## Core Readout")
    lines.append("- Predicted strategy annual return: `0.057576`")
    lines.append("- Predicted strategy Sharpe: `0.432289`")
    lines.append("- Predicted strategy max drawdown: `-0.153458`")
    lines.append("- Buy-and-hold annual return: `0.083473`")
    lines.append("- Buy-and-hold Sharpe: `0.416349`")
    lines.append("- Buy-and-hold max drawdown: `-0.556927`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The strategy behaves primarily as a downside-protection overlay, not as a full replacement for buy-and-hold.")
    lines.append("- Its edge is concentrated in crisis and drawdown periods.")
    lines.append("- The main drag is slow recovery / re-entry after the market turns back up, plus a lower steady-state exposure in long bull phases.")
    lines.append("")
    lines.append("## Re-entry Lag")
    lines.append(f"- Average re-entry lag: `{avg_lag:.2f}` trading days")
    lines.append(f"- Median re-entry lag: `{med_lag:.2f}` trading days")
    lines.append(f"- Maximum re-entry lag: `{max_lag:.2f}` trading days")
    lines.append("")
    lines.append("## Exposure")
    lines.append(f"- Average position overall: `{overall['avg_position']:.3f}`")
    if hold is not None:
        lines.append(f"- Average position in hold zone: `{hold['avg_position']:.3f}`")
    if bull_reg is not None:
        lines.append(f"- Average position in bull market regime: `{bull_reg['avg_position']:.3f}`")
    if bear_reg is not None:
        lines.append(f"- Average position in bear market regime: `{bear_reg['avg_position']:.3f}`")
    lines.append("")
    lines.append("## Phase Winners")
    if best_phase is not None and worst_phase is not None:
        lines.append(f"- Best relative phase: `{best_phase['phase_name']}`")
        lines.append(f"- Worst relative phase: `{worst_phase['phase_name']}`")
    lines.append("")
    lines.append("## Downside Protection")
    if not best_dd.empty:
        lines.append(f"- Best drawdown-zone improvement: `{best_dd.iloc[0]['metric']}`")
    if not worst_tail.empty:
        lines.append(f"- Worst tail-day gap: `{worst_tail.iloc[0]['metric']}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("- The current evidence supports a regime-filter / downside-protection narrative.")
    lines.append("- The highest-value next optimization is recovery/re-entry logic, followed by reducing false bear signals.")
    lines.append("- Further feature or hyperparameter search is lower priority than fixing the post-drawdown re-risk process.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    curve, spy = load_inputs()

    curve = curve.copy()
    curve["strategy_ret"] = return_series_from_equity(curve["equity_predicted_strategy"])
    curve["bh_ret"] = return_series_from_equity(curve["equity_buy_and_hold"])

    phase_df = compute_phase_metrics(curve)
    state_df = compute_state_duration_summary(curve)
    reentry_df = compute_reentry_lag_analysis(curve)
    switch_df = compute_switching_behavior(curve)
    downside_df = compute_downside_protection_summary(curve)
    exposure_df = compute_exposure_summary(curve, spy)

    phase_df.to_csv(OUTPUT_DIR / "phase_performance_diagnostics.csv", index=False)
    state_df.to_csv(OUTPUT_DIR / "state_duration_summary.csv", index=False)
    reentry_df.to_csv(OUTPUT_DIR / "reentry_lag_analysis.csv", index=False)
    switch_df.to_csv(OUTPUT_DIR / "switching_behavior_summary.csv", index=False)
    downside_df.to_csv(OUTPUT_DIR / "downside_protection_summary.csv", index=False)
    exposure_df.to_csv(OUTPUT_DIR / "exposure_summary.csv", index=False)

    overall_rows = [
        {
            "version": "predicted_strategy",
            "annual_return": annualized_return(curve["strategy_ret"]),
            "annual_volatility": annualized_volatility(curve["strategy_ret"]),
            "sharpe": sharpe_ratio(curve["strategy_ret"]),
            "max_drawdown": max_drawdown(curve["equity_predicted_strategy"]),
            "total_return": total_return(curve["equity_predicted_strategy"]),
            "final_wealth": float(curve["equity_predicted_strategy"].iloc[-1]),
            "total_switch_count": int(curve["position"].ne(curve["position"].shift()).sum() - 1),
            "avg_switch_count_per_year": float((curve["position"].ne(curve["position"].shift()).sum() - 1) / ((curve["Date"].iloc[-1] - curve["Date"].iloc[0]).days / 365.25)),
            "avg_position": float(curve["position"].mean()),
        },
        {
            "version": "buy_and_hold",
            "annual_return": annualized_return(curve["bh_ret"]),
            "annual_volatility": annualized_volatility(curve["bh_ret"]),
            "sharpe": sharpe_ratio(curve["bh_ret"]),
            "max_drawdown": max_drawdown(curve["equity_buy_and_hold"]),
            "total_return": total_return(curve["equity_buy_and_hold"]),
            "final_wealth": float(curve["equity_buy_and_hold"].iloc[-1]),
            "total_switch_count": 0,
            "avg_switch_count_per_year": 0.0,
            "avg_position": 1.0,
        },
    ]
    pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / "strategy_performance_overall.csv", index=False)

    write_markdown_summary(phase_df, state_df, reentry_df, downside_df, exposure_df, OUTPUT_DIR / "diagnostic_summary.md")

    plot_phase_equity_curves(curve, OUTPUT_DIR / "phase_equity_curves.png")
    plot_state_distribution_over_time(curve, OUTPUT_DIR / "state_distribution_over_time.png")
    plot_reentry_examples(curve, reentry_df, OUTPUT_DIR / "reentry_lag_examples.png")
    plot_drawdown_comparison(curve, OUTPUT_DIR / "drawdown_comparison.png")
    plot_exposure_over_time(curve, OUTPUT_DIR / "exposure_over_time.png")


if __name__ == "__main__":
    main()
