import os
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

from jumpmodels.jump import JumpModel


TICKER = "SPY"
FEATURE_COLUMNS: List[str] = [
    "log_downside_dev_hl5",
    "log_downside_dev_hl21",
    "ewm_return_hl5",
    "ewm_return_hl10",
    "ewm_return_hl21",
    "sortino_hl5",
    "sortino_hl10",
    "sortino_hl21",
]
REQUIRED_COLUMNS: List[str] = ["Date", "ret", "rf_daily", "excess_ret", *FEATURE_COLUMNS]
PENALTY_GRID: List[float] = [float(x) for x in range(6)]
TRADING_DAYS = 252
TRAIN_YEARS = 11
VALIDATION_YEARS = 5
STEP_MONTHS = 6
FEE_BPS = 5
BUY_HOLD_NAME = "buy_and_hold"


@dataclass
class MainWindow:
    rebalance_date: str
    oos_start: str
    oos_end: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_path() -> Path:
    return project_root() / "data_features" / "spy_jm_features.csv"


def results_dir() -> Path:
    output_dir = project_root() / "results" / "spy_oracle_fixed_penalty"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def penalty_label(penalty: float) -> str:
    return f"{penalty:.1f}".replace(".", "_")


def load_feature_frame() -> pd.DataFrame:
    path = input_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in REQUIRED_COLUMNS:
        if column != "Date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("Date", ascending=True)
    frame = frame.drop_duplicates(subset=["Date"], keep="last")
    frame = frame.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if frame.empty:
        raise ValueError("No valid SPY feature rows remain after cleaning")

    return frame


def first_date_on_or_after(dates: pd.Series, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = dates[dates >= target]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def build_main_windows(frame: pd.DataFrame) -> List[MainWindow]:
    dates = frame["Date"]
    earliest_date = dates.iloc[0]
    latest_date = dates.iloc[-1]

    anchor_date = first_date_on_or_after(
        dates, earliest_date + pd.DateOffset(years=TRAIN_YEARS + VALIDATION_YEARS)
    )
    if anchor_date is None:
        return []

    windows: List[MainWindow] = []
    while anchor_date is not None and anchor_date <= latest_date:
        next_anchor_target = anchor_date + pd.DateOffset(months=STEP_MONTHS)
        next_anchor_date = first_date_on_or_after(dates, next_anchor_target)

        if next_anchor_date is None:
            oos_mask = dates >= anchor_date
        else:
            oos_mask = (dates >= anchor_date) & (dates < next_anchor_date)

        if oos_mask.sum() == 0:
            anchor_date = next_anchor_date
            continue

        oos_dates = dates[oos_mask]
        windows.append(
            MainWindow(
                rebalance_date=anchor_date.strftime("%Y-%m-%d"),
                oos_start=oos_dates.iloc[0].strftime("%Y-%m-%d"),
                oos_end=oos_dates.iloc[-1].strftime("%Y-%m-%d"),
            )
        )
        anchor_date = next_anchor_date

    return windows


def standardize_features(
    train_frame: pd.DataFrame, frame_to_transform: pd.DataFrame
) -> pd.DataFrame:
    means = train_frame[FEATURE_COLUMNS].mean()
    stds = train_frame[FEATURE_COLUMNS].std(ddof=0).replace(0.0, 1.0)
    standardized = (frame_to_transform[FEATURE_COLUMNS] - means) / stds
    return standardized.replace([np.inf, -np.inf], np.nan)


def determine_bull_label(train_excess_ret: pd.Series, train_labels: pd.Series) -> int:
    cumulative_excess = train_excess_ret.groupby(train_labels).sum()
    return int(cumulative_excess.idxmax())


def labels_to_signal(labels: pd.Series, bull_label: int) -> pd.Series:
    return (labels == bull_label).astype(int)


def annualized_sharpe(excess_returns: np.ndarray) -> float:
    clean = excess_returns[~np.isnan(excess_returns)]
    if len(clean) < 2:
        return np.nan

    mean = clean.mean()
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        if mean > 0:
            return np.inf
        if mean < 0:
            return -np.inf
        return 0.0

    return float(np.sqrt(TRADING_DAYS) * mean / std)


def fit_oracle_segment_for_penalty(frame: pd.DataFrame, window: MainWindow, penalty: float) -> Dict[str, object]:
    oos_start = pd.Timestamp(window.oos_start)
    oos_end = pd.Timestamp(window.oos_end)
    train_start = oos_start - pd.DateOffset(years=TRAIN_YEARS)

    train_mask = (frame["Date"] >= train_start) & (frame["Date"] < oos_start)
    segment_mask = (frame["Date"] >= oos_start) & (frame["Date"] <= oos_end)

    train_frame = frame.loc[train_mask].copy().reset_index(drop=True)
    segment_frame = frame.loc[segment_mask].copy().reset_index(drop=True)
    combined_frame = frame.loc[train_mask | segment_mask].copy().reset_index(drop=True)

    if train_frame.empty or segment_frame.empty:
        raise ValueError(f"{window.rebalance_date} penalty {penalty}: empty train or segment frame")

    combined_features = standardize_features(train_frame, combined_frame)
    if combined_features.isna().any().any():
        raise ValueError(f"{window.rebalance_date} penalty {penalty}: standardized features contain NaN")

    train_features = combined_features.iloc[: len(train_frame)]
    model = JumpModel(n_components=2, jump_penalty=penalty, random_state=0)
    model.fit(train_features, ret_ser=train_frame["excess_ret"], sort_by="cumret")

    all_labels = pd.Series(model.predict_online(combined_features), index=combined_frame.index)
    train_labels = all_labels.iloc[: len(train_frame)]
    bull_label = determine_bull_label(train_frame["excess_ret"], train_labels)
    segment_position = labels_to_signal(all_labels.iloc[len(train_frame) :], bull_label).reset_index(drop=True)

    return {
        "rebalance_date": window.rebalance_date,
        "oos_start": window.oos_start,
        "oos_end": window.oos_end,
        "penalty": penalty,
        "date": segment_frame["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "ret": segment_frame["ret"].astype(float).tolist(),
        "rf_daily": segment_frame["rf_daily"].astype(float).tolist(),
        "position": segment_position.astype(int).tolist(),
    }


def worker_process_window(window_payload: Dict[str, str]) -> Dict[str, object]:
    frame = load_feature_frame()
    window = MainWindow(**window_payload)
    segments: List[Dict[str, object]] = []
    for penalty in PENALTY_GRID:
        segments.append(fit_oracle_segment_for_penalty(frame, window, penalty))
    return {"rebalance_date": window.rebalance_date, "segments": segments}


def stitch_penalty_segments(segment_payloads: List[Dict[str, object]], penalty: float) -> Dict[str, object]:
    fee_rate = FEE_BPS / 10000.0
    penalty_segments = sorted(
        [payload for payload in segment_payloads if float(payload["penalty"]) == penalty],
        key=lambda item: item["rebalance_date"],
    )

    previous_final_position = 0
    frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, object]] = []

    for payload in penalty_segments:
        position = np.asarray(payload["position"], dtype=np.int8)
        ret = np.asarray(payload["ret"], dtype=float)
        rf = np.asarray(payload["rf_daily"], dtype=float)

        previous_position = np.empty_like(position, dtype=np.int8)
        previous_position[0] = int(previous_final_position)
        if len(position) > 1:
            previous_position[1:] = position[:-1]

        fee = np.where(position != previous_position, fee_rate, 0.0)
        strategy_ret_gross = position * ret + (1 - position) * rf
        strategy_ret = strategy_ret_gross - fee
        strategy_excess_ret = strategy_ret - rf

        segment_frame = pd.DataFrame(
            {
                "Date": payload["date"],
                "ret": ret,
                "rf_daily": rf,
                "position": position,
                "fee": fee,
                "strategy_ret_gross": strategy_ret_gross,
                "strategy_ret": strategy_ret,
                "strategy_excess_ret": strategy_excess_ret,
                "penalty": penalty,
            }
        )
        frames.append(segment_frame)

        diagnostics.append(
            {
                "main_rebalance_date": payload["rebalance_date"],
                "penalty": penalty,
                "subwindow_start": payload["oos_start"],
                "subwindow_end": payload["oos_end"],
                "initial_position": int(previous_final_position),
                "final_position": int(position[-1]),
                "switch_count": int(np.count_nonzero(fee > 0)),
                "sharpe": annualized_sharpe(strategy_excess_ret),
            }
        )
        previous_final_position = int(position[-1])

    full_frame = pd.concat(frames, ignore_index=True)
    full_frame["Date"] = pd.to_datetime(full_frame["Date"])
    full_frame = full_frame.sort_values("Date", ascending=True).reset_index(drop=True)
    return {
        "curve": full_frame,
        "diagnostics": pd.DataFrame(diagnostics),
    }


def attach_equity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["equity"] = (1.0 + result["strategy_ret"]).cumprod()
    return result


def compute_performance_metrics(frame: pd.DataFrame, penalty: float) -> Dict[str, object]:
    curve = attach_equity(frame)
    equity = curve["equity"].to_numpy(dtype=float, copy=False)
    strategy_ret = curve["strategy_ret"].to_numpy(dtype=float, copy=False)
    strategy_excess_ret = curve["strategy_excess_ret"].to_numpy(dtype=float, copy=False)
    position = curve["position"].to_numpy(dtype=float, copy=False)
    fee = curve["fee"].to_numpy(dtype=float, copy=False)

    total_return = float(equity[-1] - 1.0)
    final_wealth = float(equity[-1])
    years = len(curve) / TRADING_DAYS
    annual_return = float(final_wealth ** (1.0 / years) - 1.0) if years > 0 else np.nan
    annual_volatility = float(np.std(strategy_ret, ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = annualized_sharpe(strategy_excess_ret)
    running_max = np.maximum.accumulate(equity)
    max_drawdown = float(np.min(equity / running_max - 1.0))
    total_switch_count = int(np.count_nonzero(fee > 0))
    avg_switch_count_per_year = float(total_switch_count / years) if years > 0 else np.nan
    avg_position = float(np.mean(position))

    return {
        "penalty": penalty,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "final_wealth": final_wealth,
        "total_switch_count": total_switch_count,
        "avg_switch_count_per_year": avg_switch_count_per_year,
        "avg_position": avg_position,
    }


def build_buy_and_hold(frame: pd.DataFrame, oos_start: str) -> pd.DataFrame:
    oos_start_ts = pd.Timestamp(oos_start)
    buyhold = frame.loc[frame["Date"] >= oos_start_ts, ["Date", "ret", "rf_daily", "excess_ret"]].copy()
    buyhold["position"] = 1
    buyhold["fee"] = 0.0
    buyhold["strategy_ret_gross"] = buyhold["ret"]
    buyhold["strategy_ret"] = buyhold["ret"]
    buyhold["strategy_excess_ret"] = buyhold["ret"] - buyhold["rf_daily"]
    return buyhold.reset_index(drop=True)


def build_daily_equity_table(
    fixed_curves: Dict[float, pd.DataFrame], buyhold_curve: pd.DataFrame
) -> pd.DataFrame:
    daily = attach_equity(buyhold_curve)[["Date", "equity"]].rename(
        columns={"equity": "equity_buy_and_hold"}
    )
    for penalty, frame in fixed_curves.items():
        daily = daily.merge(
            attach_equity(frame)[["Date", "equity"]].rename(
                columns={"equity": f"equity_penalty_{penalty_label(penalty)}"}
            ),
            on="Date",
            how="left",
            sort=True,
        )
    return daily.sort_values("Date").reset_index(drop=True)


def plot_all_fixed_penalties(fixed_curves: Dict[float, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for penalty, frame in fixed_curves.items():
        curve = attach_equity(frame)
        ax.plot(curve["Date"], curve["equity"], label=f"{penalty:.1f}", linewidth=1.4)
    ax.set_title("IVV Oracle OOS Equity Curves by Fixed Penalty")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend(title="Penalty", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_vs_buyhold(
    best_curve: pd.DataFrame, best_penalty: float, buyhold_curve: pd.DataFrame, output_path: Path
) -> None:
    best = attach_equity(best_curve)
    buyhold = attach_equity(buyhold_curve)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(best["Date"], best["equity"], label=f"penalty_{penalty_label(best_penalty)}", linewidth=2)
    ax.plot(buyhold["Date"], buyhold["equity"], label=BUY_HOLD_NAME, linewidth=2)
    ax.set_title("SPY Oracle OOS: Best Fixed Penalty vs Buy and Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_outputs(
    performance_summary: pd.DataFrame,
    daily_equity: pd.DataFrame,
    continuity: pd.DataFrame,
    fixed_curves: Dict[float, pd.DataFrame],
    best_penalty: float,
    buyhold_curve: pd.DataFrame,
    output_dir: Path,
) -> None:
    performance_summary.to_csv(output_dir / "fixed_penalty_performance_summary.csv", index=False)
    daily_equity.to_csv(output_dir / "fixed_penalty_daily_equity_curves.csv", index=False)
    continuity.to_csv(output_dir / "oos_segment_continuity_diagnostics.csv", index=False)
    plot_all_fixed_penalties(
        fixed_curves, output_dir / "oracle_oos_equity_curves_by_fixed_penalty.png"
    )
    plot_best_vs_buyhold(
        fixed_curves[best_penalty],
        best_penalty,
        buyhold_curve,
        output_dir / "oracle_oos_best_penalty_vs_buyhold.png",
    )


def main() -> None:
    start_time = time.perf_counter()
    frame = load_feature_frame()
    main_windows = build_main_windows(frame)
    if not main_windows:
        raise ValueError("No main windows could be constructed from spy_jm_features.csv")

    worker_count = min(max(os.cpu_count() or 1, 1), len(main_windows))
    window_payloads = [window.__dict__ for window in main_windows]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        window_results = list(executor.map(worker_process_window, window_payloads))

    per_penalty_segments: Dict[float, List[Dict[str, object]]] = {penalty: [] for penalty in PENALTY_GRID}
    for window_result in window_results:
        for segment_payload in window_result["segments"]:
            per_penalty_segments[float(segment_payload["penalty"])].append(segment_payload)

    fixed_curves: Dict[float, pd.DataFrame] = {}
    diagnostics_frames: List[pd.DataFrame] = []
    performance_rows: List[Dict[str, object]] = []
    for penalty in PENALTY_GRID:
        stitched = stitch_penalty_segments(per_penalty_segments[penalty], penalty)
        fixed_curves[penalty] = stitched["curve"]
        diagnostics_frames.append(stitched["diagnostics"])
        performance_rows.append(compute_performance_metrics(stitched["curve"], penalty))

    performance_summary = pd.DataFrame(performance_rows).sort_values("penalty").reset_index(drop=True)
    continuity = pd.concat(diagnostics_frames, ignore_index=True).sort_values(
        ["penalty", "main_rebalance_date"]
    ).reset_index(drop=True)

    buyhold_curve = build_buy_and_hold(frame, main_windows[0].oos_start)
    daily_equity = build_daily_equity_table(fixed_curves, buyhold_curve)
    best_penalty = float(performance_summary.sort_values(["sharpe", "penalty"], ascending=[False, True]).iloc[0]["penalty"])

    output_dir = results_dir()
    save_outputs(
        performance_summary=performance_summary,
        daily_equity=daily_equity,
        continuity=continuity,
        fixed_curves=fixed_curves,
        best_penalty=best_penalty,
        buyhold_curve=buyhold_curve,
        output_dir=output_dir,
    )

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Input file: {input_path()}")
    print(f"Main windows: {len(main_windows)}")
    print(f"Workers used: {worker_count}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Best fixed penalty by oracle OOS Sharpe: {best_penalty:.1f}")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
