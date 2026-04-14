import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


TRADING_DAYS = 252
TRAIN_YEARS = 11
STEP_MONTHS = 6
FEE_BPS = 5
JM_FEATURE_COLUMNS: List[str] = [
    "log_downside_dev_hl5",
    "log_downside_dev_hl21",
    "ewm_return_hl5",
    "ewm_return_hl10",
    "ewm_return_hl21",
    "sortino_hl5",
    "sortino_hl10",
    "sortino_hl21",
]


@dataclass
class MainWindow:
    rebalance_date: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    oos_start: str
    oos_end: str


def first_date_on_or_after(dates: pd.Series, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = dates[dates >= target]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def last_date_before(dates: pd.Series, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = dates[dates < target]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def build_main_windows(frame: pd.DataFrame, validation_years: int) -> List[MainWindow]:
    dates = frame["Date"].sort_values().reset_index(drop=True)
    earliest_date = dates.iloc[0]
    latest_date = dates.iloc[-1]

    anchor_date = first_date_on_or_after(
        dates, earliest_date + pd.DateOffset(years=TRAIN_YEARS + validation_years)
    )
    if anchor_date is None:
        return []

    windows: List[MainWindow] = []
    while anchor_date is not None and anchor_date <= latest_date:
        next_anchor_date = first_date_on_or_after(
            dates, anchor_date + pd.DateOffset(months=STEP_MONTHS)
        )
        oos_end = dates.iloc[-1] if next_anchor_date is None else last_date_before(dates, next_anchor_date)
        if oos_end is None or oos_end < anchor_date:
            anchor_date = next_anchor_date
            continue

        val_start = first_date_on_or_after(dates, anchor_date - pd.DateOffset(years=validation_years))
        train_start = first_date_on_or_after(dates, anchor_date - pd.DateOffset(years=validation_years + TRAIN_YEARS))
        train_end = last_date_before(dates, val_start) if val_start is not None else None
        val_end = last_date_before(dates, anchor_date)

        if None in [val_start, train_start, train_end, val_end]:
            anchor_date = next_anchor_date
            continue

        windows.append(
            MainWindow(
                rebalance_date=anchor_date.strftime("%Y-%m-%d"),
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
                oos_start=anchor_date.strftime("%Y-%m-%d"),
                oos_end=oos_end.strftime("%Y-%m-%d"),
            )
        )
        anchor_date = next_anchor_date

    return windows


def build_segments(start_date: pd.Timestamp, end_exclusive: pd.Timestamp) -> List[Dict[str, pd.Timestamp]]:
    segments: List[Dict[str, pd.Timestamp]] = []
    current_start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_exclusive)
    while current_start < final_end:
        next_start = min(current_start + pd.DateOffset(months=STEP_MONTHS), final_end)
        segments.append({"start": current_start, "end_exclusive": next_start})
        current_start = next_start
    return segments


def standardize_jm_features(train_frame: pd.DataFrame, frame_to_transform: pd.DataFrame) -> pd.DataFrame:
    means = train_frame[JM_FEATURE_COLUMNS].mean()
    stds = train_frame[JM_FEATURE_COLUMNS].std(ddof=0).replace(0.0, 1.0)
    standardized = (frame_to_transform[JM_FEATURE_COLUMNS] - means) / stds
    return standardized.replace([np.inf, -np.inf], np.nan)


def annualized_sharpe(excess_returns: np.ndarray) -> float:
    clean = np.asarray(excess_returns, dtype=float)
    clean = clean[~np.isnan(clean)]
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


def build_segment_supervised_bundle(
    frame: pd.DataFrame,
    feature_columns: List[str],
    segment_start: pd.Timestamp,
    segment_end_exclusive: pd.Timestamp,
    penalty: float,
) -> Dict[str, object]:
    dates = frame["Date"]
    train_start = pd.Timestamp(segment_start) - pd.DateOffset(years=TRAIN_YEARS)
    train_mask = (dates >= train_start) & (dates < segment_start)
    segment_mask = (dates >= segment_start) & (dates < segment_end_exclusive)

    train_frame = frame.loc[train_mask].copy().reset_index(drop=True)
    segment_frame = frame.loc[segment_mask].copy().reset_index(drop=True)
    if train_frame.empty or segment_frame.empty:
        raise ValueError(f"Empty train or segment window: {segment_start} -> {segment_end_exclusive}")

    combined_frame = pd.concat([train_frame, segment_frame], ignore_index=True)
    combined_jm = standardize_jm_features(train_frame, combined_frame)
    if combined_jm.isna().any().any():
        raise ValueError(f"JM standardization produced NaN for segment {segment_start} -> {segment_end_exclusive}")

    jm = JumpModel(n_components=2, jump_penalty=penalty, random_state=0)
    jm.fit(
        combined_jm.iloc[: len(train_frame)],
        ret_ser=train_frame["excess_ret"],
        sort_by="cumret",
    )

    combined_labels = pd.Series(jm.predict_online(combined_jm), index=combined_frame.index)
    train_labels = combined_labels.iloc[: len(train_frame)].reset_index(drop=True)
    bull_label = int(train_frame["excess_ret"].groupby(train_labels).sum().idxmax())

    train_frame["bull_label_t"] = (train_labels == bull_label).astype(int).to_numpy()
    segment_labels = combined_labels.iloc[len(train_frame) :].reset_index(drop=True)
    segment_frame["bull_label_t"] = (segment_labels == bull_label).astype(int).to_numpy()

    train_supervised = train_frame.copy()
    train_supervised["y_next"] = train_supervised["bull_label_t"].shift(-1)
    train_supervised = train_supervised.iloc[:-1].copy()
    train_supervised = train_supervised.dropna(subset=["y_next"])

    segment_supervised = segment_frame.copy()
    segment_supervised["y_next"] = segment_supervised["bull_label_t"].shift(-1)
    # signal_date is the decision date t. y_next is the realized label for t+1.
    # execution_date is the next-day date t+1, used by some mapped strategies.
    segment_supervised["signal_date"] = segment_supervised["Date"]
    segment_supervised["execution_date"] = segment_supervised["Date"].shift(-1)
    segment_supervised["next_ret"] = segment_supervised["ret"].shift(-1)
    segment_supervised["next_rf_daily"] = segment_supervised["rf_daily"].shift(-1)
    segment_supervised = segment_supervised.iloc[:-1].copy()
    segment_supervised = segment_supervised.dropna(subset=["y_next", "signal_date", "execution_date", "next_ret", "next_rf_daily"])

    if train_supervised.empty or segment_supervised.empty:
        raise ValueError(f"Supervised dataset empty for segment {segment_start} -> {segment_end_exclusive}")

    return {
        "train_dataset": {
            "X": train_supervised[feature_columns].to_numpy(dtype=float, copy=False),
            "y": train_supervised["y_next"].to_numpy(dtype=int, copy=False),
        },
        "segment_dataset": {
            "X": segment_supervised[feature_columns].to_numpy(dtype=float, copy=False),
            "y": segment_supervised["y_next"].to_numpy(dtype=int, copy=False),
            "base": segment_supervised[["signal_date", "execution_date", "next_ret", "next_rf_daily"]].reset_index(drop=True),
        },
    }


def simulate_next_day_strategy(
    dataset_base: pd.DataFrame,
    predicted_labels: np.ndarray,
    initial_position: int,
) -> Dict[str, object]:
    base = dataset_base.reset_index(drop=True).copy()
    position = np.asarray(predicted_labels, dtype=int)
    previous_position = np.empty_like(position)
    previous_position[0] = int(initial_position)
    if len(position) > 1:
        previous_position[1:] = position[:-1]

    fee_rate = FEE_BPS / 10000.0
    fee = np.where(position != previous_position, fee_rate, 0.0)
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
    return {
        "frame": frame,
        "final_position": int(position[-1]),
        "sharpe": annualized_sharpe(strategy_excess_ret),
    }


def stitch_oos_strategy(
    stitched_payloads: List[Dict[str, object]],
    label_key: str,
) -> Dict[str, object]:
    sorted_payloads = sorted(
        stitched_payloads, key=lambda item: item["window"]["rebalance_date"]
    )
    previous_final_position = 0
    frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, object]] = []

    for payload in sorted_payloads:
        dataset = payload["oos_dataset"]
        dataset_base = pd.DataFrame(
            {
                "execution_date": pd.to_datetime(dataset["execution_date"]),
                "next_ret": np.asarray(dataset["next_ret"], dtype=float),
                "next_rf_daily": np.asarray(dataset["next_rf_daily"], dtype=float),
            }
        )
        simulation = simulate_next_day_strategy(
            dataset_base,
            np.asarray(dataset[label_key], dtype=int),
            initial_position=previous_final_position,
        )
        previous_final_position = int(simulation["final_position"])
        frames.append(simulation["frame"])
        diagnostics.append(
            {
                "rebalance_date": payload["window"]["rebalance_date"],
                "initial_position": int(simulation["frame"]["position"].iloc[0] if len(simulation["frame"]) == 1 else simulation["frame"]["position"].shift(1).fillna(previous_final_position).iloc[0]),
                "final_position": previous_final_position,
                "oos_sharpe": simulation["sharpe"],
            }
        )

    full_frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return {"frame": full_frame, "diagnostics": pd.DataFrame(diagnostics)}


def build_strategy_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    curve = frame.copy()
    curve["equity"] = (1.0 + curve["strategy_ret"]).cumprod()
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


def build_buy_and_hold(frame: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    oos_start_ts = pd.Timestamp(oos_start)
    buyhold = frame.loc[frame["Date"] >= oos_start_ts, ["Date", "ret", "rf_daily"]].copy()
    buyhold["position"] = 1
    buyhold["fee"] = 0.0
    buyhold["strategy_ret_gross"] = buyhold["ret"]
    buyhold["strategy_ret"] = buyhold["ret"]
    buyhold["strategy_excess_ret"] = buyhold["ret"] - buyhold["rf_daily"]
    return buyhold.reset_index(drop=True)
