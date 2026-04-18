import site
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_paths() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
    preferred_vendor = Path(__file__).resolve().parents[1] / ".vendor_local"
    if preferred_vendor.exists():
        preferred_vendor_str = str(preferred_vendor)
        if preferred_vendor_str not in sys.path:
            sys.path.insert(0, preferred_vendor_str)
        return
    fallback_vendor = Path(__file__).resolve().parents[1] / ".vendor"
    if fallback_vendor.exists():
        fallback_vendor_str = str(fallback_vendor)
        if fallback_vendor_str not in sys.path:
            sys.path.insert(0, fallback_vendor_str)


configure_paths()

from jumpmodels.jump import JumpModel

from single_asset_gspc_spy_common import (
    BASE_JM_FEATURES,
    build_target_windows,
    load_research_frame,
    results_root,
)
from spy_regime_common import standardize_jm_features


RESULTS_SUBDIR = "state_semantics_penalty_grid"
TRADING_DAYS = 252
TARGET_RETURN = 0.0
FIG_DPI = 160
PENALTY_GRID = [0, 5, 10, 15, 20]


def output_dir() -> Path:
    out = results_root() / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def downside_deviation(returns: pd.Series | np.ndarray, target: float = TARGET_RETURN) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    downside = np.minimum(arr - target, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def annualized_sortino(returns: pd.Series | np.ndarray, target: float = TARGET_RETURN) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return np.nan
    dd = downside_deviation(arr, target=target)
    if dd == 0 or np.isnan(dd):
        mean_ret = float(np.mean(arr))
        if mean_ret > 0:
            return np.inf
        if mean_ret < 0:
            return -np.inf
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * np.mean(arr) / dd)


def annualized_state_cagr(returns: pd.Series | np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    gross = np.prod(1.0 + arr)
    if gross <= 0:
        return np.nan
    return float(gross ** (TRADING_DAYS / arr.size) - 1.0)


def fit_and_label_one_window(frame: pd.DataFrame, window, jump_penalty: float) -> tuple[pd.DataFrame, dict]:
    train_start = pd.Timestamp(window.train_start)
    train_end = pd.Timestamp(window.train_end)
    val_start = pd.Timestamp(window.val_start)
    val_end = pd.Timestamp(window.val_end)
    oos_start = pd.Timestamp(window.oos_start)
    oos_end = pd.Timestamp(window.oos_end)

    train_df = frame.loc[(frame["Date"] >= train_start) & (frame["Date"] <= train_end)].copy().reset_index(drop=True)
    val_df = frame.loc[(frame["Date"] >= val_start) & (frame["Date"] <= val_end)].copy().reset_index(drop=True)
    oos_df = frame.loc[(frame["Date"] >= oos_start) & (frame["Date"] <= oos_end)].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or oos_df.empty:
        raise ValueError(
            f"Empty split detected for window {window.rebalance_date}: "
            f"train={len(train_df)}, validation={len(val_df)}, oos={len(oos_df)}"
        )

    combined = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="validation"),
            oos_df.assign(split="oos"),
        ],
        ignore_index=True,
    )

    standardized = standardize_jm_features(train_df, combined)
    if standardized.isna().any().any():
        bad_cols = standardized.columns[standardized.isna().any()].tolist()
        raise ValueError(f"NaN in standardized JM features for {window.rebalance_date}: {bad_cols}")

    jm = JumpModel(n_components=2, jump_penalty=jump_penalty, random_state=0)
    jm.fit(
        standardized.iloc[: len(train_df)],
        ret_ser=train_df["excess_ret"],
        sort_by="cumret",
    )

    combined["raw_state"] = pd.Series(jm.predict_online(standardized), index=combined.index).to_numpy(dtype=int, copy=False)
    train_labels = combined.loc[combined["split"] == "train", "raw_state"].reset_index(drop=True)
    bull_state = int(train_df["excess_ret"].groupby(train_labels).sum().idxmax())
    bear_state = 1 - bull_state

    mapping_row = {
        "jump_penalty": float(jump_penalty),
        "rebalance_date": window.rebalance_date,
        "bull_state_by_train_cum_excess": bull_state,
        "bear_state_by_train_cum_excess": bear_state,
    }
    return combined, mapping_row


def summarize_split_metrics(df: pd.DataFrame, jump_penalty: float, rebalance_date: str, split_name: str, bull_state: int) -> dict:
    split_df = df.loc[df["split"] == split_name].copy()
    bear_state = 1 - bull_state
    bull_ret = split_df.loc[split_df["raw_state"] == bull_state, "ret"]
    bear_ret = split_df.loc[split_df["raw_state"] == bear_state, "ret"]

    bull_cagr = annualized_state_cagr(bull_ret)
    bear_cagr = annualized_state_cagr(bear_ret)
    bull_sortino = annualized_sortino(bull_ret)
    bear_sortino = annualized_sortino(bear_ret)

    return {
        "jump_penalty": float(jump_penalty),
        "rebalance_date": rebalance_date,
        "split": split_name,
        "bull_state": int(bull_state),
        "bear_state": int(bear_state),
        "bull_days": int((split_df["raw_state"] == bull_state).sum()),
        "bear_days": int((split_df["raw_state"] == bear_state).sum()),
        "bull_cagr": bull_cagr,
        "bear_cagr": bear_cagr,
        "cagr_spread_bull_minus_bear": bull_cagr - bear_cagr if pd.notna(bull_cagr) and pd.notna(bear_cagr) else np.nan,
        "bull_sortino": bull_sortino,
        "bear_sortino": bear_sortino,
        "sortino_spread_bull_minus_bear": bull_sortino - bear_sortino if pd.notna(bull_sortino) and pd.notna(bear_sortino) else np.nan,
    }


def plot_penalty_summary(summary_df: pd.DataFrame, metric_col: str, title: str, out_path: Path) -> None:
    split_order = ["train", "validation", "oos"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for ax, split_name in zip(axes, split_order):
        sub = summary_df.loc[summary_df["split"] == split_name].sort_values("jump_penalty")
        ax.plot(sub["jump_penalty"], sub[metric_col], marker="o", linewidth=1.8)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(f"{split_name.capitalize()} - {metric_col}")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Jump penalty")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    out_dir = output_dir()
    frame = load_research_frame()
    windows = build_target_windows(frame)

    mapping_rows: list[dict] = []
    split_rows: list[dict] = []

    for jump_penalty in PENALTY_GRID:
        for window in windows:
            combined, mapping_row = fit_and_label_one_window(frame[["Date", "ret", "excess_ret", *BASE_JM_FEATURES]].copy(), window, jump_penalty)
            mapping_rows.append(mapping_row)
            bull_state = int(mapping_row["bull_state_by_train_cum_excess"])
            for split_name in ["train", "validation", "oos"]:
                split_rows.append(
                    summarize_split_metrics(
                        combined,
                        jump_penalty=jump_penalty,
                        rebalance_date=window.rebalance_date,
                        split_name=split_name,
                        bull_state=bull_state,
                    )
                )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(["jump_penalty", "rebalance_date"]).reset_index(drop=True)
    split_df = pd.DataFrame(split_rows).sort_values(["jump_penalty", "rebalance_date", "split"]).reset_index(drop=True)
    split_df["bull_sortino_finite"] = split_df["bull_sortino"].replace([np.inf, -np.inf], np.nan)
    split_df["bear_sortino_finite"] = split_df["bear_sortino"].replace([np.inf, -np.inf], np.nan)
    split_df["sortino_spread_bull_minus_bear_finite"] = split_df["sortino_spread_bull_minus_bear"].replace([np.inf, -np.inf], np.nan)

    summary_df = (
        split_df.groupby(["jump_penalty", "split"], as_index=False)
        .agg(
            mean_bull_cagr=("bull_cagr", "mean"),
            mean_bear_cagr=("bear_cagr", "mean"),
            mean_cagr_spread_bull_minus_bear=("cagr_spread_bull_minus_bear", "mean"),
            median_cagr_spread_bull_minus_bear=("cagr_spread_bull_minus_bear", "median"),
            mean_bull_sortino=("bull_sortino_finite", "mean"),
            mean_bear_sortino=("bear_sortino_finite", "mean"),
            mean_sortino_spread_bull_minus_bear=("sortino_spread_bull_minus_bear_finite", "mean"),
            median_sortino_spread_bull_minus_bear=("sortino_spread_bull_minus_bear", "median"),
            mean_bull_days=("bull_days", "mean"),
            mean_bear_days=("bear_days", "mean"),
            n_windows=("rebalance_date", "count"),
        )
    )

    compact_summary = summary_df.pivot(index="jump_penalty", columns="split", values=["mean_cagr_spread_bull_minus_bear", "mean_sortino_spread_bull_minus_bear"])
    compact_summary.columns = [f"{metric}_{split}" for metric, split in compact_summary.columns]
    compact_summary = compact_summary.reset_index()

    mapping_df.to_csv(out_dir / "penalty_window_state_mapping.csv", index=False)
    split_df.to_csv(out_dir / "penalty_window_state_spreads.csv", index=False)
    summary_df.to_csv(out_dir / "penalty_split_summary.csv", index=False)
    compact_summary.to_csv(out_dir / "penalty_grid_headline_spreads.csv", index=False)

    plot_penalty_summary(
        summary_df,
        metric_col="mean_cagr_spread_bull_minus_bear",
        title="Bull minus bear state CAGR spread by jump penalty",
        out_path=out_dir / "jump_penalty_cagr_spread.png",
    )
    plot_penalty_summary(
        summary_df,
        metric_col="mean_sortino_spread_bull_minus_bear",
        title="Bull minus bear state Sortino spread by jump penalty",
        out_path=out_dir / "jump_penalty_sortino_spread.png",
    )

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
