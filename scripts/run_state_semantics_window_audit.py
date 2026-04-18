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
    JUMP_PENALTY,
    build_target_windows,
    load_research_frame,
    results_root,
)
from spy_regime_common import standardize_jm_features

# ============================================================
# Config
# ============================================================

RESULTS_SUBDIR = "state_semantics_window_audit"
TRADING_DAYS = 252
TARGET_RETURN = 0.0  # downside / sortino target
FIG_DPI = 160


# ============================================================
# Helpers
# ============================================================

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
        mean_ret = np.mean(arr)
        if mean_ret > 0:
            return np.inf
        if mean_ret < 0:
            return -np.inf
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * np.mean(arr) / dd)


def annualized_volatility(returns: pd.Series | np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.std(arr, ddof=1) * np.sqrt(TRADING_DAYS))


def annualized_state_cagr(returns: pd.Series | np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    gross = np.prod(1.0 + arr)
    if gross <= 0:
        return np.nan
    return float(gross ** (TRADING_DAYS / arr.size) - 1.0)


def summarize_state(df: pd.DataFrame, rebalance_date: str, split_name: str, raw_state: int) -> dict:
    state_df = df.loc[df["raw_state"] == raw_state].copy()

    if state_df.empty:
        return {
            "rebalance_date": rebalance_date,
            "split": split_name,
            "raw_state": raw_state,
            "n_obs": 0,
            "annualized_cagr": np.nan,
            "cum_excess_return_sum": np.nan,
            "ann_volatility": np.nan,
            "downside_dev_daily": np.nan,
            "downside_dev_annualized": np.nan,
            "sortino": np.nan,
        }

    ret = state_df["ret"]
    excess_ret = state_df["excess_ret"]

    dd_daily = downside_deviation(ret)
    return {
        "rebalance_date": rebalance_date,
        "split": split_name,
        "raw_state": raw_state,
        "n_obs": int(len(state_df)),
        "annualized_cagr": annualized_state_cagr(ret),
        "cum_excess_return_sum": float(excess_ret.sum()),
        "ann_volatility": annualized_volatility(ret),
        "downside_dev_daily": dd_daily,
        "downside_dev_annualized": float(dd_daily * np.sqrt(TRADING_DAYS)) if pd.notna(dd_daily) else np.nan,
        "sortino": annualized_sortino(ret),
    }


def fit_and_label_one_window(frame: pd.DataFrame, window) -> tuple[pd.DataFrame, dict]:
    train_start = pd.Timestamp(window.train_start)
    train_end = pd.Timestamp(window.train_end)
    val_start = pd.Timestamp(window.val_start)
    val_end = pd.Timestamp(window.val_end)
    oos_start = pd.Timestamp(window.oos_start)
    oos_end = pd.Timestamp(window.oos_end)

    train_mask = (frame["Date"] >= train_start) & (frame["Date"] <= train_end)
    val_mask = (frame["Date"] >= val_start) & (frame["Date"] <= val_end)
    oos_mask = (frame["Date"] >= oos_start) & (frame["Date"] <= oos_end)

    train_df = frame.loc[train_mask].copy().reset_index(drop=True)
    val_df = frame.loc[val_mask].copy().reset_index(drop=True)
    oos_df = frame.loc[oos_mask].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or oos_df.empty:
        raise ValueError(
            f"Empty split detected for window {window.rebalance_date}: "
            f"train={len(train_df)}, val={len(val_df)}, oos={len(oos_df)}"
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

    jm = JumpModel(n_components=2, jump_penalty=JUMP_PENALTY, random_state=0)
    jm.fit(
        standardized.iloc[: len(train_df)],
        ret_ser=train_df["excess_ret"],
        sort_by="cumret",
    )

    raw_states = pd.Series(jm.predict_online(standardized), index=combined.index)
    combined["raw_state"] = raw_states.to_numpy(dtype=int, copy=False)

    train_labels = combined.loc[combined["split"] == "train", "raw_state"].reset_index(drop=True)
    bull_state_train = int(train_df["excess_ret"].groupby(train_labels).sum().idxmax())

    mapping_row = {
        "rebalance_date": window.rebalance_date,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "val_start": window.val_start,
        "val_end": window.val_end,
        "oos_start": window.oos_start,
        "oos_end": window.oos_end,
        "bull_state_by_train_cum_excess": bull_state_train,
    }

    for split_name in ["train", "validation", "oos"]:
        split_df = combined.loc[combined["split"] == split_name].copy()
        cagrs = split_df.groupby("raw_state")["ret"].apply(annualized_state_cagr)
        sortinos = split_df.groupby("raw_state")["ret"].apply(annualized_sortino)
        cum_excess = split_df.groupby("raw_state")["excess_ret"].sum()

        if len(cagrs) == 2:
            mapping_row[f"better_state_by_cagr_{split_name}"] = int(cagrs.idxmax())
            mapping_row[f"better_state_by_sortino_{split_name}"] = int(sortinos.idxmax())
            mapping_row[f"better_state_by_cum_excess_{split_name}"] = int(cum_excess.idxmax())
            mapping_row[f"state0_minus_state1_cagr_{split_name}"] = float(cagrs.get(0, np.nan) - cagrs.get(1, np.nan))
            mapping_row[f"state0_minus_state1_sortino_{split_name}"] = float(sortinos.get(0, np.nan) - sortinos.get(1, np.nan))
        else:
            mapping_row[f"better_state_by_cagr_{split_name}"] = np.nan
            mapping_row[f"better_state_by_sortino_{split_name}"] = np.nan
            mapping_row[f"better_state_by_cum_excess_{split_name}"] = np.nan
            mapping_row[f"state0_minus_state1_cagr_{split_name}"] = np.nan
            mapping_row[f"state0_minus_state1_sortino_{split_name}"] = np.nan

    return combined, mapping_row


def plot_metric_by_split(stats_df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    split_order = ["train", "validation", "oos"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, split_name in zip(axes, split_order):
        sub = stats_df.loc[stats_df["split"] == split_name].copy()
        pivot = sub.pivot(index="rebalance_date", columns="raw_state", values=metric).sort_index()

        x = pd.to_datetime(pivot.index)
        if 0 in pivot.columns:
            ax.plot(x, pivot[0], label="state 0", linewidth=1.7)
        if 1 in pivot.columns:
            ax.plot(x, pivot[1], label="state 1", linewidth=1.7)

        ax.set_title(f"{split_name.capitalize()} — {metric}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_state_gap(mapping_df: pd.DataFrame, metric_key: str, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    split_order = ["train", "validation", "oos"]

    for ax, split_name in zip(axes, split_order):
        col = f"state0_minus_state1_{metric_key}_{split_name}"
        x = pd.to_datetime(mapping_df["rebalance_date"])
        y = mapping_df[col].to_numpy(dtype=float)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.plot(x, y, linewidth=1.7)
        ax.set_title(f"{split_name.capitalize()} — state0 minus state1 ({metric_key})")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_bull_state_consistency(mapping_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = pd.to_datetime(mapping_df["rebalance_date"])
    y = mapping_df["bull_state_by_train_cum_excess"].to_numpy(dtype=float)
    ax.step(x, y, where="post", linewidth=1.8)
    ax.set_yticks([0, 1], ["state 0", "state 1"])
    ax.set_title("Bull-state assignment by training cumulative excess return")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    out_dir = output_dir()
    frame = load_research_frame()
    windows = build_target_windows(frame)

    all_stats: list[dict] = []
    mapping_rows: list[dict] = []

    for window in windows:
        combined, mapping_row = fit_and_label_one_window(frame, window)
        mapping_rows.append(mapping_row)

        for split_name in ["train", "validation", "oos"]:
            split_df = combined.loc[combined["split"] == split_name].copy()
            for raw_state in [0, 1]:
                all_stats.append(
                    summarize_state(
                        split_df,
                        rebalance_date=window.rebalance_date,
                        split_name=split_name,
                        raw_state=raw_state,
                    )
                )

    stats_df = pd.DataFrame(all_stats).sort_values(
        ["rebalance_date", "split", "raw_state"]
    ).reset_index(drop=True)
    mapping_df = pd.DataFrame(mapping_rows).sort_values("rebalance_date").reset_index(drop=True)

    # Main tables
    stats_df.to_csv(out_dir / "state_window_stats.csv", index=False)
    mapping_df.to_csv(out_dir / "state_window_mapping_summary.csv", index=False)

    # Compact pivots
    pivot_mean = (
        stats_df.pivot_table(
            index=["rebalance_date", "split"],
            columns="raw_state",
            values="annualized_cagr",
        )
        .reset_index()
        .rename(columns={0: "state0_annualized_cagr", 1: "state1_annualized_cagr"})
    )
    pivot_mean.to_csv(out_dir / "state_cagr_pivot.csv", index=False)

    pivot_sortino = (
        stats_df.pivot_table(
            index=["rebalance_date", "split"],
            columns="raw_state",
            values="sortino",
        )
        .reset_index()
        .rename(columns={0: "state0_sortino", 1: "state1_sortino"})
    )
    pivot_sortino.to_csv(out_dir / "state_sortino_pivot.csv", index=False)

    # Visuals
    plot_metric_by_split(
        stats_df,
        metric="annualized_cagr",
        title="State annualized CAGR by split",
        out_path=out_dir / "state_mean_return_by_split.png",
    )
    plot_metric_by_split(
        stats_df,
        metric="ann_volatility",
        title="State annualized volatility by split",
        out_path=out_dir / "state_annualized_volatility_by_split.png",
    )
    plot_metric_by_split(
        stats_df,
        metric="sortino",
        title="State Sortino ratio by split",
        out_path=out_dir / "state_sortino_by_split.png",
    )
    plot_metric_by_split(
        stats_df,
        metric="downside_dev_annualized",
        title="State annualized downside deviation by split",
        out_path=out_dir / "state_downside_dev_by_split.png",
    )

    plot_state_gap(
        mapping_df,
        metric_key="cagr",
        title="State 0 minus state 1 CAGR gap by split",
        out_path=out_dir / "state_gap_mean_return.png",
    )
    plot_state_gap(
        mapping_df,
        metric_key="sortino",
        title="State 0 minus state 1 Sortino gap by split",
        out_path=out_dir / "state_gap_sortino.png",
    )
    plot_bull_state_consistency(
        mapping_df,
        out_path=out_dir / "bull_state_assignment_consistency.png",
    )

    # Small headline summary
    summary_rows = []
    for split_name in ["train", "validation", "oos"]:
        col = f"better_state_by_cagr_{split_name}"
        share_state1 = float((mapping_df[col] == 1).mean())
        share_state0 = float((mapping_df[col] == 0).mean())
        summary_rows.append(
            {
                "split": split_name,
                "share_windows_state0_has_higher_cagr": share_state0,
                "share_windows_state1_has_higher_cagr": share_state1,
                "share_windows_train_bull_state_equals_state1": float(
                    (mapping_df["bull_state_by_train_cum_excess"] == 1).mean()
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "state_semantics_headline_summary.csv", index=False)

    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
