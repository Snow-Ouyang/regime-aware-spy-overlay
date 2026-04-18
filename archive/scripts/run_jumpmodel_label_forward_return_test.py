from pathlib import Path

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from scipy import stats

from single_asset_gspc_spy_common import (
    BASE_JM_FEATURES,
    JUMP_PENALTY,
    build_target_windows,
    load_research_frame,
    load_trade_base,
    results_root,
)
from spy_regime_common import standardize_jm_features


RESULTS_SUBDIR = "jumpmodel_current_label_forward_return_test"
HORIZONS = [1, 5, 10, 20]


def output_dir() -> Path:
    out = results_root() / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def cumulative_simple_return(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0 or np.isnan(arr).any():
        return np.nan
    return float(np.prod(1.0 + arr) - 1.0)


def build_forward_returns(trade_base: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    trade_base:
      Date=t
      next_ret = realized return from t to t+1
      next_rf_daily = rf from t to t+1

    forward_ret_hd at Date=t means cumulative simple return over:
      next_ret[t], next_ret[t+1], ..., next_ret[t+h-1]
    """
    out = trade_base[["Date", "next_ret", "next_rf_daily"]].copy().sort_values("Date").reset_index(drop=True)

    for h in horizons:
        out[f"forward_ret_{h}d"] = (
            out["next_ret"]
            .rolling(window=h, min_periods=h)
            .apply(cumulative_simple_return, raw=False)
            .shift(-(h - 1))
        )
        out[f"forward_rf_{h}d"] = (
            out["next_rf_daily"]
            .rolling(window=h, min_periods=h)
            .apply(cumulative_simple_return, raw=False)
            .shift(-(h - 1))
        )
        out[f"forward_excess_ret_{h}d"] = (
            (1.0 + out[f"forward_ret_{h}d"]) / (1.0 + out[f"forward_rf_{h}d"]) - 1.0
        )

    return out


def build_jumpmodel_current_label_panel() -> pd.DataFrame:
    """
    Build OOS panel using only current-day observable Jump Model labels.

    For each window:
      1) fit JM on train only
      2) predict_online on train + oos combined standardized features
      3) map raw states to bull/bear using TRAIN cumulative excess return
      4) keep OOS current-day label bull_label_t at Date=t

    No future-info leakage in the label itself.
    """
    frame = load_research_frame()
    windows = build_target_windows(frame)

    rows = []

    for window in windows:
        train_start = pd.Timestamp(window.train_start)
        train_end = pd.Timestamp(window.train_end)
        oos_start = pd.Timestamp(window.oos_start)
        oos_end = pd.Timestamp(window.oos_end)

        train_mask = (frame["Date"] >= train_start) & (frame["Date"] <= train_end)
        oos_mask = (frame["Date"] >= oos_start) & (frame["Date"] <= oos_end)

        train_df = frame.loc[train_mask].copy().reset_index(drop=True)
        oos_df = frame.loc[oos_mask].copy().reset_index(drop=True)

        if train_df.empty or oos_df.empty:
            continue

        combined = pd.concat([train_df, oos_df], ignore_index=True)
        standardized = standardize_jm_features(train_df, combined)

        if standardized.isna().any().any():
            bad_cols = standardized.columns[standardized.isna().any()].tolist()
            raise ValueError(
                f"NaN in standardized JM features for rebalance_date={window.rebalance_date}: {bad_cols}"
            )

        jm = JumpModel(n_components=2, jump_penalty=JUMP_PENALTY, random_state=0)
        jm.fit(
            standardized.iloc[: len(train_df)],
            ret_ser=train_df["excess_ret"],
            sort_by="cumret",
        )

        combined_raw_states = pd.Series(jm.predict_online(standardized), index=combined.index)
        train_raw_states = combined_raw_states.iloc[: len(train_df)].reset_index(drop=True)
        oos_raw_states = combined_raw_states.iloc[len(train_df):].reset_index(drop=True)

        # Map bull state using TRAIN information only
        bull_state_train = int(train_df["excess_ret"].groupby(train_raw_states).sum().idxmax())

        panel = pd.DataFrame(
            {
                "Date": pd.to_datetime(oos_df["Date"]).reset_index(drop=True),
                "rebalance_date": window.rebalance_date,
                "raw_state_t": oos_raw_states.to_numpy(dtype=int, copy=False),
            }
        )
        panel["bull_label_t"] = (panel["raw_state_t"] == bull_state_train).astype(int)
        panel["bear_label_t"] = 1 - panel["bull_label_t"]

        rows.append(panel)

    out = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["Date", "rebalance_date"])
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return out


def summarize_group(x: pd.Series) -> dict:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "se": np.nan}
    std = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    se = float(std / np.sqrt(n)) if n > 1 and np.isfinite(std) else np.nan
    return {
        "n": int(n),
        "mean": float(np.mean(arr)),
        "std": std,
        "se": se,
    }


def run_ttest(bull: pd.Series, bear: pd.Series) -> dict:
    bull_arr = pd.to_numeric(bull, errors="coerce").dropna().to_numpy(dtype=float)
    bear_arr = pd.to_numeric(bear, errors="coerce").dropna().to_numpy(dtype=float)

    if len(bull_arr) < 2 or len(bear_arr) < 2:
        return {"t_stat": np.nan, "p_value": np.nan}

    # Welch t-test
    t_stat, p_value = stats.ttest_ind(bull_arr, bear_arr, equal_var=False, nan_policy="omit")
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def build_summary_table(panel: pd.DataFrame, horizons: list[int], return_prefix: str) -> pd.DataFrame:
    rows = []

    for h in horizons:
        col = f"{return_prefix}_{h}d"

        bull = panel.loc[panel["bull_label_t"] == 1, col]
        bear = panel.loc[panel["bull_label_t"] == 0, col]

        bull_stats = summarize_group(bull)
        bear_stats = summarize_group(bear)
        test_stats = run_ttest(bull, bear)

        spread = np.nan
        if pd.notna(bull_stats["mean"]) and pd.notna(bear_stats["mean"]):
            spread = bull_stats["mean"] - bear_stats["mean"]

        rows.append(
            {
                "horizon": f"{h}d",
                "return_type": return_prefix,
                "bull_n": bull_stats["n"],
                "bull_mean": bull_stats["mean"],
                "bull_se": bull_stats["se"],
                "bear_n": bear_stats["n"],
                "bear_mean": bear_stats["mean"],
                "bear_se": bear_stats["se"],
                "bull_minus_bear": spread,
                "t_stat": test_stats["t_stat"],
                "p_value": test_stats["p_value"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = output_dir()

    label_panel = build_jumpmodel_current_label_panel()
    forward_panel = build_forward_returns(load_trade_base(), HORIZONS)

    panel = label_panel.merge(forward_panel, on="Date", how="inner")
    panel = panel.sort_values("Date").reset_index(drop=True)

    panel.to_csv(out_dir / "jumpmodel_current_label_forward_return_panel.csv", index=False)

    raw_summary = build_summary_table(panel, HORIZONS, "forward_ret")
    excess_summary = build_summary_table(panel, HORIZONS, "forward_excess_ret")
    summary = pd.concat([raw_summary, excess_summary], ignore_index=True)

    raw_summary.to_csv(out_dir / "jumpmodel_current_label_forward_raw_return_summary.csv", index=False)
    excess_summary.to_csv(out_dir / "jumpmodel_current_label_forward_excess_return_summary.csv", index=False)
    summary.to_csv(out_dir / "jumpmodel_current_label_forward_return_ttest_summary.csv", index=False)

    print(f"Results directory: {out_dir}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nInterpretation guide:")
    print("- This version uses current-day observable JM label bull_label_t only.")
    print("- bull_mean > bear_mean and p_value < 0.05 means the current JM state has predictive separation for future returns.")
    print("- If significance disappears versus the previous script, that confirms the earlier result relied on future information.")


if __name__ == "__main__":
    main()