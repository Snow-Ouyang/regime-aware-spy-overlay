from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from single_asset_gspc_spy_common import results_root


INPUT_SUBDIR = "probability_nextday_return_regression"
OUTPUT_SUBDIR = "probability_multihorizon_return_regression"
INPUT_FILE = "probability_nextday_return_panel.csv"

HORIZONS = [1, 5, 10, 20]


def input_path() -> Path:
    return results_root() / INPUT_SUBDIR / INPUT_FILE


def output_dir() -> Path:
    out = results_root() / OUTPUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def cumulative_simple_return(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0 or np.isnan(arr).any():
        return np.nan
    return float(np.prod(1.0 + arr) - 1.0)


def load_panel() -> pd.DataFrame:
    path = input_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input panel: {path}\n"
            f"Please run the probability-nextday panel script first."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    required_cols = [
        "Date",
        "predicted_probability_smoothed",
        "predicted_probability_raw",
        "next_ret",
        "next_rf_daily",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input panel missing required columns: {missing}")

    return df


def build_multihorizon_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()

    # next_ret at Date=t already means realized return from t to t+1
    # So for horizon h, use next_ret[t], next_ret[t+1], ..., next_ret[t+h-1]
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


def run_ols(df: pd.DataFrame, y_col: str, x_col: str, cov_type: str = "HC3"):
    reg_df = df[[y_col, x_col]].dropna().copy()
    X = sm.add_constant(reg_df[x_col])
    y = reg_df[y_col]
    model = sm.OLS(y, X)
    result = model.fit(cov_type=cov_type)
    return result, reg_df


def coefficient_table(results) -> pd.DataFrame:
    conf = results.conf_int()
    return pd.DataFrame(
        {
            "term": results.params.index,
            "coef": results.params.values,
            "std_err": results.bse.values,
            "t_or_z": results.tvalues.values,
            "p_value": results.pvalues.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
        }
    )


def save_summary_text(results, path: Path) -> None:
    path.write_text(results.summary().as_text(), encoding="utf-8")


def main() -> None:
    out_dir = output_dir()

    panel = load_panel()
    panel = build_multihorizon_returns(panel, HORIZONS)
    panel.to_csv(out_dir / "probability_multihorizon_panel.csv", index=False)

    headline_rows = []

    for h in HORIZONS:
        # raw return regression
        y_ret = f"forward_ret_{h}d"
        ret_results, ret_reg_df = run_ols(
            panel,
            y_col=y_ret,
            x_col="predicted_probability_smoothed",
            cov_type="HC3",
        )
        save_summary_text(
            ret_results,
            out_dir / f"ols_{y_ret}_on_smoothed_prob_summary.txt",
        )
        coefficient_table(ret_results).to_csv(
            out_dir / f"ols_{y_ret}_on_smoothed_prob_coefficients.csv",
            index=False,
        )

        headline_rows.append(
            {
                "regression": f"{y_ret} ~ predicted_probability_smoothed",
                "n_obs": int(ret_results.nobs),
                "beta": float(ret_results.params["predicted_probability_smoothed"]),
                "p_value": float(ret_results.pvalues["predicted_probability_smoothed"]),
                "t_or_z": float(ret_results.tvalues["predicted_probability_smoothed"]),
                "r_squared": float(ret_results.rsquared),
            }
        )

        # excess return regression
        y_ex = f"forward_excess_ret_{h}d"
        ex_results, ex_reg_df = run_ols(
            panel,
            y_col=y_ex,
            x_col="predicted_probability_smoothed",
            cov_type="HC3",
        )
        save_summary_text(
            ex_results,
            out_dir / f"ols_{y_ex}_on_smoothed_prob_summary.txt",
        )
        coefficient_table(ex_results).to_csv(
            out_dir / f"ols_{y_ex}_on_smoothed_prob_coefficients.csv",
            index=False,
        )

        headline_rows.append(
            {
                "regression": f"{y_ex} ~ predicted_probability_smoothed",
                "n_obs": int(ex_results.nobs),
                "beta": float(ex_results.params["predicted_probability_smoothed"]),
                "p_value": float(ex_results.pvalues["predicted_probability_smoothed"]),
                "t_or_z": float(ex_results.tvalues["predicted_probability_smoothed"]),
                "r_squared": float(ex_results.rsquared),
            }
        )

    headline = pd.DataFrame(headline_rows)
    headline.to_csv(out_dir / "multihorizon_regression_headline_summary.csv", index=False)

    print(f"Results directory: {out_dir}")
    print("\nHeadline summary:")
    print(headline.to_string(index=False))

    print("\nInterpretation:")
    print("- beta > 0 and p_value < 0.05: higher bull probability is significantly associated with higher future cumulative return.")
    print("- If 1d is insignificant but 5d/10d/20d become significant, the signal is more medium-horizon than next-day.")
    print("- Daily and short-horizon return regressions often have low R-squared; beta sign and significance matter more.")


if __name__ == "__main__":
    main()