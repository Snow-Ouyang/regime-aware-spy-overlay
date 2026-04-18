from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from single_asset_gspc_spy_common import (
    StageConfig,
    build_target_windows,
    load_research_frame,
    load_trade_base,
    prepare_window,
    results_root,
    select_smoothing,
)


RESULTS_SUBDIR = "probability_nextday_return_regression"


def output_dir() -> Path:
    out = results_root() / RESULTS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_probability_return_panel() -> pd.DataFrame:
    """
    Build an OOS panel with:
      - Date: signal date t
      - predicted_probability_raw
      - predicted_probability_smoothed
      - next_ret: realized SPY return on t+1
      - next_rf_daily
      - next_excess_ret
      - rebalance_date
      - selected_smoothing_halflife
    """
    config = StageConfig(
        stage_name="final_model_gspc_to_spy",
        results_subdir="final_model_gspc_to_spy",
        feature_mode="enhanced",
        rule_mode="single_threshold_drawdown_overlay",
        threshold=0.55,
        drawdown_threshold=0.20,
        drawdown_prob_floor=0.52,
        output_ml_figures=False,
    )

    frame = load_research_frame()
    windows = build_target_windows(frame)
    trade_base = load_trade_base()

    panels = []

    for window in windows:
        prepared = prepare_window(window.__dict__, config.feature_mode)
        selected_smoothing = select_smoothing(prepared["validation_cache"], config)

        panel = pd.DataFrame(
            {
                "Date": pd.to_datetime(prepared["oos_signal_date"]),
                "rebalance_date": window.rebalance_date,
                "predicted_probability_raw": np.asarray(prepared["oos_raw_prob"], dtype=float),
                "predicted_probability_smoothed": np.asarray(
                    prepared["oos_smoothed"][selected_smoothing], dtype=float
                ),
                "y_true": np.asarray(prepared["oos_y_true"], dtype=int),
                "selected_smoothing_halflife": int(selected_smoothing),
            }
        )

        # trade_base has Date=t and next_ret = realized return from t to t+1
        panel = panel.merge(
            trade_base[["Date", "next_ret", "next_rf_daily"]],
            on="Date",
            how="inner",
        )
        panel["next_excess_ret"] = panel["next_ret"] - panel["next_rf_daily"]

        panels.append(panel)

    full_panel = (
        pd.concat(panels, ignore_index=True)
        .sort_values(["Date", "rebalance_date"])
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return full_panel


def run_ols(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    use_hc3: bool = True,
):
    """
    OLS: y = alpha + beta * x + eps
    """
    reg_df = df[[y_col, x_col]].dropna().copy()
    X = sm.add_constant(reg_df[x_col])
    y = reg_df[y_col]

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3") if use_hc3 else model.fit()
    return results, reg_df


def save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def coefficient_table(results, x_name: str) -> pd.DataFrame:
    conf = results.conf_int()
    return pd.DataFrame(
        {
            "term": results.params.index,
            "coef": results.params.values,
            "std_err": results.bse.values,
            "t_value": results.tvalues.values,
            "p_value": results.pvalues.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
        }
    )


def main() -> None:
    out_dir = output_dir()

    panel = build_probability_return_panel()
    panel.to_csv(out_dir / "probability_nextday_return_panel.csv", index=False)

    # Regression 1: next raw return on smoothed probability
    ret_results, ret_reg_df = run_ols(
        panel,
        y_col="next_ret",
        x_col="predicted_probability_smoothed",
        use_hc3=True,
    )
    save_text(ret_results.summary().as_text(), out_dir / "ols_next_ret_on_prob_summary.txt")
    coefficient_table(ret_results, "predicted_probability_smoothed").to_csv(
        out_dir / "ols_next_ret_on_prob_coefficients.csv", index=False
    )

    # Regression 2: next excess return on smoothed probability
    ex_results, ex_reg_df = run_ols(
        panel,
        y_col="next_excess_ret",
        x_col="predicted_probability_smoothed",
        use_hc3=True,
    )
    save_text(ex_results.summary().as_text(), out_dir / "ols_next_excess_ret_on_prob_summary.txt")
    coefficient_table(ex_results, "predicted_probability_smoothed").to_csv(
        out_dir / "ols_next_excess_ret_on_prob_coefficients.csv", index=False
    )

    # Optional: also check raw probability
    raw_ret_results, _ = run_ols(
        panel,
        y_col="next_ret",
        x_col="predicted_probability_raw",
        use_hc3=True,
    )
    save_text(raw_ret_results.summary().as_text(), out_dir / "ols_next_ret_on_raw_prob_summary.txt")
    coefficient_table(raw_ret_results, "predicted_probability_raw").to_csv(
        out_dir / "ols_next_ret_on_raw_prob_coefficients.csv", index=False
    )

    headline = pd.DataFrame(
        [
            {
                "regression": "next_ret ~ smoothed_prob",
                "n_obs": int(ret_results.nobs),
                "beta": float(ret_results.params["predicted_probability_smoothed"]),
                "p_value": float(ret_results.pvalues["predicted_probability_smoothed"]),
                "r_squared": float(ret_results.rsquared),
            },
            {
                "regression": "next_excess_ret ~ smoothed_prob",
                "n_obs": int(ex_results.nobs),
                "beta": float(ex_results.params["predicted_probability_smoothed"]),
                "p_value": float(ex_results.pvalues["predicted_probability_smoothed"]),
                "r_squared": float(ex_results.rsquared),
            },
            {
                "regression": "next_ret ~ raw_prob",
                "n_obs": int(raw_ret_results.nobs),
                "beta": float(raw_ret_results.params["predicted_probability_raw"]),
                "p_value": float(raw_ret_results.pvalues["predicted_probability_raw"]),
                "r_squared": float(raw_ret_results.rsquared),
            },
        ]
    )
    headline.to_csv(out_dir / "regression_headline_summary.csv", index=False)

    print(f"Results directory: {out_dir}")
    print("\nHeadline summary:")
    print(headline.to_string(index=False))

    print("\nInterpretation guide:")
    print("- beta > 0 and p_value < 0.05: higher predicted bull probability is significantly associated with higher next-day return.")
    print("- beta <= 0 or p_value not significant: probability may classify regimes reasonably, but it does not translate cleanly into next-day return prediction.")
    print("- R-squared will likely be small for daily returns; focus more on beta sign and significance than on R-squared.")


if __name__ == "__main__":
    main()