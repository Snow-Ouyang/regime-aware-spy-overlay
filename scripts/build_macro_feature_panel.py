from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


MACRO_EWM_HALFLIFE = 21
EPS = 1e-12


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_data_dir() -> Path:
    return project_root() / "data_raw"


def features_dir() -> Path:
    output_dir = project_root() / "data_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def macro_data_dir() -> Path:
    return features_dir() / "macro data"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
    return frame


def detect_date_column(columns: List[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in ["date", "observation_date", "datetime"]:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        if "date" in column.lower():
            return column
    raise ValueError("Could not identify a date column")


def detect_value_column(columns: List[str], date_column: str) -> str:
    non_date_columns = [column for column in columns if column != date_column]
    if len(non_date_columns) == 1:
        return non_date_columns[0]
    raise ValueError(f"Could not identify unique value column from: {columns}")


def load_macro_series(filename: str, output_name: str) -> Tuple[pd.DataFrame, str, str]:
    input_path = macro_data_dir() / filename
    frame = pd.read_csv(input_path, na_values=[".", ""])
    frame = normalize_columns(frame)
    date_column = detect_date_column(frame.columns.tolist())
    value_column = detect_value_column(frame.columns.tolist(), date_column)
    frame = frame[[date_column, value_column]].copy()
    frame = frame.rename(columns={date_column: "Date", value_column: output_name})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[output_name] = pd.to_numeric(frame[output_name], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values("Date", ascending=True)
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return frame, date_column, value_column


def build_macro_raw_panel() -> pd.DataFrame:
    dgs2, dgs2_date_col, dgs2_value_col = load_macro_series("DGS2.csv", "dgs2")
    dgs10, dgs10_date_col, dgs10_value_col = load_macro_series("DGS10.csv", "dgs10")
    vix, vix_date_col, vix_value_col = load_macro_series("VIX.csv", "vix")
    dbaa, dbaa_date_col, dbaa_value_col = load_macro_series("DBAA.csv", "dbaa")
    daaa, daaa_date_col, daaa_value_col = load_macro_series("DAAA.csv", "daaa")
    print(
        f"DGS2: using date column `{dgs2_date_col}` and value column `{dgs2_value_col}`"
    )
    print(
        f"DGS10: using date column `{dgs10_date_col}` and value column `{dgs10_value_col}`"
    )
    print(f"VIX: using date column `{vix_date_col}` and value column `{vix_value_col}`")
    print(f"DBAA: using date column `{dbaa_date_col}` and value column `{dbaa_value_col}`")
    print(f"DAAA: using date column `{daaa_date_col}` and value column `{daaa_value_col}`")

    date_union = set(dgs2["Date"]).union(dgs10["Date"]).union(vix["Date"]).union(dbaa["Date"]).union(daaa["Date"])
    dates = pd.DataFrame({"Date": sorted(date_union)})
    panel = dates.merge(dgs2, on="Date", how="left")
    panel = panel.merge(dgs10, on="Date", how="left")
    panel = panel.merge(vix, on="Date", how="left")
    panel = panel.merge(dbaa, on="Date", how="left")
    panel = panel.merge(daaa, on="Date", how="left")
    panel = panel.sort_values("Date", ascending=True).reset_index(drop=True)

    panel["dgs2"] = panel["dgs2"].ffill()
    panel["dgs10"] = panel["dgs10"].ffill()
    panel["vix"] = panel["vix"].ffill()
    panel["dbaa"] = panel["dbaa"].ffill()
    panel["daaa"] = panel["daaa"].ffill()
    panel["credit_spread_baa_aaa"] = panel["dbaa"] - panel["daaa"]
    return panel


def build_macro_feature_panel(raw_panel: pd.DataFrame) -> pd.DataFrame:
    feature_panel = raw_panel.copy()

    # 2Y yield change + EWMA:
    # dgs2_change_t = dgs2_t - dgs2_{t-1}
    # dgs2_change_ewm21_t = EWM_21(dgs2_change_t)
    feature_panel["dgs2_change"] = feature_panel["dgs2"].diff()
    feature_panel["dgs2_change_ewm21"] = feature_panel["dgs2_change"].ewm(
        halflife=MACRO_EWM_HALFLIFE, adjust=False
    ).mean()

    # 10Y-2Y slope EWMA:
    # slope_t = dgs10_t - dgs2_t
    # slope_ewm21_t = EWM_21(slope_t)
    feature_panel["slope_10y_2y"] = feature_panel["dgs10"] - feature_panel["dgs2"]
    feature_panel["slope_10y_2y_ewm21"] = feature_panel["slope_10y_2y"].ewm(
        halflife=MACRO_EWM_HALFLIFE, adjust=False
    ).mean()

    # 10Y-2Y slope change EWMA:
    # slope_change_t = slope_t - slope_{t-1}
    # slope_change_ewm21_t = EWM_21(slope_change_t)
    feature_panel["slope_10y_2y_change"] = feature_panel["slope_10y_2y"].diff()
    feature_panel["slope_10y_2y_change_ewm21"] = feature_panel["slope_10y_2y_change"].ewm(
        halflife=MACRO_EWM_HALFLIFE, adjust=False
    ).mean()

    # VIX log-difference EWMA:
    # vix_log_diff_t = log(vix_t + eps) - log(vix_{t-1} + eps)
    # vix_log_diff_ewm21_t = EWM_21(vix_log_diff_t)
    feature_panel["vix_log_diff"] = np.log(feature_panel["vix"] + EPS).diff()
    feature_panel["vix_log_diff_ewm21"] = feature_panel["vix_log_diff"].ewm(
        halflife=MACRO_EWM_HALFLIFE, adjust=False
    ).mean()

    # Credit spread EWMA:
    # credit_spread_t = DBAA_t - DAAA_t
    # credit_spread_baa_aaa_ewm21_t = EWM_21(credit_spread_t)
    feature_panel["credit_spread_baa_aaa_ewm21"] = feature_panel["credit_spread_baa_aaa"].ewm(
        halflife=MACRO_EWM_HALFLIFE, adjust=False
    ).mean()

    feature_panel["Date"] = feature_panel["Date"].dt.strftime("%Y-%m-%d")
    output_columns = [
        "Date",
        "dgs2_change_ewm21",
        "slope_10y_2y_ewm21",
        "slope_10y_2y_change_ewm21",
        "vix_log_diff_ewm21",
        "credit_spread_baa_aaa_ewm21",
    ]
    return feature_panel[output_columns].copy()


def main() -> None:
    raw_panel = build_macro_raw_panel()
    raw_panel_output = raw_panel.copy()
    raw_panel_output["Date"] = raw_panel_output["Date"].dt.strftime("%Y-%m-%d")

    raw_output_path = features_dir() / "macro_raw_panel.csv"
    raw_panel_output.to_csv(raw_output_path, index=False)
    print(f"Saved macro raw panel to {raw_output_path} with {len(raw_panel_output)} rows")

    feature_panel = build_macro_feature_panel(raw_panel)
    feature_output_path = features_dir() / "macro_feature_panel.csv"
    try:
        feature_panel.to_csv(feature_output_path, index=False)
        print(f"Saved macro feature panel to {feature_output_path} with {len(feature_panel)} rows")
    except PermissionError:
        fallback_path = features_dir() / "macro_feature_panel_creditspread.csv"
        feature_panel.to_csv(fallback_path, index=False)
        print(
            "macro_feature_panel.csv is locked by another process; "
            f"saved updated credit-spread panel to {fallback_path} with {len(feature_panel)} rows"
        )


if __name__ == "__main__":
    main()
