from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


TRAIN_YEARS = 11
VALIDATION_YEARS = 4
ASSET_UNIVERSE: List[str] = [
    "SPY",
    "IVV",
    "IJH",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "SPTL",
    "HYG",
    "SPBO",
    "IYR",
    "DBC",
    "GLD",
]
BASE_JM_FEATURES: List[str] = [
    "log_downside_dev_hl5",
    "log_downside_dev_hl21",
    "ewm_return_hl5",
    "ewm_return_hl10",
    "ewm_return_hl21",
    "sortino_hl5",
    "sortino_hl10",
    "sortino_hl21",
]
A1_REFINED_FEATURES: List[str] = [
    "ret_1_5d",
    "ret_6_20d",
    "ret_21_60d",
]
A2_ORIGINAL_FEATURES: List[str] = [
    "close_over_ma20",
    "close_over_ma60",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_dir() -> Path:
    return project_root() / "data_raw"


def features_dir() -> Path:
    return project_root() / "data_features"


def results_dir() -> Path:
    output_dir = project_root() / "results" / "multi_asset_availability_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def macro_panel_path() -> Path:
    m0 = features_dir() / "macro_feature_panel_m0.csv"
    if m0.exists():
        return m0
    return features_dir() / "macro_feature_panel.csv"


def normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values("Date", ascending=True)
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return frame


def first_date_on_or_after(dates: pd.Series, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = dates[dates >= target]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def build_price_feature_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    for column in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Close"])

    price = frame["Close"]
    ma20 = price.rolling(20).mean()
    ma60 = price.rolling(60).mean()
    return pd.DataFrame(
        {
            "Date": frame["Date"],
            "ret_1_5d": price / price.shift(5) - 1.0,
            "ret_6_20d": price.shift(5) / price.shift(20) - 1.0,
            "ret_21_60d": price.shift(20) / price.shift(60) - 1.0,
            "close_over_ma20": price / ma20 - 1.0,
            "close_over_ma60": price / ma60 - 1.0,
        }
    )


def infer_issue_reason(
    row: Dict[str, object],
    spy_first_possible_oos_date: Optional[pd.Timestamp],
    macro_start_date: Optional[pd.Timestamp],
) -> str:
    if row["issue_flag"] == "missing":
        return "file missing"
    raw_start = row["raw_data_start_date"]
    feature_start = row["feature_start_date"]
    merged_start = row["merged_dataset_start_date"]
    merged_end = row["merged_dataset_end_date"]

    if row["first_possible_oos_date"] is None:
        if merged_start is not None and merged_end is not None:
            total_years = (merged_end - merged_start).days / 365.25
            if total_years < (TRAIN_YEARS + VALIDATION_YEARS):
                return "raw history too short"
        return "other"
    if spy_first_possible_oos_date is None:
        return "other"
    if row["first_possible_oos_date"] <= spy_first_possible_oos_date:
        return ""

    if raw_start is None or feature_start is None or merged_start is None:
        return "other"
    if macro_start_date is not None and merged_start == macro_start_date:
        return "macro merge too late"
    if merged_start == feature_start:
        return "feature history too short"
    if raw_start >= feature_start:
        return "raw history too short"
    return "other"


def audit_asset(asset: str, macro_frame: pd.DataFrame) -> Dict[str, object]:
    raw_path = raw_dir() / f"{asset.lower()}.csv"
    feature_path = features_dir() / f"{asset.lower()}_jm_features.csv"

    result: Dict[str, object] = {
        "asset": asset,
        "raw_data_start_date": None,
        "raw_data_end_date": None,
        "feature_start_date": None,
        "feature_end_date": None,
        "merged_dataset_start_date": None,
        "merged_dataset_end_date": None,
        "first_possible_oos_date": None,
        "can_match_spy_oos_start": False,
        "gap_vs_spy_oos_start_days": None,
        "issue_flag": "",
        "issue_reason": "",
    }

    if not raw_path.exists() or not feature_path.exists():
        result["issue_flag"] = "missing"
        return result

    raw_frame = normalize_date_column(pd.read_csv(raw_path))
    feature_frame = normalize_date_column(pd.read_csv(feature_path))
    price_features = build_price_feature_frame(raw_frame)

    result["raw_data_start_date"] = raw_frame["Date"].min()
    result["raw_data_end_date"] = raw_frame["Date"].max()
    result["feature_start_date"] = feature_frame["Date"].min()
    result["feature_end_date"] = feature_frame["Date"].max()

    macro_use = macro_frame.copy()
    for column in macro_use.columns:
        if column != "Date":
            macro_use[column] = pd.to_numeric(macro_use[column], errors="coerce")
    macro_columns = [column for column in macro_use.columns if column != "Date"]

    merged = feature_frame.merge(macro_use, on="Date", how="left", sort=True)
    merged = merged.merge(price_features, on="Date", how="left", sort=True)
    merged = merged.sort_values("Date", ascending=True).reset_index(drop=True)
    merged[macro_columns] = merged[macro_columns].ffill()

    required_columns = [
        "Date",
        "ret",
        "rf_daily",
        "excess_ret",
        *BASE_JM_FEATURES,
        *A1_REFINED_FEATURES,
        *A2_ORIGINAL_FEATURES,
        *macro_columns,
    ]
    merged = merged.dropna(subset=required_columns).reset_index(drop=True)
    if merged.empty:
        result["issue_flag"] = "merged_empty"
        result["issue_reason"] = "other"
        return result

    result["merged_dataset_start_date"] = merged["Date"].min()
    result["merged_dataset_end_date"] = merged["Date"].max()

    target_start = result["merged_dataset_start_date"] + pd.DateOffset(
        years=TRAIN_YEARS + VALIDATION_YEARS
    )
    first_oos = first_date_on_or_after(merged["Date"], target_start)
    result["first_possible_oos_date"] = first_oos
    result["issue_flag"] = "ok" if first_oos is not None else "insufficient"
    return result


def format_date(value: Optional[pd.Timestamp]) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main() -> None:
    macro_path = macro_panel_path()
    if not macro_path.exists():
        raise FileNotFoundError(f"Missing macro panel: {macro_path}")

    macro_frame = normalize_date_column(pd.read_csv(macro_path))
    macro_start_date = macro_frame["Date"].min() if not macro_frame.empty else None

    audit_rows = [audit_asset(asset, macro_frame) for asset in ASSET_UNIVERSE]
    spy_row = next((row for row in audit_rows if row["asset"] == "SPY"), None)
    spy_first_possible_oos_date = spy_row["first_possible_oos_date"] if spy_row else None

    for row in audit_rows:
        first_oos = row["first_possible_oos_date"]
        if first_oos is not None and spy_first_possible_oos_date is not None:
            row["can_match_spy_oos_start"] = first_oos <= spy_first_possible_oos_date
            row["gap_vs_spy_oos_start_days"] = max(
                0, (first_oos - spy_first_possible_oos_date).days
            )
        else:
            row["can_match_spy_oos_start"] = False
            row["gap_vs_spy_oos_start_days"] = None

        row["issue_reason"] = infer_issue_reason(row, spy_first_possible_oos_date, macro_start_date)

    output_rows = []
    for row in audit_rows:
        output_rows.append(
            {
                "asset": row["asset"],
                "raw_data_start_date": format_date(row["raw_data_start_date"]),
                "raw_data_end_date": format_date(row["raw_data_end_date"]),
                "feature_start_date": format_date(row["feature_start_date"]),
                "feature_end_date": format_date(row["feature_end_date"]),
                "merged_dataset_start_date": format_date(row["merged_dataset_start_date"]),
                "merged_dataset_end_date": format_date(row["merged_dataset_end_date"]),
                "first_possible_oos_date": format_date(row["first_possible_oos_date"]),
                "can_match_spy_oos_start": row["can_match_spy_oos_start"],
                "gap_vs_spy_oos_start_days": row["gap_vs_spy_oos_start_days"],
                "issue_flag": row["issue_flag"],
                "issue_reason": row["issue_reason"],
            }
        )

    audit_frame = pd.DataFrame(output_rows)
    eligible = audit_frame.loc[audit_frame["can_match_spy_oos_start"]].copy()
    ineligible = audit_frame.loc[~audit_frame["can_match_spy_oos_start"]].copy()

    summary = pd.DataFrame(
        [
            {
                "spy_first_possible_oos_date": format_date(spy_first_possible_oos_date),
                "total_assets_checked": len(audit_frame),
                "assets_matching_spy_oos_start": int(eligible["asset"].count()),
                "assets_not_matching_spy_oos_start": int(ineligible["asset"].count()),
                "assets_missing_files": int((audit_frame["issue_flag"] == "missing").sum()),
            }
        ]
    )

    output_dir = results_dir()
    audit_frame.to_csv(output_dir / "asset_availability_audit.csv", index=False)
    summary.to_csv(output_dir / "asset_availability_summary.csv", index=False)
    eligible.to_csv(output_dir / "eligible_assets_for_common_oos.csv", index=False)
    ineligible.to_csv(output_dir / "ineligible_assets_for_common_oos.csv", index=False)

    print(f"Macro panel used: {macro_path}")
    print(f"SPY first possible OOS date: {format_date(spy_first_possible_oos_date)}")
    print(f"Assets checked: {len(audit_frame)}")
    print(f"Eligible assets: {len(eligible)}")
    print(f"Ineligible assets: {len(ineligible)}")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
