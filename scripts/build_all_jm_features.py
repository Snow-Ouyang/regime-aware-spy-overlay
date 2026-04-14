from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROXY_TICKERS: List[str] = [
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
EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252
ETF_REQUIRED_COLUMNS = {"Date", "Adj_Close"}
OUTPUT_COLUMNS = [
    "Date",
    "Ticker",
    "Adj_Close",
    "ret",
    "rf_daily",
    "excess_ret",
    "log_downside_dev_hl5",
    "log_downside_dev_hl21",
    "ewm_return_hl5",
    "ewm_return_hl10",
    "ewm_return_hl21",
    "sortino_hl5",
    "sortino_hl10",
    "sortino_hl21",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_data_dir() -> Path:
    return project_root() / "data_raw"


def features_dir() -> Path:
    output_dir = project_root() / "data_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def detect_rate_column(columns: List[str], date_column: str) -> str:
    non_date_columns = [column for column in columns if column != date_column]
    exact_candidates = [column for column in non_date_columns if column.lower() == "dgs3mo"]
    if exact_candidates:
        return exact_candidates[0]

    for column in non_date_columns:
        lowered = column.lower()
        if "dgs3mo" in lowered or "rate" in lowered or "yield" in lowered or "rf" in lowered:
            return column

    if len(non_date_columns) == 1:
        return non_date_columns[0]

    raise ValueError("Could not identify the risk-free rate column")


def validate_required_raw_files() -> None:
    required_files = [raw_data_dir() / f"{ticker.lower()}.csv" for ticker in PROXY_TICKERS]
    required_files.append(raw_data_dir() / "DGS3MO.csv")

    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required raw files: {missing_files}")


def load_asset_data(ticker: str) -> pd.DataFrame:
    input_path = raw_data_dir() / f"{ticker.lower()}.csv"
    frame = pd.read_csv(input_path)
    frame = normalize_columns(frame)

    missing_columns = ETF_REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{ticker}: missing required columns: {sorted(missing_columns)}")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Adj_Close"] = pd.to_numeric(frame["Adj_Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Adj_Close"])
    frame = frame.sort_values("Date", ascending=True)
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    frame["Ticker"] = ticker

    if frame.empty:
        raise ValueError(f"{ticker}: no valid rows remain after loading asset data")

    return frame[["Date", "Ticker", "Adj_Close"]].copy()


def load_risk_free_data() -> Tuple[pd.DataFrame, str, str]:
    input_path = raw_data_dir() / "DGS3MO.csv"
    frame = pd.read_csv(input_path, na_values=[".", ""])
    frame = normalize_columns(frame)

    date_column = detect_date_column(frame.columns.tolist())
    rate_column = detect_rate_column(frame.columns.tolist(), date_column)

    frame = frame[[date_column, rate_column]].copy()
    frame = frame.rename(columns={date_column: "Date", rate_column: "rf_annual_pct"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["rf_annual_pct"] = pd.to_numeric(frame["rf_annual_pct"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values("Date", ascending=True)
    frame = frame.drop_duplicates(subset=["Date"], keep="last")
    frame["rf_annual_pct"] = frame["rf_annual_pct"].ffill()
    frame["rf_annual"] = frame["rf_annual_pct"] / 100.0
    frame["rf_daily"] = (1.0 + frame["rf_annual"]) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    if frame["rf_daily"].notna().sum() == 0:
        raise ValueError("Risk-free series is empty after cleaning")

    return frame[["Date", "rf_daily"]].copy(), date_column, rate_column


def align_risk_free(asset_frame: pd.DataFrame, rf_frame: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(asset_frame, rf_frame, on="Date", how="left")
    merged = merged.sort_values("Date", ascending=True)
    merged["rf_daily"] = merged["rf_daily"].ffill()
    merged["ret"] = merged["Adj_Close"].pct_change()
    merged["excess_ret"] = merged["ret"] - merged["rf_daily"]
    merged = merged.dropna(subset=["rf_daily"]).reset_index(drop=True)
    return merged


def compute_log_downside_deviation(series: pd.Series, halflife: int) -> pd.Series:
    negative_excess_ret = series.clip(upper=0.0)
    downside_dev = np.sqrt(
        negative_excess_ret.pow(2).ewm(halflife=halflife, adjust=False).mean()
    )
    return np.log(downside_dev + EPS)


def compute_ewm_return(series: pd.Series, halflife: int) -> pd.Series:
    return series.ewm(halflife=halflife, adjust=False).mean()


def build_features(asset_frame: pd.DataFrame, rf_frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = align_risk_free(asset_frame, rf_frame)

    downside_dev_hl5 = np.exp(compute_log_downside_deviation(feature_frame["excess_ret"], 5))
    downside_dev_hl10 = np.sqrt(
        feature_frame["excess_ret"].clip(upper=0.0).pow(2).ewm(halflife=10, adjust=False).mean()
    )
    downside_dev_hl21 = np.exp(compute_log_downside_deviation(feature_frame["excess_ret"], 21))

    feature_frame["log_downside_dev_hl5"] = np.log(downside_dev_hl5 + EPS)
    feature_frame["log_downside_dev_hl21"] = np.log(downside_dev_hl21 + EPS)
    feature_frame["ewm_return_hl5"] = compute_ewm_return(feature_frame["excess_ret"], 5)
    feature_frame["ewm_return_hl10"] = compute_ewm_return(feature_frame["excess_ret"], 10)
    feature_frame["ewm_return_hl21"] = compute_ewm_return(feature_frame["excess_ret"], 21)
    feature_frame["sortino_hl5"] = feature_frame["ewm_return_hl5"] / (downside_dev_hl5 + EPS)
    feature_frame["sortino_hl10"] = feature_frame["ewm_return_hl10"] / (downside_dev_hl10 + EPS)
    feature_frame["sortino_hl21"] = feature_frame["ewm_return_hl21"] / (downside_dev_hl21 + EPS)

    feature_frame["Date"] = feature_frame["Date"].dt.strftime("%Y-%m-%d")
    feature_frame = feature_frame.sort_values("Date", ascending=True).reset_index(drop=True)
    return feature_frame[OUTPUT_COLUMNS].copy()


def save_feature_file(frame: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    output_path = output_dir / f"{ticker.lower()}_jm_features.csv"
    frame.to_csv(output_path, index=False)
    print(f"{ticker}: saved {len(frame)} rows to {output_path}")
    return output_path


def build_panel(frames_by_ticker: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = pd.concat(frames_by_ticker.values(), ignore_index=True)
    panel = panel.sort_values(["Date", "Ticker"], ascending=True).reset_index(drop=True)
    return panel


def main() -> None:
    validate_required_raw_files()

    rf_frame, detected_date_column, detected_rate_column = load_risk_free_data()
    print(
        "DGS3MO: loaded risk-free data with "
        f"date column `{detected_date_column}` and rate column `{detected_rate_column}`"
    )

    output_dir = features_dir()
    success: List[str] = []
    failed: List[str] = []
    frames_by_ticker: Dict[str, pd.DataFrame] = {}

    for ticker in PROXY_TICKERS:
        try:
            asset_frame = load_asset_data(ticker)
            feature_frame = build_features(asset_frame, rf_frame)
            save_feature_file(feature_frame, ticker, output_dir)
            frames_by_ticker[ticker] = feature_frame
            success.append(ticker)
        except Exception as exc:
            failed.append(ticker)
            print(f"{ticker}: feature build failed with error: {exc}")

    if frames_by_ticker:
        panel = build_panel(frames_by_ticker)
        panel_path = output_dir / "jm_features_panel.csv"
        panel.to_csv(panel_path, index=False)
        print(f"PANEL: saved {len(panel)} rows to {panel_path}")

    print(f"Successful tickers: {success}")
    print(f"Failed tickers: {failed}")


if __name__ == "__main__":
    main()
