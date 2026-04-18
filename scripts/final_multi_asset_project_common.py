import os
import site
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def configure_paths() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
    vendor_override = os.environ.get("BULL_BEAR_VENDOR_PATH")
    if vendor_override:
        vendor_override_path = Path(vendor_override)
        if vendor_override_path.exists():
            vendor_override_str = str(vendor_override_path)
            if vendor_override_str not in sys.path:
                sys.path.insert(0, vendor_override_str)
            return
    vendor_site = Path(__file__).resolve().parents[1] / ".vendor"
    if vendor_site.exists():
        vendor_site_str = str(vendor_site)
        if vendor_site_str not in sys.path:
            sys.path.append(vendor_site_str)


configure_paths()

from build_all_jm_features import build_features, load_risk_free_data


RESEARCH_ASSETS: List[Tuple[str, str]] = [
    ("^GSPC", "gspc"),
    ("^MID", "mid"),
    ("^RUT", "rut"),
]
TRADE_ASSETS: List[Tuple[str, str]] = [
    ("SPY", "spy_trade"),
    ("IJH", "ijh_trade"),
    ("IWM", "iwm_trade"),
]
RESEARCH_TO_TRADE_MAP: Dict[str, str] = {
    "gspc": "spy_trade",
    "mid": "ijh_trade",
    "rut": "iwm_trade",
}
TARGET_OOS_START = pd.Timestamp("2008-04-28")
VALIDATION_YEARS = 4
TRAIN_YEARS = 11
STEP_MONTHS = 6
TRADING_DAYS = 252
TRANSACTION_COST = 0.0005
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
A2_FEATURES: List[str] = [
    "close_over_ma20",
    "close_over_ma60",
]
FIXED_XGB_PARAMS: Dict[str, object] = {
    "max_depth": 4,
    "learning_rate": 0.10,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 0,
    "n_jobs": 1,
    "verbosity": 0,
}
SMOOTHING_HALFLIFE_GRID: List[int] = [0, 4, 8, 12]
THRESHOLD_GRID: List[float] = [0.60]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_data_dir() -> Path:
    output_dir = project_root() / "data_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def features_dir() -> Path:
    output_dir = project_root() / "data_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def results_root_dir() -> Path:
    output_dir = project_root() / "results" / "final_multi_asset_project"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def signal_results_dir() -> Path:
    output_dir = results_root_dir() / "signal_research"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def signal_asset_results_dir(stem: str) -> Path:
    output_dir = signal_results_dir() / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def signal_panels_dir() -> Path:
    output_dir = signal_results_dir() / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def trade_execution_dir() -> Path:
    output_dir = results_root_dir() / "trade_execution"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def macro_feature_path() -> Path:
    return features_dir() / "macro_feature_panel_m0.csv"


def raw_path_for_stem(stem: str) -> Path:
    return raw_data_dir() / f"{stem}.csv"


def research_feature_path_for_stem(stem: str) -> Path:
    return features_dir() / f"{stem}_features_final.csv"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(part) for part in column if str(part) != "").strip("_")
            for column in frame.columns
        ]
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
    return frame


def ensure_raw_data(ticker: str, stem: str) -> Path:
    output_path = raw_path_for_stem(stem)
    if output_path.exists():
        return output_path

    import yfinance as yf

    frame = yf.download(
        tickers=ticker,
        period="max",
        auto_adjust=False,
        interval="1d",
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"{ticker}: yfinance returned no rows")

    frame = frame.reset_index()
    frame = normalize_columns(frame)
    rename_map = {}
    for column in frame.columns:
        lowered = column.lower()
        if lowered.startswith("adj_close"):
            rename_map[column] = "Adj_Close"
        elif lowered == "date":
            rename_map[column] = "Date"
        elif lowered.startswith("open"):
            rename_map[column] = "Open"
        elif lowered.startswith("high"):
            rename_map[column] = "High"
        elif lowered.startswith("low"):
            rename_map[column] = "Low"
        elif lowered == "close" or lowered.startswith("close_"):
            rename_map[column] = "Close"
        elif lowered.startswith("volume"):
            rename_map[column] = "Volume"
    frame = frame.rename(columns=rename_map)

    required_columns = ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{ticker}: missing download columns {missing_columns}")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in required_columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", "Adj_Close"]).sort_values("Date").reset_index(drop=True)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    frame.to_csv(output_path, index=False)
    return output_path


def ensure_research_feature_file(ticker: str, stem: str) -> Path:
    output_path = research_feature_path_for_stem(stem)
    if output_path.exists():
        return output_path

    ensure_raw_data(ticker, stem)
    rf_frame, _, _ = load_risk_free_data()
    asset_frame = pd.read_csv(raw_path_for_stem(stem))
    asset_frame.columns = [str(column).strip() for column in asset_frame.columns]
    asset_frame["Date"] = pd.to_datetime(asset_frame["Date"], errors="coerce")
    asset_frame["Adj_Close"] = pd.to_numeric(asset_frame["Adj_Close"], errors="coerce")
    asset_frame = asset_frame.dropna(subset=["Date", "Adj_Close"]).sort_values("Date").reset_index(drop=True)
    asset_frame["Ticker"] = ticker
    asset_frame = asset_frame[["Date", "Ticker", "Adj_Close"]].copy()
    feature_frame = build_features(asset_frame, rf_frame)
    feature_frame.to_csv(output_path, index=False)
    return output_path


def ensure_research_inputs() -> Dict[str, Dict[str, Path]]:
    outputs: Dict[str, Dict[str, Path]] = {}
    for ticker, stem in RESEARCH_ASSETS:
        outputs[stem] = {
            "raw_path": ensure_raw_data(ticker, stem),
            "feature_path": ensure_research_feature_file(ticker, stem),
        }
    return outputs


def ensure_trade_inputs() -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for ticker, stem in TRADE_ASSETS:
        outputs[stem] = ensure_raw_data(ticker, stem)
    return outputs


def load_research_experiment_frame(stem: str) -> pd.DataFrame:
    feature_frame = pd.read_csv(research_feature_path_for_stem(stem))
    macro_frame = pd.read_csv(macro_feature_path())
    raw_frame = pd.read_csv(raw_path_for_stem(stem))

    for frame in [feature_frame, macro_frame, raw_frame]:
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")

    for column in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        raw_frame[column] = pd.to_numeric(raw_frame[column], errors="coerce")
    for column in macro_frame.columns:
        if column != "Date":
            macro_frame[column] = pd.to_numeric(macro_frame[column], errors="coerce")

    feature_frame = feature_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    macro_frame = macro_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    raw_frame = raw_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    price = raw_frame["Close"]
    ma20 = price.rolling(20).mean()
    ma60 = price.rolling(60).mean()
    price_features = pd.DataFrame(
        {
            "Date": raw_frame["Date"],
            "ret_1_5d": price / price.shift(5) - 1.0,
            "ret_6_20d": price.shift(5) / price.shift(20) - 1.0,
            "ret_21_60d": price.shift(20) / price.shift(60) - 1.0,
            "close_over_ma20": price / ma20 - 1.0,
            "close_over_ma60": price / ma60 - 1.0,
        }
    )

    merged = feature_frame.merge(macro_frame, on="Date", how="left", sort=True)
    merged = merged.merge(price_features, on="Date", how="left", sort=True)
    merged = merged.sort_values("Date").reset_index(drop=True)
    macro_columns = [column for column in macro_frame.columns if column != "Date"]
    merged[macro_columns] = merged[macro_columns].ffill()
    required_columns = [
        "Date",
        "ret",
        "rf_daily",
        "excess_ret",
        *BASE_JM_FEATURES,
        *A1_REFINED_FEATURES,
        *A2_FEATURES,
        *macro_columns,
    ]
    merged = merged.dropna(subset=required_columns).reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"{stem}: merged dataset is empty after dropping required NaN")
    return merged


def research_feature_columns(frame: pd.DataFrame) -> List[str]:
    macro_columns = [
        column
        for column in frame.columns
        if column.startswith("dgs")
        or column.startswith("slope_")
        or column.startswith("vix_")
        or column.startswith("credit_spread_")
    ]
    return [*BASE_JM_FEATURES, *A1_REFINED_FEATURES, *A2_FEATURES, *macro_columns]


def load_trade_price_frame(stem: str) -> pd.DataFrame:
    frame = pd.read_csv(raw_path_for_stem(stem))
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
    return frame


def load_risk_free_daily_series() -> pd.DataFrame:
    rf_frame, _, _ = load_risk_free_data()
    rf_frame = rf_frame.copy()
    rf_frame["Date"] = pd.to_datetime(rf_frame["Date"], errors="coerce")
    rf_frame = rf_frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    rf_frame["rf_daily"] = pd.to_numeric(rf_frame["rf_daily"], errors="coerce")
    rf_frame["rf_daily"] = rf_frame["rf_daily"].ffill()
    return rf_frame[["Date", "rf_daily"]].copy()
