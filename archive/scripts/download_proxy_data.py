import site
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_vendor_path() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)


configure_vendor_path()

import yfinance as yf


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
EXTRA_TICKERS: List[str] = ["SPY"]
DOWNLOAD_TICKERS: List[str] = PROXY_TICKERS + EXTRA_TICKERS
DOWNLOAD_PERIOD = "max"
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]


def data_raw_dir() -> Path:
    output_dir = project_root() / "data_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(part) for part in col if str(part) != "").strip("_")
            for col in frame.columns
        ]
    else:
        frame.columns = [str(col) for col in frame.columns]
    return frame


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    base_columns = {"Open", "High", "Low", "Close", "Volume"}

    for column in frame.columns:
        normalized = column.strip().replace(" ", "_")
        if normalized.startswith("Adj_Close"):
            rename_map[column] = "Adj_Close"
        elif normalized.startswith("Date"):
            rename_map[column] = "Date"
        else:
            base_name = normalized.split("_")[0]
            if base_name in base_columns:
                rename_map[column] = base_name

    return frame.rename(columns=rename_map)


def ensure_required_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{ticker}: missing required columns: {missing_columns}")
    return frame[EXPECTED_COLUMNS].copy()


def clean_downloaded_data(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{ticker}: downloaded data is empty")

    frame = flatten_columns(frame)
    frame = frame.reset_index()
    frame = normalize_column_names(frame)

    if "Date" not in frame.columns:
        raise ValueError(f"{ticker}: Date column not found after reset_index")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Date"])
    frame = ensure_required_columns(frame, ticker)
    frame = frame.sort_values("Date", ascending=True)
    frame = frame.drop_duplicates(subset=["Date"], keep="last")
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")

    if frame.empty:
        raise ValueError(f"{ticker}: no valid rows remain after cleaning")

    return frame


def download_ticker_data(ticker: str) -> pd.DataFrame:
    frame = yf.download(
        tickers=ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    return clean_downloaded_data(frame, ticker)


def save_single_ticker_csv(frame: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    output_path = output_dir / f"{ticker.lower()}.csv"
    frame.to_csv(output_path, index=False)
    print(f"{ticker}: saved {len(frame)} rows to {output_path}")
    return output_path


def main() -> None:
    output_dir = data_raw_dir()
    success: List[str] = []
    failed: List[str] = []

    for ticker in DOWNLOAD_TICKERS:
        try:
            frame = download_ticker_data(ticker)
            save_single_ticker_csv(frame, ticker, output_dir)
            success.append(ticker)
        except Exception as exc:
            failed.append(ticker)
            print(f"{ticker}: download failed with error: {exc}")

    print(f"Successful tickers: {success}")
    print(f"Failed tickers: {failed}")


if __name__ == "__main__":
    main()
