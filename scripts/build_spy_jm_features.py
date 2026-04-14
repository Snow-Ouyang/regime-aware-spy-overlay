from pathlib import Path

from build_all_jm_features import (
    build_features,
    features_dir,
    load_asset_data,
    load_risk_free_data,
    save_feature_file,
)


TICKER = "SPY"


def main() -> None:
    rf_frame, detected_date_column, detected_rate_column = load_risk_free_data()
    print(
        "DGS3MO: loaded risk-free data with "
        f"date column `{detected_date_column}` and rate column `{detected_rate_column}`"
    )
    asset_frame = load_asset_data(TICKER)
    feature_frame = build_features(asset_frame, rf_frame)
    output_path = save_feature_file(feature_frame, TICKER, features_dir())
    print(f"{TICKER}: feature file ready at {Path(output_path)}")


if __name__ == "__main__":
    main()
