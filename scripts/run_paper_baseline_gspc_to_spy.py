from single_asset_gspc_spy_common import StageConfig, run_ml_stage, write_stage_outputs


def main() -> None:
    config = StageConfig(
        stage_name="paper_baseline_gspc_to_spy",
        results_subdir="paper_baseline_gspc_to_spy",
        feature_mode="paper",
        rule_mode="single_threshold",
        threshold=0.50,
    )
    result = run_ml_stage(config)
    out_dir = write_stage_outputs(
        results_subdir=config.results_subdir,
        version_name=config.stage_name,
        mapped_frame=result["mapped_frame"],
        buyhold_frame=result["buyhold_frame"],
        signal_frame=result["signal_frame"],
        selection_log=result["selection_log"],
        prediction_metrics=result["prediction_metrics"],
    )
    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
