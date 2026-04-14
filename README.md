# Regime-Aware ML Overlay for SPY

This repository is a single-asset machine learning project that studies whether a regime-aware overlay can improve the risk-adjusted profile of holding `SPY`.

The project is intentionally focused on one research-to-trade mapping:

- Research asset: `^GSPC`
- Trade asset: `SPY`
- Unified backtest start date: `2000-05-26`

All mainline stages in this repository were rerun under that same protocol so the reported results are directly comparable.

## Project Overview

The goal is not to build a perfect market-timing strategy that simply beats `SPY` in absolute return. The goal is narrower and more defensible:

- improve downside protection
- preserve participation in recoveries
- evaluate the strategy as a regime-aware overlay / risk filter on top of long-only equity exposure

The project uses `^GSPC` as the research asset because it provides the signal-generation context, while `SPY` remains the execution asset because it is the tradable ETF.

## Main Takeaways

1. The oracle stage confirms that the JumpModel regime labels have economic value under the mapped `^GSPC -> SPY` research-to-trade structure.
2. Relative to the paper baseline, the final model improves annual return and Sharpe while preserving the core downside-protection behavior.
3. The final model does not beat buy-and-hold on absolute annual return, but it does provide materially better downside protection and a higher Sharpe.

| Stage | Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Paper baseline | 0.0694 | 0.5244 | -0.1873 |
| Final model | 0.0719 | 0.5361 | -0.1899 |
| Buy-and-hold | 0.0836 | 0.4172 | -0.5569 |

## Research Question

Can a regime-aware ML overlay built on `^GSPC`:

- reduce left-tail damage and drawdowns
- improve recovery / re-entry after major selloffs
- maintain enough upside participation to remain useful over a long sample

## Methodology

The repository preserves the full mainline evolution from label validation to the final overlay.

### 1. Oracle

`oracle_gspc_to_spy` is not a real trading model. It is a mapped label economic-value test.

- JumpModel produces the latent bull/bear regime labels on the research asset.
- The oracle stage is evaluated under the same `^GSPC -> SPY` structure used by the rest of the project.
- Its purpose is to test whether the label itself contains economic value before ML prediction is introduced.
- It is intentionally more conservative than a pure full-information upper-bound oracle and should not be read as the final tradeable strategy.

Entry point:
- [`scripts/run_oracle_gspc_to_spy.py`](scripts/run_oracle_gspc_to_spy.py)

Output:
- [`results/single_asset_mainline/oracle_gspc_to_spy`](results/single_asset_mainline/oracle_gspc_to_spy)

### 2. Paper Baseline

`paper_baseline_gspc_to_spy` is the closest approximation to a first-pass paper-style baseline.

- JumpModel + XGBoost
- baseline technical features only
- M0 macro features
- simple single-threshold execution rule

Entry point:
- [`scripts/run_paper_baseline_gspc_to_spy.py`](scripts/run_paper_baseline_gspc_to_spy.py)

Output:
- [`results/single_asset_mainline/paper_baseline_gspc_to_spy`](results/single_asset_mainline/paper_baseline_gspc_to_spy)

### 3. Feature Enhancement

`feature_enhanced_gspc_to_spy` adds the validated feature improvements while keeping the simpler execution rule.

- baseline technical features
- refined return-window features
- recovery-oriented trend features
- M0 macro features

This stage matters because it adds recovery-oriented structure without yet changing the execution logic.

Entry point:
- [`scripts/run_feature_enhanced_gspc_to_spy.py`](scripts/run_feature_enhanced_gspc_to_spy.py)

Output:
- [`results/single_asset_mainline/feature_enhanced_gspc_to_spy`](results/single_asset_mainline/feature_enhanced_gspc_to_spy)

### 4. Decision-Rule Enhancement

`decision_rule_enhanced_gspc_to_spy` upgrades the execution layer.

- dynamic smoothing
- searched lower / upper thresholds
- double-threshold rule
- inertia hold in the middle zone

This stage matters because it improves the translation from probabilities into trade decisions.

Entry point:
- [`scripts/run_decision_rule_enhanced_gspc_to_spy.py`](scripts/run_decision_rule_enhanced_gspc_to_spy.py)

Output:
- [`results/single_asset_mainline/decision_rule_enhanced_gspc_to_spy`](results/single_asset_mainline/decision_rule_enhanced_gspc_to_spy)

### 5. Final Model

`final_recovery_overlay_gspc_to_spy` is the current single-asset mainline.

Research layer:
- `^GSPC`

Execution layer:
- `SPY`

Features:
- baseline technical features
- refined return-window features
- recovery-oriented trend features
- M0 macro features

Model:
- JumpModel with `n_components=2`, `jump_penalty=0.0`
- XGBoost with fixed production parameters

Post-processing:
- dynamic smoothing over `{0, 4, 8, 12}`
- searched lower / upper thresholds on validation

Final extra-entry rules:
- two-day rising confirmation and `probability > 0.52`
- `drawdown_from_peak <= -20%` and `probability > 0.44`

Entry point:
- [`scripts/run_final_model_gspc_to_spy.py`](scripts/run_final_model_gspc_to_spy.py)

Output:
- [`results/single_asset_mainline/final_recovery_overlay_gspc_to_spy`](results/single_asset_mainline/final_recovery_overlay_gspc_to_spy)

### 6. Diagnostics

The final stage is analyzed against the paper baseline using one unified diagnostic framework.

- phase performance
- bull / bear / hold state statistics
- re-entry lag analysis
- switching behavior
- downside protection summary
- exposure profile

Entry point:
- [`scripts/run_diagnostics_baseline_vs_final.py`](scripts/run_diagnostics_baseline_vs_final.py)

Output:
- [`results/single_asset_mainline/diagnostics_baseline_vs_final`](results/single_asset_mainline/diagnostics_baseline_vs_final)

## Data

Research asset:
- `^GSPC`

Trade asset:
- `SPY`

Macro inputs:
- current M0 macro panel in [`data_features/macro_feature_panel_m0.csv`](data_features/macro_feature_panel_m0.csv)

Unified protocol:
- OOS start: `2000-05-26`
- train lookback: `11 years`
- validation lookback: `4 years`
- validation subwindow: `6 months`
- OOS block: `6 months`

## Results

Full stage summary under the unified `2000-05-26` protocol:

| Stage | Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Oracle | 0.0217 | 0.0770 | -0.3089 |
| Paper baseline | 0.0694 | 0.5244 | -0.1873 |
| Feature enhanced | 0.0709 | 0.5368 | -0.1834 |
| Decision-rule enhanced | 0.0558 | 0.4108 | -0.2063 |
| Final model | 0.0719 | 0.5361 | -0.1899 |
| Buy-and-hold | 0.0836 | 0.4172 | -0.5569 |

### Interpretation

- The final model does not beat buy-and-hold on absolute annual return.
- Relative to the paper baseline, the final model improves annual return and Sharpe while preserving the same broad downside-protection profile.
- The final model keeps Sharpe above buy-and-hold.
- The final model’s main value is downside protection plus improved recovery participation.
- The strategy is best interpreted as a downside protection overlay / risk filter.

Key outputs:

- final summary table:
  - [`results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/strategy_performance_summary.csv`](results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/strategy_performance_summary.csv)
- final strategy vs buy-and-hold:
  - [`results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/strategy_vs_buyhold.png`](results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/strategy_vs_buyhold.png)
- final ML figures:
  - [`results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/confusion_matrix.png`](results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/confusion_matrix.png)
  - [`results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/roc_curve.png`](results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/roc_curve.png)
  - [`results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/classification_metrics_over_time.png`](results/single_asset_mainline/final_recovery_overlay_gspc_to_spy/classification_metrics_over_time.png)

## Diagnostics

The unified diagnostic result supports the following interpretation:

- this is best understood as a downside protection overlay / risk filter
- the final model improves on the baseline through better recovery participation
- it still trails buy-and-hold over long uninterrupted bull markets because average equity exposure remains below 1

See:
- [`results/single_asset_mainline/diagnostics_baseline_vs_final/diagnostic_summary.md`](results/single_asset_mainline/diagnostics_baseline_vs_final/diagnostic_summary.md)

## Project Structure

Top-level structure after cleanup:

- `data_raw/`: raw market and macro inputs
- `data_features/`: feature files and macro panels
- `scripts/`: active mainline and shared support scripts
- `results/single_asset_mainline/`: all unified mainline outputs
- `archive/scripts/`: archived exploratory scripts
- `archive/results/`: archived exploratory results

Active mainline scripts:

- [`scripts/single_asset_gspc_spy_common.py`](scripts/single_asset_gspc_spy_common.py)
- [`scripts/run_oracle_gspc_to_spy.py`](scripts/run_oracle_gspc_to_spy.py)
- [`scripts/run_paper_baseline_gspc_to_spy.py`](scripts/run_paper_baseline_gspc_to_spy.py)
- [`scripts/run_feature_enhanced_gspc_to_spy.py`](scripts/run_feature_enhanced_gspc_to_spy.py)
- [`scripts/run_decision_rule_enhanced_gspc_to_spy.py`](scripts/run_decision_rule_enhanced_gspc_to_spy.py)
- [`scripts/run_final_model_gspc_to_spy.py`](scripts/run_final_model_gspc_to_spy.py)
- [`scripts/run_diagnostics_baseline_vs_final.py`](scripts/run_diagnostics_baseline_vs_final.py)

## Reproducibility

To reproduce the mainline:

1. Ensure the required raw files and macro panel exist.
2. Run the stages in order:
   - `python scripts\run_oracle_gspc_to_spy.py`
   - `python scripts\run_paper_baseline_gspc_to_spy.py`
   - `python scripts\run_feature_enhanced_gspc_to_spy.py`
   - `python scripts\run_decision_rule_enhanced_gspc_to_spy.py`
   - `python scripts\run_final_model_gspc_to_spy.py`
   - `python scripts\run_diagnostics_baseline_vs_final.py`

All outputs will be written under:
- [`results/single_asset_mainline`](results/single_asset_mainline)

## Future Work

Multi-asset extension is future work. The current repository intentionally focuses on the single-asset `^GSPC -> SPY` problem.

Some portfolio-level risk control ideas, including drawdown-stop overlays and similar execution-layer experiments, were explored but are not part of the current single-asset mainline. Those belong to the archived exploratory track rather than the README’s main narrative.

The repo still retains future extension interfaces through the shared download / mapping utilities in:

- [`scripts/final_multi_asset_project_common.py`](scripts/final_multi_asset_project_common.py)

That means future research can add more research tickers and trade mappings without rebuilding the entire project layout.

## Conclusion

The final model is not a pure market-beating timing alpha strategy. It is a regime-aware downside protection overlay. It improves on the baseline mainly through better recovery / re-entry, while still accepting some long-bull-market underperformance relative to full buy-and-hold beta.
