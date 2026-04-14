# Diagnostics Summary

## Main finding
Relative to the paper baseline, the new final model improves annual return and Sharpe while preserving the same broad downside-protection behavior.

## Overall comparison
- `paper_baseline_gspc_to_spy`: annual return 0.0694, Sharpe 0.5244, max drawdown -0.1873, excess annual return vs buy-and-hold -0.0143
- `final_model_gspc_to_spy`: annual return 0.0737, Sharpe 0.5731, max drawdown -0.1684, excess annual return vs buy-and-hold -0.0099

## Interpretation
- The new final model is still best understood as a downside protection overlay / risk filter, not a pure timing-alpha replacement for buy-and-hold.
- The main improvement comes from a simpler execution layer: dynamic smoothing, a fixed 0.55 single threshold, and a drawdown-conditioned extra entry rule based on `drawdown_from_peak <= -20%` and `probability > 0.52`.
- The older double-threshold plus inertia-hold line was explored but did not remain the best solution under the unified 2000-05-26 protocol.
- Rising-2d extra entry was tested on top of the new baseline and rejected because it did not improve on the drawdown-only extra-entry version.

## Re-entry
- Paper baseline mean / median re-entry lag: 50.11 / 59.00 days
- Final model mean / median re-entry lag: 48.92 / 51.00 days

## Decision
The unified protocol supports using `final_model_gspc_to_spy` as the single-asset mainline model.