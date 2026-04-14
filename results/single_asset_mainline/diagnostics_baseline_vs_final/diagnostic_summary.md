# Diagnostics Summary

## Main finding
The final model improves on the paper baseline mainly through faster recovery participation while preserving the project's downside-protection profile.

## Interpretation
- The final model is still best understood as a downside protection overlay / risk filter, not a pure timing-alpha replacement for buy-and-hold.
- The strongest relative value comes from crisis and post-crisis recovery windows.
- The final model still lags buy-and-hold during long uninterrupted bull markets because average market exposure remains below 1.

## Decision
The unified 2000-05-26 protocol supports using `final_recovery_overlay_gspc_to_spy` as the single-asset mainline model.