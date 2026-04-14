# SPY Long Sample Diagnostics

## Core Readout
- Predicted strategy annual return: `0.057576`
- Predicted strategy Sharpe: `0.432289`
- Predicted strategy max drawdown: `-0.153458`
- Buy-and-hold annual return: `0.083473`
- Buy-and-hold Sharpe: `0.416349`
- Buy-and-hold max drawdown: `-0.556927`

## Interpretation
- The strategy behaves primarily as a downside-protection overlay, not as a full replacement for buy-and-hold.
- Its edge is concentrated in crisis and drawdown periods.
- The main drag is slow recovery / re-entry after the market turns back up, plus a lower steady-state exposure in long bull phases.

## Re-entry Lag
- Average re-entry lag: `29.40` trading days
- Median re-entry lag: `32.00` trading days
- Maximum re-entry lag: `69.00` trading days

## Exposure
- Average position overall: `0.585`
- Average position in hold zone: `0.491`
- Average position in bull market regime: `0.753`
- Average position in bear market regime: `0.117`

## Phase Winners
- Best relative phase: `2007-11_to_2009-06`
- Worst relative phase: `2003-04_to_2007-10`

## Downside Protection
- Best drawdown-zone improvement: `dd_leq_30pct`
- Worst tail-day gap: `worst_10pct`

## Conclusion
- The current evidence supports a regime-filter / downside-protection narrative.
- The highest-value next optimization is recovery/re-entry logic, followed by reducing false bear signals.
- Further feature or hyperparameter search is lower priority than fixing the post-drawdown re-risk process.