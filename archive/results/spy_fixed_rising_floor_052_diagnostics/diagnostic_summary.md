# fixed_rising_floor_052 Diagnostics

## Core Readout
- fixed_rising_floor_052 annual return: `0.076319`
- fixed_rising_floor_052 Sharpe: `0.582064`
- fixed_rising_floor_052 max drawdown: `-0.163226`
- buy-and-hold annual return: `0.083473`
- buy-and-hold Sharpe: `0.416349`
- buy-and-hold max drawdown: `-0.556927`

## Main Answers
- The strategy's main edge still comes from risk reduction and crisis handling, but this version improves recovery participation relative to the old baseline.
- The rising-entry floor improves long-run return and Sharpe mainly by allowing earlier re-risking after market improvement.
- It still lags buy-and-hold on raw annual return because average exposure remains well below 1.0.

## Recovery / Re-entry
- fixed_rising_floor_052 average re-entry lag: `18.93` days
- fixed_rising_floor_052 median re-entry lag: `20.50` days
- fixed_rising_floor_052 max re-entry lag: `51.00` days
- current baseline average re-entry lag: `29.40` days
- current baseline median re-entry lag: `32.00` days

## Interpretation
- This version still fits best as a downside-protection overlay / risk filter, not a full buy-and-hold replacement.
- The recovery enhancement is real if re-entry lag statistics improve versus the old baseline.
- The remaining weakness is still underexposure during persistent bull trends.

## Crisis Phase
- In 2007-11 to 2009-06, excess annual return vs buy-and-hold was `0.3475` and Sharpe difference was `1.5156`.

## Long Bull / Recovery Drag
- 2003-04 to 2007-10 excess annual return vs buy-and-hold: `-0.0590`.
- 2009-07 to 2019-12 excess annual return vs buy-and-hold: `-0.0644`.

## State Mix
- Hold share of days: `0.140`
- Hold average segment length: `3.97` days

## Recommendation
- Yes, the current evidence supports promoting `fixed_rising_floor_052` to the SPY single-asset mainline.
- It improves on the old baseline in return and Sharpe while keeping drawdown control far better than buy-and-hold.
- The next optimization target should still be recovery / re-entry efficiency, not another broad feature or parameter search.