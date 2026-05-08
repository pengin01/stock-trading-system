# v624 Robust Low Cap Daily

2万円運用の paper trading bot です。

## Files

- v624_robust_low_cap_daily.py
- v624_weekly_report.py
- .github/workflows/v624-robust-low-cap-daily.yml

## Strategy

- Initial capital: 20,000 JPY
- Lot size: 100 shares
- Max position: 1
- Tickers:
  - 9432.T
  - 9424.T
- Hold days:
  - 9432.T: 5
  - 9424.T: 10
- TP: +6%
- SL: -3%
- MIN_SCORE: 0.012
- PRICE: 50 - 200 JPY
- MIN_VALUE20: 50,000,000

## CSV State Files

- positions_v624.csv
- trades_v624.csv
- equity_v624.csv
- candidates_v624.csv

## GitHub Secret

Set this repository secret:

```text
DISCORD_WEBHOOK_URL
```
