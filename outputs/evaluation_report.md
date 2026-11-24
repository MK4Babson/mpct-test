# Evaluation report (synthetic data)

Engine run with default tolerances: `{'qty_review_pct': 1.5, 'qty_hold_pct': 5.0, 'rate_round_usd': 0.005, 'pct_review_pts': 1.0, 'pct_hold_pts': 5.0, 'tm_review_pct': 0.0, 'tm_hold_pct': 10.0, 'money_tol': 1.0}`

## Detection by rule

| rule                  |   planted |   flagged |   true_pos |   false_pos |   missed |   precision_pct |   recall_pct |
|:----------------------|----------:|----------:|-----------:|------------:|---------:|----------------:|-------------:|
| DUPLICATE_INVOICE     |         1 |         1 |          1 |           0 |        0 |             100 |          100 |
| EXTENSION_ERROR       |         6 |         6 |          6 |           0 |        0 |             100 |          100 |
| PCT_COMPLETE_OVERBILL |        14 |        14 |         14 |           0 |        0 |             100 |          100 |
| PER_DIEM_OVER_POLICY  |        11 |        11 |         11 |           0 |        0 |             100 |          100 |
| QTY_OVER_RECEIPT      |        17 |        17 |         17 |           0 |        0 |             100 |          100 |
| RATE_OVER_CONTRACT    |         8 |         8 |          8 |           0 |        0 |             100 |          100 |
| RETAINAGE_ERROR       |         8 |         8 |          8 |           0 |        0 |             100 |          100 |
| TM_OVER_APPROVED      |        10 |        10 |         10 |           0 |        0 |             100 |          100 |
| UNAPPROVED_CO         |         5 |         5 |          5 |           0 |        0 |             100 |          100 |
| ALL RULES             |        80 |        80 |         80 |           0 |        0 |             100 |          100 |

## Invoice-level gate outcome

| has_error         |   PASS |   REVIEW |   HOLD |
|:------------------|-------:|---------:|-------:|
| Clean             |     70 |        0 |      0 |
| Has planted error |      0 |        1 |     44 |

**Planted overbilling caught:** $7,900,209 of $7,900,209 (100.0%)

## Missed errors

None.

## False alarms

None.

## Quantity tolerance trade-off

|   tolerance_pct |   precision_pct |   recall_pct |   false_pos |   missed |
|----------------:|----------------:|-------------:|------------:|---------:|
|            0    |             9.7 |        100   |         158 |        0 |
|            0.25 |            14.7 |        100   |          99 |        0 |
|            0.5  |            25.4 |        100   |          50 |        0 |
|            0.75 |            45.9 |        100   |          20 |        0 |
|            1    |            81   |        100   |           4 |        0 |
|            1.25 |           100   |        100   |           0 |        0 |
|            1.5  |           100   |        100   |           0 |        0 |
|            1.75 |           100   |        100   |           0 |        0 |
|            2    |           100   |        100   |           0 |        0 |
|            2.25 |           100   |        100   |           0 |        0 |
|            2.5  |           100   |         94.1 |           0 |        1 |
|            2.75 |           100   |         94.1 |           0 |        1 |
|            3    |           100   |         88.2 |           0 |        2 |
|            3.25 |           100   |         82.4 |           0 |        3 |
|            3.5  |           100   |         82.4 |           0 |        3 |
|            3.75 |           100   |         82.4 |           0 |        3 |
|            4    |           100   |         82.4 |           0 |        3 |
|            4.25 |           100   |         82.4 |           0 |        3 |
|            4.5  |           100   |         82.4 |           0 |        3 |
|            4.75 |           100   |         82.4 |           0 |        3 |
|            5    |           100   |         82.4 |           0 |        3 |
|            5.25 |           100   |         82.4 |           0 |        3 |
|            5.5  |           100   |         82.4 |           0 |        3 |
|            5.75 |           100   |         82.4 |           0 |        3 |
|            6    |           100   |         82.4 |           0 |        3 |

![tolerance trade-off](tolerance_tradeoff.png)