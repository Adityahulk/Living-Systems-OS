# Results Summary

## Cross-Validation RMSE

| model | rmse_mean | rmse_std |
| --- | --- | --- |
| hybrid | 535.0008087158203 | 84.00179536684695 |
| rf | 561.4541554807619 | 72.18836868645148 |
| mlp | 614.4958698146722 | 33.046084118696 |
| gbr | 635.0305934246912 | 62.25665864371892 |
| deepsurv | 697.3323974609375 | 131.7048528016885 |
| cox | 707.7728167835149 | 80.44349795636272 |
| hybrid_no_ode | 770.6130256951844 | 17.172491267035422 |
| lognormal_aft | 2839.886690438149 | 950.7938628981541 |
| weibull_aft | 3370.913495216907 | 516.6145071203084 |

## Statistical Tests

See `results/metrics/statistical_tests.json` for paired tests and bootstrap confidence intervals.
