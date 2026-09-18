# Credit Score model monitoring report

## Monitoring Date
2026-09-16 09:35

## Reference Population

The monitoring reference consists of the customer-level training populationused to train the final 
credit-score classification model.

- Rows: 70,000
- Customers: 8,750
- Numeric Features Monitored: 26
- Categorical Features Monitored: 5

## Monitoring Scenario

A synthetic production dataset was generated to validate the monitoring
framework. The simulated scenario introduced controlled changes in:

- Outstanding debt
- Total monthly EMI
- Credit inquiries
- Payment delays
- Credit mix distribution
- Payment behaviour distribution
- Missingness in Monthly Inhand Salary

The results below therefore demonstrate the ability of the monitoring
framework to detect known changes. They should not be interpreted as evidence
of actual production drift.

## 1. Numeric Feature Drift

Numeric drift was measured using Population Stability Index (PSI).

Interpretation:

- PSI < 0.10: No significant drift
- PSI 0.10–0.25: Moderate drift
- PSI >= 0.25: Significant drift

### Results

- Significant drift detected in 3 numeric features.
- Moderate drift detected in 1 numeric features.

### Most affected features

                 Feature      PSI       Interpretation
    Num_Credit_Inquiries 2.669338    Significant drift
  Num_of_Delayed_Payment 0.655966    Significant drift
     Delay_from_due_date 0.306051    Significant drift
        Outstanding_Debt 0.232424       Moderate drift
     Total_EMI_per_month 0.055869 No significant drift
    Changed_Credit_Limit 0.001000 No significant drift
Credit_Utilization_Ratio 0.000909 No significant drift
       Num_Bank_Accounts 0.000830 No significant drift
           Annual_Income 0.000763 No significant drift
   Monthly_Inhand_Salary 0.000713 No significant drift

## 2. Categorical Feature Drift

Categorical drift was measured using Total Variation Distance.

Interpretation:

- TV < 0.05: No significant drift
- TV 0.05–0.10: Moderate drift
- TV >= 0.10: Significant drift

### Results

- Significant drift detected in 2 categorical features.
- Moderate drift detected in 0 categorical features.

### Results

              Feature  TV_Distance       Interpretation
           Credit_Mix     0.348314    Significant drift
    Payment_Behaviour     0.301657    Significant drift
                Month     0.009600 No significant drift
           Occupation     0.007000 No significant drift
Payment_of_Min_Amount     0.006781 No significant drift

## 3. Missingness Drift

Missingness was monitored separately because missing-value patterns can
change independently of feature distributions.

Interpretation:

- Absolute change < 5 percentage points: No significant drift
- 5–10 percentage points: Moderate drift
- >= 10 percentage points: Significant drift

### Results

- Significant missingness changes: 2
- Moderate missingness changes: 1

### Most affected features

                Feature  Reference_Missing_%  Current_Missing_%  Change_pp  Absolute_Change_pp       Interpretation
  Monthly_Inhand_Salary            14.968571          36.553333  21.584762           21.584762    Significant drift
             Credit_Mix            20.230000           0.000000 -20.230000           20.230000    Significant drift
      Payment_Behaviour             7.544286           0.000000  -7.544286            7.544286       Moderate drift
                    Age             8.387143           8.053333  -0.333810            0.333810 No significant drift
Amount_invested_monthly             4.502857           4.706667   0.203810            0.203810 No significant drift
             Occupation             7.018571           7.206667   0.188095            0.188095 No significant drift
        Monthly_Balance             1.220000           1.373333   0.153333            0.153333 No significant drift
   Changed_Credit_Limit             2.108571           2.206667   0.098095            0.098095 No significant drift
 Num_of_Delayed_Payment             6.937143           6.980000   0.042857            0.042857 No significant drift
  Credit_History_Months             9.052857           9.040000  -0.012857            0.012857 No significant drift

## 4. Prediction Drift

Prediction drift was measured using Total Variation Distance between the
reference and current prediction distributions.

- Prediction TV distance: 0.1207
- Status: Significant drift

### Prediction distribution

   Class  Reference_%  Current_%
    Good    18.588571   6.873333
    Poor    28.041429  40.113333
Standard    53.370000  53.013333

## 5. Overall Monitoring Summary

        Monitoring_Signal  Significant_Drift_Count  Moderate_Drift_Count    Overall_Status
    Numeric feature drift                        3                     1 Significant drift
Categorical feature drift                        2                     0 Significant drift
        Missingness drift                        2                     1 Significant drift
         Prediction drift                        1                     0 Significant drift

## 6. Interpretation

The monitoring framework successfully identifies changes introduced in the
synthetic production population.

The strongest signals should be investigated before assuming that the model
itself has degraded. Feature drift may indicate a change in the incoming
customer population, data collection process, upstream systems, or business
processes.

Prediction drift provides an additional model-level signal. A meaningful
change in predicted credit-score categories can indicate that the population
being scored differs materially from the population used during training.

## 7. Recommended Production Actions

If similar drift were detected in a real production environment:

1. Investigate the affected features and upstream data sources.
2. Check whether changes are caused by genuine population shifts or data
   quality issues.
3. Monitor prediction distributions and confidence scores.
4. When actual customer outcomes become available, evaluate model
   performance using accuracy, macro F1, class-level recall and other
   appropriate metrics.
5. Retrain or recalibrate the model only after confirming that the observed
   change is persistent and meaningful.

## Limitations

This monitoring demonstration uses simulated production data because
historical production observations and post-deployment ground-truth labels
are not available.

Consequently, the current implementation demonstrates data and prediction
drift detection but does not establish real-world model performance
degradation.

## Monitoring Architecture

Reference training population
        |
        v
Feature distributions
        |
        +--> Numeric drift (PSI)
        |
        +--> Categorical drift (TV distance)
        |
        +--> Missingness drift
        |
        v
Production population
        |
        v
Model predictions
        |
        +--> Prediction distribution drift
        |
        v
Monitoring report
