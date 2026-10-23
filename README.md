# Delay Intelligence

Analytics repository for the regional operations team: delivery-hours estimation, delay prediction and hub segmentation. The data is synthetic
(produced by `generate_data.py`).

The logistics brand is only the setting.

## Data findings

All numbers below are copied from `outputs/part_a_delay_by_group.csv`, `outputs/part_a_correlations.csv` and `outputs/part_a_checks.csv`.
After removing 24 duplicate rows, 1,200 unique shipments remain, of which 278 are delayed (overall delay rate 0.2317).

1. **Service type.** Express parcels are delayed most often: 119 of 243 (delay rate 0.4897), against 0.2056 for Standard and 0.0857 for Economy.
2. **Hub tier.** The delay rate climbs steeply from Tier 1 to Tier 3: 0.0400 for Tier 1, 0.2150 for Tier 2 and 0.4400 for Tier 3.
3. **Distance band.** The delay rate rises with every band: 0.0219 for 0-200 km, 0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km.
4. **Strongest correlates of delivery_hours.** The three columns most strongly correlated with delivery_hours are promised_hours (0.8372),
   distance_km (0.4013) and n_stops (0.3247). All three are positive: parcels with a longer promise, a longer lane or a busier rider route take
   longer to deliver. These are associations, not proof of cause; promised_hours, for example, largely stands in for the service level, which sets the base transit time.
5. **Why declared_value_inr was log-transformed.** The raw values are right-skewed: the mean (INR 1533.0542) sits well above the median
   (INR 1121.5000), pulled up by a few expensive parcels. After the log transform the mean (7.0277) and median (7.0224) almost coincide, so the
   feature is roughly symmetric and no longer dominated by extreme values.

## Model results (regression)

Numbers come from `outputs/part_b_metrics.csv`, `part_b_tree_depth.csv`, `part_b_gb_learning_rate.csv`, `part_b_cv.csv` and `part_b_checks.csv`.
The split is 840 training / 180 validation / 180 test rows, and the scaler was fitted on the training rows only (largest absolute scaled-column mean: about 0 on train, 0.2187 on test).

**Baseline.** Predicting the training mean for every parcel gives a validation R² of -0.0212 (RMSE 18.8552 hours). The model explains none of the variation in delivery time; any real model has to beat this.

**Winner on validation.** Linear regression had the lowest validation RMSE, 4.8912 hours (R² 0.9313), narrowly ahead of gradient boosting at 4.9165 hours 
(R² 0.9306). The random forest scored 5.5095 hours and the decision tree 6.2796 hours.

**Bias and variance (decision-tree depth).**
- Depths 1 and 2 are high bias: both errors are high and close together (depth 1: train 11.6334, validation 12.1324; depth 2: 9.3687 and 10.4114).
  The tree is too simple to capture the pattern.
- Depths 12 and unlimited are high variance: training error falls to 1.3613 and 0.0000 while validation error climbs to 7.3540 and 7.8368, gaps of
  5.9927 and 7.8368 hours. The tree memorises the training rows.
- Depth 6 has the lowest validation RMSE (6.2796); depth 8 already starts to overfit (6.7901).

**Learning rate (gradient boosting, 100 trees).** A rate of 0.1 was best (validation R² 0.9306, RMSE 4.9165). At 0.01 the R² is only 0.7127: each tree
adds a tiny correction, so 100 trees are too few to converge and the model underfits. At 1.0 the R² is 0.8440: each tree takes a full-size step, the
corrections overshoot and the model becomes unstable.

**Cross-validation (5 folds, 1,020 training plus validation rows, one shared KFold).** Linear regression 0.9261 mean R² (std 0.0048); gradient boosting
0.9353 (std 0.0045). The ranking disagrees with validation: gradient boosting is ahead in cross-validation although linear regression was ahead on the
single 180-row validation split.

**Which model to ship.** I would ship gradient boosting. On the single 180-row validation split it was effectively tied with linear regression (RMSE 4.9165 versus  4.8912 hours, a gap of 0.0253 hours), so that split cannot separate the two. Cross-validation is the stronger evidence because it uses 1,020 rows across five folds, and there gradient boosting is ahead: mean R² 0.9353 against 0.9261, a gap of 0.0092, about two standard deviations (std 0.0045 and 0.0048). Gradient boosting can also capture interactions that a linear model cannot, such as Express parcels during the sale window or rain on long lanes. The trade-off is that it is slower and harder to explain than linear regression, and the margin is small, so this is a judgement call rather than a clear-cut win.

**Final test score.** The pre-set rule (lowest validation RMSE) selected linear regression as the final model, and it was scored on the 180 test rows exactly once: RMSE 4.5235 hours, MAE 3.5671 hours and R² 0.9493. Gradient boosting was never scored on the test rows, because the test set is reserved for one evaluation of the model chosen before looking at it. The test figures above therefore describe linear regression, not the model recommended for shipping, and the shipped model's test performance has not been measured. The test R² (0.9493) is also higher than the validation R² (0.9313), which with only 180 rows can simply reflect which parcels landed in which split; it is a single noisy estimate, not a guarantee.

## Model results (classification)

Numbers come from `outputs/part_c_models.csv`, `part_c_feature_scores.csv`, `part_c_selectk.csv`, `part_c_tuning.csv`, `part_c_thresholds.csv`,
`part_c_final_test.csv` and `part_c_final_config.csv`. The split is stratified: 840 training / 180 validation / 180 test rows, with 195 / 42 / 41 delayed parcels.

**Blind baseline.** Predicting "not delayed" for every parcel gives a validation accuracy of 0.7667 but catches none of the 42 delayed parcels (recall 0.0000, TP 0). It is useless for operations, because the only parcels that matter are the ones it never flags. The trained models reach accuracy 0.8833 to 0.9056, and the better measure is recall: logistic regression catches 35 of the 42 delayed parcels (recall 0.8333, F1 0.8046).

**Noise columns.** The five bookkeeping columns (label_print_batch, scanner_build_no, tape_roll_no, sticker_serial_mod, shelf_slot_no) together hold 0.2244 of the default forest's feature importance, which looks substantial but is not evidence of signal: a forest can find chance patterns in any continuous column. The F-scores are the cleaner evidence. All five fall outside the top 7 (F-rank 8, 12, 14, 16 and 17) with F-scores between 0.0230 and 4.5847, while the smallest F-score among the top seven columns is 36.7281.

**Number of selected columns (k).** Validation F1 is 0.6988 at k=5, 0.8333 at k=7, 0.7857 at k=10, 0.8049 at k=12 and 0.7848 at k=17. Cutting to five columns drops real signal (rain_flag and sale_event_flag), and adding more columns brings noise back in. With only 42 delayed validation parcels, one parcel moves F1 by roughly  0.01 to 0.02, so the gap between k=7 and the larger values is suggestive rather than conclusive.

**Tuning.** Tuning did not beat the untuned forest on validation. Grid search (best cross-validated accuracy 0.8952) reached validation F1 0.7750 and randomized search (0.8905) reached 0.7848, against 0.8333 for the default forest on the same seven columns. Both searches improved cross-validated accuracy on training data, yet validation F1 did not follow. That says a 180-row validation set, with 42 delayed parcels, is too small to rank configurations whose scores differ by a few parcels; choosing the winner by validation F1 partly rewards luck. The default forest was chosen (`chosen_from` in `part_c_final_config.csv`) following the pre-set rule.

**Alert threshold.** The rule is the highest threshold whose validation recall is at least 0.80. At 0.5 the final model has validation recall 0.8333 (35 of 42 delayed parcels caught) and precision 0.7955, with 9 false alarms. At 0.6 precision would rise to 0.8889 but recall would fall to 0.7619, missing 10 delayed parcels instead of 7, so 0.5 is the chosen threshold. Lower thresholds buy recall at the cost of precision: at 0.1 recall is 0.9762 but precision is only 0.4824. The validation F1 of this model at 0.5 is 0.8140 in `part_c_thresholds.csv`, slightly below the 0.8333 in `part_c_tuning.csv`. The tuning file uses the library's `predict`, which treats a probability of exactly 0.5 as on time, whereas the threshold rule flags a parcel at 0.5 or above, so two on-time parcels at exactly 0.5 become false alarms.

**Final test result.** Scored once on the test rows at threshold 0.5: TN 125, FP 14, FN 10, TP 31, precision 0.6889, recall 0.7561, F1 0.7209 and accuracy 0.8667. Test recall (0.7561) is lower than validation recall (0.8333), a drop of 0.0772. The test set holds only 41 delayed parcels, so each parcel is worth about 0.024 of recall and the drop is about three parcels. The threshold was chosen on validation data, so the validation figures are somewhat optimistic, and the test figures are a single noisy estimate. Treat the recall of the alert as roughly 0.75 to 0.83, not as a guarantee of 0.80.