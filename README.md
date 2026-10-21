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