import os
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

dedup = raw.drop_duplicates()
clean, medians = clean_shipments(raw)
feat = build_features(clean, hubs)
feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)
feat.to_csv("data/modelling_table.csv", index=False)

print(len(feat), feat["is_delayed"].sum())

hours = feat["delivery_hours"]
q1 = hours.quantile(0.25)
q3 = hours.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", int((dedup["parcel_weight_kg"] > 100).sum())),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(dedup["service_type"].nunique())),
    ("service_type_spellings_after", int(clean["service_type"].nunique())),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("declared_value_mean_inr", feat["declared_value_inr"].mean()),
    ("declared_value_median_inr", feat["declared_value_inr"].median()),
    ("log_declared_value_mean", feat["log_declared_value"].mean()),
    ("log_declared_value_median", feat["log_declared_value"].median()),
    ("rows_after_hub_merge", len(feat)),
    ("nulls_in_features_and_target",
     int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(feat["is_delayed"].sum())),
    ("delay_rate", feat["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((hours < lower) | (hours > upper)).sum())),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df["value"] = checks_df["value"].astype(float).round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)
print(checks_df.to_string(index=False))

# ---------- Delay by group ----------
feat["distance_band"] = pd.cut(
    feat["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
)   # right edge inclusive is the default

rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index().rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    g["group"] = g["group"].astype(str)
    rows.append(g)

by_group = pd.concat(rows, ignore_index=True).round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)
print(by_group.to_string(index=False))

corr = feat.drop(columns=["distance_band"]).corr(numeric_only=True).round(4)
corr.to_csv("outputs/part_a_correlations.csv", index_label="column")
print(corr.loc["delivery_hours", "promised_hours"],
      corr.loc["delivery_hours", "distance_km"],
      corr.loc["is_delayed", "distance_km"])

top3 = corr["delivery_hours"].drop("delivery_hours").abs().sort_values(ascending=False).head(3)
print(corr.loc[top3.index, "delivery_hours"])