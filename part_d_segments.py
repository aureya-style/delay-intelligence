import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---------- Task 1: hub profile ----------
profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)
print(profile.round(4).to_string(index=False))

# ---------- Task 2: scale and cluster (K = 3) ----------
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X_scaled = StandardScaler().fit_transform(profile[cols])

km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)

# ---------- Task 3a: separation evidence for K = 2..5 ----------
def across_over_within(X, labels):
    i, j = np.triu_indices(len(X), k=1)
    dist = np.linalg.norm(X[i] - X[j], axis=1)
    same = labels[i] == labels[j]
    return dist[~same].mean() / dist[same].mean()

sep_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).labels_
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append((k, across_over_within(X_scaled, labels), "-".join(str(s) for s in sizes)))

sep_df = pd.DataFrame(sep_rows, columns=["K", "across_over_within", "cluster_sizes"])
sep_df["across_over_within"] = sep_df["across_over_within"].round(4)
sep_df.to_csv("outputs/part_d_separation.csv", index=False)
print(sep_df.to_string(index=False))

# ---------- Task 3b: name the clusters by mean delay rate ----------
profile["cluster"] = km3.labels_
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
hub_segments = seg[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = (
    profile.groupby("segment")
    .agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    .reindex(["Standard", "Watch", "Escalate"])
    .reset_index()
    .round(4)
)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)
print(segment_profile.to_string(index=False))

# ---------- Task 3c: tier-versus-segment mismatches ----------
implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(implied)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(mismatch.to_string(index=False))