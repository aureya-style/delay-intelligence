import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in impute.iterrows()}

clean, _ = clean_shipments(new_raw, medians=medians)
feat = build_features(clean, hubs)

X_new = feat[feature_order]
assert X_new.isna().sum().sum() == 0, "missing values in features"

out = feat[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(X_new)[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)

out = out.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")

def choose_action(row):
    if row["flagged"] == 0:
        return "No action"
    if row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"

out["action"] = out.apply(choose_action, axis=1)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]

out.to_csv("outputs/new_shipment_actions.csv", index=False)

print(out.to_string(index=False))
print("rows:", len(out), "| unique ids:", out["shipment_id"].nunique())
print("flagged:", int(out["flagged"].sum()))
print("escalations:", int((out["action"] == "Escalate to hub manager").sum()))
print("calls:", int((out["action"] == "Proactive customer call").sum()))