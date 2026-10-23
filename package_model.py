import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
threshold = float(cfg["threshold"])

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(
        random_state=42,
        n_estimators=int(cfg["n_estimators"]),
        max_depth=max_depth,
        min_samples_leaf=int(cfg["min_samples_leaf"]),
    )),
])
model.fit(X_train, y_train)

joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
impute = pd.DataFrame({"column": list(medians.keys()), "median": list(medians.values())})
impute["median"] = impute["median"].round(4)
impute.to_csv("models/impute_values.csv", index=False)

loaded = joblib.load("models/delay_model.pkl")
p_memory = model.predict_proba(X_test)[:, 1]
p_loaded = loaded.predict_proba(X_test)[:, 1]
max_diff = float(np.abs(p_memory - p_loaded).max())

check = pd.DataFrame(
    [("test_rows_scored", len(X_test)),
     ("reload_max_abs_probability_diff", max_diff),
     ("threshold", threshold)],
    columns=["metric", "value"],
)
check["value"] = check["value"].astype(float).round(4)
check.to_csv("outputs/part_d_reload_check.csv", index=False)
print(check.to_string(index=False))
print(impute.to_string(index=False))