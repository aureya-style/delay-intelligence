import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import FEATURES
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

# ---------- Task 1: split ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(len(X_train), len(X_val), len(X_test))

# ---------- Task 2: scaling without leakage ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

train_max = np.abs(X_train_s.mean(axis=0)).max()
test_max = np.abs(X_test_s.mean(axis=0)).max()
print(train_max, test_max)

checks = pd.DataFrame(
    [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", train_max),
        ("test_scaled_max_abs_mean", test_max),
    ],
    columns=["metric", "value"],
)
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_b_checks.csv", index=False)

# ---------- helper used by every model ----------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }

# ---------- Task 3: mean baseline ----------
baseline_pred = np.full(len(y_val), y_train.mean())
baseline = regression_metrics(y_val, baseline_pred)
print("baseline:", baseline)

# ---------- Task 4: tree-depth diagnostic ----------
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))
    val_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
    depth_rows.append(("unlimited" if d is None else d, train_rmse, val_rmse))

depth_df = pd.DataFrame(depth_rows, columns=["max_depth", "train_rmse", "val_rmse"])
depth_df[["train_rmse", "val_rmse"]] = depth_df[["train_rmse", "val_rmse"]].round(4)
depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
print(depth_df.to_string(index=False))

best_depth, best_val = None, float("inf")
for d, (_, _, v) in zip(depths, depth_rows): 
    if v < best_val:
        best_depth, best_val = d, v 
print("best depth:", best_depth, "val_rmse:", round(best_val, 4))

# ---------- Task 5: four candidate models ----------
candidates = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

metrics_rows = [("mean_baseline", "val", baseline)]
for name, model in candidates.items():
    model.fit(X_train, y_train)
    metrics_rows.append((name, "val", regression_metrics(y_val, model.predict(X_val))))

def metrics_table(rows):
    out = pd.DataFrame(
        [{"model": n, "split": s, **m} for n, s, m in rows],
        columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
    )
    out[["MAE", "MSE", "RMSE", "R2"]] = out[["MAE", "MSE", "RMSE", "R2"]].astype(float).round(4)
    return out

metrics_df = metrics_table(metrics_rows)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
print(metrics_df.to_string(index=False))

# ---------- Task 6: learning-rate experiment ----------
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rows.append((lr, np.sqrt(mean_squared_error(y_val, pred)), r2_score(y_val, pred)))

lr_df = pd.DataFrame(lr_rows, columns=["learning_rate", "val_rmse", "val_r2"]).round(4)
lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
print(lr_df.to_string(index=False))

# ---------- Task 7: rank, cross-validate top two, final test ----------
val_rmse = {n: m["RMSE"] for n, s, m in metrics_rows if n in candidates}
ranked = sorted(val_rmse, key=val_rmse.get)
top_two = ranked[:2]
final_name = ranked[0]
print("ranking:", ranked)

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name in top_two:
    scores = cross_val_score(candidates[name], X_tv, y_tv, cv=kf)
    cv_rows.append((name, scores.mean(), scores.std()))

cv_df = pd.DataFrame(cv_rows, columns=["model", "mean_r2", "std_r2"]).round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)
print(cv_df.to_string(index=False))

final_model = candidates[final_name]
test_metrics = regression_metrics(y_test, final_model.predict(X_test))
metrics_rows.append((final_name, "test", test_metrics))

metrics_df = metrics_table(metrics_rows)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
print(metrics_df.to_string(index=False))