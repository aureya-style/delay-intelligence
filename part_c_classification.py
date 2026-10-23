import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src.shipment_prep import FEATURES, NOISE_COLUMNS
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# ---------- Task 1: stratified split ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(len(X_train), len(X_val), len(X_test))
print(y_train.sum(), y_val.sum(), y_test.sum())


# ---------- helper: confusion cells + metrics from predictions ----------
def class_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------- Task 2: blind baseline + three classifiers (default 0.5 rule) ----------
model_rows = []

model_rows.append(("blind_all_on_time", class_metrics(y_val, np.zeros(len(y_val), dtype=int))))

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logreg.fit(X_train, y_train)
model_rows.append(("logistic_regression", class_metrics(y_val, logreg.predict(X_val))))

best_d, best_f1, best_tree_metrics = None, -1.0, None
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    m = class_metrics(y_val, tree.predict(X_val))
    if m["f1"] > best_f1:
        best_d, best_f1, best_tree_metrics = d, m["f1"], m
model_rows.append((f"decision_tree_depth_{best_d}", best_tree_metrics))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
model_rows.append(("random_forest_default", class_metrics(y_val, rf.predict(X_val))))

models_df = pd.DataFrame(
    [{"model": n, **m} for n, m in model_rows],
    columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
)
num_cols = ["accuracy", "precision", "recall", "f1"]
models_df[num_cols] = models_df[num_cols].round(4)
models_df.to_csv("outputs/part_c_models.csv", index=False)
print(models_df.to_string(index=False))

# ---------- Task 3: which columns carry signal? ----------
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)

selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)

scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": rf_all.feature_importances_,
})
scores_df["f_rank"] = scores_df["f_score"].rank(ascending=False, method="first").astype(int)
scores_df["is_noise"] = scores_df["feature"].isin(NOISE_COLUMNS)
scores_df = scores_df.sort_values("f_rank")[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]
scores_out = scores_df.copy()
scores_out[["f_score", "rf_importance"]] = scores_out[["f_score", "rf_importance"]].round(4)
scores_out.to_csv("outputs/part_c_feature_scores.csv", index=False)
print(scores_out.to_string(index=False))
print("noise share of forest importance:",
      round(scores_df.loc[scores_df["is_noise"], "rf_importance"].sum(), 4))

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    selectk_rows.append((k, f1_score(y_val, pipe.predict(X_val))))

selectk_df = pd.DataFrame(selectk_rows, columns=["k", "val_f1"]).round(4)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)
print(selectk_df.to_string(index=False))

# ---------- Task 4: tuning on the 7 selected columns ----------
selector7 = SelectKBest(f_classif, k=7)
X_train_7 = selector7.fit_transform(X_train, y_train)
X_val_7 = selector7.transform(X_val)
print("selected:", [f for f, keep in zip(FEATURES, selector7.get_support()) if keep])

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_7, y_train)

rand = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_grid,
    n_iter=9, cv=5, random_state=42,
)
rand.fit(X_train_7, y_train)

default_rf = RandomForestClassifier(n_estimators=100, random_state=42)
default_rf.fit(X_train_7, y_train)

def params_str(p):
    return ", ".join(f"{k}={v}" for k, v in sorted(p.items()))

default_params = {"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}

tuning_rows = [
    ("grid_search", 27, 5, 27 * 5, params_str(grid.best_params_),
     grid.best_score_, f1_score(y_val, grid.best_estimator_.predict(X_val_7))),
    ("randomized_search", 9, 5, 9 * 5, params_str(rand.best_params_),
     rand.best_score_, f1_score(y_val, rand.best_estimator_.predict(X_val_7))),
    ("default_forest_same_7_features", 1, 0, 1, params_str(default_params),
     np.nan, f1_score(y_val, default_rf.predict(X_val_7))),
]
tuning_df = pd.DataFrame(
    tuning_rows,
    columns=["search", "candidates", "cv_folds", "total_fits",
             "best_params", "best_cv_accuracy", "val_f1"],
)
tuning_df[["best_cv_accuracy", "val_f1"]] = tuning_df[["best_cv_accuracy", "val_f1"]].round(4)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_df.to_string(index=False))

# ---------- Task 5: choose the final configuration ----------
cfg_candidates = [
    ("default_forest_same_7_features", default_params, tuning_rows[2][6]),
    ("grid_search", grid.best_params_, tuning_rows[0][6]),
    ("randomized_search", rand.best_params_, tuning_rows[1][6]),
]
chosen_from, chosen_params, best_val_f1 = None, None, -1.0
for name, params, val_f1 in cfg_candidates:
    if val_f1 > best_val_f1:
        chosen_from, chosen_params, best_val_f1 = name, params, val_f1
print("chosen:", chosen_from, chosen_params)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
])
final_model.fit(X_train, y_train)

# ---------- Task 6: alert threshold on validation ----------
val_proba = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

thr_rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    thr_rows.append({"threshold": t, **class_metrics(y_val, pred)})
thr_raw = pd.DataFrame(thr_rows)

ok = thr_raw.loc[thr_raw["recall"] >= 0.80, "threshold"]
chosen_threshold = float(ok.max()) if len(ok) > 0 else 0.1
print("chosen threshold:", chosen_threshold)

thr_out = thr_raw.copy()
thr_out[["accuracy", "precision", "recall", "f1"]] = thr_out[
    ["accuracy", "precision", "recall", "f1"]].round(4)
thr_out.to_csv("outputs/part_c_thresholds.csv", index=False)
print(thr_out.to_string(index=False))

# ---------- Task 7: ONE test evaluation ----------
test_proba = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= chosen_threshold).astype(int)
test_row = {"threshold": chosen_threshold, **class_metrics(y_test, test_pred)}
test_df = pd.DataFrame([test_row])
test_df[["accuracy", "precision", "recall", "f1"]] = test_df[
    ["accuracy", "precision", "recall", "f1"]].round(4)
test_df.to_csv("outputs/part_c_final_test.csv", index=False)
print(test_df.to_string(index=False))

config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"],
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])
config_df.to_csv("outputs/part_c_final_config.csv", index=False)
print(config_df.to_string(index=False))