import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    df = raw.copy()

    df = df.drop_duplicates()

    df["service_type"] = df["service_type"].str.strip().str.title()

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    fill_cols = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {c: float(df[c].median()) for c in fill_cols}
    for c in fill_cols:
        df[c] = df[c].fillna(medians[c])

    return df, medians