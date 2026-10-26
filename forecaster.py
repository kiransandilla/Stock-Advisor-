# Feature building and forecasting for the demand forecaster
import numpy as np
import pandas as pd
import joblib

KEYS = ["Store ID", "Product ID"]


def load_bundle(path):
    # Loads the saved model together with its settings
    return joblib.load(path)


def make_forecast(history, future_plan, bundle, origin_date):
    # history: columns Store ID, Product ID, Date, Units Sold (sales up to the origin date)
    # future_plan: columns Store ID, Product ID, Date, Price, Discount, Promotion (no sales)
    model = bundle["model"]
    features = bundle["features"]
    max_h = bundle["max_horizon"]
    origin_date = pd.Timestamp(origin_date)

    # 1. Use only sales up to and including the origin date
    hist = history[history["Date"] <= origin_date].sort_values(KEYS + ["Date"])

    # 2. Last 7 rows of every store-product combination
    last7 = hist.groupby(KEYS).tail(7)
    origin_feats = last7.groupby(KEYS).agg(
        lag_6=("Units Sold", "first"),
        lag_0=("Units Sold", "last"),
        avg_7=("Units Sold", "mean"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
    ).reset_index()

    # 3. Safety check: 7 consecutive days ending exactly on the origin date
    span = (origin_feats["last_date"] - origin_feats["first_date"]).dt.days
    if not ((origin_feats["last_date"] == origin_date).all() and (span == 6).all()):
        raise ValueError("History is incomplete around the origin date")

    # 4. One row per store-product per future day
    plan = future_plan[
        (future_plan["Date"] > origin_date)
        & (future_plan["Date"] <= origin_date + pd.Timedelta(days=max_h))
    ].copy()
    plan["horizon"] = (plan["Date"] - origin_date).dt.days
    plan["target_dow"] = plan["Date"].dt.dayofweek
    rows = plan.merge(origin_feats[KEYS + ["lag_0", "lag_6", "avg_7"]], on=KEYS, how="left")

    # 5. Predict directly for every horizon (sales cannot be negative)
    rows["predicted_units"] = np.clip(model.predict(rows[features]), 0, None)
    return rows


def recommend_restock(forecast, current_stock, lead_time_days, z, daily_error, on_order=0):
    # forecast: output of make_forecast (needs Store ID, Product ID, horizon, predicted_units)
    # current_stock: table with Store ID, Product ID, current_stock
    if lead_time_days > forecast["horizon"].max():
        raise ValueError("Lead time is longer than the forecast horizon")

    # Lead-time demand = forecast units for days 1..lead_time_days
    window = forecast[forecast["horizon"] <= lead_time_days]
    demand = (window.groupby(KEYS)["predicted_units"]
                    .sum().rename("lead_time_demand").reset_index())

    out = demand.merge(current_stock, on=KEYS, how="left")
    out["safety_stock"] = z * daily_error * np.sqrt(lead_time_days)
    out["on_order"] = on_order
    need = out["lead_time_demand"] + out["safety_stock"] - out["current_stock"] - out["on_order"]
    out["reorder_qty"] = np.ceil(np.clip(need, 0, None)).astype(int)
    return out


def recommend_restock_v2(forecast, current_stock, lead_time_days, z, error_table,
                         default_error, on_order=0):
    # error_table: Store ID, Product ID, daily_error (forecast error per store-product)
    if lead_time_days > forecast["horizon"].max():
        raise ValueError("Lead time is longer than the forecast horizon")
    window = forecast[forecast["horizon"] <= lead_time_days]
    demand = (window.groupby(KEYS)["predicted_units"]
                    .sum().rename("lead_time_demand").reset_index())
    out = demand.merge(current_stock, on=KEYS, how="left")
    out = out.merge(error_table, on=KEYS, how="left")
    out["daily_error"] = out["daily_error"].fillna(default_error)
    out["safety_stock"] = z * out["daily_error"] * np.sqrt(lead_time_days)
    out["on_order"] = on_order
    need = out["lead_time_demand"] + out["safety_stock"] - out["current_stock"] - out["on_order"]
    out["reorder_qty"] = np.ceil(np.clip(need, 0, None)).astype(int)
    return out


def build_order_list(forecast, rec, lead_time_days):
    # Adds runs_out_on_day, stock_lasts and priority (URGENT / ORDER / OK) to a restock table
    f = forecast.sort_values(KEYS + ["horizon"]).copy()
    f["cum_demand"] = f.groupby(KEYS)["predicted_units"].cumsum()
    f = f.merge(rec[KEYS + ["current_stock"]], on=KEYS, how="left")
    runs_out = (f[f["cum_demand"] >= f["current_stock"]]
                .groupby(KEYS)["horizon"].min()
                .rename("runs_out_on_day").reset_index())
    out = rec.merge(runs_out, on=KEYS, how="left")

    def priority(row):
        if row["reorder_qty"] == 0:
            return "OK"
        if pd.notna(row["runs_out_on_day"]) and row["runs_out_on_day"] <= lead_time_days:
            return "URGENT"
        return "ORDER"

    out["priority"] = out.apply(priority, axis=1)
    out["stock_lasts"] = out["runs_out_on_day"].apply(
        lambda d: "more than 14 days" if pd.isna(d) else "about " + str(int(d)) + " day(s)")
    return out


def load_k_table(path):
    # CSV with columns: Lead time, k (fitted for z = 1.28 only)
    return pd.read_csv(path)


def get_k(k_table, lead_time_days, never_shrink=True):
    # Looks up the buffer correction for a lead time; by default never shrinks the buffer
    row = k_table[k_table["Lead time"] == lead_time_days]
    if row.empty:
        raise ValueError("No k factor for this lead time")
    k = float(row["k"].iloc[0])
    return max(k, 1.0) if never_shrink else k


def recommend_restock_v3(forecast, current_stock, lead_time_days, z, error_table,
                         default_error, k_table, on_order=0, never_shrink=True):
    # Per-product safety stock, multiplied by the lead-time correction k
    if abs(z - 1.28) > 1e-9:
        raise ValueError("k factors were fitted for z = 1.28 only; refit them for another z")
    out = recommend_restock_v2(forecast, current_stock, lead_time_days, z, error_table,
                               default_error, on_order)
    k = get_k(k_table, lead_time_days, never_shrink)
    out["k_factor"] = k
    out["safety_stock"] = out["safety_stock"] * k
    need = out["lead_time_demand"] + out["safety_stock"] - out["current_stock"] - out["on_order"]
    out["reorder_qty"] = np.ceil(np.clip(need, 0, None)).astype(int)
    return out
