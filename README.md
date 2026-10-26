# Demand Forecasting and Restock Advisor for Small Shops: ML Component

This document summarizes the machine learning component of the project. It was developed and tested independently in Google Colab (Python, pandas, NumPy, Matplotlib, scikit-learn). Backend integration (Spring Boot, MySQL) has not been started.

**Every number below comes from actual notebook output.** Nothing here is estimated or illustrative unless it is explicitly labeled as such.

---

## 1. What the component does

For every store-product combination, it:

1. Takes the last 7 days of sales history up to a chosen "origin" day.
2. Takes the planned Price, Discount and Promotion for the next 14 days.
3. Predicts daily units sold for each of the next 14 days (horizons 1 to 14).
4. Combines the forecast with current stock, supplier lead time and a safety buffer to recommend a reorder quantity.

---

## 2. Dataset

| Item | Value |
|---|---|
| Source | `sales_data.csv` (synthetic data, **not** real sales from any shop) |
| Rows | 76,000 (after removing `Demand`: 15 columns) |
| Date range | 2022-01-01 to 2024-01-30 (760 days) |
| Stores / products | 5 stores, 20 products, 100 store-product combinations |
| Prediction target | `Units Sold` |

**Checks performed**

- No missing values and no duplicate rows.
- Every one of the 100 combinations has exactly 760 rows, the same first and last date, no repeated dates, and every gap between consecutive rows is exactly 1 day. This confirms that "7 rows back" really means "7 days back".
- `Demand` was removed to avoid target leakage.

**Columns not used as features, and why**

- `Inventory Level` and `Units Ordered`: their values on a future day are not known when forecasting, and their exact meaning in the synthetic data was not verified. `Inventory Level` is used only as "current stock" in the restock step.
- `Category`, `Region`, `Weather Condition`, `Competitor Pricing`, `Seasonality`, `Epidemic`: not used in this version. They are possible future experiments, but future values would also have to be known in advance.

---

## 3. Method

### 3.1 Chronological split (never random)

| Block | Dates | Rows (one-day-ahead dataset) |
|---|---|---|
| Train | 2022-01-08 to 2023-09-30 | 63,100 |
| Validation | 2023-10-01 to 2023-11-30 | 6,100 |
| Test | 2023-12-01 to 2024-01-30 | 6,100 |

The first 7 days of each history are dropped because the lag features do not exist yet. The test block was used only for final evaluation, and no tuning was done after seeing test results.

### 3.2 Experiment 1: one-day-ahead model comparison

Features: `sales_lag_1`, `sales_lag_7`, `sales_avg_7`, `Price`, `Discount`, `Promotion`, `day_of_week`.

Validation results (lower is better):

| Model | MAE | RMSE |
|---|---|---|
| Same weekday last week | 39.238 | 50.982 |
| 7-day moving average | 29.463 | 38.354 |
| Training average (reference) | 33.343 | 41.598 |
| Ridge Regression | 28.243 | 36.472 |
| Random Forest | 27.621 | 35.874 |
| HistGradientBoosting (HGB) | **27.430** | **35.659** |

- Ridge used scaling (`StandardScaler`) and one-hot encoding of the weekday, fitted on training data only.
- Random Forest: training MAE 27.331 vs validation 27.621, so no sign of overfitting.
- HGB: early stopping kept 119 trees.
- The three ML models are within about 1 MAE of each other. HGB was chosen because it had the lowest validation error, not because it is more complex. Its edge over Random Forest (about 0.2) is small.

One-day-ahead **test** results for the selected HGB model:

| Model | MAE | RMSE | WAPE % |
|---|---|---|---|
| HGB (selected) | 29.087 | 37.577 | 33.5 |
| 7-day moving average | 31.640 | 40.919 | 36.4 |
| Same weekday last week | 42.926 | 55.036 | 49.4 |
| Training average (reference) | 34.291 | 43.160 | 39.5 |

WAPE = total absolute error divided by total actual sales.

### 3.3 Experiment 2: direct multi-horizon model (the real 14-day forecast)

The one-day-ahead model uses yesterday's actual sales, which do not exist for future days. So the final model is a **direct multi-horizon** model, as specified in the DDS.

- One global model for all store-products and all horizons.
- Each training row means: "standing at origin day T, what will sales be h days later?"
- Features: `lag_0` (sales on day T), `lag_6` (sales on day T-6), `avg_7` (mean of days T-6 to T), `horizon` (h = 1 to 14), `Price`, `Discount`, `Promotion` and `target_dow` (weekday) **of the target day**.
- Target: sales on day T+h.
- Lag features come only from days up to T. **No predictions are fed back into later steps, so no recursion is needed and future actual sales are never used as inputs.**
- Dataset size: 1,045,100 rows (train 874,300 / validation 85,400 / test 85,400), split by **target date** using the same date boundaries as above.

**Choosing the number of trees.** Validation MAE was measured every 50 trees. It fell quickly up to about 150 trees and then flattened (29.447 at 300 trees, 29.431 at 400, 29.427 at 450). 400 trees was chosen. The gain from tuning further is about 0.02 MAE.

**Final settings:** `HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_depth=6, min_samples_leaf=50, early_stopping=False, random_state=42)`.

### 3.4 Final test results (horizons 1 to 14 combined)

| Model | MAE | RMSE | WAPE % |
|---|---|---|---|
| HGB multi-horizon | **31.106** | **39.617** | **35.8** |
| 7-day moving average | 34.923 | 44.286 | 40.2 |

Average Units Sold in the test period: 86.89.

Per-horizon test MAE:

| Horizon (days ahead) | HGB | Moving average |
|---|---|---|
| 1 | 28.951 | 31.640 |
| 2 | 29.286 | 32.052 |
| 3 | 29.732 | 32.618 |
| 4 | 30.098 | 33.182 |
| 5 | 30.512 | 33.772 |
| 6 | 30.893 | 34.479 |
| 7 | 31.326 | 35.216 |
| 8 | 31.690 | 35.735 |
| 9 | 31.895 | 36.174 |
| 10 | 32.104 | 36.498 |
| 11 | 32.257 | 36.755 |
| 12 | 32.290 | 36.909 |
| 13 | 32.250 | 36.970 |
| 14 | 32.196 | 36.917 |

**Interpretation**

- The model beats the moving average by about 3.8 MAE (roughly 11%) on dates it never saw.
- Error rises from about 29.0 at 1 day ahead to about 32.3 at 12 days, then is flat. The small dips after day 12 (about 0.05 to 0.09) are noise and should not be read as forecasts improving.
- **The error is large in relative terms.** WAPE of 35.8% means the typical miss is about a third of what was sold. The data appears to contain a lot of random variation that none of the available features explain. Tuning gave very small gains (see 3.3), which supports this.
- Consistency checks: horizon 1 here (MAE 28.951) is close to the separate one-day-ahead test result (29.087), and the horizon-1 moving-average error equals the earlier one-day baseline exactly.

---

## 4. Forecast function and testing

`make_forecast(history, future_plan, bundle, origin_date)` builds the features from raw history and predicts all 14 horizons.

- **Safety check:** it refuses to run if the last 7 days before the origin are not complete and consecutive. This was tested by deleting one day on purpose, and the function raised an error.
- **Consistency test:** for the backtest origin 2024-01-16, the function produced 1,400 rows (100 x 14) and its predictions differed from the big training-pipeline predictions by at most **0.0**. Two independent ways of building features agree.
- **Backtest from 2024-01-16** (one origin date, so only a check of the mechanics, not an accuracy estimate): HGB MAE 28.300 / RMSE 36.926 vs moving average 31.504 / 40.755.
- **Deployment model:** same settings, retrained on all 1,045,100 rows (target dates 2022-01-08 to 2024-01-30). **No accuracy score is reported for it**, because no unseen data is left. Its accuracy claim is borrowed from the evaluation model above.
- **Saved files:** `demand_forecaster.joblib` (1.45 MB, model plus feature list and settings) and `forecaster.py` (feature building, forecasting, restock logic and order-list builder, 6.28 KB). Two small lookup files used by the restock logic are saved beside them: `per_product_error.csv` (forecast error of each store-product) and `restock_k_factors.csv` (buffer correction for lead times 1 to 14).
- The fresh-notebook test (below) was run before the later restock additions (Steps 20 to 26). Those additions were tested by reloading `forecaster.py` in the same notebook and comparing with the notebook versions. A second fresh-notebook run with the final files has not been done.
- **Fresh-notebook test:** a brand-new Colab notebook loaded only these two files plus small CSV inputs. It matched the original forecasts (largest difference 1.4e-14, which is floating-point rounding) and gave the same restock count.

---

## 5. Restock recommendation

For each store-product:

```
lead_time_demand = sum of forecast units for days 1..L
safety_stock     = z * daily_error * sqrt(L)
reorder_qty      = ceil( max(0, lead_time_demand + safety_stock - current_stock - on_order) )
```

- `daily_error` = 39.617 (the test RMSE of the multi-horizon model, as the closest measured quantity to a standard deviation of error).
- **Illustrative settings, not from the data:** lead time L = 3 days and z = 1.28 (about a 90% target of not running out during the wait). The dataset contains no supplier lead times, so in real use the shop owner provides L.
- `current_stock` = `Inventory Level` on the origin date. `on_order` was set to 0 because the meaning of `Units Ordered` was not verified.

**Demo results (origin 2024-01-16):** 71 of 100 store-products needed a reorder. This depends entirely on the made-up lead time and service level and on the synthetic inventory values, so it demonstrates that the logic runs. It is not a finding about any real shop.

**Tests performed**

| Test | Result |
|---|---|
| More stock on hand never raises the order | Passed |
| Longer lead time never lowers the order | Passed |
| Higher z never lowers the order | Passed |
| Lead time beyond the 14-day forecast is rejected | Passed |
| Real 3-day sales stayed within forecast + safety buffer | 92.0% of 100 store-products (design target about 90%) |

The last check used one origin date and 100 store-products that share the same 3 days, so it is a rough check, not proof. Sections 5.1 to 5.4 describe later improvements, including a much stronger check over many origin dates.

### 5.1 Order list with priorities

The restock table was turned into a plain order list. Using the forecast, expected demand is added up day by day, and the first day it reaches current stock is the estimated "runs out" day. Each store-product gets a label:

- **URGENT:** needs an order and is expected to run out within the lead time.
- **ORDER:** needs an order, but stock should last until delivery (the safety buffer would be used up).
- **OK:** no order needed.

Demo result (lead time 3, one shared buffer): URGENT 56, ORDER 15, OK 29 (the 71 reorders from before, split by urgency). Total 12,219 units, spread over the stores as S001 13 items / 2,450 units, S002 13 / 1,734, S003 13 / 2,510, S004 15 / 2,643, S005 17 / 2,882. The list is saved as `order_list.csv`.

**Caution:** "URGENT" depends on the made-up lead time, the service level and the synthetic stock levels. That 56 of 100 are urgent says something about this synthetic data, not about real shops. The "runs out" day uses the average forecast, while real daily sales are noisy, so it is an estimate.

### 5.2 Per-product safety stock

One shared buffer (87.8 units) is unfair to products with different error sizes. The forecast error (RMSE) of each store-product was measured on the **validation** block using the evaluation model:

- Per-product error: min 24.6, median 36.5, mean 37.2, max 54.8 units per day. The errors differ by about 2x between products.
- Correlation between a product's sales level and its error: 0.54 (moderate; bigger sellers tend to have bigger errors).
- Safety stock now ranges from 54.4 to 121.4 units (mean 82.6) instead of 87.8 for everyone.
- Result: reorders stay at 71, total units fall from 12,219 to 11,751 (about 4%). Priorities stay at URGENT 56 / ORDER 15 / OK 29.

The per-product buffer is more sensible by design, but the coverage checks below show **no evidence that it covers demand better** than a single buffer.

### 5.3 Buffer check over many origin dates (test period)

The single-date check above (92%) was too thin. The buffer was checked for every origin date in the test period (3-day windows: 5,900; 7-day windows: 5,500). A window counts as covered if real demand stayed at or below forecast plus buffer. Errors for the buffers came from validation (version 1b uses the overall validation RMSE of 37.779, which is the fair comparison for version 1).

| Lead time | Buffer | Covered % |
|---|---|---|
| 3 days | v1 (test RMSE 39.617, slightly unfair) | 89.2 |
| 3 days | v1b (validation RMSE) | 88.4 |
| 3 days | v2 (per product) | 88.2 |
| 7 days | v1 (test RMSE 39.617, slightly unfair) | 84.6 |
| 7 days | v1b (validation RMSE) | 83.7 |
| 7 days | v2 (per product) | 83.4 |

(design target: about 90%)

- **3-day buffer is close to its target, slightly low.** 7-day buffer clearly falls short.
- **Per-product (v2) is a tie with the single buffer (v1b)**, not better.
- Coverage swings a lot between origin dates (3-day, v2: mean 88.2%, standard deviation 6.4 points, min 69%, max 99%), because all products share the same calendar days. This is why one origin date was a weak check.
- The windows overlap and are not independent trials, so the percentages are rough.

### 5.4 Correction for longer lead times (k factor)

The rule `z * daily_error * sqrt(L)` assumes errors on different days are independent. If they are related, errors add up faster and the buffer is too small for longer lead times, which is what the 7-day result shows. A correction factor `k` was fitted for each lead time from 1 to 14 days, **using validation windows only**: k is the value that makes 90% of validation windows covered. It multiplies the safety stock. It was then checked on the test windows, which were not used to fit it.

| Lead time (days) | k | Test coverage without k % | Test coverage with k % |
|---|---|---|---|
| 1 | 0.924 | 90.5 | 89.1 |
| 2 | 0.955 | 89.4 | 88.5 |
| 3 | 0.990 | 88.2 | 87.9 |
| 4 | 1.042 | 87.1 | 88.1 |
| 5 | 1.103 | 85.9 | 87.8 |
| 6 | 1.177 | 85.0 | 88.5 |
| 7 | 1.248 | 83.4 | 88.3 |
| 8 | 1.301 | 82.5 | 88.4 |
| 9 | 1.351 | 81.4 | 88.3 |
| 10 | 1.395 | 80.6 | 88.3 |
| 11 | 1.463 | 79.9 | 88.4 |
| 12 | 1.538 | 79.4 | 88.9 |
| 13 | 1.589 | 78.7 | 88.6 |
| 14 | 1.632 | 78.3 | 88.4 |

- **k rises smoothly with lead time**, which fits the idea that errors on neighbouring days are related. This is supported by the data, not proven.
- Without correction, test coverage falls from 90.5% (1 day) to 78.3% (14 days). With k it stays at about 88% to 89% for every lead time. The steady gap of about 2 points below 90% is most likely because the test period is slightly harder than validation (higher average sales and more spread). A factor fitted on validation cannot remove that.
- **Design choices (mine, not forced by the data):** the module never shrinks a buffer (`max(k, 1.0)`), because for lead times 1 to 3 shrinking made test coverage slightly worse. The k values are valid **only for z = 1.28**, and the module refuses other z values until k is refitted.
- **Cost:** a larger buffer means more stock held. At a 7-day lead time the mean buffer rises from 126.1 to 157.4 units, and total units to order rise from 46,962 to 49,929, with 95 of 100 store-products needing a reorder either way.
- Module tests passed: lead time 7 matches a by-hand calculation; lead time 3 gives results identical to version 2 (k is not applied); a larger buffer never lowers an order; z = 1.65 is rejected.

Final usage of the restock functions in `forecaster.py`:

```python
err = pd.read_csv("per_product_error.csv")                    # forecast error per store-product
k_table = forecaster.load_k_table("restock_k_factors.csv")    # buffer correction per lead time

restock = forecaster.recommend_restock_v3(
    forecast, stock, lead_time_days=7, z=1.28,
    error_table=err, default_error=39.617, k_table=k_table)
order_list = forecaster.build_order_list(forecast, restock, lead_time_days=7)
```

The older functions `recommend_restock` and `recommend_restock_v2` are still in the module.

---

## 6. How to use the saved model on new data

Requirements: `demand_forecaster.joblib` and `forecaster.py` in the same folder, and **the same scikit-learn version that was used to save the model**.

```python
import sys
import pandas as pd
sys.path.insert(0, "/path/to/folder")
import forecaster

bundle = forecaster.load_bundle("/path/to/folder/demand_forecaster.joblib")

# history: columns Store ID, Product ID, Date, Units Sold (at least 7 consecutive days up to the origin)
# plan:    columns Store ID, Product ID, Date, Price, Discount, Promotion (next 14 days, no sales)
# stock:   columns Store ID, Product ID, current_stock
history = pd.read_csv("history.csv", parse_dates=["Date"])
plan = pd.read_csv("plan.csv", parse_dates=["Date"])
stock = pd.read_csv("stock.csv")

origin = pd.Timestamp("2024-01-16")   # the last day with known sales
forecast = forecaster.make_forecast(history, plan, bundle, origin)
restock = forecaster.recommend_restock(forecast, stock, lead_time_days=3, z=1.28, daily_error=39.617)
```

---

## 7. Limitations (state these honestly)

1. **Synthetic data.** Results say nothing about real Indian shops. The high noise level is a property of this dataset.
2. **Large errors.** Typical miss is about 36% of sales (WAPE 35.8%). The model is better than the baselines, but it is not highly accurate.
3. **Future Price, Discount and Promotion must be supplied** by the shop owner as a plan. They are decisions, not forecasts.
4. **Model comparison scope.** Ridge, Random Forest and HGB were compared only on the one-day-ahead problem. The final 14-day model is HGB only.
5. **Same-weekday baseline** was evaluated only for the one-day-ahead case, not for multi-horizon.
6. **No store-product ID feature**, so the model learns each combination's level only through its recent sales (`lag_0`, `lag_6`, `avg_7`). Not tested whether adding one helps.
7. **Restock demo simplifications:** lead time and service level are made up (the dataset has no supplier lead times); `Inventory Level` has unverified meaning; `Units Ordered` is unused (`on_order` is set to 0). The URGENT/ORDER/OK counts depend on these inputs and on the synthetic stock levels.
8. **Feature importance is not causation.** It shows what the model used, not what drives sales.
9. **Safety buffer is only approximately calibrated.** Even with the k correction, test coverage is about 88% to 89% against a 90% target. Per-product buffers did not cover demand better than a single buffer. Windows overlap and all products share the same days, so the coverage percentages are rough, and coverage varied a lot between origin dates (3-day, v2: 69% to 99%). The k factors are fitted for z = 1.28 only, and the errors behind them come from the validation period.
10. **Single-origin backtest.** The 14-day forecast backtest used one origin date (2024-01-16). It checks the mechanics, not accuracy. The accuracy figures come from the full test period.
11. **Version pinning.** The saved model should be loaded with the same scikit-learn version.

---

## 8. Likely examiner questions

**Why not split the data randomly?**
Sales are a time series. A random split would let the model train on days after the ones it is tested on, which would make results look better than they would be in real use. We split by date so that testing always happens on later dates.

**What is target leakage, and how did you avoid it?**
Leakage means giving the model information it would not have at prediction time. We removed `Demand`, excluded `Inventory Level` and `Units Ordered` as features, and built all lag features only from days up to the origin. The forecast function was tested to give identical results to the training pipeline while seeing only history up to the origin.

**Why compare against baselines?**
A baseline is a simple rule with no learning. If a model cannot beat it, the model is not worth using. Here HGB beat the 7-day moving average by about 11% in MAE on the test period.

**Why is the error so large?**
Our evidence points to a high noise floor in the synthetic data. Training and validation errors were almost equal (no overfitting), more trees gave only about 0.02 MAE, and the three different ML models landed within about 1 MAE of each other. That suggests the remaining error is mostly unexplained randomness rather than a modeling mistake. This is an interpretation supported by those results, not something we proved.

**Why HistGradientBoosting?**
It had the lowest validation MAE and RMSE in the one-day-ahead comparison. The advantage over Random Forest was small, so we do not claim it is truly better, only that it is slightly ahead and fast.

**How are 7 to 14 day forecasts made without future sales?**
By a direct multi-horizon model. The `horizon` feature tells the model how many days ahead to predict, and every feature comes from days up to the origin plus the planned Price, Discount and Promotion of the target day. No predictions are fed back in. A recursive approach would instead predict day 1, use that prediction as input for day 2, and so on, which compounds errors. We did not use it.

**Why is the error higher for farther horizons?**
Older information (the last known sales) is less useful the farther ahead you look. In our test, MAE grew from about 29.0 at day 1 to about 32.3 at day 12, then stayed flat.

**Why is the final model retrained on all data, and why is it not scored?**
In real use, more recent data helps. But once it has seen all the data, no unseen period remains to score it on, and scoring it on its training data would look unrealistically good. So the reported accuracy comes from the evaluation model trained on the training block only, with identical settings.

**What does the safety stock do, and how did you size it?**
It is a buffer against forecast errors. We use `z * error * sqrt(L) * k`: z = 1.28 aims for about 90% coverage, `error` is each store-product's own forecast RMSE measured on the validation block, L is the lead time, and k is a correction fitted on validation windows (1.0 or higher, never shrinking the buffer). The lead time and z are illustrative, not from the data. Checked over every origin date in the test period, the plain `sqrt(L)` buffer covered about 88% of 3-day windows but only about 83% of 7-day windows. With k, coverage was about 88% to 89% at every lead time from 1 to 14 days, still slightly under the 90% target, most likely because the test period is harder than validation.

**Why did you add the k correction?**
The `sqrt(L)` rule assumes forecast errors on different days are independent. In the data, coverage fell as lead time grew (90.5% at 1 day to 78.3% at 14 days on test), which suggests errors on neighbouring days are related, so errors add up faster than `sqrt(L)`. k was fitted on validation only and then checked on test, so the improvement is not an artifact of fitting on the test set. The explanation is supported by the results, not proven.

**Did the per-product safety stock help?**
It is more sensible by design: errors differ by about 2x between store-products (24.6 to 54.8 units per day). But in the coverage check it did not beat a single shared buffer (88.2% vs 88.4% for 3-day windows), so we do not claim it is more accurate.

**What would you improve next?**
Add a store-product ID feature, evaluate a multi-horizon version of the same-weekday baseline, refit the buffer errors on a longer and more recent window, fit k for other service levels (other z values), take lead times per product from the shop owner, and test on real shop data. Also rerun the fresh-notebook test with the final files.

---

## 9. Next step

Integrate with the Spring Boot backend and MySQL database. Practical options are to wrap `forecaster.py` in a small Python service (for example FastAPI) that the backend calls, or to export the model to a format Java can load. Pin the scikit-learn version in either case.
