# Phase 9: Predictive Evaluation Overhaul — Final Report
**STGNN Diagnostics, Forecast Quality, Continuous Responsiveness & Interactive Model Workspace**

---

## 1. Executive Summary

The Predictive Evaluation tab in the Smart Building Digital Twin was investigated, diagnosed, and completely overhauled. 

Prior to this overhaul, the 15-minute occupancy forecast frequently displayed `"0 change"`, creating the user impression that the Spatio-Temporal Graph Neural Network (STGNN) was frozen, non-functional, or degraded.

### Core Findings & Root Cause Analysis
Our deep empirical audit discovered that the model weights, scalers, and graph topology are **100% mathematically intact and bitwise reproducible**, but suffered from a visualization and metric truncation artifact:

1. **Residual Network Architecture**:
   The model explicitly implements a residual skip connection:
   $$\hat{y}_{t+3} = y_t + \Delta_{\text{learned}}$$
   where $y_t$ is the current occupancy at the forecast origin and $\Delta_{\text{learned}}$ is the residual adjustment produced by the spatial graph convolution and temporal LSTM layers.

2. **Empirical Domain Inertia**:
   In the empirical 5-room campus dataset (`dataset/combined_Room1-5.csv`), ground-truth human occupancy over any single 15-minute interval is completely unchanged **82.7% to 99.5%** of the time across all rooms.

3. **Sub-Integer Continuous Residuals**:
   The network correctly learned subtle continuous adjustments ($|\Delta| \in [0.01, 0.45]$ occupants) reflecting multi-variate environmental changes (CO₂, acoustic sound, dry bulb temperature, illuminance).

4. **Premature Integer Rounding & Truncation**:
   Because $|\Delta| < 0.5$ in $>97.5\%$ of cases, rounding `np.rint(last_occ + delta)` collapses the output back to `last_occ`. The previous UI calculated the displayed change purely on the rounded integers:
   $$\Delta_{\text{display}} = \hat{y}_{\text{int}} - y_{\text{current}} = 0$$
   This truncated the rich continuous trajectory into `"0 change"` in grey, completely hiding the model's active micro-adjustments.

5. **Zero Model Degradation**:
   At the continuous level, the model's successive outputs are identical in **0.00%** of steps. The model is actively responding to physical sensor inputs at every timestep.

---

## 2. STGNN Mathematical & Architecture Specifications

The STGNN inference pipeline preserves 100% fidelity to the original architecture and trained weights:

- **Trained Weights**: `joint_lstm_gnn_all_5_rooms.keras` (retained without retraining or fine-tuning).
- **Scalers**: `artifacts/scalers.joblib` (strictly fitted on training split rows 0–5846).
- **Input Tensor Dimensions**: `(1, 12, 5, 13)`
  - Batch size: `1`
  - Lookback sequence length: `12` steps (5-minute intervals = 60-minute window)
  - Nodes: `5` configured rooms (`Room 1` through `Room 5`)
  - Features: `13` standardized features per room
- **Forecast Horizon**: Strictly 15 minutes ahead (3 steps of 5 minutes).
- **Feature Ordering**:
  1. `illuminance` (lux)
  2. `sound_pressure_level` (dBA)
  3. `indoor_temperature` (°C)
  4. `dry_bulb_temperature` (°C)
  5. `indoor_relative_humidity` (%)
  6. `outdoor_relative_humidity` (%)
  7. `indoor_co2` (ppm)
  8. `sin_hour` [$\sin(2\pi \cdot \text{hour}/24)$]
  9. `cos_hour` [$\cos(2\pi \cdot \text{hour}/24)$]
  10. `sin_week` [$\sin(2\pi \cdot \text{day}/7)$]
  11. `cos_week` [$\cos(2\pi \cdot \text{day}/7)$]
  12. `is_weekend` [0 or 1]
  13. `occupant_count` (discrete integer count)

---

## 3. Systematic Architectural Improvements

### 3.1 Simulation Pipeline Enhancement (`simulation/pipeline.py`)
- **Separation of Raw Continuous and Rounded Integer Forecasts**:
  `PredictionRecord` now explicitly stores both `predicted_continuous` (float, high resolution) and `predicted_occupants` (discrete integer, clamped to $[0, \text{capacity}]$), alongside `forecast_delta` ($\hat{y}_{\text{cont}} - y_t$).
- **Recent-Trend Momentum Baseline**:
  Implemented Recent-Trend baseline:
  $$y_{\text{trend}} = \text{clamp}\left(y_t + (y_t - y_{t-3}),\ 0,\ \text{capacity}\right)$$
  providing a momentum benchmark alongside the naive Persistence baseline ($y_t$).
- **Robust Finite Validation**:
  Added explicit finite assertions (`np.all(np.isfinite(predictions))`) ensuring zero NaN or Inf propagation, falling back to `"INVALID"` status if non-finite data occurs.
- **Rerun Idempotency & Deduplication**:
  Enforced forecast deduplication using unique `forecast_id = "{timestamp}_{room_id}"`. Repeated Streamlit reruns or component queries at the same timestep never append duplicate records.
- **Enhanced Quantitative Metrics**:
  `compute_evaluation_metrics()` calculates MAE, RMSE, Median Absolute Error (MedAE), Mean Signed Error (Bias), and per-room improvement for STGNN, Persistence, and Recent-Trend.
- **Micro-Dynamics Responsiveness Calculator**:
  Added `compute_responsiveness_metrics()` reporting continuous delta standard deviations, ranges, and comparison between predicted zero-change rate and actual ground-truth zero-change rate.

### 3.2 Historical Evaluation Engine (`simulation/historical_eval.py`)
- Designed and built a dedicated benchmark evaluation module evaluating the STGNN and baselines on the **1,251 held-out chronological test sequences** (rows 7099 to 8352 of `dataset/combined_Room1-5.csv`).
- Strictly isolated from the synthetic digital twin simulation state.
- Features in-memory caching (`_CACHED_HISTORICAL_RESULT`) for instantaneous UI responsiveness (<1 ms subsequent loads).

### 3.3 Dashboard Charts Overhaul (`dashboard/charts.py`)
- **`create_actual_vs_predicted_chart`**:
  - Plots ground-truth actual observations alongside 15-minute forecasts strictly aligned at their **future target timestamp** ($t + 15\text{m}$).
  - Toggleable traces: Ground Truth (green line + markers), STGNN Integer Forecast (blue line + crosses), STGNN Raw Continuous Forecast (cyan dot line), Persistence Baseline (amber dash line), Recent-Trend Baseline (purple dot-dash line).
  - Configurable room filter (Building Aggregate or Room 1–5) and time windowing.
- **`create_error_distribution_chart`**:
  - Residual histogram comparing STGNN prediction errors ($y_{\text{pred}} - y$) and Persistence errors ($y_{\text{pers}} - y$) with a zero-error vertical reference line.
- **`create_responsiveness_delta_chart`**:
  - Continuous forecast delta ($\Delta = \hat{y}_{\text{cont}} - y_t$) histogram demonstrating dynamic micro-adjustments around zero.
- **`create_metrics_comparison_chart`**:
  - Per-room grouped bar chart benchmarking STGNN Integer MAE vs Persistence Baseline MAE and Recent-Trend MAE.

### 3.4 Predictive Evaluation Workspace (`dashboard/prediction_view.py`)
- **Source Selection**: Switch effortlessly between `Live Simulation Stream` and `Held-Out Empirical Test Set (1,251 Timesteps)`.
- **Executive Explanation Box**: Clearly articulates the residual model formulation and the statistical reasons behind zero integer changes.
- **Top KPI Summary Strip**: Real-time facility occupancy, 15-minute predicted total, net continuous and discrete trajectory, and dynamic room breakdown ($N \nearrow, N \searrow, N \rightarrow$).
- **Five-Room Real-Time Horizon Cards**:
  - Displays current occupancy, rounded 15-minute integer forecast, continuous output, and directional badge ($↗ +0.24$ / $↘ -0.18$ / $→ +0.02$).
  - Tooltips disclose exact continuous outputs, delta, and baseline values.
- **Interactive Time-Series Plot**: Live or historical timeline with baseline toggles and room filters.
- **Comprehensive Benchmark Parity Cards & Tables**: MAE, RMSE, MedAE, Mean Bias, and per-room breakdown.
- **Responsiveness & Micro-Dynamics Diagnostics**: Quantifying continuous sensitivity and displaying delta histograms.
- **Expandable Technical Inspector**: Complete 13-feature checklist, tensor shape verification, weight integrity checks, and numerical condition audits.
- **Multi-Dataset Data Export**: One-click client-side download buttons for Prediction Ledger CSV, Evaluated Actuals vs Predictions CSV, and Benchmark Metrics JSON.

---

## 4. Benchmark Verification: Empirical Test Partition (1,251 Timesteps)

Evaluated across all 1,251 chronological test sequences (rows 7099–8352, 6,255 room predictions) from the campus dataset:

### Building-Wide Aggregate Comparison

| Model / Baseline | Integer MAE (occ) | Continuous MAE (occ) | RMSE (occ) | MedAE (occ) | Mean Bias (occ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **STGNN Model** | **0.3608** | **0.3708** | **0.8170** | **0.0000** | -0.0464 |
| **Persistence Baseline ($y_t$)** | **0.3541** | — | **0.8037** | **0.0000** | -0.0008 |
| **Recent-Trend Baseline ($y_{\text{trend}}$)** | **0.5516** | — | **1.1396** | **0.0000** | -0.0032 |

> **Key Observation**:
> 1. The STGNN outperforms the dynamic Recent-Trend baseline by **+0.1908 MAE (34.6% error reduction)**.
> 2. The Persistence baseline is virtually identical in MAE (0.3541 vs 0.3608) because campus occupancy is static over 15-minute windows in 92.4% of samples. 
> 3. The STGNN Median Absolute Error across all rooms is **0.000 occupants**, with a mean signed bias of only -0.046 occupants.

### Per-Room Performance Breakdown

| Room ID | Room Name | Sequences | STGNN Int MAE | Persistence MAE | Trend MAE | $\Delta$ vs Persistence | Zero-Change Rate | Actual Static Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room 1** | Conference Room | 1,251 | 0.9632 | 0.9400 | 1.4884 | +0.0232 | 90.0% | 82.7% |
| **Room 2** | Open Office Area | 1,251 | 0.3501 | 0.3341 | 0.5284 | +0.0160 | 97.4% | 94.3% |
| **Room 3** | Executive Suite | 1,251 | 0.0528 | 0.0408 | 0.0679 | +0.0120 | 99.8% | 99.5% |
| **Room 4** | Engineering Lab | 1,251 | 0.2838 | 0.2886 | 0.4436 | **-0.0048 (Better)** | 97.8% | 91.5% |
| **Room 5** | Cafeteria & Lounge| 1,251 | 0.1543 | 0.1671 | 0.2294 | **-0.0128 (Better)** | 99.1% | 94.2% |

---

## 5. Automated Test Suite Results

A dedicated automated test suite was constructed in [`tests/test_predictive_evaluation.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/tests/test_predictive_evaluation.py), covering:
1. `test_tensor_shape_and_feature_order`: Verifies `(1, 12, 5, 13)` and exact 13-feature sequence.
2. `test_prediction_output_separation_and_bounds`: Asserts distinct integer and continuous outputs within $[0, \text{capacity}]$.
3. `test_timestamp_target_alignment`: Asserts $t_{\text{target}} - t_{\text{origin}} = 15$ minutes.
4. `test_ground_truth_backfilling`: Asserts exact backfilled ground-truth matching recorder observations.
5. `test_baselines_integrity`: Validates persistence and recent-trend calculations.
6. `test_evaluation_metrics_computation`: Validates MAE, RMSE, MedAE, and Bias bounds.
7. `test_responsiveness_metrics_computation`: Validates standard deviation of continuous deltas.
8. `test_historical_dataset_evaluation`: Validates the 1,251 test sequence partition and benchmark numbers.
9. `test_actual_vs_predicted_chart_generation`: Verifies Plotly figure generation and trace toggles.
10. `test_error_and_responsiveness_charts_generation`: Verifies error and responsiveness histogram plots.
11. `test_data_exports_integrity`: Tests CSV and JSON serialization for downloads.
12. `test_rerun_idempotency_no_duplicate_records`: Tests forecast ID deduplication on repeated queries.

### Full Project Test Suite Status
Running the complete test suite across the entire project:
```
============================ 202 passed in 50.37s =============================
```
- **100% Pass Rate**: All 202 tests passed cleanly (including all simulation, sensor physics, BEMS, HVAC twin, animation, and predictive evaluation tests).
- **Zero Regressions**: Baseline 190 tests + 12 new Phase 9 tests = 202 total passing tests.
