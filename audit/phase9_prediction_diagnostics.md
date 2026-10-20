# Phase 9: STGNN Prediction Diagnostics & Root Cause Analysis

**Audit Date**: October 20, 2026  
**Artifacts Inspected**:
- [`stgnn_model.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/stgnn_model.py)
- [`simulation/pipeline.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/simulation/pipeline.py)
- [`simulation/engine.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/simulation/engine.py)
- [`artifacts/scalers.joblib`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/artifacts/scalers.joblib)
- [`joint_lstm_gnn_all_5_rooms.keras`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/joint_lstm_gnn_all_5_rooms.keras)
- [`dataset/combined_Room*.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/)
- [`notebook.ipynb`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/notebook.ipynb) (Cells 40–62)

---

## 1. Executive Diagnostic Summary

The user reported that the 15-minute occupancy forecast in the Predictive Evaluation workspace *"almost always shows zero change."*

Our investigation confirms this observation and establishes the exact chain of mathematical, architectural, and presentation causes:

1. **The Model Is Neither Frozen nor Stuck**:
   On every 5-minute timestep, the STGNN model receives changing 13-feature input tensors and computes **unique floating-point outputs** ($0.00\%$ identical successive raw forecasts).
2. **Residual Network Architecture ($Output = Last\_Occ + \Delta$)**:
   In `stgnn_model.py` (lines 238–261), the neural network architecture incorporates an explicit residual identity connection:
   $$\hat{y}_{t+3} = y_t + \Delta_{\text{STGNN}}(X_{t-11:t})$$
   The model does not predict absolute occupancy from scratch; it predicts an incremental delta ($\Delta$) on top of the last observed normalized occupancy $y_t$.
3. **High Empirical Inertia in Training Distribution**:
   In the empirical campus dataset (`dataset/combined_Room1-5.csv`), ground-truth occupancy over 15-minute intervals is overwhelmingly stationary:
   - **Room 1**: 99.52% of 15-minute steps have $\Delta_{\text{actual}} = 0$.
   - **Room 2**: 82.73% of 15-minute steps have $\Delta_{\text{actual}} = 0$.
   - **Room 3**: 93.29% of 15-minute steps have $\Delta_{\text{actual}} = 0$.
   - **Room 4**: 65.07% of 15-minute steps have $\Delta_{\text{actual}} = 0$.
   - **Room 5**: 90.89% of 15-minute steps have $\Delta_{\text{actual}} = 0$.
   Trained with MSE loss, the network learned that predicting small continuous deltas ($|\Delta| \in [0.01, 0.45]$ occupants) minimizes expected risk.
4. **Premature Integer Rounding ($\text{round}(y_t + \Delta)$)**:
   Because the continuous delta magnitude is almost always $< 0.5$ occupants, the notebook post-processing step `np.rint(pred_continuous)` rounds $y_t + \Delta$ back to $y_t$.
5. **Dashboard Conflation of Integer Display with Model Output**:
   The dashboard UI only showed the rounded integer and computed `delta = pred_int - curr_occ`. When `pred_int == curr_occ`, it displayed `"0 change"`. The genuine continuous model output, delta gradient, and time trends were suppressed.

---

## 2. Empirical Evidence & Statistical Diagnostics

### 2.1 Live Digital Twin Simulation Run ($N = 90$ evaluated steps per room)

In a 100-step simulation run ($T = 08:00$ to $16:20$ simulated time), the model produced the following output distributions:

| Room ID | Room Name | Raw Mean | Raw Std | Raw Min | Raw Max | Unique Raw | Mean Raw Delta ($\hat{y} - y_t$) | Std Raw Delta | Zero Rounded Change Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room 1** | Conference / Lobby | 1.780 | 2.364 | 0.018 | 7.170 | **90 / 90** | **+0.1137** | 0.064 | **100.00%** |
| **Room 2** | Open Office A | 11.159 | 3.714 | 5.028 | 16.219 | **88 / 90** | **+0.2589** | 0.099 | **100.00%** |
| **Room 3** | Suite | 1.113 | 1.764 | -0.187 | 4.903 | **90 / 90** | **-0.0755** | 0.079 | **100.00%** |
| **Room 4** | Open Office B | 6.217 | 1.624 | 2.529 | 10.457 | **89 / 90** | **+0.3501** | 0.128 | **88.89%** |
| **Room 5** | Lounge | 2.698 | 4.202 | 0.067 | 11.498 | **89 / 90** | **+0.3643** | 0.101 | **98.89%** |

**Key Finding**:
Across all 5 rooms, the raw model output is continuously varying on every step ($0.0\%$ identical successive raw outputs). However, the mean absolute continuous delta is between $0.07$ and $0.36$ occupants. Because $0.36 < 0.50$, integer rounding collapses $97.78\%$ of all predictions to the exact persistence value ($0$ integer change).

### 2.2 Historical Empirical Test Set ($N = 1251$ sequences)

Evaluating the model on the held-out historical test split ($15\%$ chronological holdout, 1251 sequences) reveals the identical behavior:

| Room ID | Actual Ground Truth Mean | Actual Ground Truth Std | Ground Truth 15m Zero Change % | Model Raw Delta Mean | Model Raw Delta Std | Model Rounded Zero Change % | STGNN Int MAE | Persistence MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room 1** | 0.005 | 0.098 | **99.52%** | +0.109 | 0.127 | **100.00%** | 0.0096 | 0.0096 |
| **Room 2** | 1.965 | 5.288 | **82.73%** | +0.006 | 0.086 | **100.00%** | 0.9800 | 0.9800 |
| **Room 3** | 0.400 | 0.936 | **93.29%** | +0.140 | 0.130 | **98.48%** | 0.1287 | 0.1135 |
| **Room 4** | 2.717 | 2.839 | **65.07%** | +0.088 | 0.294 | **90.09%** | 0.5596 | 0.5212 |
| **Room 5** | 0.668 | 1.007 | **90.89%** | +0.056 | 0.088 | **100.00%** | 0.1263 | 0.1263 |

---

## 3. Scaling & Preprocessing Verification

We checked the preprocessing and inverse transformation pipeline:

1. **Sensor Scalers**:
   Fitted strictly on training partition ($N = 5,846$ rows). Each room has an independent `StandardScaler` transforming features 0 through 11:
   `['illuminance', 'sound_pressure_level', 'indoor_temperature', 'dry_bulb_temperature', 'indoor_relative_humidity', 'outdoor_relative_humidity', 'indoor_co2', 'sin_hour', 'cos_hour', 'sin_week', 'cos_week', 'is_weekend']`.
2. **Target Scalers**:
   Each room has an independent target scaler transforming feature 12 (`occupant_count`):
   - Room 1: Mean = 2.2215, Scale = 5.9571
   - Room 2: Mean = 0.9550, Scale = 2.0417
   - Room 3: Mean = 1.8414, Scale = 2.6605
   - Room 4: Mean = 3.6445, Scale = 3.6010
   - Room 5: Mean = 2.0462, Scale = 3.0621
3. **Inverse Scaling Verification**:
   The denormalization formula in `stgnn_model.py`:
   $$y_{\text{denorm}} = y_{\text{norm}} \times \text{scale} + \text{mean}$$
   exactly matches scikit-learn's `inverse_transform`. Target scaling is not erroneously applied to sensor features, and sensor scalers are not applied to target outputs.
4. **Numerical Parity**:
   Re-verified against Notebook Cell 62: **100.0% sample match rate** on whole-number predictions with 0 maximum error.

---

## 4. Forecast Timestamps & Alignment Verification

1. **Lookback Sequence**:
   - $t - 11$ to $t$ (12 observations at 5-minute intervals = 60 minutes).
2. **Forecast Origin ($t_0$)**:
   - Time of current logical clock step when inference executes.
3. **Target Timestamp ($t_{\text{target}}$)**:
   - Exactly $t_0 + 15\text{ minutes}$ (3 steps $\times$ 5 min).
4. **Causality & Backfilling**:
   - At step $t_0$, `actual_occupants_at_target` is strictly `None`.
   - When the simulation clock reaches $t_0 + 15\text{ minutes}$, `actual_occupants_at_target` is backfilled from the ground truth recorded at that step. No future information is ever leaked to the input tensor.

---

## 5. Required Architectural & Dashboard Enhancements

To address this issue without altering model weights or violating physical causality:

1. **Prediction Representation**:
   Decouple continuous model output (`raw_prediction`), integer count (`display_prediction`), current ground truth (`current_occupancy`), and continuous forecast change (`forecast_delta`).
2. **Dashboard Card Display**:
   Display both:
   - The integer estimate (`display_prediction`)
   - The high-resolution continuous prediction (`raw_prediction`, e.g. `4.26 occ`)
   - The continuous forecast delta with direction badge ($\Delta = +0.26$, $\nearrow$ Accumulating, $\searrow$ Dissipating, $\rightarrow$ Steady)
3. **Interactive Aligned Time-Series Plot**:
   Plot forecasts at their **target timestamp** ($t+15\text{m}$) against ground truth and persistence, with series visibility toggles, tooltips, and time-window controls.
4. **Dual Data-Source Evaluation**:
   Allow the operator to evaluate the STGNN against:
   - **Live Simulation Stream** (real-time cyber-physical twin)
   - **Historical Empirical Benchmark** (held-out test set from physical sensors)
5. **Multiple Baselines**:
   Compare STGNN against both **Persistence** ($y_{t+3} = y_t$) and **Recent-Trend Extrapolation** ($y_{t+3} = y_t + (y_t - y_{t-3})$).
6. **Detailed Error Analysis & Responsiveness Diagnostics**:
   Provide signed error distributions, MAE/RMSE by occupancy tier, and responsiveness metrics.
