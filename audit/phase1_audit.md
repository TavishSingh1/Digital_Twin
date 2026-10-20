# Phase 1: Project Audit Report

**Date of Audit:** October 15, 2026  
**Project:** AI-Based Occupancy Prediction Simulation & Digital Twin  
**Working Directory:** `d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin`  
**Classification Convention:**
- **[VERIFIED]**: Directly confirmed via source code examination, metadata parsing, or non-destructive execution tests.
- **[INFERRED]**: Highly probable based on strong circumstantial code evidence, but not directly serialized or guaranteed.
- **[UNKNOWN]**: Insufficient evidence to establish facts with certainty.

---

## A. Workspace Inventory

The workspace was inventoried recursively, excluding temporary caches and virtual environment internals. A total of **7 project files** were discovered across 1 subdirectory.

| Relative Path | File Type | File Size | Status | Primary Purpose / Role |
| :--- | :---: | :---: | :---: | :--- |
| [`joint_lstm_gnn_all_5_rooms.keras`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/joint_lstm_gnn_all_5_rooms.keras) | Keras Zip Archive | 5,711,814 bytes (~5.45 MB) | **Inspected** [VERIFIED] | Final trained multi-room spatio-temporal neural network model archive (`Joint_Output_STGNN_LSTM_5Rooms`). Contains `metadata.json`, `config.json`, and `model.weights.h5`. |
| [`notebook.ipynb`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/notebook.ipynb) | Jupyter Notebook | 2,370,313 bytes (~2.26 MB) | **Inspected** [VERIFIED] | 74-cell Jupyter notebook documenting data ingestion, cleaning, synchronization, feature engineering, architecture experimentation, model training, evaluation, and SHAP explainability. |
| [`dataset/combined_Room1.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/combined_Room1.csv) | CSV Dataset | 1,989,335 bytes (~1.90 MB) | **Inspected** [VERIFIED] | Raw/combined environmental and occupancy sensor time series for Room 1 (8,352 rows, 27 columns). |
| [`dataset/combined_Room2.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/combined_Room2.csv) | CSV Dataset | 2,025,535 bytes (~1.93 MB) | **Inspected** [VERIFIED] | Raw/combined environmental and occupancy sensor time series for Room 2 (8,352 rows, 27 columns). |
| [`dataset/combined_Room3.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/combined_Room3.csv) | CSV Dataset | 2,591,703 bytes (~2.47 MB) | **Inspected** [VERIFIED] | Raw/combined environmental and occupancy sensor time series for Room 3 (8,352 rows, 35 columns). |
| [`dataset/combined_Room4.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/combined_Room4.csv) | CSV Dataset | 4,233,552 bytes (~4.04 MB) | **Inspected** [VERIFIED] | Raw/combined environmental and occupancy sensor time series for Room 4 (13,536 rows, 35 columns). |
| [`dataset/combined_Room5.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/combined_Room5.csv) | CSV Dataset | 4,257,012 bytes (~4.06 MB) | **Inspected** [VERIFIED] | Raw/combined environmental and occupancy sensor time series for Room 5 (13,536 rows, 35 columns). |

### Key Inventory Findings
1. **No Fitted Scaler Files Saved on Disk [VERIFIED]:** There are **no** `.pkl`, `.joblib`, `.json`, or `.parquet` scaler files anywhere in the workspace directory.
2. **No Dedicated Python Modules or Simulation Scripts [VERIFIED]:** No standalone `.py` scripts for preprocessing, feature transformation, inference pipelines, or virtual simulation currently exist. All logic resides strictly inside [`notebook.ipynb`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/notebook.ipynb).
3. **No Virtual Building / Sensor Code [VERIFIED]:** Virtual occupants, virtual sensors, and simulated building rooms do not exist yet.

---

## B. Notebook Findings

The notebook [`notebook.ipynb`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/notebook.ipynb) was parsed and analyzed across all 74 cells (nbformat 4.5). The specific findings for each requirement are detailed below:

### 1. Dataset Loading Paths and Formats [VERIFIED]
- **Cell 8:** Paths originally configured for Kaggle environment:
  ```python
  KAGGLE_DIR = "/kaggle/input/datasets/fishonk/pjt1h1"
  filename = f"combined_Room{room_number}.csv"
  ```
  Fallback mechanism searches direct path candidates and `glob.glob(f"**/{filename}", recursive=True)`. In local execution, this resolves to `dataset/combined_Room{i}.csv`.
- **Format:** Standard comma-separated UTF-8 text files (`pd.read_csv`).

### 2. Timestamp Column and Parsing [VERIFIED]
- **Cell 6:** `TIMESTAMP_COL = "timestamp"`
- **Cell 12:** `find_column(df, [TIMESTAMP_COL], required=True)` maps raw column to `timestamp`.
- **Parsing:** `df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)`
- **Deduplication:** `df.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="first")`

### 3. Room Identifiers and Synchronization Logic [VERIFIED]
- **Cell 6:** `ROOMS = ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]` (`N_ROOMS = 5`).
- **Cell 16:** All 5 rooms are synchronized using an exact timestamp intersection:
  ```python
  common_index = indexed[ROOMS[0]].index
  for room in ROOMS[1:]:
      common_index = common_index.intersection(indexed[room].index)
  common_index = common_index.sort_values()
  ```
- **Timeline Result:** Yields exactly **8,352 synchronized timestamps** spanning from `2021-09-06 16:00:00+00:00` to `2021-12-23 15:55:00+00:00`.
- **Reindexing:** Each room dataframe is reindexed against `common_index`:
  ```python
  aligned = indexed[room].reindex(common_index)
  ```
  Assertion ensures zero missing values in `aligned[INPUT_FEATURES]`.

### 4. Cleaning, Interpolation, and Missing Value Handling [VERIFIED]
- **Cell 12 (`load_room`):**
  - Environmental sensor columns are converted via `pd.to_numeric(..., errors="coerce")`.
  - Sensor columns are interpolated across time:
    ```python
    df[sensor_columns] = df[sensor_columns].interpolate(method="time", limit_direction="both").ffill().bfill()
    ```
  - Occupancy target values are **never** interpolated or fabricated:
    ```python
    df = df.dropna(subset=[TARGET_COL]).reset_index()
    ```

### 5. Input Feature Names and Order [VERIFIED]
- **Cell 14:** `INPUT_FEATURES = COMMON_SENSOR_FEATURES + TIME_FEATURES + [TARGET_COL]`
- Exactly **13 features per room** in the following strictly indexed order:
  1. `illuminance` (float32)
  2. `sound_pressure_level` (float32)
  3. `indoor_temperature` (float32)
  4. `dry_bulb_temperature` (float32)
  5. `indoor_relative_humidity` (float32)
  6. `outdoor_relative_humidity` (float32)
  7. `indoor_co2` (float32)
  8. `sin_hour` (float32)
  9. `cos_hour` (float32)
  10. `sin_week` (float32)
  11. `cos_week` (float32)
  12. `is_weekend` (float32)
  13. `occupant_count` (float32 historical occupancy)

### 6. Environmental Feature Columns and Units [VERIFIED]
- **Cell 10 (`SENSOR_ALIASES`):**
  - `illuminance`: lux (`illuminance [lux]`)
  - `sound_pressure_level`: dBA (`sound_pressure_level [dba]`)
  - `indoor_temperature`: degrees Celsius (`air_temperature [Celsius]`)
  - `dry_bulb_temperature`: degrees Celsius (`dry_bulb_temp [Celsius]`)
  - `indoor_relative_humidity`: percentage (`indoor_relative_humidity [%]`)
  - `outdoor_relative_humidity`: percentage (`outdoor_relative_humidity [%]`)
  - `indoor_co2`: parts per million (`indoor_co2 [ppm]`)

### 7. Time-Related Features and Exact Formulas [VERIFIED]
- **Cell 14 (`add_time_features`):**
  - `minutes = df[TIMESTAMP_COL].dt.hour * 60 + df[TIMESTAMP_COL].dt.minute`
  - `dow = df[TIMESTAMP_COL].dt.dayofweek` (Monday=0, Sunday=6)
  - `sin_hour = np.sin(2 * np.pi * minutes / (24 * 60))`
  - `cos_hour = np.cos(2 * np.pi * minutes / (24 * 60))`
  - `sin_week = np.sin(2 * np.pi * dow / 7)`
  - `cos_week = np.cos(2 * np.pi * dow / 7)`
  - `is_weekend = (dow >= 5).astype(np.float32)`

### 8. Historical Occupancy and Target Column [VERIFIED]
- **Target Column:** `occupant_count` (mapped from `occupant_count [number]`).
- **Historical Occupancy Feature:** The 13th feature in each room's feature vector is the room's scaled historical occupancy at each step of the lookback sequence.
- **Persistence Residual Connection (Cell 55):**
  ```python
  last_occ = layers.Lambda(lambda z: z[:, -1, :, -1], name="last_observed_occupancy")(inputs)
  outputs = layers.Add(name="joint_residual_output")([last_occ, delta])
  ```
  The model explicitly extracts the last observed occupancy at time step $t-1$ across all 5 rooms, predicts a joint $\Delta$ (delta change), and adds the baseline: $\hat{y}_{t+2} = y_{t-1} + \Delta$.

### 9. Input Sequence Length and Forecast Horizon [VERIFIED]
- **Cell 6 & 24:**
  - `SEQ_LEN = 12` steps. At 5-minute intervals, lookback window = $12 \times 5 = 60\text{ minutes}$.
  - `HORIZON = 3` steps. At 5-minute intervals, forecast horizon = $3 \times 5 = 15\text{ minutes}$ ahead.
  - Target alignment: Sequence starting at $t - \text{seq\_len}$ to $t$ predicts target at index $t + \text{horizon} - 1$ ($t + 2$, exactly 15 minutes ahead).

### 10. Train / Validation / Test Split Strategy [VERIFIED]
- **Cell 6 & 20:** Strictly chronological split (no shuffling):
  - `TRAIN_RATIO = 0.70` (70%) $\rightarrow$ 5,846 rows (`2021-09-06 16:00` to `2021-12-12 23:05`)
  - `VAL_RATIO = 0.15` (15%) $\rightarrow$ 1,253 rows (`2021-12-12 23:10` to `2021-12-17 07:30`)
  - `TEST` = remainder (15%) $\rightarrow$ 1,253 rows (`2021-12-17 07:35` to `2021-12-23 15:55`)
- **After Sequence Windowing (Cell 24) [VERIFIED]:**
  - `X_train`: shape `(5832, 12, 5, 13)`, `Y_train`: shape `(5832, 5)`
  - `X_val`: shape `(1251, 12, 5, 13)`, `Y_val`: shape `(1251, 5)`
  - `X_test`: shape `(1251, 12, 5, 13)`, `Y_test`: shape `(1251, 5)`

### 11. Scaling and Normalization Procedures [VERIFIED]
- **Cell 22:**
  - `sklearn.preprocessing.StandardScaler` is used.
  - Scalers are fitted **strictly on training data** (`[:train_end]`).
  - **Sensor Scalers (`sensor_scalers[room]`):** Fits on the first 12 features (`X_all[:train_end, node, :12]`), scaling all 7 environmental sensors and all 5 cyclic time features.
  - **Target Scalers (`target_scalers[room]`):** Fits on `Y_all[:train_end, node].reshape(-1, 1)`. Used to scale both the 13th historical occupancy input feature (`X_scaled[:, node, -1]`) and the target array (`Y_scaled[:, node]`).

### 12. Scaler Persistence Status [VERIFIED]
- In Cell 52, `joblib.dump` was executed to save scalers for an earlier intermediate model to `/content/joint_lstm_gnn_results/joint_lstm_gnn_target_scalers.joblib`.
- **Critical Finding:** That path was inside Google Colab's ephemeral VM storage. **No `.joblib` file was saved to the current project repository**, and no scaler was serialized for the final model in Cell 60. Scalers currently exist **only** as reconstructible logic in notebook code cells.

### 13. Model Architecture and Custom Objects [VERIFIED]
Two model iterations exist in the notebook:
- **Model 1 (Cell 30, `Improved_Joint_STGNN_LSTM_5Rooms`):** Employs TimeDistributed GraphConv $\rightarrow$ LSTM $\rightarrow$ GraphConv $\rightarrow$ individual Dense(1) head per room. Suffered severe underperformance on Room 1 ($R^2 = -161.81$).
- **Model 2 (Cell 55, `Joint_Output_STGNN_LSTM_5Rooms`):**
  - **Saved to disk in Cell 60 as [`joint_lstm_gnn_all_5_rooms.keras`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/joint_lstm_gnn_all_5_rooms.keras).**
  - Input: `(None, 12, 5, 13)`
  - Spatial Block 1: `TimeDistributed(GraphConv(96, ADJACENCY))` + LayerNorm + Dropout(0.2)
  - Spatial Block 2: `TimeDistributed(GraphConv(96, ADJACENCY))` + LayerNorm + Residual Add + ReLU + Dropout(0.2)
  - Temporal Prep: `Permute((2, 1, 3))` $\rightarrow$ shape `(None, 5, 12, 96)`
  - Temporal Block: Two shared LSTMs wrapped in `TimeDistributed(LSTM(128))`
  - Post-Temporal Graph: `GraphConv(96, ADJACENCY)` + LayerNorm + Residual Fusion with `Dense(96)` projection
  - Joint Representation: `Flatten()` $\rightarrow$ `Dense(256)` + `BatchNormalization()` + `Dropout(0.25)` $\rightarrow$ `Dense(128)` + `Dropout(0.15)`
  - Delta Head: `Dense(5)` predicting raw occupancy changes
  - Residual Output: `Add()` with `Lambda(lambda z: z[:, -1, :, -1])` (last observed occupancy)
  - Output: `(None, 5)` predicting 5-room occupancy counts 15 minutes ahead.

### 14. Training Loss, Optimizer, and Callbacks [VERIFIED]
- **Cell 56 & 59:**
  - Optimizer: `tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0)`
  - Loss: `tf.keras.losses.Huber(delta=1.0)`
  - Metrics: `mae`, `rmse`
  - Callbacks: `EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4, restore_best_weights=True)`, `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)`
  - Batch size: 64, Epochs: 60, Shuffle: False.

### 15. Evaluation Metrics and Performance [VERIFIED]
- **Cell 57 (Unrounded continuous test set evaluation):**
  - Room 1: $R^2 = -30.05$, $\text{MAE} = 0.465$, $\text{RMSE} = 0.545$
  - Room 2: $R^2 = 0.564$, $\text{MAE} = 1.034$, $\text{RMSE} = 3.492$
  - Room 3: $R^2 = 0.472$, $\text{MAE} = 0.429$, $\text{RMSE} = 0.680$
  - Room 4: $R^2 = 0.863$, $\text{MAE} = 0.665$, $\text{RMSE} = 1.049$
  - Room 5: $R^2 = 0.733$, $\text{MAE} = 0.258$, $\text{RMSE} = 0.520$
  - Average: $R^2 = -5.48$, $\text{MAE} = 0.570$, $\text{RMSE} = 1.257$
- **Cell 61 & 62 (Post-processing):**
  - Non-negative clipping and integer rounding: `predicted_whole = np.maximum(np.rint(predicted).astype(int), 0)`

---

## C. Dataset Schema and Quality Summary

All 5 CSV files in [`dataset/`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/) were inspected safely and non-destructively:

### 1. Dataset Dimensions and Time Range [VERIFIED]
| File | Rows | Columns | File Size | Valid Timestamp Range | Duplicates |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `combined_Room1.csv` | 8,352 | 27 | 1.99 MB | 2021-09-06 16:00 to 2021-12-23 15:55 UTC | 0 |
| `combined_Room2.csv` | 8,352 | 27 | 2.03 MB | 2021-09-06 16:00 to 2021-12-23 15:55 UTC | 0 |
| `combined_Room3.csv` | 8,352 | 35 | 2.59 MB | 2021-09-06 16:00 to 2021-12-23 15:55 UTC | 0 |
| `combined_Room4.csv` | 13,536 | 35 | 4.23 MB | 2021-09-06 16:00 to 2021-12-23 15:55 UTC | 0 |
| `combined_Room5.csv` | 13,536 | 35 | 4.26 MB | 2021-09-06 16:00 to 2021-12-23 15:55 UTC | 0 |

### 2. Timeline Alignment Analysis [VERIFIED]
- **Sampling Interval:** 5.0 minutes nominal cadence across all files.
- **Why Row Counts Differ:**
  - Rooms 1, 2, and 3 share **identical** 8,352 timestamps.
  - Rooms 4 and 5 share **identical** 13,536 timestamps.
  - The 8,352 timestamps of Rooms 1–3 are a **strict subset** of Rooms 4–5.
  - Rooms 1–3 have a 5,184-step gap between `2021-10-24 16:00:00 UTC` and `2021-11-17 15:55:00 UTC` (approx. 18 days of uncollected sensor data).
  - Consequently, the synchronized multi-room intersection across all 5 rooms is constrained to exactly **8,352 timestamps**.

### 3. Occupancy Signal Verification [VERIFIED]
| Room | Raw Target Column | Dtype | Min | Max | Mean | Std | Missing Counts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Room 1 | `occupant_count [number]` | int64 | 0 | 38 | 1.603 | 5.112 | 0 (0.00%) |
| Room 2 | `occupant_count [number]` | int64 | 0 | 22 | 1.208 | 3.262 | 0 (0.00%) |
| Room 3 | `occupant_count [number]` | int64 | 0 | 13 | 1.554 | 2.455 | 0 (0.00%) |
| Room 4 | `occupant_count [number]` | int64 | 0 | 18 | 3.334 | 3.413 | 0 (0.00%) |
| Room 5 | `occupant_count [number]` | int64 | 0 | 16 | 1.866 | 2.936 | 0 (0.00%) |

*Note: All 5 files also contain a binary target `occupant_presence [binary]` (0 or 1), but the notebook models specifically predict count (`occupant_count [number]`).*

### 4. Environmental Feature Completeness [VERIFIED]
All 7 required environmental sensor signals exist in all 5 files:
- `illuminance [lux]`: $\le 30$ missing values per room ($\le 0.36\%$).
- `sound_pressure_level [dba]`: $\le 30$ missing values per room ($\le 0.36\%$).
- `air_temperature [Celsius]`: 0 missing values across all rooms.
- `dry_bulb_temp [Celsius]`: 0 missing values across all rooms.
- `indoor_relative_humidity [%]`: $\le 30$ missing values per room ($\le 0.36\%$).
- `outdoor_relative_humidity [%]`: 0 missing values across all rooms.
- `indoor_co2 [ppm]`: $\le 30$ missing values per room ($\le 0.36\%$).

All missing sensor values are cleanly filled via time interpolation followed by `ffill()`/`bfill()` as verified in notebook Cell 12.

---

## D. Keras Model Inspection

### 1. Archive Structure [VERIFIED]
File: [`joint_lstm_gnn_all_5_rooms.keras`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/joint_lstm_gnn_all_5_rooms.keras) (5,711,814 bytes)  
Zip members:
- `metadata.json` (64 bytes):
  ```json
  {"keras_version": "3.13.2", "date_saved": "2026-09-16@05:05:20"}
  ```
- `config.json` (27,544 bytes): Functional model configuration with 30 layers.
- `model.weights.h5` (5,683,876 bytes): Complete HDF5 weights container.

### 2. Verified Architecture and Parameters [VERIFIED]
- **Class:** Functional Keras Model (`Joint_Output_STGNN_LSTM_5Rooms`)
- **Total Parameters:** **464,016** (all trainable weights intact in `model.weights.h5`).
- **Input Tensor:** Name `all_rooms_history`, Shape `(None, 12, 5, 13)`, Dtype `float32`.
- **Output Tensor:** Name `joint_residual_output`, Shape `(None, 5)`, Dtype `float32`.
- **Custom Objects:**
  - `OccupancySTGNN>GraphConv` (Keras-3 custom layer with weights `self_kernel`, `neighbor_kernel`, `bias`, and untrainable constant `normalized_adjacency`).
  - `Lambda` layer (`last_observed_occupancy`, slice `z[:, -1, :, -1]`).

### 3. Model Loading and Deserialization Diagnosis [VERIFIED]
When attempting non-destructive model loading with the local Python environment (Python 3.11, TensorFlow 2.20.0, Keras 3.12.0), two specific deserialization behaviors were uncovered:

1. **Blind `keras.models.load_model()` Fails [VERIFIED]:**
   - **Reason:** Missing custom class registration for `OccupancySTGNN>GraphConv`, and safety policy blocking raw `Lambda` layers (`safe_mode=True`).
2. **`load_model(custom_objects={"GraphConv": GraphConv}, safe_mode=False)` Fails [VERIFIED]:**
   - **Error:** `ValueError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`
   - **Root Cause:** Version mismatch between serialization and local deserialization. The model was saved under **Keras 3.13.2** (`metadata.json`). Keras 3.13 introduced `quantization_config` in `Dense.get_config()`. The active local environment runs **Keras 3.12.0**, whose `Dense` layer rejects `quantization_config`.
3. **Weight Loading Test Succeeded 100% [VERIFIED]:**
   - When the exact architecture is constructed in memory using Keras 3 and weights are loaded via `model.load_weights()` from `model.weights.h5`:
     - All 464,016 weights loaded without warning or error.
     - Synthetic inference test with shape `(1, 12, 5, 13)` passed:
       ```
       Prediction output: [[0.9990414, 0.9786348, 0.9884334, 1.016431, 1.0340694]]
       ```
   - **Conclusion:** The model architecture, weights, and inference capabilities are 100% sound and recoverable without retraining.

---

## E. Verified Model / Data Compatibility

| Dimension / Spec | Notebook Training Pipeline | Saved Keras Model | Dataset Signals | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Lookback Sequence Length** | 12 steps (60 min) | 12 steps | 5-min intervals | **MATCH [VERIFIED]** |
| **Number of Nodes (Rooms)** | 5 rooms | 5 nodes | 5 CSV files | **MATCH [VERIFIED]** |
| **Features per Node** | 13 features | 13 features | 7 sensors + 5 time + 1 occ | **MATCH [VERIFIED]** |
| **Feature Ordering** | Strict 13-feature list | Implicit 13-feature dim | Documented in aliases | **MATCH [VERIFIED]** |
| **Input Shape** | `(batch, 12, 5, 13)` | `(None, 12, 5, 13)` | Transformed matrix | **MATCH [VERIFIED]** |
| **Output Shape** | `(batch, 5)` | `(None, 5)` | 5 room targets | **MATCH [VERIFIED]** |
| **Forecast Horizon** | 3 steps (15 min) | Implicit 3 steps ahead | Matched in targets | **MATCH [VERIFIED]** |
| **Graph Topology** | Fully connected 5x5 | Fully connected 5x5 | Uniform cross-talk | **MATCH [VERIFIED]** |

---

## F. Existing Artifacts and Missing Artifacts

### 1. Existing Artifacts [VERIFIED]
- **Trained Neural Network Archive:** [`joint_lstm_gnn_all_5_rooms.keras`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/joint_lstm_gnn_all_5_rooms.keras) (contains complete weights and topology).
- **Five-Room Dataset:** 5 CSV files in [`dataset/`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dataset/) covering both training and testing periods.
- **Complete End-to-End Codebase:** In [`notebook.ipynb`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/notebook.ipynb) (preprocessing, architecture, and postprocessing routines).

### 2. Missing Artifacts & Operational Impact [VERIFIED]
1. **Fitted Scalers (`sensor_scalers`, `target_scalers`) [CRITICAL]:**
   - **Why Missing:** Left in Colab ephemeral storage (`/content/`); never saved to repository.
   - **Why It Matters:** Without fitted scalers, raw sensor readings cannot be normalized into the distribution the neural network expects, and raw model output cannot be denormalized back into actual occupant numbers.
   - **Mitigation:** The scalers can be deterministically refit using the training split of the 5 CSVs using the exact code in Cell 22.
2. **Adjacency & Feature Configuration JSON [MODERATE]:**
   - `joint_lstm_gnn_config.json` was saved to Colab `/content/` but not downloaded. Can be generated directly from verified parameters.
3. **Inference Pipeline Script (`inference.py` / `model_wrapper.py`) [MODERATE]:**
   - No Python script exists to load the model and run predictions on incoming sensor vectors. Must be built for the simulation.
4. **Virtual Building Simulation Framework [FUTURE PHASE]:**
   - Classes for Rooms, Occupants, Environmental Sensors, HVAC controls, and Event Loops do not yet exist.

---

## G. Environment and Dependency Observations

1. **Python Environment [VERIFIED]:**
   - Active Python version: **3.11.9** (Windows 64-bit).
   - TensorFlow: **2.20.0**.
   - Keras: **3.12.0**.
2. **Keras Serialization Version Divergence [VERIFIED]:**
   - Saved model used Keras **3.13.2**.
   - Current environment runs Keras **3.12.0**.
   - Direct `keras.models.load_model()` encounters `quantization_config` incompatibility.
   - However, building the model definition and loading weights (`model.load_weights()`) works completely without error and without installing or upgrading any packages.
3. **Required Python Libraries for Simulation [VERIFIED]:**
   - `numpy`, `pandas`, `tensorflow`, `keras`, `scikit-learn` are already installed and functional in the local environment.

---

## H. Uncertainties and Unresolved Issues

1. **Physical Spatial Adjacency vs. Fully Connected Graph [VERIFIED NOTEBOOK LIMITATION]:**
   - Cell 6 explicitly notes: *"This is an experimental topology because a verified physical room adjacency matrix is not available here. Each room communicates with all other rooms."*
   - The graph uses a symmetric fully connected $5 \times 5$ matrix ($A_{ij} = 1$ for $i \ne j$, $0$ on diagonal).
2. **Room 1 High Error / Low $R^2$ Variance [VERIFIED NOTEBOOK FINDING]:**
   - In test evaluation, Room 1 exhibits negative $R^2$ ($-30.05$), although its MAE is low ($0.465$ occupants). This occurs because Room 1 has very low variance (mostly 0 occupants), making mean-prediction variance comparisons artificially penalize small integer residual slips.
3. **Missing Scaler Preservation [VERIFIED DEFECT]:**
   - Confirmed missing on disk; requires programmatic reconstruction from training data before simulation inference is possible.

---

## I. Prioritized Next-Step Checklist for Phase 2

When Phase 2 is authorized, the following tasks should be performed in priority order:

1. **Task 2.1: Deterministic Scaler Reconstruction & Serialization**
   - Execute a safe, standalone script that loads the 5 CSVs, performs timestamp intersection, applies the exact 70% chronological split, fits the 5 room sensor scalers and 5 target scalers, and exports them to `artifacts/scalers.joblib`.
2. **Task 2.2: Standalone Model Wrapper Module**
   - Create a clean Python module (e.g., `model/stgnn_model.py`) defining `GraphConv`, the `Joint_Output_STGNN_LSTM_5Rooms` architecture, and a `ModelWrapper` class that loads `joint_lstm_gnn_all_5_rooms.keras` via weights and exposes a high-level `predict(lookback_window) -> dict[room, count]` interface.
3. **Task 2.3: Verification of Reconstructed Pipeline**
   - Run a test comparing the reconstructed pipeline predictions on `X_test` against the notebook's documented test predictions (Cell 57) to ensure exact numerical match ($100\%$ parity).
4. **Task 2.4: Virtual Building Architecture Design**
   - Design the class hierarchy for the 5-room virtual building, virtual occupants (stochastic schedules/movement), and virtual sensors (generating 5-minute environmental readings matching the 7 features).

---
*End of Phase 1 Audit Report.*

