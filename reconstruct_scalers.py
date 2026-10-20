import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

ROOMS = ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]
TIMESTAMP_COL = "timestamp"
TARGET_COL = "occupant_count"

SEQ_LEN = 12
HORIZON = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SENSOR_ALIASES = {
    "illuminance": ["illuminance", "illuminance [lux]"],
    "sound_pressure_level": ["sound_pressure_level", "sound_pressure_level [dba]"],
    "indoor_temperature": ["indoor_temperature", "air_temperature", "air_temperature [celsius]", "indoor air temperature"],
    "dry_bulb_temperature": ["dry_bulb_temp", "dry_bulb_temp [celsius]", "dry bulb temperature"],
    "indoor_relative_humidity": ["indoor_relative_humidity", "indoor_relative_humidity [%]"],
    "outdoor_relative_humidity": ["outdoor_relative_humidity", "outdoor_relative_humidity [%]"],
    "indoor_co2": ["indoor_co2", "indoor_co2 [ppm]"],
}

TARGET_ALIASES = [
    "occupant_count",
    "occupant_count [number]",
    "occupant count",
    "occupancy_count",
]

def normalize_name(name):
    return " ".join(str(name).strip().lower().replace("_", " ").split())

def find_column(df, aliases, required=True):
    normalized = {normalize_name(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    for alias in aliases:
        key = normalize_name(alias)
        for normalized_col, original_col in normalized.items():
            if key in normalized_col or normalized_col in key:
                return original_col
    if required:
        raise KeyError(f"Could not find any of: {aliases}")
    return None

def load_room(room_name, path):
    df = pd.read_csv(path).copy()
    rename_map = {}
    
    timestamp_source = find_column(df, [TIMESTAMP_COL], required=True)
    if timestamp_source != TIMESTAMP_COL:
        rename_map[timestamp_source] = TIMESTAMP_COL

    target_source = find_column(df, TARGET_ALIASES, required=True)
    rename_map[target_source] = TARGET_COL

    for standard_name, aliases in SENSOR_ALIASES.items():
        source = find_column(df, aliases, required=False)
        if source is not None:
            rename_map[source] = standard_name

    df = df.rename(columns=rename_map)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="first").copy()

    numeric_columns = [TARGET_COL] + [c for c in SENSOR_ALIASES if c in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sensor_columns = [c for c in SENSOR_ALIASES if c in df.columns]
    if not sensor_columns:
        raise ValueError(f"{room_name}: no usable environmental sensor columns found.")

    df = df.set_index(TIMESTAMP_COL)
    df[sensor_columns] = df[sensor_columns].interpolate(method="time", limit_direction="both").ffill().bfill()
    df = df.dropna(subset=[TARGET_COL]).reset_index()
    return df

def add_time_features(df):
    df = df.copy()
    minutes = df[TIMESTAMP_COL].dt.hour * 60 + df[TIMESTAMP_COL].dt.minute
    dow = df[TIMESTAMP_COL].dt.dayofweek
    df["sin_hour"] = np.sin(2 * np.pi * minutes / (24 * 60))
    df["cos_hour"] = np.cos(2 * np.pi * minutes / (24 * 60))
    df["sin_week"] = np.sin(2 * np.pi * dow / 7)
    df["cos_week"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(np.float32)
    return df

print("1. Loading raw dataset files...")
room_dfs = {}
for i, room in enumerate(ROOMS):
    fpath = f"dataset/combined_Room{i+1}.csv"
    room_dfs[room] = load_room(room, fpath)
    room_dfs[room] = add_time_features(room_dfs[room])
    print(f"   {room}: {len(room_dfs[room])} rows")

COMMON_SENSOR_FEATURES = [
    "illuminance",
    "sound_pressure_level",
    "indoor_temperature",
    "dry_bulb_temperature",
    "indoor_relative_humidity",
    "outdoor_relative_humidity",
    "indoor_co2",
]
TIME_FEATURES = ["sin_hour", "cos_hour", "sin_week", "cos_week", "is_weekend"]
INPUT_FEATURES = COMMON_SENSOR_FEATURES + TIME_FEATURES + [TARGET_COL]

print(f"\n2. Total input features per room ({len(INPUT_FEATURES)}):")
print(f"   {INPUT_FEATURES}")

print("\n3. Synchronizing rooms on common timeline...")
indexed = {}
for room in ROOMS:
    selected = room_dfs[room][[TIMESTAMP_COL] + INPUT_FEATURES].copy()
    selected = selected.set_index(TIMESTAMP_COL).sort_index().drop_duplicates()
    indexed[room] = selected

common_index = indexed[ROOMS[0]].index
for room in ROOMS[1:]:
    common_index = common_index.intersection(indexed[room].index)
common_index = common_index.sort_values()

print(f"   Common synchronized timestamps: {len(common_index)}")
assert len(common_index) == 8352, f"Expected 8352, got {len(common_index)}"

X_room = []
Y_room = []
for room in ROOMS:
    aligned = indexed[room].reindex(common_index)
    assert not aligned[INPUT_FEATURES].isna().any().any(), f"NaN in aligned {room}"
    X_room.append(aligned[INPUT_FEATURES].to_numpy(dtype=np.float32))
    Y_room.append(aligned[TARGET_COL].to_numpy(dtype=np.float32))

X_all = np.stack(X_room, axis=1) # [8352, 5, 13]
Y_all = np.stack(Y_room, axis=1) # [8352, 5]
print(f"   X_all: {X_all.shape}, Y_all: {Y_all.shape}")

n = len(common_index)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
print(f"\n4. Chronological Split:")
print(f"   Train: 0 to {train_end} ({train_end} rows)")
print(f"   Val  : {train_end} to {val_end} ({val_end - train_end} rows)")
print(f"   Test : {val_end} to {n} ({n - val_end} rows)")

print("\n5. Fitting scalers STRICTLY on training data...")
n_sensor_features = len(INPUT_FEATURES) - 1 # 12
sensor_scalers = {}
target_scalers = {}

for node, room in enumerate(ROOMS):
    # Fit sensor scaler on first 12 features of train split
    sensor_scaler = StandardScaler()
    sensor_scaler.fit(X_all[:train_end, node, :n_sensor_features])
    
    # Fit target scaler on target column of train split
    target_scaler = StandardScaler()
    target_scaler.fit(Y_all[:train_end, node].reshape(-1, 1))

    sensor_scalers[room] = sensor_scaler
    target_scalers[room] = target_scaler

    print(f"   [{room}]")
    print(f"      Target Scaler: mean={target_scaler.mean_[0]:.4f}, scale={target_scaler.scale_[0]:.4f}")
    print(f"      Sensor Scaler: feature means[:3]={sensor_scaler.mean_[:3].round(3)}, scales[:3]={sensor_scaler.scale_[:3].round(3)}")

artifact_payload = {
    "sensor_scalers": sensor_scalers,
    "target_scalers": target_scalers,
    "rooms": ROOMS,
    "input_features": INPUT_FEATURES,
    "common_sensor_features": COMMON_SENSOR_FEATURES,
    "time_features": TIME_FEATURES,
    "target_col": TARGET_COL,
    "timestamp_col": TIMESTAMP_COL,
    "train_rows": train_end,
    "total_synchronized_rows": n,
    "seq_len": SEQ_LEN,
    "horizon": HORIZON,
}

scaler_output_path = "artifacts/scalers.joblib"
joblib.dump(artifact_payload, scaler_output_path)
print(f"\n6. Successfully saved reconstructed scalers to {scaler_output_path} ({os.path.getsize(scaler_output_path):,} bytes)")

