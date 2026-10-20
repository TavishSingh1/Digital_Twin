import json
import re
import numpy as np
import pandas as pd
from stgnn_model import STGNNInference, ROOMS, INPUT_FEATURES, TIMESTAMP_COL, TARGET_COL, SEQ_LEN, HORIZON
from reconstruct_scalers import load_room, add_time_features

# 1. Parse Cell 62 HTML table from notebook
with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell62 = nb['cells'][62]
html_table = None
for out in cell62.get('outputs', []):
    if 'data' in out and 'text/html' in out['data']:
        html_table = ''.join(out['data']['text/html'])
        break

assert html_table is not None, "Could not find HTML table in Cell 62"

# Parse table rows using regex
rows = re.findall(r'<tr>(.*?)</tr>', html_table, re.DOTALL)
header_row = rows[0]
headers = re.findall(r'<t[hd]>(.*?)</t[hd]>', header_row)[1:] # skip index header

table_data = []
for r in rows[1:]:
    cells = re.findall(r'<t[hd]>(.*?)</t[hd]>', r)
    if not cells or cells[0] == '...':
        continue
    idx = int(cells[0])
    vals = [int(x) for x in cells[1:]]
    table_data.append((idx, vals))

print(f"Parsed {len(table_data)} rows from Cell 62 HTML table:")
for idx, vals in table_data:
    print(f"  Row {idx:4d}: {vals}")

# 2. Run our inference pipeline on test sequences
infer = STGNNInference()

# Load synchronized dataset
room_dfs = {r: add_time_features(load_room(r, f"dataset/combined_Room{i+1}.csv")) for i, r in enumerate(ROOMS)}
indexed = {r: room_dfs[r][[TIMESTAMP_COL] + INPUT_FEATURES].set_index(TIMESTAMP_COL).sort_index().drop_duplicates() for r in ROOMS}
common_index = indexed[ROOMS[0]].index
for r in ROOMS[1:]: common_index = common_index.intersection(indexed[r].index)
common_index = common_index.sort_values()

X_all = np.stack([indexed[r].reindex(common_index)[INPUT_FEATURES].to_numpy(dtype=np.float32) for r in ROOMS], axis=1)
Y_all = np.stack([indexed[r].reindex(common_index)[TARGET_COL].to_numpy(dtype=np.float32) for r in ROOMS], axis=1)

n = len(common_index)
val_end = int(n * 0.85)
n_sensor_features = len(INPUT_FEATURES) - 1

X_scaled = np.zeros_like(X_all, dtype=np.float32)
Y_scaled = np.zeros_like(Y_all, dtype=np.float32)
for node, r in enumerate(ROOMS):
    X_scaled[:, node, :n_sensor_features] = infer.sensor_scalers[r].transform(X_all[:, node, :n_sensor_features]).astype(np.float32)
    X_scaled[:, node, -1] = infer.target_scalers[r].transform(Y_all[:, node].reshape(-1, 1)).reshape(-1).astype(np.float32)
    Y_scaled[:, node] = infer.target_scalers[r].transform(Y_all[:, node].reshape(-1, 1)).reshape(-1).astype(np.float32)

def make_sequences(X, Y, start, end, seq_len, horizon):
    Xs, Ys, positions = [], [], []
    first_t = max(start + seq_len, seq_len)
    last_t = end - horizon + 1
    for t in range(first_t, last_t):
        target_t = t + horizon - 1
        Xs.append(X[t - seq_len:t])
        Ys.append(Y[target_t])
        positions.append(target_t)
    return np.asarray(Xs, dtype=np.float32), np.asarray(Ys, dtype=np.float32), np.asarray(positions, dtype=np.int64)

X_test, Y_test, _ = make_sequences(X_scaled, Y_scaled, val_end - SEQ_LEN, n, SEQ_LEN, HORIZON)

pred_int, pred_cont = infer.predict(X_test)
actual_cont = infer.inverse_transform_targets(Y_test)
actual_int = infer.postprocess_integer_counts(actual_cont)

print("\n=== Direct Comparison with Cell 62 Table Rows ===")
match_count = 0
total_compared = 0

for idx, nb_vals in table_data:
    # Construct our row: [R1 Act, R1 Pred, R2 Act, R2 Pred, R3 Act, R3 Pred, R4 Act, R4 Pred, R5 Act, R5 Pred]
    our_vals = []
    for r_idx in range(5):
        our_vals.append(actual_int[idx, r_idx])
        our_vals.append(pred_int[idx, r_idx])
    
    is_match = (our_vals == nb_vals)
    if is_match:
        match_count += 1
    total_compared += 1
    print(f"Row {idx:4d}: Match={is_match}")
    print(f"  Notebook: {nb_vals}")
    print(f"  Our code: {our_vals}")

print(f"\nResult: {match_count} / {total_compared} rows match exactly ({match_count/total_compared*100:.1f}%)")
