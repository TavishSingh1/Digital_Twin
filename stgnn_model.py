"""
STGNN Multi-Room Occupancy Prediction Model & Inference Pipeline
================================================================
Joint Improved Spatio-Temporal Graph Neural Network + LSTM for 5 Rooms.

Preserves:
- Exact Model Architecture: Joint_Output_STGNN_LSTM_5Rooms
- Input shape: (batch, 12, 5, 13) [Lookback: 12 steps x 5 min = 60 minutes]
- Output shape: (batch, 5) [Forecast horizon: 3 steps x 5 min = 15 minutes ahead]
- Exact feature ordering (13 features per room)
- Weight loading from verified Keras archive
- Preprocessing, normalization, inverse scaling, and nonnegative integer clipping
"""

import os
import zipfile
import tempfile
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

# =====================================================================
# GLOBAL CONFIGURATION & METADATA
# =====================================================================

ROOMS = ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]
N_ROOMS = len(ROOMS)

TIMESTAMP_COL = "timestamp"
TARGET_COL = "occupant_count"

SEQ_LEN = 12       # 12 x 5 minutes = 60-minute lookback sequence
HORIZON = 3        # 3 x 5 minutes = 15-minute forecast horizon
CADENCE_MINUTES = 5

COMMON_SENSOR_FEATURES = [
    "illuminance",
    "sound_pressure_level",
    "indoor_temperature",
    "dry_bulb_temperature",
    "indoor_relative_humidity",
    "outdoor_relative_humidity",
    "indoor_co2",
]

TIME_FEATURES = [
    "sin_hour",
    "cos_hour",
    "sin_week",
    "cos_week",
    "is_weekend",
]

# Strict 13-feature ordering verified from notebook training pipeline
INPUT_FEATURES = COMMON_SENSOR_FEATURES + TIME_FEATURES + [TARGET_COL]

# Fully-connected 5-node spatial adjacency matrix (0 on diagonal, 1 off-diagonal)
ADJACENCY = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
], dtype=np.float32)


# =====================================================================
# GRAPH UTILITIES & CUSTOM KERAS LAYER
# =====================================================================

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Computes symmetric degree normalization with self-loops:
    A_hat = A + I
    A_norm = D^(-1/2) * A_hat * D^(-1/2)
    """
    A = np.asarray(A, dtype=np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    degree = A_hat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    normalized = d_inv_sqrt[:, None] * A_hat * d_inv_sqrt[None, :]
    return normalized.astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="OccupancySTGNN")
class GraphConv(layers.Layer):
    """
    Keras 3 compatible Graph Convolution layer.
    Computes: Z = (X * W_self) + (A_norm * X * W_neighbor) + b
    """
    def __init__(self, units: int, adjacency: np.ndarray, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation_name = activation
        self.adjacency = np.asarray(adjacency, dtype=np.float32)
        self.normalized_adjacency = normalize_adjacency(self.adjacency)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.self_kernel = self.add_weight(
            name="self_kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.neighbor_kernel = self.add_weight(
            name="neighbor_kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.adj_tensor = self.add_weight(
            name="normalized_adjacency",
            shape=self.normalized_adjacency.shape,
            initializer=tf.keras.initializers.Constant(self.normalized_adjacency),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: [batch, nodes, features]
        self_part = tf.matmul(inputs, self.self_kernel)
        neighbor_features = tf.einsum("ij,bjf->bif", self.adj_tensor, inputs)
        neighbor_part = tf.matmul(neighbor_features, self.neighbor_kernel)
        output = self_part + neighbor_part + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "adjacency": self.adjacency.tolist(),
            "activation": self.activation_name,
        })
        return config


# =====================================================================
# MODEL ARCHITECTURE BUILDER
# =====================================================================

def build_joint_output_stgnn(
    seq_len: int = SEQ_LEN,
    n_nodes: int = N_ROOMS,
    n_features: int = len(INPUT_FEATURES),
    gnn_hidden: int = 96,
    lstm_units: int = 128,
    dropout: float = 0.20,
) -> Model:
    """
    Constructs the exact Joint_Output_STGNN_LSTM_5Rooms architecture
    matching the trained model serialized in joint_lstm_gnn_all_5_rooms.keras.
    """
    inputs = layers.Input(
        shape=(seq_len, n_nodes, n_features),
        name="all_rooms_history",
    )

    # Spatial Block 1: Graph convolution across rooms at every timestep
    x = layers.TimeDistributed(
        GraphConv(gnn_hidden, ADJACENCY, activation="relu", name="graph_1"),
        name="graph_over_time_1",
    )(inputs)
    x = layers.LayerNormalization(axis=-1, name="layer_normalization_6")(x)
    x = layers.Dropout(dropout, name="dropout_5")(x)

    # Spatial Block 2: Second Graph convolution with residual connection
    x2 = layers.TimeDistributed(
        GraphConv(gnn_hidden, ADJACENCY, activation="relu", name="graph_2"),
        name="graph_over_time_2",
    )(x)
    x2 = layers.LayerNormalization(axis=-1, name="layer_normalization_7")(x2)
    x = layers.Add(name="add_2")([x, x2])
    x = layers.Activation("relu", name="activation_2")(x)
    x = layers.Dropout(dropout, name="dropout_6")(x)

    # Temporal Preparation: Permute to (batch, nodes, seq_len, hidden)
    x = layers.Permute((2, 1, 3), name="rooms_first")(x)

    # Shared Temporal LSTM 1
    x = layers.TimeDistributed(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=0.10,
            name="shared_lstm_1",
        ),
        name="time_distributed_2",
    )(x)
    x = layers.LayerNormalization(axis=-1, name="layer_normalization_8")(x)

    # Shared Temporal LSTM 2
    x = layers.TimeDistributed(
        layers.LSTM(
            lstm_units,
            return_sequences=False,
            dropout=0.10,
            name="shared_lstm_2",
        ),
        name="time_distributed_3",
    )(x)
    x = layers.LayerNormalization(axis=-1, name="layer_normalization_9")(x)
    x = layers.Dropout(dropout, name="dropout_7")(x)

    # Post-Temporal Graph Reasoning
    graph_x = GraphConv(
        gnn_hidden,
        ADJACENCY,
        activation="relu",
        name="final_graph",
    )(x)
    graph_x = layers.LayerNormalization(axis=-1, name="layer_normalization_10")(graph_x)

    # Residual Temporal-Spatial Fusion
    temporal_x = layers.Dense(
        gnn_hidden,
        activation=None,
        name="temporal_projection",
    )(x)
    x = layers.Add(name="add_3")([temporal_x, graph_x])
    x = layers.Activation("relu", name="activation_3")(x)
    x = layers.LayerNormalization(axis=-1, name="layer_normalization_11")(x)

    # Persistence baseline: Last observed occupancy across all rooms (t - 1)
    last_occ = layers.Lambda(
        lambda z: z[:, -1, :, -1],
        name="last_observed_occupancy",
    )(inputs)

    # Joint Room Representation Head
    x = layers.Flatten(name="joint_room_embedding")(x)
    x = layers.Dense(256, activation="relu", name="joint_dense_1")(x)
    x = layers.BatchNormalization(name="joint_batch_norm")(x)
    x = layers.Dropout(0.25, name="dropout_8")(x)
    x = layers.Dense(128, activation="relu", name="joint_dense_2")(x)
    x = layers.Dropout(0.15, name="dropout_9")(x)

    # Joint Occupancy Delta Prediction
    delta = layers.Dense(
        n_nodes,
        activation=None,
        name="joint_occupancy_delta",
    )(x)

    # Final Output = Baseline + Delta
    outputs = layers.Add(name="joint_residual_output")([last_occ, delta])

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="Joint_Output_STGNN_LSTM_5Rooms",
    )


# =====================================================================
# TIME FEATURE ENGINEERING HELPER
# =====================================================================

def compute_time_features_df(timestamps: pd.Series) -> pd.DataFrame:
    """
    Computes the 5 cyclic time features exactly as implemented in the notebook:
    - sin_hour, cos_hour
    - sin_week, cos_week
    - is_weekend
    """
    ts = pd.to_datetime(timestamps, utc=True)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    dow = ts.dt.dayofweek

    df_time = pd.DataFrame(index=ts.index)
    df_time["sin_hour"] = np.sin(2 * np.pi * minutes / (24 * 60)).astype(np.float32)
    df_time["cos_hour"] = np.cos(2 * np.pi * minutes / (24 * 60)).astype(np.float32)
    df_time["sin_week"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df_time["cos_week"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    df_time["is_weekend"] = (dow >= 5).astype(np.float32)
    return df_time


# =====================================================================
# STANDALONE INFERENCE ENGINE
# =====================================================================

class STGNNInference:
    """
    High-level inference engine for 5-room occupancy forecasting.
    Loads trained weights directly from the .keras zip archive, integrates
    reconstructed sensor & target scalers, and exposes clean prediction APIs.
    """
    def __init__(
        self,
        model_path: str = "joint_lstm_gnn_all_5_rooms.keras",
        scalers_path: str = "artifacts/scalers.joblib",
    ):
        self.model_path = model_path
        self.scalers_path = scalers_path

        # 1. Load fitted scalers
        if not os.path.exists(self.scalers_path):
            raise FileNotFoundError(
                f"Scalers artifact not found at '{self.scalers_path}'. "
                "Run reconstruct_scalers.py first."
            )
        scaler_payload = joblib.load(self.scalers_path)
        self.sensor_scalers = scaler_payload["sensor_scalers"]
        self.target_scalers = scaler_payload["target_scalers"]
        self.input_features = scaler_payload.get("input_features", INPUT_FEATURES)
        self.rooms = scaler_payload.get("rooms", ROOMS)

        # 2. Build model architecture in memory
        self.model = build_joint_output_stgnn(
            seq_len=SEQ_LEN,
            n_nodes=N_ROOMS,
            n_features=len(self.input_features),
        )

        # 3. Load weights safely from the Keras archive
        self._load_weights_from_archive(self.model_path)

    def _load_weights_from_archive(self, archive_path: str):
        """Extracts model.weights.h5 safely to a temporary location and loads weights."""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Model archive not found: {archive_path}")

        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp_path = tmp.name
            with zipfile.ZipFile(archive_path, "r") as zf:
                if "model.weights.h5" not in zf.namelist():
                    raise KeyError("Archive does not contain 'model.weights.h5'")
                tmp.write(zf.read("model.weights.h5"))

        try:
            self.model.load_weights(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def predict_normalized(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Runs model forward pass on already scaled features.
        Parameters:
            X_normalized: np.ndarray of shape (batch, 12, 5, 13)
        Returns:
            np.ndarray of shape (batch, 5) in normalized target scale
        """
        X = np.asarray(X_normalized, dtype=np.float32)
        if X.ndim == 3 and X.shape == (SEQ_LEN, N_ROOMS, len(self.input_features)):
            X = np.expand_dims(X, axis=0)
        assert X.shape[1:] == (SEQ_LEN, N_ROOMS, len(self.input_features)), (
            f"Expected shape (batch, {SEQ_LEN}, {N_ROOMS}, {len(self.input_features)}), "
            f"got {X.shape}"
        )
        return self.model.predict(X, verbose=0)

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """
        Denormalizes predictions back to continuous occupant counts using
        each room's fitted target scaler.
        """
        scaled = np.asarray(scaled_targets, dtype=np.float32)
        if scaled.ndim == 1:
            scaled = scaled.reshape(1, -1)
        denorm = np.zeros_like(scaled, dtype=np.float32)
        for i, room in enumerate(self.rooms):
            denorm[:, i] = (
                self.target_scalers[room]
                .inverse_transform(scaled[:, i].reshape(-1, 1))
                .reshape(-1)
            )
        return denorm

    @staticmethod
    def postprocess_integer_counts(continuous_counts: np.ndarray) -> np.ndarray:
        """
        Applies notebook verified post-processing:
        Clips to non-negative and rounds to nearest whole integer.
        """
        return np.maximum(np.rint(continuous_counts).astype(int), 0)

    def predict(self, X_normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs full pipeline on normalized inputs.
        Returns:
            (integer_occupant_counts, continuous_occupant_counts)
            both with shape (batch, 5)
        """
        pred_scaled = self.predict_normalized(X_normalized)
        pred_continuous = self.inverse_transform_targets(pred_scaled)
        pred_integer = self.postprocess_integer_counts(pred_continuous)
        return pred_integer, pred_continuous

    def transform_raw_window(self, window_dfs: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Transforms raw lookback records (12 steps of 5 min) into normalized tensor.
        Parameters:
            window_dfs: dict mapping room names to DataFrame of length 12 containing
                        environmental columns and historical occupant_count.
        Returns:
            X_normalized of shape (1, 12, 5, 13)
        """
        n_sensor_features = len(self.input_features) - 1 # 12
        X_room = []

        for room in self.rooms:
            if room not in window_dfs:
                raise KeyError(f"Missing lookback DataFrame for '{room}'")
            df = window_dfs[room].copy()
            if len(df) != SEQ_LEN:
                raise ValueError(
                    f"Expected window length {SEQ_LEN} for {room}, got {len(df)}"
                )

            # Ensure time features are present
            if not all(col in df.columns for col in TIME_FEATURES):
                if TIMESTAMP_COL not in df.columns:
                    raise KeyError(
                        f"Missing '{TIMESTAMP_COL}' column to compute time features for {room}"
                    )
                time_feats = compute_time_features_df(df[TIMESTAMP_COL])
                for col in TIME_FEATURES:
                    df[col] = time_feats[col].values

            # Verify all input features exist
            for feat in self.input_features:
                if feat not in df.columns:
                    raise KeyError(f"Feature '{feat}' missing from DataFrame for {room}")

            # Extract raw numpy arrays
            raw_node = df[self.input_features].to_numpy(dtype=np.float32)

            # Scale first 12 features with sensor scaler
            scaled_node = np.zeros_like(raw_node, dtype=np.float32)
            scaled_node[:, :n_sensor_features] = self.sensor_scalers[room].transform(
                raw_node[:, :n_sensor_features]
            )

            # Scale 13th feature (historical occupancy) with target scaler
            scaled_node[:, -1] = (
                self.target_scalers[room]
                .transform(raw_node[:, -1].reshape(-1, 1))
                .reshape(-1)
            )

            X_room.append(scaled_node)

        # Stack into [1, 12, 5, 13]
        X_tensor = np.stack(X_room, axis=1) # shape: (12, 5, 13)
        return np.expand_dims(X_tensor, axis=0)

    def forecast_15min_ahead(self, window_dfs: dict[str, pd.DataFrame]) -> dict[str, int]:
        """
        End-to-end forecasting method:
        Takes 12 raw observations (60 min history) for all 5 rooms and returns
        a dictionary mapping room names to predicted occupant counts 15 minutes ahead.
        """
        X_norm = self.transform_raw_window(window_dfs)
        pred_int, _ = self.predict(X_norm)
        return {room: int(pred_int[0, i]) for i, room in enumerate(self.rooms)}

