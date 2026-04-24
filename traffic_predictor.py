"""
traffic_predictor.py - Machine Learning based traffic flow prediction.
Uses historical and real-time data to forecast:
  - Traffic volume at each intersection
  - Expected congestion levels
  - Peak demand periods
  - Anomaly detection (accidents, events)

Supports multiple model backends:
  - Gradient Boosting (default, lightweight)
  - Simple Neural Network (numpy-based)
  - Moving average baseline
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from config import PredictorConfig, HOURLY_DEMAND_MULTIPLIER, PEAK_HOURS


@dataclass
class TrafficDataPoint:
    """Single observation of traffic state."""
    timestamp: float = 0.0
    hour: int = 0
    day_of_week: int = 0            # 0=Monday, 6=Sunday
    is_peak: int = 0
    vehicle_count: int = 0
    avg_speed: float = 0.0
    queue_length: int = 0
    weather_factor: float = 1.0      # 1.0=clear, 0.5=heavy rain
    event_factor: float = 1.0        # 1.0=normal, 2.0=festival
    congestion_level: float = 0.0    # target variable
    intersection_id: str = ""

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.hour / 24.0,
            self.day_of_week / 7.0,
            self.is_peak,
            self.vehicle_count / 100.0,
            self.avg_speed / 60.0,
            self.queue_length / 50.0,
            self.weather_factor,
            self.event_factor,
        ])


class SimpleNeuralNet:
    """
    Lightweight numpy-based neural network for traffic prediction.
    Architecture: Input(8) -> Hidden(32) -> Hidden(16) -> Output(1)
    """

    def __init__(self, input_size: int = 8, hidden1: int = 32,
                 hidden2: int = 16, output_size: int = 1,
                 learning_rate: float = 0.001):
        self.lr = learning_rate
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, output_size))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.output = self._sigmoid(self.z3)
        return self.output

    def backward(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]

        # Output layer
        dz3 = (self.output - y) * self.output * (1 - self.output)
        dW3 = self.a2.T @ dz3 / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Hidden layer 2
        dz2 = (dz3 @ self.W3.T) * self._relu_derivative(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer 1
        dz1 = (dz2 @ self.W2.T) * self._relu_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.forward(X)
        loss = np.mean((predictions - y) ** 2)
        self.backward(X, y)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class GradientBoostingSimple:
    """
    Simplified gradient boosting with decision stumps for traffic prediction.
    Lightweight alternative when sklearn is not available.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1,
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.stumps: List[Dict] = []
        self.initial_prediction = 0.0

    def _find_best_split(self, X: np.ndarray, residuals: np.ndarray
                         ) -> Dict:
        """Find the best feature and threshold for splitting."""
        best = {"feature": 0, "threshold": 0, "left_val": 0,
                "right_val": 0, "mse": float('inf')}

        n_features = X.shape[1]
        for feat in range(n_features):
            thresholds = np.percentile(X[:, feat], [25, 50, 75])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_val = residuals[left_mask].mean()
                right_val = residuals[right_mask].mean()

                predictions = np.where(left_mask, left_val, right_val)
                mse = np.mean((residuals - predictions) ** 2)

                if mse < best["mse"]:
                    best = {
                        "feature": feat, "threshold": thresh,
                        "left_val": left_val, "right_val": right_val,
                        "mse": mse
                    }
        return best

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.initial_prediction = y.mean()
        predictions = np.full(len(y), self.initial_prediction)
        self.stumps = []

        for _ in range(self.n_estimators):
            residuals = y - predictions
            stump = self._find_best_split(X, residuals)
            self.stumps.append(stump)

            left_mask = X[:, stump["feature"]] <= stump["threshold"]
            update = np.where(left_mask, stump["left_val"], stump["right_val"])
            predictions += self.lr * update

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_prediction)

        for stump in self.stumps:
            left_mask = X[:, stump["feature"]] <= stump["threshold"]
            update = np.where(left_mask, stump["left_val"], stump["right_val"])
            predictions += self.lr * update

        return np.clip(predictions, 0, 1)


class TrafficPredictor:
    """
    Main traffic prediction system that:
    1. Collects real-time traffic data
    2. Trains ML models on historical patterns
    3. Predicts future traffic flow and congestion
    4. Detects anomalies (accidents, unusual patterns)
    5. Accounts for Indian traffic specifics (festivals, weather, time patterns)
    """

    def __init__(self, config: PredictorConfig = None):
        self.config = config or PredictorConfig()

        # Data storage per intersection
        self.data_buffer: Dict[str, deque] = {}
        self.max_buffer = 5000

        # Models per intersection
        self.models: Dict[str, object] = {}
        self.model_type = self.config.model_type

        # Global patterns
        self.global_model = None
        self.training_count = 0

        # Prediction cache
        self.last_predictions: Dict[str, float] = {}

        # Anomaly detection
        self.baseline_stats: Dict[str, Dict] = {}

        # Indian calendar events (simplified)
        self.special_events = {
            "diwali": 2.5,       # Major festival
            "holi": 1.8,
            "ganesh_chaturthi": 2.0,
            "republic_day": 1.5,  # Parade routes
            "cricket_match": 1.8,
            "election": 1.3,
            "bandh": 0.2,        # Strike — very low traffic
            "rain_heavy": 0.6,
            "rain_light": 0.85,
            "fog": 0.7,
        }

    def register_intersection(self, intersection_id: str):
        """Register intersection for prediction."""
        self.data_buffer[intersection_id] = deque(maxlen=self.max_buffer)

        if self.model_type == "neural_net":
            self.models[intersection_id] = SimpleNeuralNet(
                input_size=8, learning_rate=0.001
            )
        else:
            self.models[intersection_id] = GradientBoostingSimple(
                n_estimators=30, learning_rate=0.1
            )

    def add_observation(self, intersection_id: str, data: TrafficDataPoint):
        """Add real-time observation."""
        if intersection_id not in self.data_buffer:
            self.register_intersection(intersection_id)
        self.data_buffer[intersection_id].append(data)

    def _prepare_training_data(self, intersection_id: str
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector from buffer."""
        buffer = self.data_buffer.get(intersection_id, [])
        if len(buffer) < 20:
            return np.array([]), np.array([])

        X = np.array([dp.to_feature_vector() for dp in buffer])
        y = np.array([dp.congestion_level for dp in buffer])

        return X, y

    def train(self, intersection_id: str) -> float:
        """Train/retrain model for a specific intersection."""
        X, y = self._prepare_training_data(intersection_id)
        if len(X) < 20:
            return -1.0

        model = self.models.get(intersection_id)
        if model is None:
            return -1.0

        if isinstance(model, SimpleNeuralNet):
            # Train for multiple epochs
            losses = []
            for epoch in range(50):
                loss = model.train_step(X, y.reshape(-1, 1))
                losses.append(loss)
            return losses[-1]

        elif isinstance(model, GradientBoostingSimple):
            model.fit(X, y)
            predictions = model.predict(X)
            mse = np.mean((predictions - y) ** 2)
            return mse

        return -1.0

    def predict_congestion(self, intersection_id: str,
                           current_data: TrafficDataPoint,
                           horizon_steps: int = 6
                           ) -> List[float]:
        """
        Predict future congestion levels.
        Returns list of predicted congestion values for future time steps.
        """
        model = self.models.get(intersection_id)
        if model is None:
            return [self._baseline_prediction(current_data)] * horizon_steps

        # If not enough training data, use baseline
        buffer = self.data_buffer.get(intersection_id, [])
        if len(buffer) < 20:
            return [self._baseline_prediction(current_data)] * horizon_steps

        predictions = []
        current_features = current_data.to_feature_vector()

        for step in range(horizon_steps):
            # Predict
            features = current_features.reshape(1, -1)

            if isinstance(model, SimpleNeuralNet):
                pred = model.predict(features)[0][0]
            else:
                pred = model.predict(features)[0]

            predictions.append(float(np.clip(pred, 0, 1)))

            # Shift features forward in time
            future_hour = (current_data.hour +
                           (step + 1) * self.config.time_step_minutes // 60)
            future_hour = future_hour % 24
            current_features = current_features.copy()
            current_features[0] = future_hour / 24.0  # Update hour

        self.last_predictions[intersection_id] = predictions[0]
        return predictions

    def _baseline_prediction(self, data: TrafficDataPoint) -> float:
        """Fallback prediction using hourly demand multipliers."""
        base = HOURLY_DEMAND_MULTIPLIER.get(data.hour, 0.5)
        return min(1.0, base * data.event_factor * data.weather_factor)

    def detect_anomaly(self, intersection_id: str,
                       current_data: TrafficDataPoint) -> Dict:
        """
        Detect traffic anomalies (accidents, unusual congestion, etc.)
        Uses Z-score method against historical baseline.
        """
        buffer = self.data_buffer.get(intersection_id, [])
        if len(buffer) < 50:
            return {"is_anomaly": False, "score": 0.0, "type": None}

        # Calculate baseline statistics
        recent = list(buffer)[-100:]
        hist_counts = [dp.vehicle_count for dp in recent]
        hist_speeds = [dp.avg_speed for dp in recent]

        mean_count = np.mean(hist_counts)
        std_count = max(np.std(hist_counts), 1.0)
        mean_speed = np.mean(hist_speeds)
        std_speed = max(np.std(hist_speeds), 1.0)

        z_count = abs(current_data.vehicle_count - mean_count) / std_count
        z_speed = abs(current_data.avg_speed - mean_speed) / std_speed

        is_anomaly = z_count > 3.0 or z_speed > 3.0
        anomaly_score = max(z_count, z_speed)

        anomaly_type = None
        if is_anomaly:
            if current_data.avg_speed < mean_speed * 0.3 and z_count > 2:
                anomaly_type = "possible_accident"
            elif current_data.vehicle_count > mean_count * 2:
                anomaly_type = "unusual_surge"
            elif current_data.vehicle_count < mean_count * 0.3:
                anomaly_type = "unusual_drop"
            else:
                anomaly_type = "general_anomaly"

        return {
            "is_anomaly": is_anomaly,
            "score": float(anomaly_score),
            "type": anomaly_type,
            "z_count": float(z_count),
            "z_speed": float(z_speed),
        }

    def get_demand_forecast(self, hour: int, day_of_week: int,
                            weather: float = 1.0,
                            event: str = None) -> float:
        """
        Get expected traffic demand multiplier for planning.
        Combines time patterns with events and weather.
        """
        base = HOURLY_DEMAND_MULTIPLIER.get(hour, 0.5)

        # Weekend adjustment
        if day_of_week >= 5:  # Saturday, Sunday
            base *= 0.7

        # Weather
        base *= weather

        # Special events
        if event and event in self.special_events:
            base *= self.special_events[event]

        return min(2.0, base)

    def periodic_retrain(self, step: int):
        """Retrain all models periodically."""
        if step % self.config.retrain_interval == 0 and step > 0:
            for intersection_id in self.models:
                self.train(intersection_id)
            self.training_count += 1

    def get_prediction_summary(self) -> Dict:
        """Get summary of all predictions."""
        return {
            "predictions": dict(self.last_predictions),
            "models_trained": self.training_count,
            "data_points": {
                iid: len(buf) for iid, buf in self.data_buffer.items()
            },
        }