"""
config.py - Configuration for Indian AI Traffic Control System
Tuned for realistic Indian urban traffic conditions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# Vehicle Types specific to Indian traffic
# ─────────────────────────────────────────────
class VehicleType(Enum):
    TWO_WHEELER = "two_wheeler"
    AUTO_RICKSHAW = "auto_rickshaw"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    CYCLE = "cycle"
    PEDESTRIAN = "pedestrian"
    EMERGENCY = "emergency"
    VIP = "vip"


# PCU (Passenger Car Unit) — IRC standards
PCU_FACTORS: Dict[VehicleType, float] = {
    VehicleType.TWO_WHEELER: 0.5,
    VehicleType.AUTO_RICKSHAW: 0.75,
    VehicleType.CAR: 1.0,
    VehicleType.BUS: 3.0,
    VehicleType.TRUCK: 3.5,
    VehicleType.CYCLE: 0.3,
    VehicleType.PEDESTRIAN: 0.2,
    VehicleType.EMERGENCY: 1.0,
    VehicleType.VIP: 1.5,
}

# Typical speed ranges (km/h) — Indian urban realistic
SPEED_RANGES: Dict[VehicleType, Tuple[float, float]] = {
    VehicleType.TWO_WHEELER: (15, 45),
    VehicleType.AUTO_RICKSHAW: (10, 30),
    VehicleType.CAR: (10, 50),
    VehicleType.BUS: (8, 35),
    VehicleType.TRUCK: (8, 30),
    VehicleType.CYCLE: (5, 18),
    VehicleType.PEDESTRIAN: (3, 6),
    VehicleType.EMERGENCY: (20, 70),
    VehicleType.VIP: (15, 55),
}

# Indian traffic composition
INDIAN_TRAFFIC_MIX = {
    VehicleType.TWO_WHEELER: 0.40,
    VehicleType.AUTO_RICKSHAW: 0.10,
    VehicleType.CAR: 0.25,
    VehicleType.BUS: 0.08,
    VehicleType.TRUCK: 0.05,
    VehicleType.CYCLE: 0.07,
    VehicleType.PEDESTRIAN: 0.04,
    VehicleType.EMERGENCY: 0.005,
    VehicleType.VIP: 0.005,
}


# ─────────────────────────────────────────────
# Grid / Network Configuration
# ─────────────────────────────────────────────
@dataclass
class GridConfig:
    rows: int = 4
    cols: int = 4
    road_length: float = 200.0             # 200m urban block
    lanes_per_direction: int = 2
    max_capacity_per_lane: int = 15        # realistic Indian lane capacity
    speed_limit: float = 40.0              # urban speed limit


# ─────────────────────────────────────────────
# Signal Configuration (Indian realistic)
# ─────────────────────────────────────────────
@dataclass
class SignalConfig:
    min_green_time: int = 18               # Indian min green ≈ 15–25 sec
    max_green_time: int = 60               # peak hour extended green
    yellow_time: int = 3
    all_red_time: int = 2
    cycle_time_range: Tuple[int, int] = (60, 180)
    num_phases: int = 4
    emergency_preemption: bool = True
    pedestrian_phase: bool = True
    pedestrian_crossing_time: int = 20


# ─────────────────────────────────────────────
# RL / Learning Configuration
# ─────────────────────────────────────────────
@dataclass
class RLConfig:
    learning_rate: float = 0.08            # slightly reduced for stability
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    state_bins: int = 5
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "waiting_time": -1.2,              # stronger waiting penalty
        "throughput": 2.5,                 # reward flow
        "queue_length": -0.7,
        "emergency_delay": -12.0,
        "congestion_spread": -0.3,
    })


# ─────────────────────────────────────────────
# ML Predictor Configuration
# ─────────────────────────────────────────────
@dataclass
class PredictorConfig:
    history_window: int = 12
    prediction_horizon: int = 6
    time_step_minutes: int = 5
    features: List[str] = field(default_factory=lambda: [
        "hour", "day_of_week", "is_peak",
        "vehicle_count", "avg_speed", "queue_length",
        "weather_factor", "event_factor"
    ])
    model_type: str = "gradient_boosting"
    retrain_interval: int = 100


# ─────────────────────────────────────────────
# Indian Peak Hour Profiles
# ─────────────────────────────────────────────
PEAK_HOURS = {
    "morning_peak": (8, 10),
    "school_peak": (13, 14),
    "evening_peak": (17, 20),
    "night_low": (23, 5),
}

HOURLY_DEMAND_MULTIPLIER = {
    0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.2,
    6: 0.4, 7: 0.7, 8: 1.0, 9: 1.0, 10: 0.85, 11: 0.75,
    12: 0.65, 13: 0.75, 14: 0.65, 15: 0.55,
    16: 0.65, 17: 0.95, 18: 1.0, 19: 0.9,
    20: 0.75, 21: 0.55, 22: 0.3, 23: 0.15,
}


# ─────────────────────────────────────────────
# Simulation Configuration
# ─────────────────────────────────────────────
@dataclass
class SimulationConfig:
    total_steps: int = 800                 # longer run improves completion
    step_duration: float = 1.0
    vehicle_spawn_rate: float = 0.015      # realistic urban inflow
    random_seed: int = 42
    enable_visualization: bool = True
    log_interval: int = 50
    grid: GridConfig = field(default_factory=GridConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)