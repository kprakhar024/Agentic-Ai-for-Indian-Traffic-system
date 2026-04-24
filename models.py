"""
models.py - Data models for vehicles, roads, intersections, and signals.
Models Indian traffic characteristics including mixed traffic and non-lane discipline.
"""

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
from config import (
    VehicleType, PCU_FACTORS, SPEED_RANGES,
    INDIAN_TRAFFIC_MIX, SignalConfig
)


class Direction(Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"


class SignalPhase(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    FLASHING = "FLASHING"


class TurnType(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    U_TURN = "u_turn"


# ─────────────────────────────────────────────
# Vehicle Model
# ─────────────────────────────────────────────
@dataclass
class Vehicle:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    vehicle_type: VehicleType = VehicleType.CAR
    current_road: Optional[str] = None
    current_position: float = 0.0        # position along road (0 to road_length)
    speed: float = 0.0                   # current speed km/h
    max_speed: float = 40.0
    origin: Optional[Tuple[int, int]] = None
    destination: Optional[Tuple[int, int]] = None
    route: List[Tuple[int, int]] = field(default_factory=list)
    route_index: int = 0
    waiting_time: float = 0.0           # total time spent waiting
    travel_time: float = 0.0            # total travel time
    is_waiting: bool = False
    lane_discipline: float = 0.5         # 0=chaotic, 1=perfect (Indian avg ~0.5)
    spawn_time: float = 0.0

    @property
    def pcu(self) -> float:
        return PCU_FACTORS.get(self.vehicle_type, 1.0)

    @property
    def is_emergency(self) -> bool:
        return self.vehicle_type == VehicleType.EMERGENCY

    @property
    def is_vip(self) -> bool:
        return self.vehicle_type == VehicleType.VIP

    @property
    def priority(self) -> int:
        """Higher = more priority."""
        if self.is_emergency:
            return 100
        elif self.is_vip:
            return 50
        return 1

    def update_position(self, dt: float):
        """Move vehicle forward based on current speed."""
        if self.speed > 0:
            displacement = (self.speed / 3.6) * dt  # convert km/h to m/s
            self.current_position += displacement
            self.travel_time += dt
            self.is_waiting = False
        else:
            self.waiting_time += dt
            self.is_waiting = True

    @staticmethod
    def spawn_random(spawn_time: float = 0.0) -> 'Vehicle':
        """Spawn a vehicle with Indian traffic distribution."""
        rand = random.random()
        cumulative = 0.0
        chosen_type = VehicleType.CAR

        for vtype, probability in INDIAN_TRAFFIC_MIX.items():
            cumulative += probability
            if rand <= cumulative:
                chosen_type = vtype
                break

        speed_range = SPEED_RANGES[chosen_type]
        max_speed = random.uniform(speed_range[0], speed_range[1])

        # Indian lane discipline factor: most vehicles moderate discipline
        discipline = random.betavariate(2, 3)  # skewed toward lower discipline

        return Vehicle(
            vehicle_type=chosen_type,
            max_speed=max_speed,
            speed=random.uniform(speed_range[0], max_speed),
            lane_discipline=discipline,
            spawn_time=spawn_time,
        )


# ─────────────────────────────────────────────
# Road Segment Model
# ─────────────────────────────────────────────
@dataclass
class RoadSegment:
    id: str = ""
    from_intersection: Tuple[int, int] = (0, 0)
    to_intersection: Tuple[int, int] = (0, 0)
    length: float = 500.0                 # meters
    lanes: int = 2
    max_capacity: int = 80                # max vehicles
    speed_limit: float = 40.0             # km/h
    vehicles: List[Vehicle] = field(default_factory=list)
    congestion_level: float = 0.0         # 0 to 1

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicles)

    @property
    def total_pcu(self) -> float:
        return sum(v.pcu for v in self.vehicles)

    @property
    def density(self) -> float:
        """Vehicles per km (PCU-adjusted)."""
        if self.length == 0:
            return 0
        return self.total_pcu / (self.length / 1000.0)

    @property
    def avg_speed(self) -> float:
        if not self.vehicles:
            return self.speed_limit
        speeds = [v.speed for v in self.vehicles]
        return sum(speeds) / len(speeds) if speeds else self.speed_limit

    @property
    def occupancy(self) -> float:
        """0.0 to 1.0 occupancy ratio."""
        if self.max_capacity == 0:
            return 0
        return min(1.0, self.vehicle_count / self.max_capacity)

    def update_congestion(self):
        """
        Calculate congestion using occupancy ratio.
        More sensitive than BPR for simulation visualization.
        """
        # ─── FIX: Direct occupancy-based congestion ───
        # This gives meaningful values even at moderate loads
        occ = self.occupancy
        
        # Sigmoid-like curve: slow start, rapid rise after 50% occupancy
        if occ < 0.3:
            self.congestion_level = occ * 0.5          # 0 to 0.15
        elif occ < 0.6:
            self.congestion_level = 0.15 + (occ - 0.3) * 1.5  # 0.15 to 0.6
        elif occ < 0.85:
            self.congestion_level = 0.6 + (occ - 0.6) * 1.2   # 0.6 to 0.9
        else:
            self.congestion_level = 0.9 + (occ - 0.85) * 0.67  # 0.9 to 1.0
        
        self.congestion_level = min(1.0, max(0.0, self.congestion_level))

    def get_travel_time(self) -> float:
        """Estimated travel time in seconds considering congestion."""
        free_flow_speed = max(self.speed_limit, 5.0)
        free_flow_time = (self.length / 1000.0) / (free_flow_speed / 3600.0)
        
        # Increase travel time with congestion
        congestion_factor = 1.0 + 3.0 * self.congestion_level
        return free_flow_time * congestion_factor

    def add_vehicle(self, vehicle: Vehicle) -> bool:
        if self.vehicle_count >= self.max_capacity:
            return False
        vehicle.current_road = self.id
        vehicle.current_position = 0.0
        self.vehicles.append(vehicle)
        self.update_congestion()
        return True

    def remove_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        for i, v in enumerate(self.vehicles):
            if v.id == vehicle_id:
                removed = self.vehicles.pop(i)
                self.update_congestion()
                return removed
        return None


# ─────────────────────────────────────────────
# Intersection Model
# ─────────────────────────────────────────────
@dataclass
class IntersectionState:
    """Real-time state of an intersection."""
    queue_lengths: Dict[Direction, int] = field(
        default_factory=lambda: {d: 0 for d in Direction}
    )
    waiting_times: Dict[Direction, float] = field(
        default_factory=lambda: {d: 0.0 for d in Direction}
    )
    throughput: Dict[Direction, int] = field(
        default_factory=lambda: {d: 0 for d in Direction}
    )
    emergency_present: Dict[Direction, bool] = field(
        default_factory=lambda: {d: False for d in Direction}
    )
    total_pcu: Dict[Direction, float] = field(
        default_factory=lambda: {d: 0.0 for d in Direction}
    )


@dataclass
class TrafficSignal:
    """Traffic signal at one direction of an intersection."""
    direction: Direction = Direction.NORTH
    phase: SignalPhase = SignalPhase.RED
    green_time: int = 30
    remaining_time: int = 30
    total_green_given: float = 0.0
    total_red_given: float = 0.0

    def tick(self, dt: float = 1.0):
        self.remaining_time -= dt
        if self.phase == SignalPhase.GREEN:
            self.total_green_given += dt
        elif self.phase == SignalPhase.RED:
            self.total_red_given += dt


@dataclass
class Intersection:
    id: str = ""
    position: Tuple[int, int] = (0, 0)     # grid position
    coordinates: Tuple[float, float] = (0.0, 0.0)  # physical coordinates

    signals: Dict[Direction, TrafficSignal] = field(default_factory=dict)
    incoming_roads: Dict[Direction, Optional[str]] = field(
        default_factory=lambda: {d: None for d in Direction}
    )
    outgoing_roads: Dict[Direction, Optional[str]] = field(
        default_factory=lambda: {d: None for d in Direction}
    )

    state: IntersectionState = field(default_factory=IntersectionState)
    current_green_direction: Optional[Direction] = None
    phase_index: int = 0
    signal_config: SignalConfig = field(default_factory=SignalConfig)

    # Tracking
    total_vehicles_passed: int = 0
    total_waiting_time: float = 0.0
    emergency_preemptions: int = 0

    def __post_init__(self):
        if not self.signals:
            for d in Direction:
                self.signals[d] = TrafficSignal(direction=d)

    @property
    def total_queue(self) -> int:
        return sum(self.state.queue_lengths.values())

    @property
    def max_queue_direction(self) -> Direction:
        return max(self.state.queue_lengths,
                   key=self.state.queue_lengths.get)

    @property
    def has_emergency(self) -> bool:
        return any(self.state.emergency_present.values())

    def get_emergency_direction(self) -> Optional[Direction]:
        for d, has_em in self.state.emergency_present.items():
            if has_em:
                return d
        return None

    def update_state(self, roads: Dict[str, RoadSegment]):
        """Update intersection state from connected roads."""
        for direction, road_id in self.incoming_roads.items():
            if road_id and road_id in roads:
                road = roads[road_id]
                self.state.queue_lengths[direction] = road.vehicle_count
                self.state.total_pcu[direction] = road.total_pcu

                waiting = [v for v in road.vehicles if v.is_waiting]
                self.state.waiting_times[direction] = (
                    sum(v.waiting_time for v in waiting) / max(len(waiting), 1)
                )
                self.state.emergency_present[direction] = any(
                    v.is_emergency for v in road.vehicles
                )