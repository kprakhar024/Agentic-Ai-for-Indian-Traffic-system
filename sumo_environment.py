"""
sumo_environment.py — Bridge between SUMO simulator and AI agents.
Uses TraCI (Traffic Control Interface) to:
  - Control traffic signals in real-time
  - Read vehicle/detector data
  - Inject/remove vehicles
  - Query routes and travel times
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

try:
    import traci
    import traci.constants as tc
    HAS_TRACI = True
except ImportError:
    HAS_TRACI = False
    print("❌ TraCI not found! Install with: pip install traci")
    print("   Also ensure SUMO_HOME environment variable is set.")

from config import (
    SimulationConfig, VehicleType, HOURLY_DEMAND_MULTIPLIER,
    INDIAN_TRAFFIC_MIX, PCU_FACTORS
)
from models import Direction, SignalPhase


# ─────────────────────────────────────────────
# SUMO Vehicle Type to Our VehicleType Mapping
# ─────────────────────────────────────────────
SUMO_TYPE_MAP = {
    "two_wheeler": VehicleType.TWO_WHEELER,
    "auto_rickshaw": VehicleType.AUTO_RICKSHAW,
    "car": VehicleType.CAR,
    "bus": VehicleType.BUS,
    "truck": VehicleType.TRUCK,
    "cycle": VehicleType.CYCLE,
    "emergency": VehicleType.EMERGENCY,
}

# Direction mapping: SUMO TLS phase index to our Direction
# SUMO typically orders: N, E, S, W or based on edge ordering
PHASE_TO_DIRECTION = {
    0: Direction.NORTH,
    1: Direction.EAST,
    2: Direction.SOUTH,
    3: Direction.WEST,
}


@dataclass
class SUMOIntersectionData:
    """Real-time data extracted from SUMO for one intersection."""
    tls_id: str = ""
    queue_lengths: Dict[Direction, int] = field(
        default_factory=lambda: {d: 0 for d in Direction}
    )
    waiting_times: Dict[Direction, float] = field(
        default_factory=lambda: {d: 0.0 for d in Direction}
    )
    vehicle_counts: Dict[Direction, int] = field(
        default_factory=lambda: {d: 0 for d in Direction}
    )
    avg_speeds: Dict[Direction, float] = field(
        default_factory=lambda: {d: 0.0 for d in Direction}
    )
    total_pcu: Dict[Direction, float] = field(
        default_factory=lambda: {d: 0.0 for d in Direction}
    )
    emergency_present: Dict[Direction, bool] = field(
        default_factory=lambda: {d: False for d in Direction}
    )
    throughput: int = 0
    current_phase: int = 0
    current_phase_duration: float = 0.0


class SUMOEnvironment:
    """
    Interface between SUMO simulator and our AI agents.
    Handles all TraCI communication for:
    - Reading traffic state
    - Controlling traffic signals
    - Collecting metrics
    """

    def __init__(self, config: SimulationConfig,
                 sumo_cfg_path: str = None,
                 gui: bool = True):
        self.config = config
        self.gui = gui

        # Find SUMO config file
        if sumo_cfg_path:
            self.sumo_cfg = sumo_cfg_path
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.sumo_cfg = os.path.join(
                base_dir, "sumo_network", "indian_grid.sumocfg"
            )

        # SUMO connection state
        self.connected = False
        self.current_step = 0

        # Traffic light IDs
        self.tls_ids: List[str] = []

        # Edge data cache
        self.tls_incoming_edges: Dict[str, Dict[Direction, List[str]]] = {}

        # Metrics
        self.total_departed = 0
        self.total_arrived = 0
        self.total_waiting_time = 0.0
        self.travel_times: List[float] = []
        self.step_data: List[Dict] = []

    def start(self):
        """Start SUMO simulation."""
        if not HAS_TRACI:
            raise RuntimeError("TraCI not available. Install SUMO first.")

        if not os.path.exists(self.sumo_cfg):
            raise FileNotFoundError(
                f"SUMO config not found: {self.sumo_cfg}\n"
                f"Run: python sumo_network/generate_network.py"
            )

        # Choose SUMO binary
        sumo_binary = "sumo-gui" if self.gui else "sumo"

        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--start",                      # Auto-start
            "--quit-on-end",                # Quit when done
            "--waiting-time-memory", "100",  # Track waiting
            "--time-to-teleport", "-1",     # Disable teleporting stuck vehicles
            "--no-step-log", "true",
        ]

        print(f"  🚀 Starting SUMO: {sumo_binary}")
        print(f"  📁 Config: {self.sumo_cfg}")

        traci.start(sumo_cmd)
        self.connected = True

        # Discover traffic lights
        self.tls_ids = list(traci.trafficlight.getIDList())
        print(f"  ✅ Connected! Found {len(self.tls_ids)} traffic lights")

        # Map incoming edges to directions for each TLS
        self._map_tls_edges()

    def _map_tls_edges(self):
        """Map incoming edges to directions for each traffic light."""
        for tls_id in self.tls_ids:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            incoming_edges = defaultdict(list)

            for link_group in controlled_links:
                for link in link_group:
                    if len(link) >= 2:
                        incoming_lane = link[0]  # e.g., "E_01_to_00_0"
                        edge_id = incoming_lane.rsplit('_', 1)[0]

                        # Determine direction from edge name
                        direction = self._edge_to_direction(edge_id, tls_id)
                        if edge_id not in incoming_edges[direction]:
                            incoming_edges[direction].append(edge_id)

            self.tls_incoming_edges[tls_id] = dict(incoming_edges)

    def _edge_to_direction(self, edge_id: str, tls_id: str) -> Direction:
        """Determine which direction an edge approaches from."""
        # Parse edge format: E_RC_to_RC (e.g., E_01_to_00)
        try:
            parts = edge_id.split('_')
            # Find 'to' index
            to_idx = parts.index('to')
            from_node = parts[to_idx - 1]
            to_node = parts[to_idx + 1]

            # Extract row, col from node IDs
            fr = int(from_node[0])
            fc = int(from_node[1])
            tr = int(to_node[0])
            tc = int(to_node[1])

            # Determine direction
            if fr < tr:
                return Direction.NORTH  # Coming from north (row decreasing)
            elif fr > tr:
                return Direction.SOUTH
            elif fc < tc:
                return Direction.WEST
            else:
                return Direction.EAST

        except (ValueError, IndexError):
            return Direction.NORTH  # Default

    def get_intersection_data(self, tls_id: str) -> SUMOIntersectionData:
        """Extract real-time traffic data for an intersection from SUMO."""
        data = SUMOIntersectionData(tls_id=tls_id)

        # Current signal phase
        data.current_phase = traci.trafficlight.getPhase(tls_id)
        data.current_phase_duration = traci.trafficlight.getPhaseDuration(tls_id)

        # Get data from incoming edges
        for direction, edge_ids in self.tls_incoming_edges.get(
                tls_id, {}).items():
            total_vehicles = 0
            total_waiting = 0.0
            total_speed = 0.0
            total_pcu = 0.0
            has_emergency = False
            speed_count = 0

            for edge_id in edge_ids:
                try:
                    # Vehicle count
                    veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                    total_vehicles += len(veh_ids)

                    # Halting vehicles (speed < 0.1 m/s)
                    halting = traci.edge.getLastStepHaltingNumber(edge_id)
                    data.queue_lengths[direction] += halting

                    # Waiting time
                    waiting = traci.edge.getWaitingTime(edge_id)
                    total_waiting += waiting

                    # Speed
                    speed = traci.edge.getLastStepMeanSpeed(edge_id)
                    if speed >= 0:
                        total_speed += speed
                        speed_count += 1

                    # Check for emergency vehicles
                    for vid in veh_ids:
                        vtype = traci.vehicle.getTypeID(vid)
                        if vtype == "emergency":
                            has_emergency = True

                        # Calculate PCU
                        pcu = PCU_FACTORS.get(
                            SUMO_TYPE_MAP.get(vtype, VehicleType.CAR), 1.0
                        )
                        total_pcu += pcu

                except traci.exceptions.TraCIException:
                    pass

            data.vehicle_counts[direction] = total_vehicles
            data.waiting_times[direction] = total_waiting
            data.avg_speeds[direction] = (
                total_speed / max(speed_count, 1)
            )
            data.total_pcu[direction] = total_pcu
            data.emergency_present[direction] = has_emergency

        return data

    def set_signal_phase(self, tls_id: str, phase_index: int):
        """Set traffic light to a specific phase."""
        try:
            traci.trafficlight.setPhase(tls_id, phase_index)
        except traci.exceptions.TraCIException as e:
            print(f"  ⚠ Could not set phase for {tls_id}: {e}")

    def set_signal_state(self, tls_id: str, state: str):
        """
        Set traffic light state string directly.
        State chars: 'G'=green, 'g'=green-minor, 'y'=yellow, 'r'=red
        Example: "GGGrrrGGGrrr" for N-S green, E-W red
        """
        try:
            traci.trafficlight.setRedYellowGreenState(tls_id, state)
        except traci.exceptions.TraCIException as e:
            print(f"  ⚠ Could not set state for {tls_id}: {e}")

    def set_phase_duration(self, tls_id: str, duration: float):
        """Set remaining duration for current phase."""
        try:
            traci.trafficlight.setPhaseDuration(tls_id, duration)
        except traci.exceptions.TraCIException:
            pass

    def get_vehicle_data(self) -> Dict:
        """Get global vehicle statistics."""
        try:
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            running = traci.vehicle.getIDCount()
            mean_speed = traci.vehicle.getIDCount()

            # Calculate mean speed from all vehicles
            all_vehicles = traci.vehicle.getIDList()
            speeds = []
            waiting_count = 0
            total_waiting_time = 0.0

            for vid in all_vehicles:
                try:
                    speed = traci.vehicle.getSpeed(vid)
                    speeds.append(speed)
                    wt = traci.vehicle.getAccumulatedWaitingTime(vid)
                    total_waiting_time += wt
                    if speed < 0.1:
                        waiting_count += 1
                except traci.exceptions.TraCIException:
                    pass

            avg_speed = sum(speeds) / max(len(speeds), 1)

            self.total_departed += departed
            self.total_arrived += arrived

            return {
                "departed_this_step": departed,
                "arrived_this_step": arrived,
                "total_departed": self.total_departed,
                "total_arrived": self.total_arrived,
                "running": running,
                "avg_speed_ms": avg_speed,
                "avg_speed_kmh": avg_speed * 3.6,
                "waiting_count": waiting_count,
                "total_waiting_time": total_waiting_time,
            }

        except traci.exceptions.TraCIException:
            return {"running": 0, "avg_speed_kmh": 0}

    def get_edge_data(self, edge_id: str) -> Dict:
        """Get data for a specific edge/road."""
        try:
            return {
                "vehicle_count": traci.edge.getLastStepVehicleNumber(edge_id),
                "mean_speed": traci.edge.getLastStepMeanSpeed(edge_id),
                "occupancy": traci.edge.getLastStepOccupancy(edge_id),
                "waiting_time": traci.edge.getWaitingTime(edge_id),
                "halting": traci.edge.getLastStepHaltingNumber(edge_id),
                "travel_time": traci.edge.getTraveltime(edge_id),
            }
        except traci.exceptions.TraCIException:
            return {}

    def get_all_edge_ids(self) -> List[str]:
        """Get all edge IDs in the network."""
        try:
            return list(traci.edge.getIDList())
        except traci.exceptions.TraCIException:
            return []

    def reroute_vehicle(self, vehicle_id: str, new_route: List[str]):
        """Reroute a vehicle to a new path."""
        try:
            traci.vehicle.setRoute(vehicle_id, new_route)
        except traci.exceptions.TraCIException:
            pass

    def simulation_step(self):
        """Advance SUMO by one step."""
        traci.simulationStep()
        self.current_step += 1

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return traci.simulation.getTime()

    def is_running(self) -> bool:
        """Check if simulation is still running."""
        try:
            return traci.simulation.getMinExpectedNumber() > 0
        except traci.exceptions.TraCIException:
            return False

    def get_state_summary(self) -> Dict:
        """Get comprehensive state summary."""
        veh_data = self.get_vehicle_data()
        sim_time = self.get_simulation_time()
        hour = int(8 + sim_time / 3600) % 24  # Starting from 8 AM

        # Get congestion from all edges
        congestion_levels = []
        for edge_id in self.get_all_edge_ids():
            if edge_id.startswith(":"):  # Skip internal edges
                continue
            edata = self.get_edge_data(edge_id)
            if edata:
                congestion_levels.append(
                    edata.get("occupancy", 0) / 100.0
                )

        avg_congestion = (
            sum(congestion_levels) / max(len(congestion_levels), 1)
        )

        return {
            "step": self.current_step,
            "sim_time": sim_time,
            "hour": hour,
            "total_vehicles": veh_data.get("running", 0),
            "vehicles_waiting": veh_data.get("waiting_count", 0),
            "avg_speed": veh_data.get("avg_speed_kmh", 0),
            "avg_congestion": avg_congestion,
            "vehicles_spawned": veh_data.get("total_departed", 0),
            "vehicles_completed": veh_data.get("total_arrived", 0),
            "avg_travel_time": 0,  # Calculated from arrived vehicles
            "avg_waiting_time": veh_data.get("total_waiting_time", 0) / max(
                veh_data.get("running", 1), 1
            ),
        }

    def close(self):
        """Close SUMO connection."""
        if self.connected:
            try:
                traci.close()
                self.connected = False
                print("  ✅ SUMO connection closed")
            except Exception:
                pass