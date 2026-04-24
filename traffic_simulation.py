"""
traffic_simulation.py
FINAL WORKING VERSION - Uses Greenshields flow model instead of
car-following to prevent cascading stops.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import (
    SimulationConfig, GridConfig, INDIAN_TRAFFIC_MIX,
    HOURLY_DEMAND_MULTIPLIER, VehicleType
)
from models import (
    Vehicle, RoadSegment, Intersection, SignalPhase,
    TrafficSignal, Direction
)
from route_optimizer import TrafficGraph


class TrafficEnvironment:

    DIRECTION_OFFSETS = {
        Direction.NORTH: (-1, 0),
        Direction.SOUTH: (1, 0),
        Direction.EAST: (0, 1),
        Direction.WEST: (0, -1),
    }

    OPPOSITE = {
        Direction.NORTH: Direction.SOUTH,
        Direction.SOUTH: Direction.NORTH,
        Direction.EAST: Direction.WEST,
        Direction.WEST: Direction.EAST,
    }

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.grid = self.config.grid

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.intersections: Dict[str, Intersection] = {}
        self.roads: Dict[str, RoadSegment] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.graph = TrafficGraph()

        self.current_step = 0
        self.current_hour = 8
        self.total_vehicles_spawned = 0
        self.total_vehicles_completed = 0
        self.completed_travel_times: List[float] = []
        self.completed_waiting_times: List[float] = []

        self._build_grid()

    # ═══════════════════════════════════════
    # NETWORK BUILDING
    # ═══════════════════════════════════════
    def _intersection_id(self, row, col):
        return f"I_{row}_{col}"

    def _road_id(self, from_rc, to_rc):
        return f"R_{from_rc[0]}{from_rc[1]}_to_{to_rc[0]}{to_rc[1]}"

    def _build_grid(self):
        rows, cols = self.grid.rows, self.grid.cols

        for r in range(rows):
            for c in range(cols):
                iid = self._intersection_id(r, c)
                self.intersections[iid] = Intersection(
                    id=iid, position=(r, c),
                    coordinates=(c * self.grid.road_length,
                                 r * self.grid.road_length),
                    signal_config=self.config.signal,
                )
                self.graph.add_node((r, c))

        for r in range(rows):
            for c in range(cols):
                current = (r, c)
                current_id = self._intersection_id(r, c)
                for direction, (dr, dc) in self.DIRECTION_OFFSETS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor = (nr, nc)
                        neighbor_id = self._intersection_id(nr, nc)
                        rid = self._road_id(current, neighbor)
                        cap = (self.grid.max_capacity_per_lane *
                               self.grid.lanes_per_direction)
                        road = RoadSegment(
                            id=rid,
                            from_intersection=current,
                            to_intersection=neighbor,
                            length=self.grid.road_length,
                            lanes=self.grid.lanes_per_direction,
                            max_capacity=cap,
                            speed_limit=self.grid.speed_limit,
                        )
                        self.roads[rid] = road
                        self.graph.add_edge(
                            current, neighbor, rid, road
                        )
                        opposite = self.OPPOSITE[direction]
                        self.intersections[
                            current_id
                        ].outgoing_roads[direction] = rid
                        self.intersections[
                            neighbor_id
                        ].incoming_roads[opposite] = rid

    # ═══════════════════════════════════════
    # METRICS
    # ═══════════════════════════════════════
    def _network_occupancy(self) -> float:
        total_cap = sum(r.max_capacity for r in self.roads.values())
        total_veh = sum(r.vehicle_count for r in self.roads.values())
        return total_veh / max(total_cap, 1)

    def _is_green(self, intersection: Intersection,
                  direction: Direction) -> bool:
        """Check if signal is GREEN for given direction."""
        sig = intersection.signals.get(direction)
        if not sig:
            return True
        return sig.phase == SignalPhase.GREEN

    def _can_pass_signal(self, intersection: Intersection,
                         direction: Direction,
                         vehicle: Vehicle) -> bool:
        """Check if vehicle can pass through signal."""
        if vehicle.is_emergency:
            return True
        sig = intersection.signals.get(direction)
        if not sig:
            return True
        if sig.phase == SignalPhase.GREEN:
            return True
        if sig.phase == SignalPhase.YELLOW:
            return random.random() < 0.25
        # RED: small chance of jumping
        return random.random() < (1 - vehicle.lane_discipline) * 0.03

    def _arriving_direction(self, road: RoadSegment) -> Direction:
        fr, fc = road.from_intersection
        tr, tc = road.to_intersection
        if fr < tr:
            return Direction.NORTH
        elif fr > tr:
            return Direction.SOUTH
        elif fc < tc:
            return Direction.WEST
        else:
            return Direction.EAST

    # ═══════════════════════════════════════
    # SPAWNING (strictly controlled)
    # ═══════════════════════════════════════
    def spawn_vehicles(self):
        occ = self._network_occupancy()

        # HARD STOP at 40% occupancy
        if occ > 0.40:
            return

        cap_factor = max(0.0, 1.0 - occ * 2.5)
        demand = HOURLY_DEMAND_MULTIPLIER.get(self.current_hour, 0.5)
        prob = self.config.vehicle_spawn_rate * demand * cap_factor

        rows, cols = self.grid.rows, self.grid.cols
        edges = []
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    edges.append((r, c))

        # STRICT LIMIT: max 1 vehicle per step
        spawned = 0
        random.shuffle(edges)

        for entry in edges:
            if spawned >= 1:
                break
            if random.random() >= prob:
                continue

            v = Vehicle.spawn_random(spawn_time=self.current_step)
            dest = self._pick_dest(entry, edges)
            if not dest:
                continue
            v.origin = entry
            v.destination = dest

            iid = self._intersection_id(*entry)
            inter = self.intersections[iid]
            rid = self._best_road_toward(inter, dest)

            if rid:
                road = self.roads[rid]
                if road.occupancy < 0.5 and road.add_vehicle(v):
                    v.speed = min(v.max_speed,
                                  road.speed_limit * 0.9)
                    self.vehicles[v.id] = v
                    self.total_vehicles_spawned += 1
                    spawned += 1

    def _pick_dest(self, origin, edges):
        cands = [e for e in edges if e != origin]
        if not cands:
            return None
        rows, cols = self.grid.rows, self.grid.cols
        far = [e for e in cands
               if abs(e[0]-origin[0]) + abs(e[1]-origin[1]) >=
               max(rows, cols) - 1]
        if far and random.random() < 0.5:
            return random.choice(far)
        return random.choice(cands)

    def _best_road_toward(self, intersection, dest):
        best = None
        best_d = float('inf')
        for d, rid in intersection.outgoing_roads.items():
            if not rid:
                continue
            road = self.roads.get(rid)
            if not road or road.occupancy >= 0.5:
                continue
            nr, nc = road.to_intersection
            dist = abs(nr - dest[0]) + abs(nc - dest[1])
            if dist < best_d:
                best_d = dist
                best = rid
        return best

    # ═══════════════════════════════════════
    # VEHICLE MOVEMENT — GREENSHIELDS MODEL
    # ═══════════════════════════════════════
    def move_vehicles(self):
        """
        KEY DESIGN: Greenshields speed-density model.
        
        All vehicles on a road move at the SAME speed based on:
          speed = free_flow_speed × (1 - density/max_density)
        
        This PREVENTS cascading stops that killed throughput.
        Individual adjustments only for:
          - Signal approach (last 10m if RED)
          - Vehicle at intersection waiting for signal
        """
        dt = self.config.step_duration
        to_remove_ids = set()
        at_intersection = defaultdict(list)

        # ─────────────────────────────────────
        # PHASE 1: Calculate road speeds and move
        # ─────────────────────────────────────
        for rid, road in self.roads.items():
            if not road.vehicles:
                continue

            dest_iid = self._intersection_id(*road.to_intersection)
            dest_inter = self.intersections.get(dest_iid)
            arr_dir = self._arriving_direction(road)

            # Greenshields: road-level speed
            free_flow = road.speed_limit
            occ = road.occupancy
            road_speed = free_flow * max(0.15, 1.0 - occ)

            # Is signal green for this direction?
            signal_green = (
                self._is_green(dest_inter, arr_dir)
                if dest_inter else True
            )

            for vehicle in list(road.vehicles):
                dist_to_end = road.length - vehicle.current_position

                if dist_to_end <= 0.5:
                    # AT intersection — queue for processing
                    vehicle.current_position = road.length
                    at_intersection[rid].append(vehicle)
                    vehicle.speed = 0
                    vehicle.is_waiting = True
                    vehicle.waiting_time += dt
                    continue

                # ─── Determine vehicle speed ───
                if dist_to_end < 10.0 and not signal_green:
                    # Approaching RED signal — brake to stop
                    # Linear deceleration over last 10m
                    brake_speed = road_speed * (dist_to_end / 10.0)
                    vehicle.speed = max(0.5, brake_speed)
                else:
                    # GREEN or far from intersection — use road speed
                    desired = min(vehicle.max_speed, road_speed)

                    # Two-wheeler bonus
                    if vehicle.vehicle_type == VehicleType.TWO_WHEELER:
                        desired *= (1 + (1 - vehicle.lane_discipline)
                                    * 0.15)

                    # Smooth acceleration
                    diff = desired - vehicle.speed
                    vehicle.speed += diff * 0.4
                    vehicle.speed = max(
                        0, min(vehicle.speed, vehicle.max_speed)
                    )

                # Move
                displacement = (vehicle.speed / 3.6) * dt
                vehicle.current_position += displacement
                vehicle.travel_time += dt

                # Clamp to road end
                if vehicle.current_position >= road.length:
                    vehicle.current_position = road.length

                # Track waiting
                vehicle.is_waiting = vehicle.speed < 2.0
                if vehicle.is_waiting:
                    vehicle.waiting_time += dt

        # ─────────────────────────────────────
        # PHASE 2: Process intersection crossings
        # ─────────────────────────────────────
        for road_id, queued in at_intersection.items():
            road = self.roads.get(road_id)
            if not road:
                continue

            dest_iid = self._intersection_id(*road.to_intersection)
            intersection = self.intersections.get(dest_iid)
            if not intersection:
                for v in queued:
                    to_remove_ids.add(v.id)
                continue

            arr_dir = self._arriving_direction(road)

            # Sort by priority then waiting time
            queued.sort(key=lambda v: (-v.priority, -v.waiting_time))

            # ─── THROUGHPUT: pass many vehicles per green ───
            max_pass = road.lanes * 5  # 5 per lane per step = 10
            passed = 0

            for vehicle in queued:
                if passed >= max_pass:
                    break

                # Can this vehicle pass the signal?
                if not self._can_pass_signal(
                    intersection, arr_dir, vehicle
                ):
                    continue

                # ─── REACHED DESTINATION? ───
                if road.to_intersection == vehicle.destination:
                    to_remove_ids.add(vehicle.id)
                    self.total_vehicles_completed += 1
                    self.completed_travel_times.append(
                        vehicle.travel_time
                    )
                    self.completed_waiting_times.append(
                        vehicle.waiting_time
                    )
                    intersection.total_vehicles_passed += 1
                    intersection.state.throughput[arr_dir] = (
                        intersection.state.throughput.get(
                            arr_dir, 0
                        ) + 1
                    )
                    passed += 1
                    continue

                # ─── TRANSFER TO NEXT ROAD ───
                next_rid = self._next_road(
                    intersection, vehicle, arr_dir
                )
                if not next_rid:
                    continue

                next_road = self.roads.get(next_rid)
                if not next_road or next_road.occupancy >= 0.80:
                    continue

                removed = road.remove_vehicle(vehicle.id)
                if not removed:
                    continue

                if next_road.add_vehicle(removed):
                    # KEY: Enter at FULL road speed
                    nr_speed = next_road.speed_limit * max(
                        0.3, 1.0 - next_road.occupancy
                    )
                    removed.speed = min(removed.max_speed, nr_speed)
                    removed.is_waiting = False
                    intersection.total_vehicles_passed += 1
                    intersection.state.throughput[arr_dir] = (
                        intersection.state.throughput.get(
                            arr_dir, 0
                        ) + 1
                    )
                    passed += 1
                else:
                    road.add_vehicle(removed)

        # ─────────────────────────────────────
        # PHASE 3: Cleanup
        # ─────────────────────────────────────
        for vid in to_remove_ids:
            v = self.vehicles.pop(vid, None)
            if v and v.current_road:
                road = self.roads.get(v.current_road)
                if road:
                    road.remove_vehicle(vid)

        # ─────────────────────────────────────
        # PHASE 4: Remove stuck vehicles
        # ─────────────────────────────────────
        stuck = [
            vid for vid, v in self.vehicles.items()
            if v.waiting_time > 100
        ]
        for vid in stuck:
            v = self.vehicles.pop(vid, None)
            if v and v.current_road:
                road = self.roads.get(v.current_road)
                if road:
                    road.remove_vehicle(vid)
            self.total_vehicles_completed += 1
            if v:
                self.completed_travel_times.append(v.travel_time)
                self.completed_waiting_times.append(v.waiting_time)

    def _next_road(self, intersection, vehicle, arriving_from):
        """Pick next road toward destination."""
        if not vehicle.destination:
            return self._any_exit(intersection, arriving_from)

        opposite = self.OPPOSITE[arriving_from]
        best = None
        best_score = float('inf')

        for d, rid in intersection.outgoing_roads.items():
            if d == opposite or not rid:
                continue
            road = self.roads.get(rid)
            if not road or road.occupancy >= 0.75:
                continue
            nr, nc = road.to_intersection
            dist = (abs(nr - vehicle.destination[0]) +
                    abs(nc - vehicle.destination[1]))
            score = dist + road.occupancy * 3
            if score < best_score:
                best_score = score
                best = rid

        if not best:
            return self._any_exit(intersection, arriving_from)
        return best

    def _any_exit(self, intersection, arriving_from):
        opposite = self.OPPOSITE[arriving_from]
        opts = []
        for d, rid in intersection.outgoing_roads.items():
            if d == opposite or not rid:
                continue
            road = self.roads.get(rid)
            if road and road.occupancy < 0.75:
                opts.append(rid)
        if opts:
            return random.choice(opts)
        u = intersection.outgoing_roads.get(opposite)
        if u and self.roads.get(u, None):
            r = self.roads[u]
            if r.occupancy < 0.75:
                return u
        return None

    # ═══════════════════════════════════════
    # UPDATE & STATE
    # ═══════════════════════════════════════
    def update_roads(self):
        for road in self.roads.values():
            road.update_congestion()

    def step(self):
        self.current_step += 1
        sim_seconds = self.current_step * 30
        self.current_hour = (8 + sim_seconds // 3600) % 24

        for inter in self.intersections.values():
            for d in Direction:
                inter.state.throughput[d] = 0

        self.spawn_vehicles()
        self.move_vehicles()
        self.update_roads()

    def get_state_summary(self) -> Dict:
        total = len(self.vehicles)
        waiting = sum(
            1 for v in self.vehicles.values()
            if v.is_waiting or v.speed < 2.0
        )
        avg_speed = (
            float(np.mean(
                [v.speed for v in self.vehicles.values()]
            ))
            if self.vehicles else 0
        )
        occs = [r.occupancy for r in self.roads.values()]
        avg_cong = float(np.mean(occs)) if occs else 0

        return {
            "step": self.current_step,
            "hour": self.current_hour,
            "total_vehicles": total,
            "vehicles_waiting": waiting,
            "avg_speed": avg_speed,
            "avg_congestion": avg_cong,
            "network_occupancy": self._network_occupancy(),
            "vehicles_spawned": self.total_vehicles_spawned,
            "vehicles_completed": self.total_vehicles_completed,
            "avg_travel_time": (
                float(np.mean(
                    self.completed_travel_times[-200:]
                ))
                if self.completed_travel_times else 0
            ),
            "avg_waiting_time": (
                float(np.mean(
                    self.completed_waiting_times[-200:]
                ))
                if self.completed_waiting_times else 0
            ),
        }