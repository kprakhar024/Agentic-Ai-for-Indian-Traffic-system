"""
coordinator.py - Central coordinator for multi-agent traffic system.
Manages:
  - Agent lifecycle
  - Global traffic optimization
  - Emergency corridor management
  - Performance monitoring
  - ML predictor integration
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from config import SimulationConfig
from models import Intersection, RoadSegment, Direction
from traffic_simulation import TrafficEnvironment
from signal_agent import TrafficSignalAgent
from communication import MessageBus, GreenWaveProtocol, AgentMessage, MessageType
from traffic_predictor import TrafficPredictor, TrafficDataPoint
from route_optimizer import RouteOptimizer, TrafficGraph


class TrafficCoordinator:
    """
    Central coordinator that orchestrates all components:
    - Traffic signal agents (one per intersection)
    - ML predictor for demand forecasting
    - Route optimizer for dynamic guidance
    - Inter-agent communication
    """
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

        # Core components
        self.environment = TrafficEnvironment(self.config)
        self.message_bus = MessageBus()
        self.predictor = TrafficPredictor(self.config.predictor)
        self.route_optimizer = RouteOptimizer(self.environment.graph)

        # Protocols
        self.green_wave = GreenWaveProtocol(self.message_bus)

        # ─── Performance tracking (MUST come BEFORE _create_agents) ───
        self.step_metrics: List[Dict] = []
        self.agent_metrics: Dict[str, List[Dict]] = {}

        # ─── Create agents AFTER agent_metrics exists ───
        self.agents: Dict[str, TrafficSignalAgent] = {}
        self._create_agents()

    def _create_agents(self):
        """Create an autonomous agent for each intersection."""
        for iid, intersection in self.environment.intersections.items():
            agent = TrafficSignalAgent(
                intersection=intersection,
                message_bus=self.message_bus,
                rl_config=self.config.rl,
                signal_config=self.config.signal,
            )
            self.agents[iid] = agent
            self.predictor.register_intersection(iid)
            self.agent_metrics[iid] = []

    def _collect_predictor_data(self):
        """Feed current traffic data to the ML predictor."""
        for iid, intersection in self.environment.intersections.items():
            state = intersection.state
            total_queue = sum(state.queue_lengths.values())

            # Compute average speed from incoming roads
            speeds = []
            for d, rid in intersection.incoming_roads.items():
                if rid and rid in self.environment.roads:
                    road = self.environment.roads[rid]
                    speeds.append(road.avg_speed)
            avg_speed = np.mean(speeds) if speeds else 30.0

            # Compute congestion level
            pcus = list(state.total_pcu.values())
            congestion = min(1.0, sum(pcus) / (
                self.config.grid.max_capacity_per_lane *
                self.config.grid.lanes_per_direction * 4  # 4 directions
            ))

            data_point = TrafficDataPoint(
                timestamp=self.environment.current_step,
                hour=self.environment.current_hour,
                day_of_week=0,  # Simplified
                is_peak=1 if self.environment.current_hour in [8, 9, 17, 18, 19] else 0,
                vehicle_count=total_queue,
                avg_speed=float(avg_speed),
                queue_length=total_queue,
                weather_factor=1.0,
                event_factor=1.0,
                congestion_level=congestion,
                intersection_id=iid,
            )

            self.predictor.add_observation(iid, data_point)

    def _run_predictions(self):
        """Run ML predictions and share with agents."""
        for iid in self.agents:
            buffer = self.predictor.data_buffer.get(iid, [])
            if len(buffer) < 5:
                continue

            latest = buffer[-1]
            predictions = self.predictor.predict_congestion(
                iid, latest, horizon_steps=3
            )

            # Detect anomalies
            anomaly = self.predictor.detect_anomaly(iid, latest)

            # Share predictions with agent via message bus
            self.message_bus.send(AgentMessage(
                msg_type=MessageType.DEMAND_FORECAST,
                sender_id="coordinator",
                receiver_id=iid,
                payload={
                    "predictions": predictions,
                    "anomaly": anomaly,
                },
            ))

    def _check_green_wave_opportunities(self):
        """Identify corridors where green wave would help."""
        rows, cols = self.config.grid.rows, self.config.grid.cols

        # Check each row (East-West corridors)
        for r in range(rows):
            corridor = [
                self.environment._intersection_id(r, c)
                for c in range(cols)
            ]
            total_queue = sum(
                self.environment.intersections[iid].total_queue
                for iid in corridor
            )

            if total_queue > cols * 15:  # High demand on this corridor
                self.green_wave.request_green_wave(
                    corridor_id=f"EW_row_{r}",
                    intersections=corridor,
                    direction="EAST",
                    speed=35.0,
                )

    def _update_route_guidance(self):
        """Update route optimizer with current road conditions."""
        # Graph roads are already references to the same objects
        # so congestion is automatically updated. Just update
        # any vehicle routes that need re-routing.
        pass

    def step(self) -> Dict:
        """
        Execute one complete system step:
        1. Environment step (spawn, move vehicles)
        2. Collect data for ML predictor
        3. Run agent decisions
        4. Run predictions
        5. Check coordination opportunities
        6. Collect metrics
        """
        # 1. Environment simulation step
        self.environment.step()

        # 2. Feed data to predictor
        self._collect_predictor_data()

        # 3. Each agent makes autonomous decisions
        for agent_id, agent in self.agents.items():
            agent.step(
                self.environment.roads,
                self.environment.current_hour
            )

        # 4. ML predictions (every 10 steps)
        if self.environment.current_step % 10 == 0:
            self._run_predictions()
            self.predictor.periodic_retrain(self.environment.current_step)

        # 5. Green wave coordination (every 30 steps)
        if self.environment.current_step % 30 == 0:
            self._check_green_wave_opportunities()

        # 6. Collect metrics
        metrics = self._collect_metrics()
        self.step_metrics.append(metrics)

        return metrics

    def _collect_metrics(self) -> Dict:
        """Collect comprehensive metrics for this step."""
        env_summary = self.environment.get_state_summary()
        msg_stats = self.message_bus.get_stats()

        agent_data = {}
        for aid, agent in self.agents.items():
            am = agent.get_metrics()
            agent_data[aid] = am
            self.agent_metrics[aid].append(am)

        all_queues = [
            self.environment.intersections[iid].total_queue
            for iid in self.environment.intersections
        ]
        all_rewards = [
            a.total_reward for a in self.agents.values()
        ]

        return {
            **env_summary,
            "max_queue": max(all_queues) if all_queues else 0,
            "avg_queue": float(np.mean(all_queues)) if all_queues else 0,
            "total_messages": msg_stats.get("total_sent", 0),
            "avg_agent_reward": (
                float(np.mean(all_rewards)) if all_rewards else 0
            ),
            "prediction_summary": self.predictor.get_prediction_summary(),
        }

    def run(self, steps: Optional[int] = None, callback=None) -> List[Dict]:
        """
        Run the complete simulation.
        Optional callback for real-time monitoring.
        """
        total_steps = steps or self.config.total_steps

        print("=" * 70)
        print("  🚦 Indian AI Traffic Control System — Simulation Starting")
        print("=" * 70)
        print(f"  Grid: {self.config.grid.rows}x{self.config.grid.cols}")
        print(f"  Intersections: {len(self.agents)}")
        print(f"  Roads: {len(self.environment.roads)}")
        print(f"  Steps: {total_steps}")
        print("=" * 70)

        for step in range(total_steps):
            metrics = self.step()

            # Progress reporting
            if step % self.config.log_interval == 0:
                self._print_step_report(step, metrics)

            if callback:
                callback(step, metrics)

        # Final report
        self._print_final_report()

        return self.step_metrics

    def _print_step_report(self, step: int, metrics: Dict):
        """Print periodic step report."""
        print(f"\n  Step {step:4d} | "
              f"Hour: {metrics['hour']:2d}:00 | "
              f"Vehicles: {metrics['total_vehicles']:3d} | "
              f"Waiting: {metrics['vehicles_waiting']:3d} | "
              f"Avg Speed: {metrics['avg_speed']:5.1f} km/h | "
              f"Avg Queue: {metrics['avg_queue']:5.1f} | "
              f"Congestion: {metrics['avg_congestion']:.3f} | "
              f"Completed: {metrics['vehicles_completed']:4d}")

    def _print_final_report(self):
        """Print comprehensive final report."""
        if not self.step_metrics:
            return

        print("\n" + "=" * 70)
        print("  📊 FINAL SIMULATION REPORT")
        print("=" * 70)

        # Overall statistics
        final = self.step_metrics[-1]
        avg_metrics = {
            key: np.mean([m[key] for m in self.step_metrics
                          if isinstance(m.get(key), (int, float))])
            for key in ["total_vehicles", "vehicles_waiting",
                        "avg_speed", "avg_congestion", "avg_queue"]
        }

        print(f"\n  Total Steps Simulated:     {len(self.step_metrics)}")
        print(f"  Total Vehicles Spawned:    {final['vehicles_spawned']}")
        print(f"  Total Vehicles Completed:  {final['vehicles_completed']}")
        print(f"  Completion Rate:           "
              f"{final['vehicles_completed'] / max(final['vehicles_spawned'], 1) * 100:.1f}%")
        print(f"\n  Average Metrics:")
        print(f"    Vehicles in Network:     {avg_metrics['total_vehicles']:.1f}")
        print(f"    Vehicles Waiting:        {avg_metrics['vehicles_waiting']:.1f}")
        print(f"    Average Speed:           {avg_metrics['avg_speed']:.1f} km/h")
        print(f"    Average Congestion:      {avg_metrics['avg_congestion']:.4f}")
        print(f"    Average Queue Length:     {avg_metrics['avg_queue']:.1f}")

        if final['avg_travel_time'] > 0:
            print(f"\n  Travel Time Stats:")
            print(f"    Avg Travel Time:         {final['avg_travel_time']:.1f}s")
            print(f"    Avg Waiting Time:        {final['avg_waiting_time']:.1f}s")

        # Agent performance
        print(f"\n  Agent Performance:")
        for aid, agent in self.agents.items():
            m = agent.get_metrics()
            print(f"    {aid}: Reward={m['total_reward']:.1f}, "
                  f"ε={m['epsilon']:.3f}, "
                  f"Q-states={m['q_table_size']}")

        # Message bus stats
        msg_stats = self.message_bus.get_stats()
        print(f"\n  Communication Stats:")
        print(f"    Total Messages Sent:     {msg_stats['total_sent']}")
        print(f"    Total Delivered:         {msg_stats['total_delivered']}")
        print(f"    Broadcasts:              {msg_stats['total_broadcast']}")

        print("\n" + "=" * 70)

    def get_route_recommendation(self, origin: Tuple[int, int],
                                 destination: Tuple[int, int]) -> Dict:
        """Get route recommendation for a vehicle."""
        return self.route_optimizer.get_recommended_routes(
            origin, destination
        )