"""
sumo_coordinator.py — Coordinator that uses SUMO instead of custom simulator.
Integrates RL agents, ML predictor, and message bus with SUMO TraCI.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import SimulationConfig, RLConfig, SignalConfig
from models import Direction, Intersection, IntersectionState, TrafficSignal
from sumo_environment import SUMOEnvironment, SUMOIntersectionData
from signal_agent import TrafficSignalAgent
from communication import (
    MessageBus, GreenWaveProtocol, AgentMessage, MessageType
)
from traffic_predictor import TrafficPredictor, TrafficDataPoint
from metrics import PerformanceAnalyzer


class SUMOAgentAdapter:
    """
    Adapts the RL SignalAgent to work with SUMO data
    instead of custom simulation data.
    """

    def __init__(self, tls_id: str, agent: TrafficSignalAgent,
                 sumo_env: SUMOEnvironment):
        self.tls_id = tls_id
        self.agent = agent
        self.sumo_env = sumo_env
        self.prev_throughput = 0

    def update_intersection_from_sumo(self, data: SUMOIntersectionData):
        """Update the agent's intersection state from SUMO data."""
        intersection = self.agent.intersection
        state = intersection.state

        for direction in Direction:
            state.queue_lengths[direction] = data.queue_lengths[direction]
            state.waiting_times[direction] = data.waiting_times[direction]
            state.total_pcu[direction] = data.total_pcu[direction]
            state.emergency_present[direction] = data.emergency_present[direction]

            # Update throughput (difference from last step)
            total_now = sum(data.vehicle_counts.values())
            state.throughput[direction] = max(
                0, self.prev_throughput - total_now
            )

        self.prev_throughput = sum(data.vehicle_counts.values())

    def apply_action_to_sumo(self, action: int):
        """Translate agent action to SUMO signal control."""
        from signal_agent import SignalAction

        tls_id = self.tls_id

        if action == SignalAction.KEEP_CURRENT:
            pass  # Let SUMO continue current phase

        elif action == SignalAction.SWITCH_NEXT:
            # Advance to next phase
            try:
                current = self.sumo_env.get_intersection_data(tls_id)
                num_phases = len(
                    self.sumo_env._get_tls_program_phases(tls_id)
                )
                next_phase = (current.current_phase + 1) % max(num_phases, 1)
                self.sumo_env.set_signal_phase(tls_id, next_phase)
            except Exception:
                pass

        elif action == SignalAction.EXTEND_GREEN:
            # Extend current phase duration
            self.sumo_env.set_phase_duration(tls_id, 10.0)

        elif action == SignalAction.SKIP_TO_BUSIEST:
            # Find busiest direction and set its phase
            data = self.sumo_env.get_intersection_data(tls_id)
            busiest = max(data.queue_lengths, key=data.queue_lengths.get)
            phase_idx = list(Direction).index(busiest) * 2  # Approximate
            self.sumo_env.set_signal_phase(tls_id, phase_idx % 4)

        elif action == SignalAction.EMERGENCY_PREEMPT:
            data = self.sumo_env.get_intersection_data(tls_id)
            for d, has_em in data.emergency_present.items():
                if has_em:
                    phase_idx = list(Direction).index(d) * 2
                    self.sumo_env.set_signal_phase(tls_id, phase_idx % 4)
                    break

        elif action == SignalAction.ALL_RED:
            # Set all red
            state_len = len(
                self.sumo_env._get_tls_state(tls_id)
            )
            self.sumo_env.set_signal_state(tls_id, "r" * state_len)


class SUMOTrafficCoordinator:
    """
    Main coordinator using SUMO for simulation.
    Manages agents, predictor, and message bus.
    """

    def __init__(self, config: SimulationConfig = None,
                 sumo_cfg: str = None, gui: bool = True):
        self.config = config or SimulationConfig()

        # Core components
        self.sumo_env = SUMOEnvironment(self.config, sumo_cfg, gui=gui)
        self.message_bus = MessageBus()
        self.predictor = TrafficPredictor(self.config.predictor)
        self.green_wave = GreenWaveProtocol(self.message_bus)

        # Tracking
        self.step_metrics: List[Dict] = []
        self.agent_metrics: Dict[str, List[Dict]] = {}

        # Agents (created after SUMO starts)
        self.agents: Dict[str, TrafficSignalAgent] = {}
        self.adapters: Dict[str, SUMOAgentAdapter] = {}

    def start(self):
        """Start SUMO and create agents."""
        self.sumo_env.start()
        self._create_agents()

    def _create_agents(self):
        """Create an RL agent for each SUMO traffic light."""
        for tls_id in self.sumo_env.tls_ids:
            # Create intersection model
            intersection = Intersection(
                id=tls_id,
                position=self._parse_tls_position(tls_id),
                signal_config=self.config.signal,
            )

            # Create RL agent
            agent = TrafficSignalAgent(
                intersection=intersection,
                message_bus=self.message_bus,
                rl_config=self.config.rl,
                signal_config=self.config.signal,
            )

            # Create adapter
            adapter = SUMOAgentAdapter(tls_id, agent, self.sumo_env)

            self.agents[tls_id] = agent
            self.adapters[tls_id] = adapter
            self.agent_metrics[tls_id] = []
            self.predictor.register_intersection(tls_id)

        print(f"  ✅ Created {len(self.agents)} RL agents")

    def _parse_tls_position(self, tls_id: str) -> Tuple[int, int]:
        """Parse grid position from TLS ID (e.g., 'I_2_3' -> (2,3))."""
        try:
            parts = tls_id.split('_')
            return (int(parts[1]), int(parts[2]))
        except (IndexError, ValueError):
            return (0, 0)

    def step(self) -> Dict:
        """Execute one complete system step."""
        # 1. Advance SUMO
        self.sumo_env.simulation_step()

        # 2. Read state from SUMO for each intersection
        for tls_id, adapter in self.adapters.items():
            sumo_data = self.sumo_env.get_intersection_data(tls_id)
            adapter.update_intersection_from_sumo(sumo_data)

        # 3. Run RL agent decisions
        sim_time = self.sumo_env.get_simulation_time()
        hour = int(8 + sim_time / 3600) % 24

        for tls_id, agent in self.agents.items():
            # Agent observes, decides, and learns
            # We pass empty roads dict since data comes from SUMO
            agent.step({}, hour)

            # Apply the agent's chosen action to SUMO
            if agent.prev_action is not None:
                self.adapters[tls_id].apply_action_to_sumo(
                    agent.prev_action
                )

        # 4. Feed data to ML predictor (every 10 steps)
        if self.sumo_env.current_step % 10 == 0:
            self._collect_predictor_data(hour)

        # 5. Periodic ML retraining
        if self.sumo_env.current_step % 100 == 0:
            self.predictor.periodic_retrain(self.sumo_env.current_step)

        # 6. Collect metrics
        metrics = self._collect_metrics()
        self.step_metrics.append(metrics)
        return metrics

    def _collect_predictor_data(self, hour: int):
        """Feed SUMO data to ML predictor."""
        for tls_id in self.agents:
            data = self.sumo_env.get_intersection_data(tls_id)

            total_queue = sum(data.queue_lengths.values())
            avg_speed = sum(data.avg_speeds.values()) / max(
                len(data.avg_speeds), 1
            )
            total_pcu = sum(data.total_pcu.values())
            max_pcu = self.config.grid.max_capacity_per_lane * \
                      self.config.grid.lanes_per_direction * 4
            congestion = min(1.0, total_pcu / max(max_pcu, 1))

            dp = TrafficDataPoint(
                timestamp=self.sumo_env.get_simulation_time(),
                hour=hour,
                day_of_week=0,
                is_peak=1 if hour in [8, 9, 17, 18, 19] else 0,
                vehicle_count=total_queue,
                avg_speed=float(avg_speed * 3.6),  # m/s to km/h
                queue_length=total_queue,
                congestion_level=congestion,
                intersection_id=tls_id,
            )
            self.predictor.add_observation(tls_id, dp)

    def _collect_metrics(self) -> Dict:
        """Collect metrics from SUMO."""
        summary = self.sumo_env.get_state_summary()

        # Agent metrics
        all_queues = []
        for tls_id in self.agents:
            data = self.sumo_env.get_intersection_data(tls_id)
            all_queues.append(sum(data.queue_lengths.values()))

        all_rewards = [a.total_reward for a in self.agents.values()]

        summary["max_queue"] = max(all_queues) if all_queues else 0
        summary["avg_queue"] = float(np.mean(all_queues)) if all_queues else 0
        summary["total_messages"] = self.message_bus.get_stats().get(
            "total_sent", 0
        )
        summary["avg_agent_reward"] = (
            float(np.mean(all_rewards)) if all_rewards else 0
        )

        return summary

    def run(self, max_steps: int = None, callback=None) -> List[Dict]:
        """Run the complete SUMO simulation with AI control."""
        max_steps = max_steps or self.config.total_steps

        print("\n" + "=" * 70)
        print("  🚦 SUMO + AI Traffic Control — Simulation Starting")
        print("=" * 70)
        print(f"  Traffic Lights: {len(self.agents)}")
        print(f"  Max Steps: {max_steps}")
        print("=" * 70)

        step = 0
        while step < max_steps and self.sumo_env.is_running():
            metrics = self.step()

            if step % self.config.log_interval == 0:
                print(
                    f"  Step {step:5d} | "
                    f"Time: {metrics.get('sim_time', 0):.0f}s | "
                    f"Vehicles: {metrics.get('total_vehicles', 0):3d} | "
                    f"Waiting: {metrics.get('vehicles_waiting', 0):3d} | "
                    f"Speed: {metrics.get('avg_speed', 0):5.1f} km/h | "
                    f"Queue: {metrics.get('avg_queue', 0):5.1f} | "
                    f"Completed: {metrics.get('vehicles_completed', 0):4d}"
                )

            if callback:
                callback(step, metrics)

            step += 1

        self._print_final_report()
        return self.step_metrics

    def _print_final_report(self):
        """Print final report."""
        if not self.step_metrics:
            return

        print("\n" + "=" * 70)
        print("  📊 SUMO + AI FINAL REPORT")
        print("=" * 70)

        final = self.step_metrics[-1]
        avg = lambda key: np.mean([
            m[key] for m in self.step_metrics
            if isinstance(m.get(key), (int, float))
        ])

        print(f"\n  Steps Simulated:        {len(self.step_metrics)}")
        print(f"  Vehicles Spawned:       {final.get('vehicles_spawned', 0)}")
        print(f"  Vehicles Completed:     {final.get('vehicles_completed', 0)}")
        print(f"  Avg Speed:              {avg('avg_speed'):.1f} km/h")
        print(f"  Avg Congestion:         {avg('avg_congestion'):.4f}")
        print(f"  Avg Queue:              {avg('avg_queue'):.1f}")

        print(f"\n  Agent Performance:")
        for aid, agent in self.agents.items():
            m = agent.get_metrics()
            print(f"    {aid}: Reward={m['total_reward']:.1f}, "
                  f"ε={m['epsilon']:.3f}")

        msg = self.message_bus.get_stats()
        print(f"\n  Messages: {msg.get('total_sent', 0)} sent, "
              f"{msg.get('total_delivered', 0)} delivered")
        print("=" * 70)

    def close(self):
        """Clean shutdown."""
        self.sumo_env.close()