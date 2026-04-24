"""
signal_agent.py - RL-based traffic signal agent.
FIXED: Enforces minimum green time, prevents rapid switching,
       starts with sensible cycling before RL takes over.
"""

import random
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from config import RLConfig, SignalConfig, VehicleType
from models import (
    Intersection, IntersectionState, TrafficSignal,
    RoadSegment, Vehicle, SignalPhase, Direction
)
from communication import (
    MessageBus, AgentMessage, MessageType, MessagePriority,
    NeighborProtocol
)


@dataclass
class AgentState:
    """Discretized state for Q-learning."""
    queue_n: int = 0
    queue_s: int = 0
    queue_e: int = 0
    queue_w: int = 0
    current_phase: int = 0
    time_of_day: int = 0
    emergency: int = 0
    neighbor_congestion: int = 0

    def to_tuple(self) -> tuple:
        return (
            self.queue_n, self.queue_s, self.queue_e, self.queue_w,
            self.current_phase, self.time_of_day,
            self.emergency, self.neighbor_congestion
        )


class SignalAction:
    """Possible actions for the signal agent."""
    KEEP_CURRENT = 0
    SWITCH_NEXT = 1
    EXTEND_GREEN = 2
    SKIP_TO_BUSIEST = 3
    EMERGENCY_PREEMPT = 4
    ALL_RED = 5

    ALL_ACTIONS = [0, 1, 2, 3, 4, 5]
    NORMAL_ACTIONS = [0, 1, 2, 3]


class TrafficSignalAgent:
    """
    Autonomous RL agent controlling one intersection.
    FIXED: Enforces minimum green time to prevent rapid switching.
    """

    PHASE_ORDER = [Direction.NORTH, Direction.EAST,
                   Direction.SOUTH, Direction.WEST]

    def __init__(self, intersection: Intersection,
                 message_bus: MessageBus,
                 rl_config: RLConfig = None,
                 signal_config: SignalConfig = None):

        self.intersection = intersection
        self.agent_id = intersection.id
        self.bus = message_bus
        self.rl_config = rl_config or RLConfig()
        self.signal_config = signal_config or SignalConfig()

        # Register with message bus
        self.bus.register_agent(self.agent_id)
        self.neighbor_protocol = NeighborProtocol(self.bus)

        # Q-table
        self.q_table: Dict[tuple, Dict[int, float]] = defaultdict(
            lambda: {a: 0.0 for a in SignalAction.ALL_ACTIONS}
        )

        # Exploration
        self.epsilon = self.rl_config.epsilon_start

        # ─── Signal timing state ───
        self.current_phase_idx = 0
        self.green_timer = 0
        self.yellow_timer = 0
        self.all_red_timer = 0
        self.in_yellow = False
        self.in_all_red = False
        self._pending_phase = None

        # ─── FIX: Track minimum green enforcement ───
        self.min_green_elapsed = 0      # Steps of green given in current phase
        self.steps_since_switch = 0     # Steps since last phase change

        # Neighbor state
        self.neighbor_states: Dict[str, Dict] = {}
        self.neighbor_congestion: Dict[str, float] = {}

        # Performance tracking
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []
        self.actions_taken: List[int] = []
        self.vehicles_passed = 0
        self.prev_state: Optional[tuple] = None
        self.prev_action: Optional[int] = None

        self._initialize_signals()

    def _initialize_signals(self):
        """Set initial signal phases."""
        for i, direction in enumerate(self.PHASE_ORDER):
            signal = self.intersection.signals[direction]
            if i == 0:
                signal.phase = SignalPhase.GREEN
                signal.green_time = self.signal_config.min_green_time
                signal.remaining_time = self.signal_config.min_green_time
            else:
                signal.phase = SignalPhase.RED
                signal.remaining_time = 0

        self.current_phase_idx = 0
        self.green_timer = self.signal_config.min_green_time
        self.min_green_elapsed = 0
        self.steps_since_switch = 0

    def _discretize_queue(self, count: int) -> int:
        """Bin queue length into discrete states."""
        if count == 0:
            return 0
        elif count <= 5:
            return 1
        elif count <= 15:
            return 2
        elif count <= 30:
            return 3
        else:
            return 4

    def _get_time_of_day_bin(self, hour: int) -> int:
        if 8 <= hour <= 10:
            return 1
        elif 17 <= hour <= 20:
            return 2
        elif 13 <= hour <= 14:
            return 3
        else:
            return 0

    def _get_neighbor_congestion_bin(self) -> int:
        if not self.neighbor_congestion:
            return 0
        avg = sum(self.neighbor_congestion.values()) / len(
            self.neighbor_congestion
        )
        if avg < 0.3:
            return 0
        elif avg < 0.7:
            return 1
        else:
            return 2

    def observe(self, roads: Dict[str, RoadSegment],
                current_hour: int = 12) -> AgentState:
        """Observe current traffic state."""
        self.intersection.update_state(roads)
        state = self.intersection.state

        return AgentState(
            queue_n=self._discretize_queue(
                state.queue_lengths[Direction.NORTH]
            ),
            queue_s=self._discretize_queue(
                state.queue_lengths[Direction.SOUTH]
            ),
            queue_e=self._discretize_queue(
                state.queue_lengths[Direction.EAST]
            ),
            queue_w=self._discretize_queue(
                state.queue_lengths[Direction.WEST]
            ),
            current_phase=self.current_phase_idx,
            time_of_day=self._get_time_of_day_bin(current_hour),
            emergency=1 if self.intersection.has_emergency else 0,
            neighbor_congestion=self._get_neighbor_congestion_bin(),
        )

    def select_action(self, state: AgentState) -> int:
        """
        Epsilon-greedy action selection.
        FIX: Filter out invalid actions based on timing constraints.
        """
        state_key = state.to_tuple()

        # Emergency override
        if state.emergency == 1 and self.signal_config.emergency_preemption:
            return SignalAction.EMERGENCY_PREEMPT

        # ─── FIX: Determine which actions are VALID right now ───
        valid_actions = self._get_valid_actions()

        if not valid_actions:
            return SignalAction.KEEP_CURRENT

        # Epsilon-greedy among valid actions only
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploit
        q_values = self.q_table[state_key]
        valid_q = {a: q_values[a] for a in valid_actions}
        max_q = max(valid_q.values())
        best = [a for a, q in valid_q.items() if q == max_q]
        return random.choice(best)

    def _get_valid_actions(self) -> List[int]:
        """
        FIX: Only allow actions that make sense given current timing.
        Prevents rapid switching that causes gridlock.
        """
        valid = []

        # KEEP_CURRENT — always valid
        valid.append(SignalAction.KEEP_CURRENT)

        # SWITCH_NEXT — only if minimum green time has elapsed
        if self.min_green_elapsed >= self.signal_config.min_green_time:
            valid.append(SignalAction.SWITCH_NEXT)

        # EXTEND_GREEN — only if not already at max
        if self.green_timer < self.signal_config.max_green_time:
            valid.append(SignalAction.EXTEND_GREEN)

        # SKIP_TO_BUSIEST — only if minimum green elapsed AND
        # busiest direction is different from current
        if self.min_green_elapsed >= self.signal_config.min_green_time:
            busiest = self.intersection.max_queue_direction
            busiest_idx = self.PHASE_ORDER.index(busiest)
            if busiest_idx != self.current_phase_idx:
                # Only skip if busiest has significantly more vehicles
                current_dir = self.PHASE_ORDER[self.current_phase_idx]
                current_q = self.intersection.state.queue_lengths[current_dir]
                busiest_q = self.intersection.state.queue_lengths[busiest]
                if busiest_q > current_q * 1.5 + 5:
                    valid.append(SignalAction.SKIP_TO_BUSIEST)

        # ALL_RED — only rarely valid (safety situations)
        # Don't include in normal actions — wastes green time

        return valid

    def calculate_reward(self, roads: Dict[str, RoadSegment]) -> float:
        """Multi-objective reward function."""
        state = self.intersection.state
        weights = self.rl_config.reward_weights
        reward = 0.0

        # 1. Reward throughput (most important)
        total_throughput = sum(state.throughput.values())
        reward += weights["throughput"] * total_throughput * 3.0

        # 2. Penalize total waiting time
        total_waiting = sum(state.waiting_times.values())
        reward += weights["waiting_time"] * min(total_waiting, 50)

        # 3. Penalize queue length
        total_queue = sum(state.queue_lengths.values())
        reward += weights["queue_length"] * min(total_queue, 100)

        # 4. Emergency delay
        for d in Direction:
            if (state.emergency_present[d] and
                    self.intersection.signals[d].phase != SignalPhase.GREEN):
                reward += weights["emergency_delay"]

        # 5. Reward balanced queues
        queues = list(state.queue_lengths.values())
        if max(queues) > 0:
            balance = min(queues) / max(max(queues), 1)
            reward += balance * 2.0  # Reward balance

        # 6. Bonus for low total queue
        if total_queue < 20:
            reward += 10.0
        elif total_queue < 40:
            reward += 3.0

        return reward

    def learn(self, state: AgentState, action: int, reward: float,
              next_state: AgentState):
        """Q-learning update."""
        state_key = state.to_tuple()
        next_key = next_state.to_tuple()

        current_q = self.q_table[state_key][action]
        next_q_values = self.q_table[next_key]
        max_next_q = max(next_q_values.values())

        new_q = current_q + self.rl_config.learning_rate * (
            reward + self.rl_config.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action] = new_q

        self.epsilon = max(
            self.rl_config.epsilon_end,
            self.epsilon * self.rl_config.epsilon_decay
        )

        self.total_reward += reward

    def execute_action(self, action: int):
        """Execute action with minimum green enforcement."""

        if action == SignalAction.KEEP_CURRENT:
            self.green_timer -= 1
            self.min_green_elapsed += 1

            # Auto-switch after max green time
            if self.green_timer <= 0:
                self._start_yellow()

        elif action == SignalAction.SWITCH_NEXT:
            # ─── FIX: Double-check minimum green ───
            if self.min_green_elapsed < self.signal_config.min_green_time:
                # Override: keep current instead
                self.green_timer -= 1
                self.min_green_elapsed += 1
            else:
                self._start_yellow()

        elif action == SignalAction.EXTEND_GREEN:
            max_green = self.signal_config.max_green_time
            current_total = self.min_green_elapsed + self.green_timer
            extension = min(10, max_green - current_total)
            self.green_timer += max(extension, 0)
            self.min_green_elapsed += 1

        elif action == SignalAction.SKIP_TO_BUSIEST:
            if self.min_green_elapsed < self.signal_config.min_green_time:
                self.green_timer -= 1
                self.min_green_elapsed += 1
            else:
                busiest = self.intersection.max_queue_direction
                target_idx = self.PHASE_ORDER.index(busiest)
                if target_idx != self.current_phase_idx:
                    self._pending_phase = target_idx
                    self._start_yellow()
                else:
                    self.green_timer -= 1
                    self.min_green_elapsed += 1

        elif action == SignalAction.EMERGENCY_PREEMPT:
            em_dir = self.intersection.get_emergency_direction()
            if em_dir:
                target_idx = self.PHASE_ORDER.index(em_dir)
                self._activate_phase(target_idx)
                self.intersection.emergency_preemptions += 1

        elif action == SignalAction.ALL_RED:
            if self.min_green_elapsed >= self.signal_config.min_green_time:
                self._set_all_red()
            else:
                self.green_timer -= 1
                self.min_green_elapsed += 1

        self.actions_taken.append(action)

    def _start_yellow(self):
        """Transition current green to yellow."""
        current_dir = self.PHASE_ORDER[self.current_phase_idx]
        self.intersection.signals[current_dir].phase = SignalPhase.YELLOW
        self.in_yellow = True
        self.yellow_timer = self.signal_config.yellow_time

    def _advance_phase(self):
        """Move to next phase after yellow + all-red."""
        current_dir = self.PHASE_ORDER[self.current_phase_idx]
        self.intersection.signals[current_dir].phase = SignalPhase.RED

        if self._pending_phase is not None:
            self._activate_phase(self._pending_phase)
            self._pending_phase = None
        else:
            self.current_phase_idx = (
                (self.current_phase_idx + 1) % len(self.PHASE_ORDER)
            )
            self._activate_phase(self.current_phase_idx)

    def _activate_phase(self, phase_idx: int):
        """Activate a specific phase with fresh green timer."""
        for d in Direction:
            self.intersection.signals[d].phase = SignalPhase.RED

        target_dir = self.PHASE_ORDER[phase_idx]
        self.intersection.signals[target_dir].phase = SignalPhase.GREEN

        # ─── FIX: Adaptive green time based on queue ───
        queue = self.intersection.state.queue_lengths.get(target_dir, 0)
        adaptive_green = max(
            self.signal_config.min_green_time,
            min(
                self.signal_config.max_green_time,
                self.signal_config.min_green_time + queue * 2
            )
        )

        self.green_timer = adaptive_green
        self.current_phase_idx = phase_idx
        self.min_green_elapsed = 0  # ─── RESET minimum green counter ───
        self.in_yellow = False
        self.in_all_red = False
        self.steps_since_switch = 0

    def _set_all_red(self):
        """Brief all-red clearance."""
        for d in Direction:
            self.intersection.signals[d].phase = SignalPhase.RED
        self.in_all_red = True
        self.all_red_timer = self.signal_config.all_red_time

    def process_messages(self):
        """Process incoming messages."""
        messages = self.bus.receive_all(self.agent_id)
        for msg in messages:
            if msg.msg_type == MessageType.STATE_UPDATE:
                self.neighbor_states[msg.sender_id] = msg.payload
            elif msg.msg_type == MessageType.CONGESTION_ALERT:
                self.neighbor_congestion[msg.sender_id] = msg.payload.get(
                    "congestion_level", 0
                )
            elif msg.msg_type == MessageType.GREEN_WAVE_REQUEST:
                self.bus.send(AgentMessage(
                    msg_type=MessageType.GREEN_WAVE_ACK,
                    sender_id=self.agent_id,
                    receiver_id=msg.sender_id,
                    payload={"status": "accepted"},
                ))

    def broadcast_state(self):
        """Share state with neighbors (every 5 steps to reduce spam)."""
        if self.steps_since_switch % 5 != 0:
            return

        state = self.intersection.state
        self.neighbor_protocol.share_state(
            self.agent_id,
            {
                "queue_lengths": {
                    d.value: q for d, q in state.queue_lengths.items()
                },
                "total_queue": self.intersection.total_queue,
                "current_phase": self.current_phase_idx,
            }
        )

        # Alert if heavily congested
        for direction in Direction:
            pcu = state.total_pcu.get(direction, 0)
            if pcu > 30:
                self.neighbor_protocol.alert_congestion(
                    self.agent_id,
                    direction.value,
                    min(1.0, pcu / 40.0),
                    state.queue_lengths.get(direction, 0),
                )

    def step(self, roads: Dict[str, RoadSegment], current_hour: int = 12):
        """
        Complete agent step.
        FIX: Handle transitions properly, maintain timing invariants.
        """
        self.steps_since_switch += 1

        # ─── Handle yellow transition ───
        if self.in_yellow:
            self.yellow_timer -= 1
            if self.yellow_timer <= 0:
                self._set_all_red()
                self.in_yellow = False
            return  # No decisions during yellow

        # ─── Handle all-red transition ───
        if self.in_all_red:
            self.all_red_timer -= 1
            if self.all_red_timer <= 0:
                self._advance_phase()
                self.in_all_red = False
            return  # No decisions during all-red

        # ─── Normal operation (GREEN phase) ───

        # 1. Process messages (throttled)
        if self.steps_since_switch % 3 == 0:
            self.process_messages()

        # 2. Observe
        state = self.observe(roads, current_hour)

        # 3. Learn from previous step
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.calculate_reward(roads)
            self.learn(self.prev_state, self.prev_action, reward, state)
            self.episode_rewards.append(reward)

        # 4. Select and execute action
        action = self.select_action(state)
        self.execute_action(action)

        self.prev_state = state
        self.prev_action = action

        # 5. Broadcast state (throttled)
        self.broadcast_state()

    def get_metrics(self) -> Dict:
        """Get agent performance metrics."""
        return {
            "agent_id": self.agent_id,
            "total_reward": self.total_reward,
            "avg_reward": (
                float(np.mean(self.episode_rewards[-100:]))
                if self.episode_rewards else 0
            ),
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "vehicles_passed": self.intersection.total_vehicles_passed,
            "emergency_preemptions": self.intersection.emergency_preemptions,
            "total_queue": self.intersection.total_queue,
            "actions_distribution": dict(
                zip(*np.unique(self.actions_taken[-100:],
                               return_counts=True))
            ) if self.actions_taken else {},
        }