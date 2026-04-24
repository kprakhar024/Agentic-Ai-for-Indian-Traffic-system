"""
communication.py - Inter-agent communication system.
Implements a message bus for traffic signal agents to share state,
coordinate green waves, and handle emergencies across the network.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import defaultdict
import threading
import queue


class MessageType(Enum):
    # State sharing
    STATE_UPDATE = "state_update"
    CONGESTION_ALERT = "congestion_alert"
    QUEUE_OVERFLOW = "queue_overflow"

    # Coordination
    GREEN_WAVE_REQUEST = "green_wave_request"
    GREEN_WAVE_ACK = "green_wave_ack"
    PHASE_SYNC_REQUEST = "phase_sync_request"
    PHASE_SYNC_RESPONSE = "phase_sync_response"

    # Emergency
    EMERGENCY_APPROACHING = "emergency_approaching"
    EMERGENCY_CLEARED = "emergency_cleared"
    VIP_CORRIDOR = "vip_corridor"

    # Prediction
    DEMAND_FORECAST = "demand_forecast"
    CONGESTION_PREDICTION = "congestion_prediction"

    # Route guidance
    ROUTE_DIVERSION = "route_diversion"
    ROAD_BLOCKED = "road_blocked"


class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3      # Emergency/VIP


@dataclass
class AgentMessage:
    """Message passed between traffic agents."""
    id: str = ""
    msg_type: MessageType = MessageType.STATE_UPDATE
    sender_id: str = ""
    receiver_id: str = ""           # "" = broadcast
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    ttl: int = 5                    # Time-to-live in hops
    hop_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"msg_{id(self)}_{time.time():.0f}"
        if self.timestamp == 0:
            self.timestamp = time.time()

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value > other.priority.value


class MessageBus:
    """
    Central message bus for inter-agent communication.
    Supports: direct messaging, broadcasting, topic subscriptions,
    priority queuing, and message history for analysis.
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._agent_queues: Dict[str, queue.PriorityQueue] = {}
        self._message_history: List[AgentMessage] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._stats = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_broadcast": 0,
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
        }

    def register_agent(self, agent_id: str):
        """Register an agent to receive messages."""
        with self._lock:
            if agent_id not in self._agent_queues:
                self._agent_queues[agent_id] = queue.PriorityQueue()

    def unregister_agent(self, agent_id: str):
        with self._lock:
            self._agent_queues.pop(agent_id, None)

    def subscribe(self, msg_type: MessageType, callback: Callable):
        """Subscribe to a message type with a callback."""
        self._subscribers[msg_type].append(callback)

    def send(self, message: AgentMessage):
        """Send a message (direct or broadcast)."""
        with self._lock:
            self._stats["total_sent"] += 1
            self._stats["by_type"][message.msg_type.value] += 1
            self._stats["by_priority"][message.priority.value] += 1

            # Store in history
            self._message_history.append(message)
            if len(self._message_history) > self._max_history:
                self._message_history.pop(0)

        if message.receiver_id:
            # Direct message
            self._deliver_to(message.receiver_id, message)
        else:
            # Broadcast to all agents except sender
            self._broadcast(message)

        # Notify subscribers
        for callback in self._subscribers.get(message.msg_type, []):
            try:
                callback(message)
            except Exception as e:
                pass  # Log in production

    def _deliver_to(self, agent_id: str, message: AgentMessage):
        """Deliver message to specific agent's queue."""
        q = self._agent_queues.get(agent_id)
        if q:
            q.put(message)
            self._stats["total_delivered"] += 1

    def _broadcast(self, message: AgentMessage):
        """Broadcast message to all registered agents."""
        self._stats["total_broadcast"] += 1
        for agent_id, q in self._agent_queues.items():
            if agent_id != message.sender_id:
                q.put(AgentMessage(
                    msg_type=message.msg_type,
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    priority=message.priority,
                    timestamp=message.timestamp,
                    payload=message.payload.copy(),
                    ttl=message.ttl,
                    hop_count=message.hop_count,
                ))
                self._stats["total_delivered"] += 1

    def receive(self, agent_id: str, timeout: float = 0.0) -> Optional[AgentMessage]:
        """Receive next message for an agent (highest priority first)."""
        q = self._agent_queues.get(agent_id)
        if not q:
            return None
        try:
            return q.get(timeout=timeout) if timeout > 0 else q.get_nowait()
        except queue.Empty:
            return None

    def receive_all(self, agent_id: str) -> List[AgentMessage]:
        """Receive all pending messages for an agent."""
        messages = []
        while True:
            msg = self.receive(agent_id)
            if msg is None:
                break
            messages.append(msg)
        return sorted(messages, key=lambda m: m.priority.value, reverse=True)

    def get_stats(self) -> Dict:
        return dict(self._stats)

    def get_recent_messages(self, msg_type: Optional[MessageType] = None,
                            count: int = 10) -> List[AgentMessage]:
        """Get recent messages, optionally filtered by type."""
        if msg_type:
            filtered = [m for m in self._message_history if m.msg_type == msg_type]
        else:
            filtered = self._message_history
        return filtered[-count:]


class GreenWaveProtocol:
    """
    Protocol for coordinating green waves across consecutive intersections.
    Enables a 'corridor' of green signals for smooth traffic flow.
    """

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.active_corridors: Dict[str, Dict] = {}

    def request_green_wave(self, corridor_id: str, intersections: List[str],
                           direction: str, speed: float = 40.0,
                           priority: MessagePriority = MessagePriority.HIGH):
        """
        Request a green wave along a corridor.
        Calculates offset timing based on distance and speed.
        """
        self.active_corridors[corridor_id] = {
            "intersections": intersections,
            "direction": direction,
            "speed": speed,
            "status": "requested",
        }

        for i, intersection_id in enumerate(intersections):
            # Calculate time offset for each intersection
            offset = i * (500.0 / (speed / 3.6))  # distance/speed = time offset

            msg = AgentMessage(
                msg_type=MessageType.GREEN_WAVE_REQUEST,
                sender_id="green_wave_controller",
                receiver_id=intersection_id,
                priority=priority,
                payload={
                    "corridor_id": corridor_id,
                    "direction": direction,
                    "offset_seconds": offset,
                    "sequence_index": i,
                    "total_intersections": len(intersections),
                },
            )
            self.bus.send(msg)

    def handle_emergency_corridor(self, route: List[str], direction: str):
        """Create emergency green corridor."""
        self.request_green_wave(
            corridor_id=f"emergency_{time.time():.0f}",
            intersections=route,
            direction=direction,
            speed=60.0,
            priority=MessagePriority.CRITICAL,
        )


class NeighborProtocol:
    """
    Protocol for neighboring intersection agents to share state
    and coordinate signal timing to prevent spillback.
    """

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus

    def share_state(self, agent_id: str, state: Dict):
        """Share intersection state with all neighbors."""
        self.bus.send(AgentMessage(
            msg_type=MessageType.STATE_UPDATE,
            sender_id=agent_id,
            payload=state,
            priority=MessagePriority.NORMAL,
        ))

    def alert_congestion(self, agent_id: str, direction: str,
                         congestion_level: float, queue_length: int):
        """Alert neighbors about congestion buildup."""
        priority = (MessagePriority.HIGH if congestion_level > 0.8
                    else MessagePriority.NORMAL)

        self.bus.send(AgentMessage(
            msg_type=MessageType.CONGESTION_ALERT,
            sender_id=agent_id,
            priority=priority,
            payload={
                "direction": direction,
                "congestion_level": congestion_level,
                "queue_length": queue_length,
            },
        ))

    def notify_queue_overflow(self, agent_id: str, direction: str,
                              overflow_count: int):
        """Notify when queue exceeds capacity (spillback risk)."""
        self.bus.send(AgentMessage(
            msg_type=MessageType.QUEUE_OVERFLOW,
            sender_id=agent_id,
            priority=MessagePriority.HIGH,
            payload={
                "direction": direction,
                "overflow_count": overflow_count,
            },
        ))