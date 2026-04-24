# test_imports.py — put this in the same folder and run it
print("Testing imports...")

from config import (
    VehicleType, PCU_FACTORS, SPEED_RANGES, INDIAN_TRAFFIC_MIX,
    GridConfig, SignalConfig, RLConfig, PredictorConfig,
    SimulationConfig, HOURLY_DEMAND_MULTIPLIER, PEAK_HOURS
)
print("  ✅ config.py OK")

from models import (
    Direction, SignalPhase, TurnType,
    Vehicle, RoadSegment, IntersectionState, TrafficSignal, Intersection
)
print("  ✅ models.py OK")

from communication import MessageBus, AgentMessage, MessageType
print("  ✅ communication.py OK")

from traffic_predictor import TrafficPredictor, TrafficDataPoint
print("  ✅ traffic_predictor.py OK")

from route_optimizer import RouteOptimizer, TrafficGraph
print("  ✅ route_optimizer.py OK")

from traffic_simulation import TrafficEnvironment
print("  ✅ traffic_simulation.py OK")

from signal_agent import TrafficSignalAgent
print("  ✅ signal_agent.py OK")

from coordinator import TrafficCoordinator
print("  ✅ coordinator.py OK")

from metrics import PerformanceAnalyzer
print("  ✅ metrics.py OK")

from dashboard import Dashboard
print("  ✅ dashboard.py OK")

print("\n🎉 All imports successful! Run: python main.py")