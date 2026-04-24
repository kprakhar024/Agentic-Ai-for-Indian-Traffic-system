"""
main.py - Entry point for the Indian AI Traffic Control System.

Runs the complete simulation with:
  ✔ Agentic AI signal control (Q-Learning RL)
  ✔ ML traffic prediction (Neural Net / Gradient Boosting)
  ✔ Inter-agent communication (Message Bus)
  ✔ Dynamic route guidance (A* / Dijkstra)
  ✔ Real-time visualization dashboard

Usage:
    python main.py                    # Run with defaults
    python main.py --steps 1000       # Custom step count
    python main.py --grid 5 5         # Custom grid size
    python main.py --no-viz           # Disable visualization
    python main.py --compare          # Compare AI vs baseline
"""

import sys
import argparse
import time
from typing import List, Dict

from config import SimulationConfig, GridConfig, RLConfig, SignalConfig
from coordinator import TrafficCoordinator
from metrics import PerformanceAnalyzer
from dashboard import Dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚦 Indian AI Traffic Control System"
    )
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps (default: 500)")
    parser.add_argument("--grid", nargs=2, type=int, default=[4, 4],
                        help="Grid size: rows cols (default: 4 4)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--compare", action="store_true",
                        help="Run baseline comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--spawn-rate", type=float, default=0.3,
                        help="Vehicle spawn rate (default: 0.3)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="RL learning rate (default: 0.1)")
    return parser.parse_args()


def run_ai_simulation(config: SimulationConfig,
                      enable_viz: bool = True) -> List[Dict]:
    """Run the AI-controlled traffic simulation."""

    # Create coordinator (orchestrates everything)
    coordinator = TrafficCoordinator(config)

    # Create dashboard
    dashboard = Dashboard(
        grid_rows=config.grid.rows,
        grid_cols=config.grid.cols
    ) if enable_viz else None

    def step_callback(step: int, metrics: Dict):
        if dashboard:
            dashboard.update(
                step, metrics,
                coordinator.environment.intersections,
                coordinator.environment.roads
            )

    # Run simulation
    metrics_history = coordinator.run(
        steps=config.total_steps,
        callback=step_callback
    )

    # Route recommendation demo
    print("\n" + "=" * 70)
    print("  🗺 ROUTE GUIDANCE DEMO")
    print("=" * 70)

    origin = (0, 0)
    destination = (config.grid.rows - 1, config.grid.cols - 1)
    recommendation = coordinator.get_route_recommendation(origin, destination)

    print(f"\n  Route from {origin} to {destination}:")
    for rec in recommendation.get("recommendations", []):
        print(f"    [{rec['rank']}] {rec['label']}")
        print(f"        Path: {' → '.join(str(p) for p in rec['path'])}")
        print(f"        Time: {rec['estimated_time_min']:.1f} min | "
              f"Distance: {rec['distance_km']:.1f} km | "
              f"Congested segments: {rec['congested_segments']}")

    if dashboard:
        dashboard.close()

    return metrics_history


def run_baseline_simulation(config: SimulationConfig) -> List[Dict]:
    """
    Run fixed-time signal baseline for comparison.
    Signals cycle through phases at fixed intervals (no AI).
    """
    print("\n" + "=" * 70)
    print("  ⏱ Running BASELINE (Fixed-Time Signals) for comparison...")
    print("=" * 70)

    # Disable RL learning for baseline
    baseline_config = SimulationConfig(
        total_steps=config.total_steps,
        grid=config.grid,
        signal=config.signal,
        random_seed=config.random_seed,
        vehicle_spawn_rate=config.vehicle_spawn_rate,
        enable_visualization=False,
        rl=RLConfig(
            learning_rate=0.0,    # No learning
            epsilon_start=0.0,    # No exploration — always keep current
            epsilon_end=0.0,
        ),
    )

    coordinator = TrafficCoordinator(baseline_config)

    # Override agents to use fixed cycling
    for agent in coordinator.agents.values():
        agent.epsilon = 0.0  # Always exploit (but Q-table is empty, so random)

    metrics_history = coordinator.run(steps=config.total_steps)
    return metrics_history


def main():
    args = parse_args()

    # Build configuration
    config = SimulationConfig(
        total_steps=args.steps,
        random_seed=args.seed,
        vehicle_spawn_rate=args.spawn_rate,
        enable_visualization=not args.no_viz,
        grid=GridConfig(rows=args.grid[0], cols=args.grid[1]),
        rl=RLConfig(learning_rate=args.lr),
    )

    # Performance analyzer
    analyzer = PerformanceAnalyzer()

    # Run AI simulation
    print("\n🚀 Starting AI-Controlled Traffic Simulation...\n")
    start_time = time.time()

    ai_metrics = run_ai_simulation(config, enable_viz=not args.no_viz)
    ai_time = time.time() - start_time

    print(f"\n  ⏱ AI Simulation completed in {ai_time:.1f}s")

    for m in ai_metrics:
        analyzer.add_step_metrics(m, is_baseline=False)

    # Run baseline comparison if requested
    if args.compare:
        baseline_metrics = run_baseline_simulation(config)
        for m in baseline_metrics:
            analyzer.add_step_metrics(m, is_baseline=True)

    # Generate final performance report
    report = analyzer.generate_report()
    print(report)

    # Print comparison if available
    comparison = analyzer.compare_with_baseline()
    if comparison.get("comparison_available"):
        print("\n  🏆 AI SYSTEM IMPROVEMENT OVER BASELINE:")
        print(f"     Speed:      +{comparison['speed_improvement_%']:.1f}%")
        print(f"     Waiting:    -{comparison['waiting_reduction_%']:.1f}%")
        print(f"     Congestion: -{comparison['congestion_reduction_%']:.1f}%")

    print("\n  ✅ Simulation complete! Thank you for using the "
          "Indian AI Traffic Control System.\n")


if __name__ == "__main__":
    main()