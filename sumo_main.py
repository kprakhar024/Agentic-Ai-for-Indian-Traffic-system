"""
main_sumo.py — Entry point for SUMO-integrated AI Traffic Control.

Usage:
    python main_sumo.py                        # GUI mode
    python main_sumo.py --no-gui               # Headless (faster)
    python main_sumo.py --steps 3600           # 1 hour simulation
    python main_sumo.py --generate-network     # Regenerate SUMO files
    python main_sumo.py --compare              # AI vs fixed-time comparison
"""

import sys
import os
import argparse
import time

from config import SimulationConfig, GridConfig, RLConfig
from metrics import PerformanceAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚦 Indian AI Traffic Control — SUMO Edition"
    )
    parser.add_argument("--steps", type=int, default=3600,
                        help="Simulation steps/seconds (default: 3600)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without SUMO GUI (faster)")
    parser.add_argument("--sumo-cfg", type=str, default=None,
                        help="Path to .sumocfg file")
    parser.add_argument("--generate-network", action="store_true",
                        help="Regenerate SUMO network files")
    parser.add_argument("--compare", action="store_true",
                        help="Compare AI vs default SUMO signals")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="RL learning rate")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Steps between log outputs")
    return parser.parse_args()


def generate_network():
    """Generate SUMO network files."""
    print("\n  📐 Generating SUMO network files...")
    network_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sumo_network"
    )
    os.makedirs(network_dir, exist_ok=True)

    gen_script = os.path.join(network_dir, "generate_network.py")
    if os.path.exists(gen_script):
        os.system(f"{sys.executable} {gen_script}")
    else:
        print(f"  ❌ Generator script not found: {gen_script}")
        print(f"     Please create the sumo_network/generate_network.py file")


def run_ai_simulation(args) -> list:
    """Run AI-controlled SUMO simulation."""
    from sumo_coordinator import SUMOTrafficCoordinator

    config = SimulationConfig(
        total_steps=args.steps,
        log_interval=args.log_interval,
        rl=RLConfig(learning_rate=args.lr),
    )

    coordinator = SUMOTrafficCoordinator(
        config=config,
        sumo_cfg=args.sumo_cfg,
        gui=not args.no_gui,
    )

    try:
        coordinator.start()
        metrics = coordinator.run(max_steps=args.steps)
        return metrics
    except FileNotFoundError as e:
        print(f"\n  ❌ {e}")
        print(f"  💡 Run: python main_sumo.py --generate-network")
        return []
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        coordinator.close()


def run_baseline_simulation(args) -> list:
    """Run SUMO with default fixed-time signals (no AI)."""
    import traci

    print("\n" + "=" * 70)
    print("  ⏱ Running BASELINE (SUMO default signals)...")
    print("=" * 70)

    sumo_cfg = args.sumo_cfg
    if not sumo_cfg:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sumo_cfg = os.path.join(
            base_dir, "sumo_network", "indian_grid.sumocfg"
        )

    sumo_binary = "sumo" if args.no_gui else "sumo-gui"
    traci.start([
        sumo_binary, "-c", sumo_cfg,
        "--start", "--quit-on-end",
        "--no-step-log", "true"
    ])

    metrics = []
    step = 0
    total_departed = 0
    total_arrived = 0

    while step < args.steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        total_departed += traci.simulation.getDepartedNumber()
        total_arrived += traci.simulation.getArrivedNumber()

        running = traci.vehicle.getIDCount()
        all_veh = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(v) for v in all_veh] if all_veh else [0]
        waiting = sum(1 for v in all_veh if traci.vehicle.getSpeed(v) < 0.1)

        m = {
            "step": step,
            "total_vehicles": running,
            "vehicles_waiting": waiting,
            "avg_speed": float(sum(speeds) / max(len(speeds), 1) * 3.6),
            "avg_congestion": waiting / max(running, 1),
            "vehicles_spawned": total_departed,
            "vehicles_completed": total_arrived,
            "avg_queue": waiting / max(len(traci.trafficlight.getIDList()), 1),
            "avg_travel_time": 0,
            "avg_waiting_time": 0,
        }
        metrics.append(m)

        if step % args.log_interval == 0:
            print(f"  [Baseline] Step {step:5d} | "
                  f"Vehicles: {running:3d} | "
                  f"Waiting: {waiting:3d} | "
                  f"Speed: {m['avg_speed']:5.1f} km/h")

        step += 1

    traci.close()
    return metrics


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  🚦 Indian AI Traffic Control System — SUMO Edition")
    print("=" * 70)

    # Check SUMO installation
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        print(f"  SUMO_HOME: {sumo_home}")
    else:
        print("  ⚠ SUMO_HOME not set! Set it to your SUMO installation path.")
        print("    Windows: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
        print("    Linux:   export SUMO_HOME=/usr/share/sumo")

    # Generate network if requested
    if args.generate_network:
        generate_network()
        return

    # Check if network exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(base_dir, "sumo_network", "indian_grid.sumocfg")

    if not args.sumo_cfg and not os.path.exists(default_cfg):
        print(f"\n  ⚠ Network files not found!")
        print(f"  Generating network files first...\n")
        generate_network()

    # Performance analyzer
    analyzer = PerformanceAnalyzer()

    # Run AI simulation
    print(f"\n  🤖 Running AI-Controlled Simulation ({args.steps} steps)...\n")
    start_time = time.time()

    ai_metrics = run_ai_simulation(args)
    elapsed = time.time() - start_time

    print(f"\n  ⏱ AI simulation completed in {elapsed:.1f}s")

    for m in ai_metrics:
        analyzer.add_step_metrics(m, is_baseline=False)

    # Run baseline comparison
    if args.compare and ai_metrics:
        baseline_metrics = run_baseline_simulation(args)
        for m in baseline_metrics:
            analyzer.add_step_metrics(m, is_baseline=True)

    # Final report
    if ai_metrics:
        report = analyzer.generate_report()
        print(report)

        comparison = analyzer.compare_with_baseline()
        if comparison.get("comparison_available"):
            print("\n  🏆 AI vs BASELINE:")
            print(f"    Speed:      +{comparison['speed_improvement_%']:.1f}%")
            print(f"    Waiting:    -{comparison['waiting_reduction_%']:.1f}%")
            print(f"    Congestion: -{comparison['congestion_reduction_%']:.1f}%")

    print("\n  ✅ Done! 🚦\n")


if __name__ == "__main__":
    main()