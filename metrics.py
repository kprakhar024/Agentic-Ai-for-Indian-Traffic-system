"""
metrics.py - Comprehensive KPI tracking and analytics.
Tracks performance metrics for evaluating the AI traffic system.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import os


class PerformanceAnalyzer:
    """
    Analyzes traffic system performance and compares
    AI-controlled vs baseline (fixed-time) signals.
    """

    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.baseline_metrics: List[Dict] = []

    def add_step_metrics(self, metrics: Dict, is_baseline: bool = False):
        if is_baseline:
            self.baseline_metrics.append(metrics)
        else:
            self.metrics_history.append(metrics)

    def calculate_kpis(self) -> Dict:
        """Calculate Key Performance Indicators."""
        if not self.metrics_history:
            return {}

        # Extract time series
        vehicles = [m["total_vehicles"] for m in self.metrics_history]
        waiting = [m["vehicles_waiting"] for m in self.metrics_history]
        speeds = [m["avg_speed"] for m in self.metrics_history]
        congestion = [m["avg_congestion"] for m in self.metrics_history]
        queues = [m["avg_queue"] for m in self.metrics_history]

        kpis = {
            "throughput": {
                "total_completed": self.metrics_history[-1].get(
                    "vehicles_completed", 0
                ),
                "completion_rate": (
                    self.metrics_history[-1].get("vehicles_completed", 0) /
                    max(self.metrics_history[-1].get("vehicles_spawned", 1), 1)
                ),
            },
            "efficiency": {
                "avg_speed_kmh": float(np.mean(speeds)),
                "speed_std": float(np.std(speeds)),
                "avg_travel_time_s": self.metrics_history[-1].get(
                    "avg_travel_time", 0
                ),
                "avg_waiting_time_s": self.metrics_history[-1].get(
                    "avg_waiting_time", 0
                ),
            },
            "congestion": {
                "avg_congestion": float(np.mean(congestion)),
                "max_congestion": float(np.max(congestion)),
                "congestion_duration": sum(
                    1 for c in congestion if c > 0.5
                ),
                "avg_queue_length": float(np.mean(queues)),
                "max_queue": float(np.max(queues)),
            },
            "waiting": {
                "avg_vehicles_waiting": float(np.mean(waiting)),
                "max_vehicles_waiting": float(np.max(waiting)),
                "waiting_ratio": float(np.mean(
                    [w / max(v, 1) for w, v in zip(waiting, vehicles)]
                )),
            },
            "network": {
                "avg_vehicles_in_network": float(np.mean(vehicles)),
                "peak_vehicles": float(np.max(vehicles)),
            },
        }

        return kpis

    def compare_with_baseline(self) -> Dict:
        """Compare AI system with fixed-time baseline."""
        if not self.baseline_metrics or not self.metrics_history:
            return {"comparison_available": False}

        ai_speeds = [m["avg_speed"] for m in self.metrics_history]
        base_speeds = [m["avg_speed"] for m in self.baseline_metrics]

        ai_waiting = [m["vehicles_waiting"] for m in self.metrics_history]
        base_waiting = [m["vehicles_waiting"] for m in self.baseline_metrics]

        ai_congestion = [m["avg_congestion"] for m in self.metrics_history]
        base_congestion = [m["avg_congestion"] for m in self.baseline_metrics]

        return {
            "comparison_available": True,
            "speed_improvement_%": (
                (np.mean(ai_speeds) - np.mean(base_speeds)) /
                max(np.mean(base_speeds), 0.01) * 100
            ),
            "waiting_reduction_%": (
                (np.mean(base_waiting) - np.mean(ai_waiting)) /
                max(np.mean(base_waiting), 0.01) * 100
            ),
            "congestion_reduction_%": (
                (np.mean(base_congestion) - np.mean(ai_congestion)) /
                max(np.mean(base_congestion), 0.01) * 100
            ),
        }

    def generate_report(self) -> str:
        """Generate a text report of all KPIs."""
        kpis = self.calculate_kpis()
        if not kpis:
            return "No data available for analysis."

        lines = [
            "=" * 60,
            "  📊 PERFORMANCE ANALYSIS REPORT",
            "=" * 60,
            "",
            "  THROUGHPUT",
            f"    Total Vehicles Completed:  {kpis['throughput']['total_completed']}",
            f"    Completion Rate:           {kpis['throughput']['completion_rate']:.1%}",
            "",
            "  EFFICIENCY",
            f"    Average Speed:             {kpis['efficiency']['avg_speed_kmh']:.1f} km/h",
            f"    Speed Std Dev:             {kpis['efficiency']['speed_std']:.1f} km/h",
            f"    Avg Travel Time:           {kpis['efficiency']['avg_travel_time_s']:.1f}s",
            f"    Avg Waiting Time:          {kpis['efficiency']['avg_waiting_time_s']:.1f}s",
            "",
            "  CONGESTION",
            f"    Average Congestion:        {kpis['congestion']['avg_congestion']:.4f}",
            f"    Peak Congestion:           {kpis['congestion']['max_congestion']:.4f}",
            f"    Steps with High Congestion:{kpis['congestion']['congestion_duration']}",
            f"    Average Queue Length:       {kpis['congestion']['avg_queue_length']:.1f}",
            "",
            "  WAITING",
            f"    Avg Vehicles Waiting:      {kpis['waiting']['avg_vehicles_waiting']:.1f}",
            f"    Max Vehicles Waiting:      {kpis['waiting']['max_vehicles_waiting']}",
            f"    Waiting Ratio:             {kpis['waiting']['waiting_ratio']:.1%}",
            "",
            "=" * 60,
        ]

        comparison = self.compare_with_baseline()
        if comparison.get("comparison_available"):
            lines.extend([
                "",
                "  AI vs BASELINE COMPARISON",
                f"    Speed Improvement:         {comparison['speed_improvement_%']:.1f}%",
                f"    Waiting Reduction:         {comparison['waiting_reduction_%']:.1f}%",
                f"    Congestion Reduction:      {comparison['congestion_reduction_%']:.1f}%",
                "=" * 60,
            ])

        return "\n".join(lines)