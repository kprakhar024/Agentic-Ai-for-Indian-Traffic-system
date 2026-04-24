"""
dashboard.py - Real-time visualization dashboard using matplotlib.
Fixed:
✅ No overlapping Y-axes
✅ Proper twinx handling
✅ Stable heatmap
✅ Clean legends
✅ Professional rendering
"""

import numpy as np
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[!] matplotlib not found. Install: pip install matplotlib")


class TextDashboard:
    """Fallback text dashboard."""

    def __init__(self, grid_rows=4, grid_cols=4):
        self.rows = grid_rows
        self.cols = grid_cols

    def update(self, step, metrics, intersections, roads):
        if step % 50 != 0:
            return
        print(f"\nStep {step} | Vehicles: {metrics.get('total_vehicles')} "
              f"| Speed: {metrics.get('avg_speed'):.1f}")

    def close(self):
        pass


class Dashboard:
    """
    Real-time matplotlib dashboard.
    Panels:
    1. Grid
    2. Heatmap
    3. Performance
    4. Learning (vehicles + rewards)
    """

    def __init__(self, grid_rows=4, grid_cols=4):
        if not HAS_MATPLOTLIB:
            self.fallback = TextDashboard(grid_rows, grid_cols)
            self.active = False
            return

        self.active = True
        self.rows = grid_rows
        self.cols = grid_cols

        # Data storage
        self.steps: List[int] = []
        self.speed_history: List[float] = []
        self.congestion_history: List[float] = []
        self.queue_history: List[float] = []
        self.vehicle_history: List[float] = []
        self.waiting_history: List[float] = []
        self.reward_history: List[float] = []

        # Heatmap colormap
        self.cmap = LinearSegmentedColormap.from_list(
            'traffic', ['#2ecc71', '#f1c40f', '#e74c3c']
        )

        # Figure layout
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.suptitle(
            'Indian AI Traffic Control System -- Live Dashboard',
            fontsize=14, fontweight='bold'
        )

        gs = self.fig.add_gridspec(
            2, 3,
            width_ratios=[1, 1, 0.05],
            hspace=0.35,
            wspace=0.35,
            left=0.07,
            right=0.93,
            top=0.92,
            bottom=0.08
        )

        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_cbar = self.fig.add_subplot(gs[0, 2])
        self.ax_perf = self.fig.add_subplot(gs[1, 0])
        self.ax_learn = self.fig.add_subplot(gs[1, 1:])

        # Heatmap image (created once)
        self._heatmap_img = self.ax_heat.imshow(
            np.zeros((self.rows, self.cols)),
            cmap=self.cmap,
            interpolation='bilinear',
            aspect='equal',
            vmin=0,
            vmax=50
        )

        self._colorbar = self.fig.colorbar(
            self._heatmap_img,
            cax=self.ax_cbar,
            label='Total PCU'
        )

        self._heat_texts = []
        self._ax_reward = None   # ✅ Secondary axis created only once

        plt.ion()
        self.fig.show()

    # ─────────────────────────────────────────
    # UPDATE
    # ─────────────────────────────────────────
    def update(self, step: int, metrics: Dict,
               intersections: Dict, roads: Dict):

        if not self.active:
            return

        if step % 10 != 0:
            return

        # Store data
        self.steps.append(step)
        self.speed_history.append(metrics.get("avg_speed", 0))
        self.congestion_history.append(metrics.get("avg_congestion", 0))
        self.queue_history.append(metrics.get("avg_queue", 0))
        self.vehicle_history.append(metrics.get("total_vehicles", 0))
        self.waiting_history.append(metrics.get("vehicles_waiting", 0))
        self.reward_history.append(metrics.get("avg_agent_reward", 0))

        try:
            self._draw_grid(self.ax_grid, intersections, roads)
            self._update_heatmap(intersections)
            self._draw_performance(self.ax_perf)
            self._draw_learning(self.ax_learn)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        except Exception as e:
            print("[Dashboard error]", e)

    # ─────────────────────────────────────────
    # GRID PANEL
    # ─────────────────────────────────────────
    def _draw_grid(self, ax, intersections, roads):
        ax.clear()
        ax.set_title('Traffic Grid -- Queue Sizes')
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Roads
        for road in roads.values():
            fr, fc = road.from_intersection
            tr, tc = road.to_intersection
            ax.plot([fc, tc], [fr, tr],
                    color=self.cmap(min(road.congestion_level, 1.0)),
                    linewidth=2,
                    alpha=0.5)

        # Intersections
        for inter in intersections.values():
            r, c = inter.position
            q = inter.total_queue

            if q < 5:
                color = '#2ecc71'
            elif q < 15:
                color = '#f1c40f'
            elif q < 30:
                color = '#e67e22'
            else:
                color = '#e74c3c'

            ax.scatter(c, r, s=300, c=color, edgecolors='black')
            ax.annotate(str(q), (c, r),
                        ha='center', va='center',
                        fontsize=9, fontweight='bold')

        ax.grid(True, alpha=0.2)

    # ─────────────────────────────────────────
    # HEATMAP PANEL
    # ─────────────────────────────────────────
    def _update_heatmap(self, intersections):
        # Remove old labels
        for txt in self._heat_texts:
            txt.remove()
        self._heat_texts = []

        heatmap = np.zeros((self.rows, self.cols))

        for inter in intersections.values():
            r, c = inter.position
            heatmap[r, c] = sum(inter.state.total_pcu.values())

        self._heatmap_img.set_data(heatmap)
        vmax = max(heatmap.max(), 1.0)
        self._heatmap_img.set_clim(0, vmax)

        for r in range(self.rows):
            for c in range(self.cols):
                val = heatmap[r, c]
                t = self.ax_heat.text(
                    c, r, f'{val:.0f}',
                    ha='center', va='center',
                    color='white' if val > vmax/2 else 'black',
                    fontsize=8
                )
                self._heat_texts.append(t)

    # ─────────────────────────────────────────
    # PERFORMANCE PANEL
    # ─────────────────────────────────────────
    def _draw_performance(self, ax):
        ax.clear()
        ax.set_title('Performance Over Time')

        if len(self.steps) < 2:
            return

        ax.plot(self.steps, self.speed_history,
                'g-', label='Avg Speed', linewidth=2)
        ax.plot(self.steps, self.queue_history,
                'r-', label='Avg Queue', linewidth=2)
        ax.plot(self.steps, self.waiting_history,
                'm--', label='Waiting', linewidth=1.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ─────────────────────────────────────────
    # LEARNING PANEL (FIXED VERSION)
    # ─────────────────────────────────────────
    def _draw_learning(self, ax):
        ax.clear()
        ax.set_title('Vehicles & Agent Rewards')

        if len(self.steps) < 2:
            return

        # Primary axis
        line1, = ax.plot(self.steps, self.vehicle_history,
                         'b-', label='Total Vehicles', linewidth=2)
        line2, = ax.plot(self.steps, self.waiting_history,
                         'r--', label='Waiting', linewidth=1.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Vehicle Count')
        ax.grid(True, alpha=0.3)

        # ✅ Secondary axis created only once
        if self._ax_reward is None:
            self._ax_reward = ax.twinx()

        self._ax_reward.clear()

        line3, = self._ax_reward.plot(
            self.steps, self.reward_history,
            'g-', label='Avg Reward', linewidth=2
        )

        self._ax_reward.set_ylabel('Avg Reward', color='green')

        # Combined legend
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

    # ─────────────────────────────────────────
    # CLOSE
    # ─────────────────────────────────────────
    def close(self):
        if self.active:
            plt.ioff()
            plt.show(block=False)
            plt.pause(1)
            plt.close(self.fig)