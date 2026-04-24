import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1️⃣ Throughput Comparison Graph
# ─────────────────────────────────────────────

initial_completion = 3.9
final_completion = 79.7

plt.figure(figsize=(6,4))
plt.bar(["Initial Model", "Final Model"], 
        [initial_completion, final_completion],
        color=["red", "green"])
plt.title("Completion Rate Comparison")
plt.ylabel("Completion Rate (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("completion_comparison.png")
plt.close()

# ─────────────────────────────────────────────
# 2️⃣ Speed vs Step
# ─────────────────────────────────────────────

steps = np.arange(0, 1200, 50)

avg_speed = [
22.0,17.4,14.5,14.4,15.0,15.1,12.2,13.1,
13.4,13.3,13.9,13.9,13.5,13.4,13.5,11.1,
13.6,11.0,12.6,11.9,13.8,14.5,12.8
]

plt.figure(figsize=(7,4))
plt.plot(steps[:len(avg_speed)], avg_speed, marker='o')
plt.title("Average Speed vs Simulation Step")
plt.xlabel("Simulation Step")
plt.ylabel("Speed (km/h)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("speed_vs_step.png")
plt.close()

# ─────────────────────────────────────────────
# 3️⃣ Congestion vs Step
# ─────────────────────────────────────────────

congestion = [
0.001,0.035,0.067,0.095,0.121,0.131,0.142,
0.144,0.137,0.133,0.128,0.137,0.140,
0.144,0.147,0.151,0.151,0.149,0.144,
0.135,0.132,0.135,0.140
]

plt.figure(figsize=(7,4))
plt.plot(steps[:len(congestion)], congestion, marker='s', color='orange')
plt.title("Congestion Level vs Simulation Step")
plt.xlabel("Simulation Step")
plt.ylabel("Congestion Ratio")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("congestion_vs_step.png")
plt.close()

# ─────────────────────────────────────────────
# 4️⃣ Completion Rate Over Time
# ─────────────────────────────────────────────

completed = [
0,1,5,13,23,54,85,123,175,216,267,295,
332,366,405,442,480,523,569,619,662,697,733,782
]

spawned = 1023
completion_percent = [(c/spawned)*100 for c in completed]

plt.figure(figsize=(7,4))
plt.plot(np.linspace(0,1200,len(completion_percent)),
         completion_percent, marker='^', color='green')
plt.title("Completion Rate Growth Over Time")
plt.xlabel("Simulation Step")
plt.ylabel("Completion Rate (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("completion_growth.png")
plt.close()

# ─────────────────────────────────────────────
# 5️⃣ Queue Length Trend
# ─────────────────────────────────────────────

queue = [
0.1,3.1,6.1,8.2,10.8,11.9,12.8,12.6,
12.1,11.9,11.6,12.2,12.6,13.0,
13.2,13.6,13.6,13.2,12.9,12.1,
11.8
]

plt.figure(figsize=(7,4))
plt.plot(steps[:len(queue)], queue, marker='o', color='blue')
plt.title("Average Queue Length vs Step")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (vehicles)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("queue_trend.png")
plt.close()

# ─────────────────────────────────────────────
# 6️⃣ Waiting Ratio Trend
# ─────────────────────────────────────────────

waiting_ratio = [
0,12,35,45,55,57,78,75,64,58,
52,53,52,57,59,81,58,82,65,71,
53,43,58
]

plt.figure(figsize=(7,4))
plt.plot(steps[:len(waiting_ratio)], waiting_ratio, marker='d', color='purple')
plt.title("Waiting Vehicles vs Simulation Step")
plt.xlabel("Simulation Step")
plt.ylabel("Vehicles Waiting")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("waiting_trend.png")
plt.close()

print("✅ All graphs generated successfully!")