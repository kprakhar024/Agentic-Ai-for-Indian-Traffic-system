"""
generate_network.py — Generates SUMO network files for Indian urban grid.
Creates:
  1. indian_grid.net.xml   — Road network (intersections + roads)
  2. indian_grid.rou.xml   — Vehicle types (Indian mix) + routes
  3. indian_grid.add.xml   — Induction loop detectors
  4. indian_grid.sumocfg   — SUMO configuration file

Run: python generate_network.py
"""

import os
import sys
import subprocess
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
GRID_ROWS = 4
GRID_COLS = 4
ROAD_LENGTH = 300           # meters between intersections
LANES = 2                   # lanes per direction
SPEED_LIMIT = 13.89         # m/s (~50 km/h)
SIMULATION_END = 3600       # seconds (1 hour)
VEHICLE_COUNT = 2000        # total vehicles to generate

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(OUTPUT_DIR, "indian_grid.net.xml")
ROU_FILE = os.path.join(OUTPUT_DIR, "indian_grid.rou.xml")
ADD_FILE = os.path.join(OUTPUT_DIR, "indian_grid.add.xml")
CFG_FILE = os.path.join(OUTPUT_DIR, "indian_grid.sumocfg")
NOD_FILE = os.path.join(OUTPUT_DIR, "indian_grid.nod.xml")
EDG_FILE = os.path.join(OUTPUT_DIR, "indian_grid.edg.xml")
TLL_FILE = os.path.join(OUTPUT_DIR, "indian_grid.tll.xml")
TYP_FILE = os.path.join(OUTPUT_DIR, "indian_grid.typ.xml")


def generate_nodes():
    """Generate intersection nodes."""
    root = ET.Element("nodes")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            node_id = f"I_{r}_{c}"
            x = c * ROAD_LENGTH
            y = (GRID_ROWS - 1 - r) * ROAD_LENGTH  # Invert Y for SUMO

            # All internal nodes are traffic lights
            # Edge nodes can be priority (no signal)
            is_edge = (r == 0 or r == GRID_ROWS - 1 or
                       c == 0 or c == GRID_COLS - 1)
            node_type = "traffic_light" if not is_edge else "traffic_light"

            node = ET.SubElement(root, "node",
                                 id=node_id,
                                 x=str(x),
                                 y=str(y),
                                 type=node_type)

    # Write
    tree = ET.ElementTree(root)
    write_pretty_xml(tree, NOD_FILE)
    print(f"  ✅ Generated {GRID_ROWS * GRID_COLS} nodes → {NOD_FILE}")


def generate_edges():
    """Generate road edges between adjacent intersections."""
    root = ET.Element("edges")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            node_id = f"I_{r}_{c}"

            # East connection
            if c + 1 < GRID_COLS:
                neighbor = f"I_{r}_{c + 1}"
                # Forward (East)
                ET.SubElement(root, "edge",
                              id=f"E_{r}{c}_to_{r}{c + 1}",
                              **{"from": node_id, "to": neighbor},
                              numLanes=str(LANES),
                              speed=str(SPEED_LIMIT),
                              length=str(ROAD_LENGTH))
                # Reverse (West)
                ET.SubElement(root, "edge",
                              id=f"E_{r}{c + 1}_to_{r}{c}",
                              **{"from": neighbor, "to": node_id},
                              numLanes=str(LANES),
                              speed=str(SPEED_LIMIT),
                              length=str(ROAD_LENGTH))

            # South connection
            if r + 1 < GRID_ROWS:
                neighbor = f"I_{r + 1}_{c}"
                # Forward (South)
                ET.SubElement(root, "edge",
                              id=f"E_{r}{c}_to_{r + 1}{c}",
                              **{"from": node_id, "to": neighbor},
                              numLanes=str(LANES),
                              speed=str(SPEED_LIMIT),
                              length=str(ROAD_LENGTH))
                # Reverse (North)
                ET.SubElement(root, "edge",
                              id=f"E_{r + 1}{c}_to_{r}{c}",
                              **{"from": neighbor, "to": node_id},
                              numLanes=str(LANES),
                              speed=str(SPEED_LIMIT),
                              length=str(ROAD_LENGTH))

    tree = ET.ElementTree(root)
    write_pretty_xml(tree, EDG_FILE)
    edge_count = len(root.findall("edge"))
    print(f"  ✅ Generated {edge_count} edges → {EDG_FILE}")


def generate_edge_types():
    """Generate edge type definitions."""
    root = ET.Element("types")
    ET.SubElement(root, "type",
                  id="urban_road",
                  numLanes=str(LANES),
                  speed=str(SPEED_LIMIT),
                  priority="3")
    tree = ET.ElementTree(root)
    write_pretty_xml(tree, TYP_FILE)
    print(f"  ✅ Generated edge types → {TYP_FILE}")


def build_network():
    """Run SUMO netconvert to build .net.xml from nodes and edges."""
    sumo_home = os.environ.get("SUMO_HOME", "")

    # Try to find netconvert
    netconvert = "netconvert"
    if sumo_home:
        candidate = os.path.join(sumo_home, "bin", "netconvert")
        if os.path.exists(candidate) or os.path.exists(candidate + ".exe"):
            netconvert = candidate

    cmd = [
        netconvert,
        "--node-files", NOD_FILE,
        "--edge-files", EDG_FILE,
        "--output-file", NET_FILE,
        "--tls.default-type", "static",
        "--tls.cycle.time", "90",
        "--tls.green.time", "31",
        "--tls.yellow.time", "3",
        "--no-turnarounds", "true",
    ]

    print(f"\n  🔧 Running netconvert...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Network built → {NET_FILE}")
        else:
            print(f"  ⚠ netconvert warnings: {result.stderr[:200]}")
            if os.path.exists(NET_FILE):
                print(f"  ✅ Network file exists → {NET_FILE}")
            else:
                print(f"  ❌ Network generation failed!")
                print(f"     Make sure SUMO is installed and SUMO_HOME is set.")
                print(f"     Command: {' '.join(cmd)}")
                generate_network_fallback()
    except FileNotFoundError:
        print(f"  ❌ netconvert not found! Generating fallback network...")
        print(f"     Install SUMO: https://sumo.dlr.de/docs/Downloads.php")
        print(f"     Set SUMO_HOME environment variable")
        generate_network_fallback()


def generate_network_fallback():
    """Generate a minimal net.xml if netconvert is not available."""
    print(f"  ⚙ Generating minimal network XML directly...")

    # This is a simplified .net.xml — netconvert produces better ones
    root = ET.Element("net", version="1.16",
                      junctionCornerDetail="5",
                      limitTurnSpeed="5.50")

    # Nodes/Junctions
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            jid = f"I_{r}_{c}"
            x = c * ROAD_LENGTH
            y = (GRID_ROWS - 1 - r) * ROAD_LENGTH
            ET.SubElement(root, "junction",
                          id=jid, type="traffic_light",
                          x=str(x), y=str(y), z="0.00",
                          incLanes="", intLanes="",
                          shape=f"{x-5},{y-5} {x+5},{y-5} {x+5},{y+5} {x-5},{y+5}")

    # Edges
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if c + 1 < GRID_COLS:
                for direction, fr, to in [
                    ("E", f"I_{r}_{c}", f"I_{r}_{c+1}"),
                    ("W", f"I_{r}_{c+1}", f"I_{r}_{c}")
                ]:
                    eid = f"E_{fr[2]}{fr[4]}_to_{to[2]}{to[4]}"
                    edge = ET.SubElement(root, "edge", id=eid,
                                         **{"from": fr, "to": to},
                                         priority="3", numLanes=str(LANES),
                                         speed=str(SPEED_LIMIT),
                                         length=str(ROAD_LENGTH))
                    for lane_idx in range(LANES):
                        ET.SubElement(edge, "lane",
                                      id=f"{eid}_{lane_idx}",
                                      index=str(lane_idx),
                                      speed=str(SPEED_LIMIT),
                                      length=str(ROAD_LENGTH))

            if r + 1 < GRID_ROWS:
                for direction, fr, to in [
                    ("S", f"I_{r}_{c}", f"I_{r+1}_{c}"),
                    ("N", f"I_{r+1}_{c}", f"I_{r}_{c}")
                ]:
                    eid = f"E_{fr[2]}{fr[4]}_to_{to[2]}{to[4]}"
                    edge = ET.SubElement(root, "edge", id=eid,
                                         **{"from": fr, "to": to},
                                         priority="3", numLanes=str(LANES),
                                         speed=str(SPEED_LIMIT),
                                         length=str(ROAD_LENGTH))
                    for lane_idx in range(LANES):
                        ET.SubElement(edge, "lane",
                                      id=f"{eid}_{lane_idx}",
                                      index=str(lane_idx),
                                      speed=str(SPEED_LIMIT),
                                      length=str(ROAD_LENGTH))

    tree = ET.ElementTree(root)
    write_pretty_xml(tree, NET_FILE)
    print(f"  ✅ Fallback network → {NET_FILE}")


def generate_vehicle_types_and_routes():
    """
    Generate Indian vehicle types and random routes.
    Models realistic Indian traffic composition.
    """
    root = ET.Element("routes")

    # ─── Indian Vehicle Types ───
    vehicle_types = {
        "two_wheeler": {
            "accel": "2.6", "decel": "4.5", "sigma": "0.8",
            "length": "2.2", "minGap": "1.0", "maxSpeed": "16.67",
            "speedFactor": "1.1", "speedDev": "0.3",
            "vClass": "motorcycle",
            "color": "1,0.8,0",
            "probability": 0.40,
            "guiShape": "motorcycle",
        },
        "auto_rickshaw": {
            "accel": "1.5", "decel": "3.5", "sigma": "0.9",
            "length": "3.0", "minGap": "1.5", "maxSpeed": "11.11",
            "speedFactor": "0.8", "speedDev": "0.2",
            "vClass": "passenger",
            "color": "0,1,0",
            "probability": 0.10,
            "guiShape": "passenger/sedan",
        },
        "car": {
            "accel": "2.6", "decel": "4.5", "sigma": "0.5",
            "length": "4.5", "minGap": "2.5", "maxSpeed": "16.67",
            "speedFactor": "1.0", "speedDev": "0.1",
            "vClass": "passenger",
            "color": "0,0,1",
            "probability": 0.25,
            "guiShape": "passenger",
        },
        "bus": {
            "accel": "1.2", "decel": "4.0", "sigma": "0.5",
            "length": "12.0", "minGap": "3.0", "maxSpeed": "11.11",
            "speedFactor": "0.9", "speedDev": "0.05",
            "vClass": "bus",
            "color": "1,0,0",
            "probability": 0.08,
            "guiShape": "bus",
        },
        "truck": {
            "accel": "1.0", "decel": "3.5", "sigma": "0.5",
            "length": "10.0", "minGap": "3.5", "maxSpeed": "11.11",
            "speedFactor": "0.8", "speedDev": "0.05",
            "vClass": "truck",
            "color": "0.5,0.3,0",
            "probability": 0.05,
            "guiShape": "truck",
        },
        "cycle": {
            "accel": "1.2", "decel": "3.0", "sigma": "0.9",
            "length": "1.8", "minGap": "0.8", "maxSpeed": "5.56",
            "speedFactor": "0.7", "speedDev": "0.3",
            "vClass": "bicycle",
            "color": "0,0.6,0.6",
            "probability": 0.07,
            "guiShape": "bicycle",
        },
        "emergency": {
            "accel": "3.0", "decel": "5.0", "sigma": "0.3",
            "length": "6.0", "minGap": "2.5", "maxSpeed": "22.22",
            "speedFactor": "1.3", "speedDev": "0.1",
            "vClass": "emergency",
            "color": "1,1,1",
            "probability": 0.005,
            "guiShape": "emergency",
        },
    }

    # Write vehicle types
    for vtype_id, params in vehicle_types.items():
        prob = params.pop("probability")
        vtype = ET.SubElement(root, "vType", id=vtype_id, **params)

    # ─── Collect all edges for routing ───
    edges = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if c + 1 < GRID_COLS:
                edges.append(f"E_{r}{c}_to_{r}{c + 1}")
                edges.append(f"E_{r}{c + 1}_to_{r}{c}")
            if r + 1 < GRID_ROWS:
                edges.append(f"E_{r}{c}_to_{r + 1}{c}")
                edges.append(f"E_{r + 1}{c}_to_{r}{c}")

    # ─── Identify entry edges (edges from border nodes) ───
    entry_edges = []
    exit_edges = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            is_border = (r == 0 or r == GRID_ROWS - 1 or
                         c == 0 or c == GRID_COLS - 1)
            if is_border:
                # Outgoing edges from border = entry points
                if c + 1 < GRID_COLS:
                    entry_edges.append(f"E_{r}{c}_to_{r}{c + 1}")
                    exit_edges.append(f"E_{r}{c + 1}_to_{r}{c}")
                if c - 1 >= 0:
                    entry_edges.append(f"E_{r}{c}_to_{r}{c - 1}")
                    exit_edges.append(f"E_{r}{c - 1}_to_{r}{c}")
                if r + 1 < GRID_ROWS:
                    entry_edges.append(f"E_{r}{c}_to_{r + 1}{c}")
                    exit_edges.append(f"E_{r + 1}{c}_to_{r}{c}")
                if r - 1 >= 0:
                    entry_edges.append(f"E_{r}{c}_to_{r - 1}{c}")
                    exit_edges.append(f"E_{r - 1}{c}_to_{r}{c}")

    entry_edges = list(set(entry_edges))
    exit_edges = list(set(exit_edges))

    # ─── Generate vehicles with Indian traffic distribution ───
    random.seed(42)
    probabilities = {k: v["probability"] for k, v
                     in {**vehicle_types,
                         # Restore probabilities
                         "two_wheeler": {"probability": 0.40},
                         "auto_rickshaw": {"probability": 0.10},
                         "car": {"probability": 0.25},
                         "bus": {"probability": 0.08},
                         "truck": {"probability": 0.05},
                         "cycle": {"probability": 0.07},
                         "emergency": {"probability": 0.005},
                         }.items()}

    type_list = list(probabilities.keys())
    type_probs = [probabilities[t] for t in type_list]
    # Normalize
    total = sum(type_probs)
    type_probs = [p / total for p in type_probs]

    # Peak hour traffic pattern (vehicles per 5-minute bin)
    # Simulating 1 hour = 3600 seconds
    peak_pattern = []
    for minute in range(0, 60, 1):
        hour = 8 + minute / 60  # Simulate 8:00 - 9:00 AM (peak)
        if 8 <= hour < 8.5:
            rate = 1.0    # Building up
        elif 8.5 <= hour < 9.0:
            rate = 1.5    # Peak
        else:
            rate = 0.8    # Declining
        peak_pattern.append(rate)

    vehicle_id = 0
    for second in range(0, SIMULATION_END, max(1, SIMULATION_END // VEHICLE_COUNT)):
        # Determine spawn rate based on time
        minute = second // 60
        rate = peak_pattern[min(minute, len(peak_pattern) - 1)]

        if random.random() > rate * 0.8:
            continue

        # Choose vehicle type
        vtype = random.choices(type_list, weights=type_probs, k=1)[0]

        # Choose random origin and destination edges
        origin = random.choice(entry_edges)
        dest = random.choice(exit_edges)

        # Avoid same origin/destination
        attempts = 0
        while dest == origin and attempts < 10:
            dest = random.choice(exit_edges)
            attempts += 1

        if dest == origin:
            continue

        # Create vehicle with route
        trip = ET.SubElement(root, "trip",
                             id=f"v_{vehicle_id}",
                             type=vtype,
                             depart=str(float(second)),
                             departLane="best",
                             departSpeed="max",
                             **{"from": origin, "to": dest})

        vehicle_id += 1

    tree = ET.ElementTree(root)
    write_pretty_xml(tree, ROU_FILE)
    print(f"  ✅ Generated {vehicle_id} vehicles → {ROU_FILE}")


def generate_detectors():
    """Generate induction loop detectors for data collection."""
    root = ET.Element("additional")

    detector_id = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if c + 1 < GRID_COLS:
                for eid in [f"E_{r}{c}_to_{r}{c + 1}",
                            f"E_{r}{c + 1}_to_{r}{c}"]:
                    for lane in range(LANES):
                        ET.SubElement(root, "inductionLoop",
                                      id=f"det_{detector_id}",
                                      lane=f"{eid}_{lane}",
                                      pos=str(ROAD_LENGTH - 10),
                                      freq="60",
                                      file="detector_output.xml")
                        detector_id += 1

            if r + 1 < GRID_ROWS:
                for eid in [f"E_{r}{c}_to_{r + 1}{c}",
                            f"E_{r + 1}{c}_to_{r}{c}"]:
                    for lane in range(LANES):
                        ET.SubElement(root, "inductionLoop",
                                      id=f"det_{detector_id}",
                                      lane=f"{eid}_{lane}",
                                      pos=str(ROAD_LENGTH - 10),
                                      freq="60",
                                      file="detector_output.xml")
                        detector_id += 1

    tree = ET.ElementTree(root)
    write_pretty_xml(tree, ADD_FILE)
    print(f"  ✅ Generated {detector_id} detectors → {ADD_FILE}")


def generate_sumo_config():
    """Generate .sumocfg file."""
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value="indian_grid.net.xml")
    ET.SubElement(inp, "route-files", value="indian_grid.rou.xml")
    ET.SubElement(inp, "additional-files", value="indian_grid.add.xml")

    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value=str(SIMULATION_END))
    ET.SubElement(time_el, "step-length", value="1.0")

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "lateral-resolution", value="0.8")

    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")
    ET.SubElement(report, "no-warnings", value="true")

    gui = ET.SubElement(root, "gui_only")
    ET.SubElement(gui, "start", value="true")

    tree = ET.ElementTree(root)
    write_pretty_xml(tree, CFG_FILE)
    print(f"  ✅ Generated SUMO config → {CFG_FILE}")


def write_pretty_xml(tree, filepath):
    """Write XML with pretty formatting."""
    rough = ET.tostring(tree.getroot(), encoding='unicode')
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="    ")
    # Remove extra XML declaration if present
    lines = pretty.split('\n')
    with open(filepath, 'w') as f:
        for line in lines:
            if line.strip() and not line.strip().startswith('<?xml'):
                f.write(line + '\n')
            elif line.strip().startswith('<?xml'):
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')


def main():
    print("=" * 60)
    print("  🚦 SUMO Network Generator — Indian Traffic Grid")
    print("=" * 60)
    print(f"  Grid: {GRID_ROWS}×{GRID_COLS}")
    print(f"  Road Length: {ROAD_LENGTH}m")
    print(f"  Lanes: {LANES} per direction")
    print(f"  Vehicles: ~{VEHICLE_COUNT}")
    print(f"  Duration: {SIMULATION_END}s")
    print("=" * 60)

    generate_nodes()
    generate_edges()
    generate_edge_types()
    build_network()
    generate_vehicle_types_and_routes()
    generate_detectors()
    generate_sumo_config()

    print("\n" + "=" * 60)
    print("  ✅ All files generated!")
    print(f"  📂 Output directory: {OUTPUT_DIR}")
    print(f"\n  To test: sumo-gui -c {CFG_FILE}")
    print("=" * 60)

    # Cleanup temp files
    for f in [NOD_FILE, EDG_FILE, TLL_FILE, TYP_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


if __name__ == "__main__":
    main()