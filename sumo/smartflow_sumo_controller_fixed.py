#!/usr/bin/env python3

import argparse
import csv
import math
import traci

AGG_INTERVAL = 5.0          # seconds between metric updates
VEH_LENGTH_M = 5.0          # assumed vehicle length (for queue length in meters)

THRESHOLD_DENSITY = 0.0005
PTP_BUFFER = 0.0002
PTP_MIN_SEC = 3.0           # min duration above threshold to count a PTP event

CSV_FIELDS = [
    "sim_time_s",
    "scenario_id",
    "tls_id",
    "phase_index",
    "approach_id",          # "N", "S", "E", "W"
    "queue_length_m",
    "density",
    "flow_veh_per_hr",
    "avg_dwell_s",
    "ptp_state",
    "avg_speed_mps",
    "avg_speed_kph",
]

def init_csv(path):
    if not path:
        return None, None
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(CSV_FIELDS)
    f.flush()
    return f, w


def log_metrics(writer, sim_time, scenario_id, tls_id, phase_index, approach_id, metrics):
    if writer is None:
        return
    row = [
        sim_time,
        scenario_id,
        tls_id,
        phase_index,
        approach_id,
        metrics.get("queue_length_m", 0.0),
        metrics.get("density", 0.0),
        metrics.get("flow_veh_per_hr", 0.0),
        metrics.get("avg_dwell_s", 0.0),
        metrics.get("ptp_state", ""),
        metrics.get("avg_speed_mps", 0.0),
        metrics.get("avg_speed_kph", 0.0),
    ]
    writer.writerow(row)

class DwellTracker:
    def __init__(self):
        self.first_seen = {}    

    def update(self, sim_time, vehicle_ids):
        vehicle_ids = set(vehicle_ids)
        for vid in vehicle_ids:
            if vid not in self.first_seen:
                self.first_seen[vid] = sim_time

    def avg_dwell(self, sim_time, vehicle_ids):
        vehicle_ids = set(vehicle_ids)
        if not vehicle_ids:
            return 0.0
        total = 0.0
        count = 0
        for vid in vehicle_ids:
            t0 = self.first_seen.get(vid, sim_time)
            total += (sim_time - t0)
            count += 1
        return total / count if count > 0 else 0.0


class FlowTracker:
    def __init__(self):
        self.prev_ids = set()
        self.last_flow_per_hr = 0.0

    def update(self, vehicle_ids, agg_interval=AGG_INTERVAL):
        vehicle_ids = set(vehicle_ids)
        new_ids = vehicle_ids - self.prev_ids
        self.prev_ids = vehicle_ids

        if agg_interval <= 0:
            self.last_flow_per_hr = 0.0
        else:
            self.last_flow_per_hr = len(new_ids) * 3600.0 / agg_interval

        return self.last_flow_per_hr


class PTPTracker:
    def __init__(self):
        self.density_history = []   # recent densities for smoothing
        self.flag = 0               # 0 = below threshold, 1 = currently above
        self.start_time = 0.0       # when we crossed above threshold
        self.ptp_durations = []     # completed events

    def update(self, sim_time, density, window=30):
        self.density_history.append(density)
        if len(self.density_history) > window:
            self.density_history.pop(0)
        smoothed = sum(self.density_history) / len(self.density_history)

        if smoothed > THRESHOLD_DENSITY and self.flag == 0:
            self.flag = 1
            self.start_time = sim_time
        elif smoothed <= (THRESHOLD_DENSITY - PTP_BUFFER) and self.flag == 1:
            duration = sim_time - self.start_time
            if duration >= PTP_MIN_SEC:
                self.ptp_durations.append(duration)
            self.flag = 0

        if smoothed < THRESHOLD_DENSITY:
            ptp_state = "FREE"
        elif smoothed < 0.002:
            ptp_state = "MODERATE"
        else:
            ptp_state = "HEAVY"

        return smoothed, ptp_state

def compute_smartflow_metrics(sim_time, lane_ids, dwell_tracker, flow_tracker, ptp_tracker):
    total_stopped = 0
    occ_values = []
    speeds = []
    veh_ids_all = []

    for lane in lane_ids:
        halted = traci.lane.getLastStepHaltingNumber(lane)
        occ    = traci.lane.getLastStepOccupancy(lane) / 100.0   # 0–1
        speed  = traci.lane.getLastStepMeanSpeed(lane)           # m/s
        vids   = traci.lane.getLastStepVehicleIDs(lane)

        total_stopped += halted
        occ_values.append(occ)
        veh_ids_all.extend(vids)
        if not math.isnan(speed):
            speeds.append(speed)

    dwell_tracker.update(sim_time, veh_ids_all)
    avg_dwell_s = dwell_tracker.avg_dwell(sim_time, veh_ids_all)

    queue_length_m = total_stopped * VEH_LENGTH_M

    density = sum(occ_values) / len(occ_values) if occ_values else 0.0

    flow_per_hr = flow_tracker.update(veh_ids_all, AGG_INTERVAL)

    avg_speed_mps = sum(speeds) / len(speeds) if speeds else 0.0
    avg_speed_kph = avg_speed_mps * 3.6

    smoothed_density, ptp_state = ptp_tracker.update(sim_time, density)

    return {
        "queue_length_m": queue_length_m,
        "density": smoothed_density,   
        "flow_veh_per_hr": flow_per_hr,
        "avg_dwell_s": avg_dwell_s,
        "ptp_state": ptp_state,
        "avg_speed_mps": avg_speed_mps,
        "avg_speed_kph": avg_speed_kph,
    }


def normalize_cycle_to_120s(tls_id, target_cycle=120.0):
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    total = sum(p.duration for p in logic.phases)
    if total <= 0:
        print(f"[CYCLE] TLS {tls_id}: invalid total cycle {total}, skipping normalization")
        return

    factor = target_cycle / float(total)
    print(f"[CYCLE] TLS {tls_id}: current cycle {total:.1f}s -> scaling by {factor:.3f} to reach ~{target_cycle}s")

    for p in logic.phases:
        p.duration *= factor
        if hasattr(p, "minDur"):
            p.minDur *= factor
        if hasattr(p, "maxDur") and p.maxDur > 0:
            p.maxDur *= factor

    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)

def build_phase_to_approach(tls_id, approaches):
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    lane_to_idx = {lane: i for i, lane in enumerate(controlled_lanes)}

    print(f"\n[DEBUG] TLS {tls_id} controlled_lanes:")
    for i, ln in enumerate(controlled_lanes):
        print(f"   idx {i:2d}: {ln}")
    print("[DEBUG] approaches config:")
    for k, v in approaches.items():
        print(f"   {k}: {v}")

    for approach_name, lane_list in approaches.items():
        for lane in lane_list:
            if lane not in lane_to_idx:
                print(f"[WARN] lane {lane} (approach {approach_name}) "
                      f"is NOT controlled by TLS {tls_id}")

    phase_to_approach = {}

    for p_idx, phase in enumerate(logic.phases):
        state = phase.state  
        active = []

        print(f"\n[DEBUG] phase {p_idx} state = {state}")

        for approach_name, lane_list in approaches.items():
            for lane in lane_list:
                idx = lane_to_idx.get(lane)
                if idx is None:
                    continue
                sig = state[idx]
                if sig in ("G", "g"): 
                    print(f"  approach {approach_name} sees GREEN on lane {lane} (idx {idx})")
                    active.append(approach_name)
                    break  

        phase_to_approach[p_idx] = active
        print(f"[DEBUG] phase {p_idx} serves approaches: {active}")

    print("\n[DEBUG] final phase_to_approach:", phase_to_approach)
    return phase_to_approach

def run_controller(args):
    sumo_cmd = [
        args.sumo_binary,
        "-c", args.config,
        "--step-length", str(args.step_length),
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
    ]

    traci.start(sumo_cmd)

    sim_time = 0.0
    next_agg_time = 0.0

    tls_id = args.tls_id
    scenario_id = args.scenario_id

    ids = traci.trafficlight.getIDList()
    print("TLS IDs in this network:", ids)
    for tid in ids:
        lanes = traci.trafficlight.getControlledLanes(tid)
        print("  TLS", tid, "controls lanes:")
        for ln in lanes:
            print("     ", ln)
            
    if args.normalize_cycle:
        normalize_cycle_to_120s(tls_id, target_cycle=args.target_cycle)

    # CSV
    csv_file, csv_writer = init_csv(args.csv_path)

    approaches = {
        "N": [lane for lane in args.north_lanes.split(",") if lane] if args.north_lanes else [],
        "S": [lane for lane in args.south_lanes.split(",") if lane] if args.south_lanes else [],
        "E": [lane for lane in args.east_lanes.split(",") if lane] if args.east_lanes else [],
        "W": [lane for lane in args.west_lanes.split(",") if lane] if args.west_lanes else [],
    }

    if args.debug_mapping:
        build_phase_to_approach(tls_id, approaches)

    dwell_trackers = {k: DwellTracker() for k in approaches.keys()}
    flow_trackers  = {k: FlowTracker()  for k in approaches.keys()}
    ptp_trackers   = {k: PTPTracker()   for k in approaches.keys()}

    try:
        while sim_time < args.sim_duration:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            if sim_time >= next_agg_time:
                next_agg_time += AGG_INTERVAL

                phase_index = traci.trafficlight.getPhase(tls_id)

                for key, lanes in approaches.items():
                    if not lanes:
                        continue

                    metrics = compute_smartflow_metrics(
                        sim_time,
                        lanes,
                        dwell_trackers[key],
                        flow_trackers[key],
                        ptp_trackers[key]
                    )

                    log_metrics(csv_writer, sim_time, scenario_id, tls_id, phase_index, key, metrics)

        traci.close()
    finally:
        if csv_file:
            csv_file.close()


def parse_args():
    p = argparse.ArgumentParser(description="SmartFlow-style SUMO fixed-time logger")

    p.add_argument("-c", "--config", required=True, help="SUMO .sumocfg file")
    p.add_argument("--sumo-binary", default="sumo", help="Path to sumo or sumo-gui")
    p.add_argument("--step-length", type=float, default=1.0)
    p.add_argument("--sim-duration", type=float, default=3600.0)
    p.add_argument("--tls-id", required=True, help="Traffic light ID to observe")
    p.add_argument("--scenario-id", default="fixed", help="Label for this experiment")
    p.add_argument("--csv-path", default=None, help="CSV file to log metrics")

    p.add_argument("--north-lanes", default="", help="Comma-separated lane IDs for northbound approach")
    p.add_argument("--south-lanes", default="", help="Comma-separated lane IDs for southbound approach")
    p.add_argument("--east-lanes",  default="", help="Comma-separated lane IDs for eastbound approach")
    p.add_argument("--west-lanes",  default="", help="Comma-separated lane IDs for westbound approach")

    p.add_argument("--normalize-cycle", action="store_true")
    p.add_argument("--target-cycle", type=float, default=120.0)

    p.add_argument("--debug-mapping", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_controller(args)
