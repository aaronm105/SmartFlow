#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import csv
import argparse

def parse_lanedata(lanedata_path):
    tree = ET.parse(lanedata_path)
    root = tree.getroot()

    rows = []

    for interval in root.findall("interval"):
        begin = float(interval.get("begin", "0"))
        end   = float(interval.get("end",   begin))
        duration = max(1e-6, end - begin)

        densities = []
        speeds    = []
        waiting_times = []
        dwell_estimates = []
        total_veh_moved = 0  # departed + left summed over lanes

        for edge in interval.findall("edge"):
            for lane in edge.findall("lane"):
                dens = float(lane.get("density", "0") or 0.0)
                spd  = float(lane.get("speed",   "0") or 0.0)
                wt   = float(lane.get("waitingTime", "0") or 0.0)

                departed = int(lane.get("departed", "0") or 0)
                left     = int(lane.get("left",     "0") or 0)

                densities.append(dens)
                # only count positive speeds when averaging
                if spd > 0:
                    speeds.append(spd)
                waiting_times.append(wt)

                # simple per-lane dwell estimate: waitingTime / vehicles that came through
                veh_through = departed + left
                if veh_through > 0:
                    dwell_estimates.append(wt / veh_through)

                total_veh_moved += veh_through

        if densities:
            avg_density = sum(densities) / len(densities)
        else:
            avg_density = 0.0

        if speeds:
            avg_speed = sum(speeds) / len(speeds)
        else:
            avg_speed = 0.0

        if waiting_times:
            avg_wait = sum(waiting_times) / len(waiting_times)
        else:
            avg_wait = 0.0

        if dwell_estimates:
            avg_dwell = sum(dwell_estimates) / len(dwell_estimates)
        else:
            avg_dwell = 0.0

        # vehicles per hour over this 120-s interval
        flow_per_hour = (total_veh_moved * 3600.0) / duration if total_veh_moved > 0 else 0.0

        rows.append({
            "begin_s": begin,
            "end_s": end,
            "duration_s": duration,
            "avg_density_veh_per_km_per_lane": avg_density,
            "avg_speed_m_s": avg_speed,
            "flow_veh_per_hour": flow_per_hour,
            "avg_waiting_time_s": avg_wait,
            "avg_dwell_s": avg_dwell,
        })

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanedata", required=True)
    ap.add_argument("--out_csv", default="sumo_metrics.csv")
    args = ap.parse_args()

    rows = parse_lanedata(args.lanedata)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "begin_s",
                "end_s",
                "duration_s",
                "avg_density_veh_per_km_per_lane",
                "avg_speed_m_s",
                "flow_veh_per_hour",
                "avg_waiting_time_s",
                "avg_dwell_s",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"wrote {len(rows)} intervals to {args.out_csv}")

if __name__ == "__main__":
    main()
