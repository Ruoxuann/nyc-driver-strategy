"""Query the optimal repositioning action for a given zone and time.

Usage:
    python -m nyc_taxi_strategy.query --zone 161 --time 14:00
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TIME_SLOT_MINUTES = 30


def _time_to_slot(time_str: str, shift_start: str) -> int:
    """Convert a wall-clock time string to a shift time slot index."""
    th, tm = map(int, time_str.split(":"))
    sh, sm = map(int, shift_start.split(":"))
    minutes = (th * 60 + tm) - (sh * 60 + sm)
    if minutes < 0:
        raise ValueError(f"Time {time_str} is before shift start {shift_start}")
    return minutes // TIME_SLOT_MINUTES


def _slot_to_time(slot: int, shift_start: str) -> str:
    sh, sm = map(int, shift_start.split(":"))
    total = sh * 60 + sm + slot * TIME_SLOT_MINUTES
    return f"{total // 60:02d}:{total % 60:02d}"


def run_query(cfg, zone: int, time_str: str):
    from nyc_taxi_strategy.graph.zone_graph import (
        build_zone_graph, compute_all_pairs_travel_time,
    )
    from nyc_taxi_strategy.graph.dp_engine import DPEngine, ZoneStats
    from nyc_taxi_strategy.models.train import load_models
    from nyc_taxi_strategy.features.transformers import (
        CyclicTimeEncoder, LagFeatureBuilder, RollingStatsBuilder,
        HolidayFeature, FeaturePipeline,
    )
    from nyc_taxi_strategy.data.etl import query_demand

    # --- Load models ---
    model_dir = Path(cfg.evaluation.output_dir) / "models"
    wait_model, fare_model, feature_cols = load_models(model_dir)

    # --- Build graph ---
    G = build_zone_graph(
        cfg.data.db_path,
        travel_time_agg=cfg.graph.travel_time_source,
        fuel_cost_per_mile=cfg.simulation.fuel_cost_per_mile,
    )
    active_zones = [z for z in G.nodes() if G.degree(z) > 0]
    travel_times = compute_all_pairs_travel_time(G, active_zones)
    travel_distances = {(u, v): G.edges[u, v]["weight"].distance_miles for u, v in G.edges()}

    # --- Build features ---
    df = query_demand(cfg.data.db_path)
    df["wait_time_s"] = 3600.0 / df["trip_count"].clip(lower=1)
    feat_cfg = cfg.features
    transformers = [CyclicTimeEncoder()]
    if feat_cfg.lag_hours:
        transformers.append(LagFeatureBuilder(lag_hours=feat_cfg.lag_hours))
    if feat_cfg.rolling_windows:
        transformers.append(RollingStatsBuilder(windows=feat_cfg.rolling_windows))
    if feat_cfg.use_holidays:
        transformers.append(HolidayFeature())
    df_feat = FeaturePipeline(transformers).fit_transform(df).dropna()

    model_zones = [z for z in active_zones if not df_feat[df_feat["pickup_zone"] == z].empty]

    if zone not in model_zones:
        print(f"Zone {zone} has no historical data. Available zones: {sorted(model_zones)}")
        return None

    # --- Build zone stats for DP ---
    sh, sm = map(int, cfg.simulation.shift_start.split(":"))
    eh, em = map(int, cfg.simulation.shift_end.split(":"))
    n_slots = ((eh * 60 + em) - (sh * 60 + sm)) // TIME_SLOT_MINUTES
    start_hour = sh

    conn = sqlite3.connect(cfg.data.db_path)
    trans = pd.read_sql_query(
        "SELECT pickup_zone, dropoff_zone, SUM(trip_count) as cnt "
        "FROM zone_transitions GROUP BY pickup_zone, dropoff_zone", conn,
    )
    conn.close()

    dropoff_probs: dict = {}
    for z, grp in trans.groupby("pickup_zone"):
        total = grp["cnt"].sum()
        dropoff_probs[z] = {int(r.dropoff_zone): r.cnt / total for r in grp.itertuples()}

    zone_stats: dict = {}
    for z in model_zones:
        zdf = df_feat[df_feat["pickup_zone"] == z]
        zone_stats[z] = {}
        for slot in range(n_slots):
            hour = (start_hour + slot * TIME_SLOT_MINUTES // 60) % 24
            hdf = zdf[zdf["hour_start"].dt.hour == hour]
            if hdf.empty:
                hdf = zdf
            X = hdf[feature_cols].mean().to_frame().T
            pred_wait = max(60.0, float(wait_model.predict(X)[0]))
            pred_fare = max(5.0, float(fare_model.predict(X)[0]))
            pred_dur = max(120.0, float(hdf["avg_duration_s"].mean()))
            probs = dropoff_probs.get(z, {z: 1.0}) or {z: 1.0}
            best_drop = max(probs, key=probs.get)
            zone_stats[z][slot] = ZoneStats(
                expected_wait_s=pred_wait,
                expected_fare=pred_fare,
                expected_trip_duration_s=pred_dur,
                expected_dropoff_zone=best_drop,
                dropoff_distribution=probs,
            )

    # --- Solve DP ---
    engine = DPEngine(zones=model_zones, n_time_slots=n_slots,
                      time_slot_minutes=TIME_SLOT_MINUTES)
    dp_result = engine.solve(
        zone_stats=zone_stats,
        travel_times=travel_times,
        travel_distances=travel_distances,
        fuel_cost_per_mile=cfg.simulation.fuel_cost_per_mile,
    )

    # --- Query result ---
    current_slot = _time_to_slot(time_str, cfg.simulation.shift_start)
    if current_slot >= n_slots:
        print(f"Time {time_str} is at or after shift end {cfg.simulation.shift_end}.")
        return

    current_stats = zone_stats.get(zone, {}).get(current_slot)
    if current_stats is None:
        print(f"No data for zone {zone} at {time_str}.")
        return

    best_action = dp_result.policy.get(zone, {}).get(current_slot, -1)

    # Compute forced-stay value: fare + V[dropoff zone after trip]
    wait_slots = int(__import__("math").ceil(
        current_stats.expected_wait_s / (TIME_SLOT_MINUTES * 60)))
    trip_slots = int(__import__("math").ceil(
        current_stats.expected_trip_duration_s / (TIME_SLOT_MINUTES * 60)))
    dropoff_arrival = current_slot + wait_slots + trip_slots
    dropoff_zone = current_stats.expected_dropoff_zone
    if dropoff_arrival < n_slots:
        forced_stay_value = (current_stats.expected_fare
                             + dp_result.value_table.get(dropoff_zone, {}).get(dropoff_arrival, 0.0))
    else:
        forced_stay_value = current_stats.expected_fare * 0.5

    print()
    print("=" * 55)
    print(f"  Current zone : {zone}")
    print(f"  Current time : {time_str}  (slot {current_slot} of {n_slots})")
    print(f"  Shift ends   : {cfg.simulation.shift_end}")
    print("=" * 55)
    print()
    print("  [Stay in current zone]")
    print(f"    Expected wait     : {current_stats.expected_wait_s/60:.1f} min")
    print(f"    Expected fare     : ${current_stats.expected_fare:.2f}")
    print(f"    Expected value    : ${forced_stay_value:.2f}  (remaining shift)")
    print()

    if best_action == -1 or best_action == zone:
        print("  Recommendation: STAY in zone", zone)
        print("  Repositioning to any other zone does not improve expected revenue.")
    else:
        travel_t = travel_times.get((zone, best_action))
        travel_d = travel_distances.get((zone, best_action), 0.0)
        fuel_cost = travel_d * cfg.simulation.fuel_cost_per_mile
        # Use the slot after travel time to get the correct target state value
        travel_slots = int(__import__("math").ceil(travel_t / (TIME_SLOT_MINUTES * 60))) if travel_t else 1
        arrival_slot = current_slot + travel_slots
        target_stats = zone_stats.get(best_action, {}).get(arrival_slot)
        reposition_value = dp_result.value_table.get(best_action, {}).get(arrival_slot, 0.0) - fuel_cost

        print(f"  [Reposition to zone {best_action}]")
        if travel_t:
            print(f"    Travel time       : {travel_t/60:.1f} min  (arrive at {_slot_to_time(arrival_slot, cfg.simulation.shift_start)})")
        print(f"    Fuel cost         : ${fuel_cost:.2f}")
        if target_stats:
            print(f"    Expected wait     : {target_stats.expected_wait_s/60:.1f} min")
            print(f"    Expected fare     : ${target_stats.expected_fare:.2f}")
        print(f"    Expected value    : ${reposition_value:.2f}  (remaining shift after travel cost)")
        gain = reposition_value - forced_stay_value
        print()
        print(f"  Recommendation: REPOSITION to zone {best_action}")
        print(f"  Expected gain over staying: ${gain:+.2f}")

    # --- Show full policy for remaining slots ---
    print()
    print("  Optimal policy from current time onwards:")
    print(f"  {'Time':<8} {'Zone':<8} {'Action':<20} {'Expected Value':>15}")
    print("  " + "-" * 53)
    current_z = zone
    route = []
    for slot in range(current_slot, n_slots):
        t = _slot_to_time(slot, cfg.simulation.shift_start)
        action = dp_result.policy.get(current_z, {}).get(slot, -1)
        val = dp_result.value_table.get(current_z, {}).get(slot, 0.0)
        action_str = "stay" if action == -1 or action == current_z else f"go to zone {action}"
        route.append((t, current_z, action_str, val))
        if slot < current_slot + 8:
            print(f"  {t:<8} {current_z:<8} {action_str:<20} ${val:>14.2f}")
        elif slot == current_slot + 8:
            print(f"  ... ({n_slots - current_slot - 8} more slots until shift end)")
        if action != -1 and action != current_z:
            current_z = action
    print()
    return route, dp_result, zone_stats, n_slots


if __name__ == "__main__":
    import argparse
    from nyc_taxi_strategy.utils.config import load_config

    parser = argparse.ArgumentParser(
        description="Query optimal repositioning action for a given zone and time"
    )
    parser.add_argument("--zone", type=int, required=True, help="Current taxi zone ID (1-263)")
    parser.add_argument("--time", required=True, help="Current time in HH:MM format")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(args.config)
    run_query(cfg, args.zone, args.time)
