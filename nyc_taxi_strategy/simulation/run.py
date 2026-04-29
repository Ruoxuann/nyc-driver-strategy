"""Run Monte Carlo simulations for one or all strategies."""

import logging
import pickle
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TIME_SLOT_MINUTES = 30


def _shift_slots(shift_start: str, shift_end: str) -> int:
    sh, sm = map(int, shift_start.split(":"))
    eh, em = map(int, shift_end.split(":"))
    total_minutes = (eh * 60 + em) - (sh * 60 + sm)
    return total_minutes // TIME_SLOT_MINUTES


def _shift_start_hour(shift_start: str) -> int:
    return int(shift_start.split(":")[0])


def build_zone_models(cfg, wait_model, fare_model, feature_cols, active_zones, n_time_slots):
    from nyc_taxi_strategy.data.etl import query_demand
    from nyc_taxi_strategy.features.transformers import (
        CyclicTimeEncoder, LagFeatureBuilder, RollingStatsBuilder,
        HolidayFeature, FeaturePipeline,
    )
    from nyc_taxi_strategy.simulation.simulator import ZoneModel

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

    conn = sqlite3.connect(cfg.data.db_path)
    trans = pd.read_sql_query(
        "SELECT pickup_zone, dropoff_zone, SUM(trip_count) as cnt "
        "FROM zone_transitions GROUP BY pickup_zone, dropoff_zone", conn,
    )
    conn.close()

    dropoff_probs: dict = {}
    for zone, grp in trans.groupby("pickup_zone"):
        total = grp["cnt"].sum()
        dropoff_probs[zone] = {int(r.dropoff_zone): r.cnt / total for r in grp.itertuples()}

    start_hour = _shift_start_hour(cfg.simulation.shift_start)
    zone_models: dict = {}
    for zone in active_zones:
        zdf = df_feat[df_feat["pickup_zone"] == zone]
        if zdf.empty:
            continue
        zone_models[zone] = {}
        for slot in range(n_time_slots):
            hour = (start_hour + slot * TIME_SLOT_MINUTES // 60) % 24
            hdf = zdf[zdf["hour_start"].dt.hour == hour]
            if hdf.empty:
                hdf = zdf
            X_slot = hdf[feature_cols].mean().to_frame().T
            pred_wait = float(wait_model.predict(X_slot)[0])
            pred_fare = float(fare_model.predict(X_slot)[0])
            probs = dropoff_probs.get(zone, {zone: 1.0}) or {zone: 1.0}
            zone_models[zone][slot] = ZoneModel(
                expected_wait_s=max(60.0, pred_wait),
                expected_fare=max(5.0, pred_fare),
                fare_std=max(1.0, float(hdf["avg_fare"].std()) if len(hdf) > 1 else 2.0),
                expected_duration_s=max(120.0, float(hdf["avg_duration_s"].mean())),
                dropoff_probs=probs,
            )
    logger.info("Zone models built: %d zones × %d slots", len(zone_models), n_time_slots)
    return zone_models


def run_strategy(cfg, strategy_name: str, start_zone: int | None = None) -> None:
    from nyc_taxi_strategy.graph.zone_graph import (
        build_zone_graph, compute_all_pairs_travel_time,
    )
    from nyc_taxi_strategy.graph.dp_engine import DPEngine, ZoneStats
    from nyc_taxi_strategy.simulation.simulator import (
        ShiftSimulator, RandomStrategy, StayPutStrategy,
        GreedyDemandStrategy, GreedyRevenueStrategy, DPStrategy,
    )
    from nyc_taxi_strategy.simulation.parallel import run_monte_carlo
    from nyc_taxi_strategy.models.train import load_models

    model_dir = Path(cfg.evaluation.output_dir) / "models"
    wait_model, fare_model, feature_cols = load_models(model_dir)

    G = build_zone_graph(
        cfg.data.db_path,
        travel_time_agg=cfg.graph.travel_time_source,
        fuel_cost_per_mile=cfg.simulation.fuel_cost_per_mile,
    )
    active_zones = [z for z in G.nodes() if G.degree(z) > 0]
    travel_times = compute_all_pairs_travel_time(G, active_zones)
    travel_distances = {(u, v): G.edges[u, v]["weight"].distance_miles for u, v in G.edges()}
    logger.info("Graph: %d nodes, %d edges, %d active zones",
                G.number_of_nodes(), G.number_of_edges(), len(active_zones))

    n_slots = _shift_slots(cfg.simulation.shift_start, cfg.simulation.shift_end)
    zone_models = build_zone_models(cfg, wait_model, fare_model, feature_cols, active_zones, n_slots)

    simulator = ShiftSimulator(
        n_time_slots=n_slots,
        time_slot_minutes=TIME_SLOT_MINUTES,
        zone_models=zone_models,
        travel_times=travel_times,
        travel_distances=travel_distances,
        fuel_cost_per_mile=cfg.simulation.fuel_cost_per_mile,
    )

    model_zones = list(zone_models.keys())
    if start_zone is None:
        start_zone = model_zones[0]
    elif start_zone not in model_zones:
        raise ValueError(f"Zone {start_zone} has no historical data. Available zones: {sorted(model_zones)}")

    def make_strategy(name):
        if name == "stay_put":
            return StayPutStrategy()
        if name == "random":
            return RandomStrategy(zones=model_zones, move_probability=0.3)
        if name == "greedy_demand":
            demand_lookup = {
                z: {s: 3600 / zm.expected_wait_s for s, zm in sd.items()}
                for z, sd in zone_models.items()
            }
            return GreedyDemandStrategy(demand_lookup=demand_lookup)
        if name == "greedy_revenue":
            fare_lookup = {
                z: {s: zm.expected_fare for s, zm in sd.items()}
                for z, sd in zone_models.items()
            }
            return GreedyRevenueStrategy(fare_lookup=fare_lookup)
        if name == "dp_optimal":
            zone_stats: dict = {}
            for zone, slot_dict in zone_models.items():
                zone_stats[zone] = {}
                for slot, zm in slot_dict.items():
                    best_drop = max(zm.dropoff_probs, key=zm.dropoff_probs.get) if zm.dropoff_probs else zone
                    zone_stats[zone][slot] = ZoneStats(
                        expected_wait_s=zm.expected_wait_s,
                        expected_fare=zm.expected_fare,
                        expected_trip_duration_s=zm.expected_duration_s,
                        expected_dropoff_zone=best_drop,
                    )
            engine = DPEngine(zones=model_zones, n_time_slots=n_slots,
                              time_slot_minutes=TIME_SLOT_MINUTES)
            dp_result = engine.solve(zone_stats=zone_stats, travel_times=travel_times,
                                     travel_distances=travel_distances,
                                     fuel_cost_per_mile=cfg.simulation.fuel_cost_per_mile)
            return DPStrategy(policy=dp_result.policy)
        raise ValueError(f"Unknown strategy: {name}")

    names = cfg.simulation.strategies if strategy_name == "all" else [strategy_name]
    results = {}
    for name in names:
        logger.info("Running strategy: %s", name)
        mc = run_monte_carlo(
            simulator=simulator,
            strategy=make_strategy(name),
            start_zone=start_zone,
            n_simulations=cfg.simulation.n_simulations,
            n_workers=cfg.simulation.parallel_workers,
        )
        results[name] = mc
        logger.info("  mean=$%.2f ± $%.2f  trips=%.1f", mc.mean_revenue, mc.std_revenue, mc.mean_trips)

    out_dir = Path(cfg.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulation_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    import argparse
    from nyc_taxi_strategy.utils.config import load_config

    parser = argparse.ArgumentParser(description="Run shift simulations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strategy", default="all",
                        help="Strategy name or 'all' (default: all from config)")
    parser.add_argument("--start-zone", type=int, default=None,
                        help="Starting zone ID for the shift (default: zone 1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    run_strategy(cfg, args.strategy, start_zone=args.start_zone)
