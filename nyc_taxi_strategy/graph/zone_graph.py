"""Model NYC taxi zones as a weighted directed graph.

Nodes are taxi zones (1-263). Edge weights represent the estimated
travel time (seconds) and cost (dollars) to reposition between zones.
These are derived from historical trip data in the zone_transitions table.
"""

import logging
import sqlite3
from dataclasses import dataclass

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class EdgeWeight:
    """Weights on a zone-to-zone edge."""

    travel_time_s: float
    distance_miles: float
    fuel_cost: float

    @property
    def total_cost(self) -> float:
        return self.fuel_cost


def build_zone_graph(
    db_path: str,
    hour_of_day: int | None = None,
    day_of_week: int | None = None,
    travel_time_agg: str = "median",
    fuel_cost_per_mile: float = 0.15,
) -> nx.DiGraph:
    """Build a directed graph of taxi zones from historical transition data.

    Args:
        db_path: Path to the SQLite database.
        hour_of_day: Filter transitions to a specific hour (0-23), or None for all.
        day_of_week: Filter to a specific day (0=Mon, 6=Sun), or None for all.
        travel_time_agg: How to aggregate travel times ('mean' or 'median').
        fuel_cost_per_mile: Cost per mile for empty repositioning.

    Returns:
        NetworkX DiGraph with EdgeWeight data on each edge.
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            pickup_zone,
            dropoff_zone,
            SUM(trip_count) as total_trips,
            SUM(avg_duration_s * trip_count) / SUM(trip_count) as weighted_avg_duration,
            SUM(avg_distance * trip_count) / SUM(trip_count) as weighted_avg_distance
        FROM zone_transitions
        WHERE 1=1
    """
    params: list = []
    if hour_of_day is not None:
        query += " AND hour_of_day = ?"
        params.append(hour_of_day)
    if day_of_week is not None:
        query += " AND day_of_week = ?"
        params.append(day_of_week)

    query += """
        GROUP BY pickup_zone, dropoff_zone
        HAVING total_trips >= 5
    """

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    G = nx.DiGraph()

    # Add all zones as nodes
    for zone_id in range(1, 264):
        G.add_node(zone_id)

    for pickup, dropoff, trips, avg_time, avg_dist in rows:
        weight = EdgeWeight(
            travel_time_s=avg_time,
            distance_miles=avg_dist,
            fuel_cost=avg_dist * fuel_cost_per_mile,
        )
        G.add_edge(pickup, dropoff, weight=weight, trips=trips)

    logger.info(f"Built zone graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_nearest_zones(
    G: nx.DiGraph,
    source_zone: int,
    max_zones: int = 10,
) -> list[tuple[int, EdgeWeight]]:
    """Get the nearest zones by travel time from a source zone.

    Args:
        G: The zone graph.
        source_zone: Starting zone ID.
        max_zones: Maximum number of zones to return.

    Returns:
        List of (zone_id, EdgeWeight) sorted by travel time, excluding self.
    """
    neighbors = []
    for target in G.successors(source_zone):
        if target == source_zone:
            continue
        edge_data = G.edges[source_zone, target]["weight"]
        neighbors.append((target, edge_data))

    neighbors.sort(key=lambda x: x[1].travel_time_s)
    return neighbors[:max_zones]


def shortest_travel_time(
    G: nx.DiGraph,
    source: int,
    target: int,
) -> float | None:
    """Find shortest travel time between two zones using Dijkstra.

    Args:
        G: The zone graph.
        source: Source zone ID.
        target: Target zone ID.

    Returns:
        Shortest travel time in seconds, or None if no path exists.
    """
    try:
        path_length = nx.dijkstra_path_length(
            G, source, target, weight=lambda u, v, d: d["weight"].travel_time_s
        )
        return path_length
    except nx.NetworkXNoPath:
        return None


def compute_all_pairs_travel_time(G: nx.DiGraph, zones: list[int] | None = None) -> dict:
    """Compute all-pairs shortest travel times for a subset of zones.

    Args:
        G: The zone graph.
        zones: List of zone IDs to include. None for all.

    Returns:
        Dict of {(source, target): travel_time_seconds}.
    """
    if zones is None:
        zones = list(G.nodes())

    subgraph = G.subgraph(zones)
    result = {}

    for source in zones:
        try:
            lengths = nx.single_source_dijkstra_path_length(
                subgraph, source, weight=lambda u, v, d: d["weight"].travel_time_s
            )
            for target, time in lengths.items():
                if target != source:
                    result[(source, target)] = time
        except nx.NodeNotFound:
            continue

    return result
