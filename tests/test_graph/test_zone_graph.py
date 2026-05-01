"""Tests for zone graph and DP engine."""

import sqlite3

import networkx as nx
import pytest

from nyc_taxi_strategy.graph.dp_engine import DPEngine, ZoneStats
from nyc_taxi_strategy.graph.zone_graph import (
    EdgeWeight,
    build_zone_graph,
    compute_all_pairs_travel_time,
    get_nearest_zones,
    shortest_travel_time,
)


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE zone_transitions (
            pickup_zone INTEGER, dropoff_zone INTEGER,
            hour_of_day INTEGER, day_of_week INTEGER,
            trip_count INTEGER, avg_duration_s REAL,
            avg_distance REAL, avg_fare REAL,
            PRIMARY KEY (pickup_zone, dropoff_zone, hour_of_day, day_of_week)
        )
    """)
    rows = [
        (1, 2, 8, 0, 20, 600.0, 3.0, 15.0),
        (2, 1, 8, 0, 20, 600.0, 3.0, 15.0),
        (1, 3, 8, 0, 20, 1200.0, 6.0, 25.0),
        (3, 1, 8, 0, 20, 1200.0, 6.0, 25.0),
        (2, 3, 8, 0, 20, 400.0, 2.0, 10.0),
        (3, 2, 8, 0, 20, 400.0, 2.0, 10.0),
    ]
    conn.executemany(
        "INSERT INTO zone_transitions VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(db_path)


class TestBuildZoneGraph:
    def test_has_all_nodes(self, tmp_db):
        G = build_zone_graph(tmp_db)
        assert G.number_of_nodes() == 263

    def test_has_edges(self, tmp_db):
        G = build_zone_graph(tmp_db)
        assert G.number_of_edges() > 0

    def test_edge_has_weight(self, tmp_db):
        G = build_zone_graph(tmp_db)
        assert G.has_edge(1, 2)
        w = G.edges[1, 2]["weight"]
        assert isinstance(w, EdgeWeight)
        assert w.travel_time_s > 0
        assert w.distance_miles > 0

    def test_fuel_cost_applied(self, tmp_db):
        G = build_zone_graph(tmp_db, fuel_cost_per_mile=0.20)
        w = G.edges[1, 2]["weight"]
        assert w.fuel_cost == pytest.approx(w.distance_miles * 0.20)

    def test_filter_by_hour(self, tmp_db):
        G = build_zone_graph(tmp_db, hour_of_day=8)
        assert G.number_of_edges() > 0

    def test_filter_by_day(self, tmp_db):
        G = build_zone_graph(tmp_db, day_of_week=0)
        assert G.number_of_edges() > 0

    def test_no_matching_rows_empty_edges(self, tmp_db):
        G = build_zone_graph(tmp_db, hour_of_day=23)
        assert G.number_of_edges() == 0


class TestComputeAllPairsTravelTime:
    def test_returns_dict(self, tmp_db):
        G = build_zone_graph(tmp_db)
        zones = [1, 2, 3]
        result = compute_all_pairs_travel_time(G, zones)
        assert isinstance(result, dict)

    def test_no_self_pairs(self, tmp_db):
        G = build_zone_graph(tmp_db)
        zones = [1, 2, 3]
        result = compute_all_pairs_travel_time(G, zones)
        for (src, tgt) in result:
            assert src != tgt

    def test_values_positive(self, tmp_db):
        G = build_zone_graph(tmp_db)
        zones = [1, 2, 3]
        result = compute_all_pairs_travel_time(G, zones)
        for v in result.values():
            assert v > 0

    def test_none_zones_uses_all(self, tmp_db):
        G = build_zone_graph(tmp_db)
        result = compute_all_pairs_travel_time(G)
        assert len(result) > 0

    def test_edge_weight_total_cost(self):
        w = EdgeWeight(travel_time_s=300, distance_miles=2.0, fuel_cost=0.30)
        assert w.total_cost == pytest.approx(0.30)


@pytest.fixture
def simple_graph():
    """A small 4-zone graph for testing."""
    G = nx.DiGraph()
    for z in [1, 2, 3, 4]:
        G.add_node(z)

    edges = [
        (1, 2, EdgeWeight(travel_time_s=300, distance_miles=2.0, fuel_cost=0.30)),
        (2, 1, EdgeWeight(travel_time_s=300, distance_miles=2.0, fuel_cost=0.30)),
        (1, 3, EdgeWeight(travel_time_s=600, distance_miles=4.0, fuel_cost=0.60)),
        (3, 1, EdgeWeight(travel_time_s=600, distance_miles=4.0, fuel_cost=0.60)),
        (2, 3, EdgeWeight(travel_time_s=200, distance_miles=1.5, fuel_cost=0.23)),
        (3, 2, EdgeWeight(travel_time_s=200, distance_miles=1.5, fuel_cost=0.23)),
        (3, 4, EdgeWeight(travel_time_s=400, distance_miles=3.0, fuel_cost=0.45)),
        (4, 3, EdgeWeight(travel_time_s=400, distance_miles=3.0, fuel_cost=0.45)),
    ]
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight, trips=100)

    return G


class TestGetNearestZones:
    def test_returns_sorted_by_time(self, simple_graph):
        nearest = get_nearest_zones(simple_graph, source_zone=1)
        times = [e.travel_time_s for _, e in nearest]
        assert times == sorted(times)

    def test_max_zones_limit(self, simple_graph):
        nearest = get_nearest_zones(simple_graph, source_zone=1, max_zones=1)
        assert len(nearest) == 1

    def test_excludes_self(self, simple_graph):
        # Add self-loop
        simple_graph.add_edge(1, 1, weight=EdgeWeight(0, 0, 0), trips=10)
        nearest = get_nearest_zones(simple_graph, source_zone=1)
        zones = [z for z, _ in nearest]
        assert 1 not in zones

    def test_isolated_node(self, simple_graph):
        nearest = get_nearest_zones(simple_graph, source_zone=4)
        # Zone 4 only connects to zone 3
        assert len(nearest) == 1
        assert nearest[0][0] == 3


class TestShortestTravelTime:
    def test_direct_path(self, simple_graph):
        t = shortest_travel_time(simple_graph, 1, 2)
        assert t == 300

    def test_indirect_path_is_shorter(self, simple_graph):
        # 1->3 direct is 600, but 1->2->3 is 300+200=500
        t = shortest_travel_time(simple_graph, 1, 3)
        assert t == 500  # via zone 2

    def test_no_path(self):
        G = nx.DiGraph()
        G.add_node(1)
        G.add_node(2)
        t = shortest_travel_time(G, 1, 2)
        assert t is None


class TestDPEngine:
    def test_stay_when_no_reposition_possible(self):
        """If there's only one zone, DP should always say stay."""
        engine = DPEngine(zones=[1], n_time_slots=4, time_slot_minutes=60)
        zone_stats = {
            1: {
                t: ZoneStats(
                    expected_wait_s=300,
                    expected_fare=20.0,
                    expected_trip_duration_s=600,
                    expected_dropoff_zone=1,
                )
                for t in range(4)
            }
        }
        result = engine.solve(zone_stats, travel_times={})
        for t in range(4):
            assert engine.get_action(result, 1, t) == -1

    def test_reposition_to_better_zone(self):
        """DP should recommend going to a zone with much higher fare."""
        engine = DPEngine(zones=[1, 2], n_time_slots=6, time_slot_minutes=30)

        zone_stats = {
            1: {
                t: ZoneStats(
                    expected_wait_s=60,
                    expected_fare=5.0,  # low fare
                    expected_trip_duration_s=600,
                    expected_dropoff_zone=1,
                )
                for t in range(6)
            },
            2: {
                t: ZoneStats(
                    expected_wait_s=60,
                    expected_fare=50.0,  # much higher fare
                    expected_trip_duration_s=600,
                    expected_dropoff_zone=2,
                )
                for t in range(6)
            },
        }

        travel_times = {
            (1, 2): 300,  # 5 min to get there
            (2, 1): 300,
        }
        travel_distances = {
            (1, 2): 2.0,
            (2, 1): 2.0,
        }

        result = engine.solve(
            zone_stats, travel_times,
            fuel_cost_per_mile=0.15,
            travel_distances=travel_distances,
        )

        # At early time slots, zone 1 should reposition to zone 2
        action = engine.get_action(result, 1, 0)
        assert action == 2

    def test_value_table_non_negative(self):
        engine = DPEngine(zones=[1], n_time_slots=4, time_slot_minutes=60)
        zone_stats = {
            1: {
                t: ZoneStats(
                    expected_wait_s=300,
                    expected_fare=20.0,
                    expected_trip_duration_s=600,
                    expected_dropoff_zone=1,
                )
                for t in range(4)
            }
        }
        result = engine.solve(zone_stats, travel_times={})
        for t, v in result.value_table[1].items():
            assert v >= 0
