"""Simulate a taxi driver's shift under different repositioning strategies.

The simulator steps through a shift in discrete time steps. At each step,
the driver is in a zone and must choose: stay and wait, or reposition.
Trip outcomes (wait time, fare, dropoff zone) are sampled from historical
distributions or model predictions.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TripEvent:
    """Record of a single completed trip during the simulation."""

    pickup_zone: int
    dropoff_zone: int
    pickup_time_slot: int
    fare: float
    wait_time_s: float
    trip_duration_s: float
    repositioned_from: int | None = None  # zone driver came from, if repositioned
    reposition_cost: float = 0.0


@dataclass
class ShiftResult:
    """Summary of a simulated shift."""

    trips: list[TripEvent] = field(default_factory=list)
    total_revenue: float = 0.0
    total_reposition_cost: float = 0.0
    total_idle_time_s: float = 0.0
    total_driving_time_s: float = 0.0
    total_empty_miles: float = 0.0

    @property
    def net_revenue(self) -> float:
        return self.total_revenue - self.total_reposition_cost

    @property
    def trips_completed(self) -> int:
        return len(self.trips)

    @property
    def idle_time_pct(self) -> float:
        total = self.total_idle_time_s + self.total_driving_time_s
        return (self.total_idle_time_s / total * 100) if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "net_revenue": round(self.net_revenue, 2),
            "gross_revenue": round(self.total_revenue, 2),
            "reposition_cost": round(self.total_reposition_cost, 2),
            "trips_completed": self.trips_completed,
            "idle_time_pct": round(self.idle_time_pct, 1),
            "empty_miles": round(self.total_empty_miles, 1),
        }


class Strategy(Protocol):
    """Protocol for driver repositioning strategies."""

    def choose_action(self, zone: int, time_slot: int) -> int:
        """Choose next action: return zone to go to, or -1 to stay.

        Args:
            zone: Current zone ID.
            time_slot: Current time slot index.

        Returns:
            Target zone ID to reposition to, or -1 to stay in place.
        """
        ...


class RandomStrategy:
    """Randomly choose to stay or go to a random neighboring zone."""

    def __init__(self, zones: list[int], move_probability: float = 0.3):
        self.zones = zones
        self.move_probability = move_probability

    def choose_action(self, zone: int, time_slot: int) -> int:
        if random.random() < self.move_probability:
            return random.choice(self.zones)
        return -1


class StayPutStrategy:
    """Always stay in the current zone and wait."""

    def choose_action(self, zone: int, time_slot: int) -> int:
        return -1


class GreedyDemandStrategy:
    """Go to the zone with the highest expected demand."""

    def __init__(self, demand_lookup: dict[int, dict[int, float]], top_k: int = 5):
        """
        Args:
            demand_lookup: demand_lookup[zone][time_slot] = expected_trip_count
            top_k: Only consider top-k zones to add some randomness.
        """
        self.demand_lookup = demand_lookup
        self.top_k = top_k

    def choose_action(self, zone: int, time_slot: int) -> int:
        scores = []
        for z, ts_dict in self.demand_lookup.items():
            demand = ts_dict.get(time_slot, 0)
            scores.append((z, demand))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: self.top_k]

        if top and top[0][0] != zone:
            return top[0][0]
        return -1


class GreedyRevenueStrategy:
    """Go to the zone with the highest expected fare."""

    def __init__(self, fare_lookup: dict[int, dict[int, float]]):
        """
        Args:
            fare_lookup: fare_lookup[zone][time_slot] = expected_fare
        """
        self.fare_lookup = fare_lookup

    def choose_action(self, zone: int, time_slot: int) -> int:
        best_zone = -1
        best_fare = self.fare_lookup.get(zone, {}).get(time_slot, 0)

        for z, ts_dict in self.fare_lookup.items():
            fare = ts_dict.get(time_slot, 0)
            if fare > best_fare:
                best_fare = fare
                best_zone = z

        return best_zone


class DPStrategy:
    """Use the precomputed DP optimal policy."""

    def __init__(self, policy: dict[int, dict[int, int]]):
        self.policy = policy

    def choose_action(self, zone: int, time_slot: int) -> int:
        return self.policy.get(zone, {}).get(time_slot, -1)


@dataclass
class ZoneModel:
    """Encapsulates predictions for a zone at a time slot.

    Used by the simulator to sample trip outcomes.
    """

    expected_wait_s: float
    expected_fare: float
    fare_std: float
    expected_duration_s: float
    dropoff_probs: dict[int, float]  # zone -> probability

    def sample_wait(self, rng: np.random.Generator) -> float:
        """Sample a wait time from exponential distribution."""
        return rng.exponential(self.expected_wait_s)

    def sample_fare(self, rng: np.random.Generator) -> float:
        """Sample a fare (clipped to non-negative)."""
        return max(0, rng.normal(self.expected_fare, self.fare_std))

    def sample_dropoff(self, rng: np.random.Generator) -> int:
        """Sample a dropoff zone from the distribution."""
        zones = list(self.dropoff_probs.keys())
        probs = list(self.dropoff_probs.values())
        return rng.choice(zones, p=probs)


class ShiftSimulator:
    """Simulate a driver's shift under a given strategy.

    Steps through discrete time slots, applying the strategy's decisions
    and sampling trip outcomes from the zone models.
    """

    def __init__(
        self,
        n_time_slots: int,
        time_slot_minutes: int,
        zone_models: dict[int, dict[int, ZoneModel]],
        travel_times: dict[tuple[int, int], float],
        travel_distances: dict[tuple[int, int], float],
        fuel_cost_per_mile: float = 0.15,
    ):
        """
        Args:
            n_time_slots: Total time slots in the shift.
            time_slot_minutes: Duration of each slot.
            zone_models: zone_models[zone][time_slot] = ZoneModel.
            travel_times: (from, to) -> seconds.
            travel_distances: (from, to) -> miles.
            fuel_cost_per_mile: Fuel cost per mile for repositioning.
        """
        self.n_time_slots = n_time_slots
        self.time_slot_minutes = time_slot_minutes
        self.zone_models = zone_models
        self.travel_times = travel_times
        self.travel_distances = travel_distances
        self.fuel_cost_per_mile = fuel_cost_per_mile

    def run(
        self,
        strategy: Strategy,
        start_zone: int,
        seed: int | None = None,
    ) -> ShiftResult:
        """Run a single shift simulation.

        Args:
            strategy: The repositioning strategy to follow.
            start_zone: Zone where the driver starts.
            seed: Random seed for reproducibility.

        Returns:
            ShiftResult with all trip events and summary stats.
        """
        rng = np.random.default_rng(seed)
        result = ShiftResult()
        current_zone = start_zone
        current_slot = 0

        while current_slot < self.n_time_slots:
            action = strategy.choose_action(current_zone, current_slot)

            # Reposition if needed
            repositioned_from = None
            if action != -1 and action != current_zone:
                travel_t = self.travel_times.get((current_zone, action))
                if travel_t is None:
                    action = -1  # can't reach, stay
                else:
                    travel_slots = int(np.ceil(travel_t / (self.time_slot_minutes * 60)))
                    dist = self.travel_distances.get((current_zone, action), 0)
                    cost = dist * self.fuel_cost_per_mile

                    result.total_reposition_cost += cost
                    result.total_empty_miles += dist
                    result.total_driving_time_s += travel_t

                    repositioned_from = current_zone
                    current_zone = action
                    current_slot += travel_slots

                    if current_slot >= self.n_time_slots:
                        break

            # Try to pick up a passenger
            zm = self.zone_models.get(current_zone, {}).get(current_slot)
            if zm is None:
                current_slot += 1
                result.total_idle_time_s += self.time_slot_minutes * 60
                continue

            wait_s = zm.sample_wait(rng)
            wait_slots = int(np.ceil(wait_s / (self.time_slot_minutes * 60)))

            result.total_idle_time_s += wait_s
            current_slot += wait_slots

            if current_slot >= self.n_time_slots:
                break

            # Complete the trip
            fare = zm.sample_fare(rng)
            dropoff = zm.sample_dropoff(rng)
            trip_duration_s = zm.expected_duration_s

            trip_slots = int(np.ceil(trip_duration_s / (self.time_slot_minutes * 60)))

            event = TripEvent(
                pickup_zone=current_zone,
                dropoff_zone=dropoff,
                pickup_time_slot=current_slot,
                fare=fare,
                wait_time_s=wait_s,
                trip_duration_s=trip_duration_s,
                repositioned_from=repositioned_from,
                reposition_cost=0,
            )
            result.trips.append(event)
            result.total_revenue += fare
            result.total_driving_time_s += trip_duration_s

            current_zone = dropoff
            current_slot += trip_slots

        return result
