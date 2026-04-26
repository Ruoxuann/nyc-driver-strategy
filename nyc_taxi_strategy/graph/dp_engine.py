"""Dynamic programming engine for optimal driver repositioning.

Solves the problem: given current zone and time, what action (stay or
reposition to zone X) maximizes expected remaining revenue in the shift?

State: (zone, time_slot)
Actions: stay in current zone, or reposition to any reachable zone
Transitions: after completing a trip, end up in dropoff_zone at time + trip_duration
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ZoneStats:
    """Expected stats for a zone at a given time slot."""

    expected_wait_s: float       # seconds until next pickup
    expected_fare: float         # expected fare of the trip
    expected_trip_duration_s: float  # expected duration of trip
    expected_dropoff_zone: int   # most likely dropoff zone
    dropoff_distribution: dict[int, float] | None = None  # zone -> probability


@dataclass
class DPResult:
    """Result of the DP computation."""

    # value_table[zone][time_slot] = expected remaining revenue
    value_table: dict[int, dict[int, float]]
    # policy[zone][time_slot] = best action (zone to go to, or -1 for stay)
    policy: dict[int, dict[int, int]]
    time_slot_minutes: int


class DPEngine:
    """Dynamic programming engine for optimal repositioning strategy.

    Discretizes time into slots and computes the optimal policy via
    backward induction.
    """

    def __init__(
        self,
        zones: list[int],
        n_time_slots: int,
        time_slot_minutes: int = 30,
    ):
        """Initialize the DP engine.

        Args:
            zones: List of zone IDs to consider.
            n_time_slots: Number of time slots in the shift.
            time_slot_minutes: Duration of each time slot in minutes.
        """
        self.zones = zones
        self.n_time_slots = n_time_slots
        self.time_slot_minutes = time_slot_minutes
        self._zone_idx = {z: i for i, z in enumerate(zones)}

    def solve(
        self,
        zone_stats: dict[int, dict[int, ZoneStats]],
        travel_times: dict[tuple[int, int], float],
        fuel_cost_per_mile: float = 0.15,
        travel_distances: dict[tuple[int, int], float] | None = None,
    ) -> DPResult:
        """Solve for the optimal policy using backward induction.

        Args:
            zone_stats: zone_stats[zone][time_slot] = ZoneStats for that state.
            travel_times: travel_times[(from_zone, to_zone)] = seconds.
            fuel_cost_per_mile: Fuel cost for empty repositioning.
            travel_distances: travel_distances[(from, to)] = miles. Optional.

        Returns:
            DPResult with value table and optimal policy.
        """
        n_zones = len(self.zones)
        n_t = self.n_time_slots

        # V[zone_idx][t] = expected remaining revenue from state (zone, t)
        V = np.zeros((n_zones, n_t + 1))
        # policy[zone_idx][t] = zone to reposition to (-1 = stay)
        policy_arr = np.full((n_zones, n_t), -1, dtype=int)

        # Backward induction
        for t in range(n_t - 1, -1, -1):
            for zi, zone in enumerate(self.zones):
                stats = zone_stats.get(zone, {}).get(t)
                if stats is None:
                    continue

                # Option 1: Stay and wait for a pickup
                wait_slots = int(np.ceil(stats.expected_wait_s / (self.time_slot_minutes * 60)))
                trip_slots = int(np.ceil(stats.expected_trip_duration_s / (self.time_slot_minutes * 60)))
                arrival_t = t + wait_slots + trip_slots

                if arrival_t < n_t:
                    dropoff_zi = self._zone_idx.get(stats.expected_dropoff_zone, zi)
                    stay_value = stats.expected_fare + V[dropoff_zi, arrival_t]
                else:
                    # Trip would finish after shift ends
                    stay_value = stats.expected_fare * 0.5  # partial credit

                best_value = stay_value
                best_action = -1  # stay

                # Option 2: Reposition to another zone
                for zj, other_zone in enumerate(self.zones):
                    if other_zone == zone:
                        continue

                    travel_t = travel_times.get((zone, other_zone))
                    if travel_t is None:
                        continue

                    travel_slots = int(np.ceil(travel_t / (self.time_slot_minutes * 60)))
                    reposition_cost = 0.0

                    if travel_distances is not None:
                        dist = travel_distances.get((zone, other_zone), 0)
                        reposition_cost = dist * fuel_cost_per_mile

                    arrival_after_reposition = t + travel_slots
                    if arrival_after_reposition >= n_t:
                        continue

                    # Value of being in the other zone after repositioning
                    other_stats = zone_stats.get(other_zone, {}).get(arrival_after_reposition)
                    if other_stats is None:
                        continue

                    wait_slots_other = int(np.ceil(
                        other_stats.expected_wait_s / (self.time_slot_minutes * 60)
                    ))
                    trip_slots_other = int(np.ceil(
                        other_stats.expected_trip_duration_s / (self.time_slot_minutes * 60)
                    ))
                    final_t = arrival_after_reposition + wait_slots_other + trip_slots_other

                    if final_t < n_t:
                        dropoff_zi_other = self._zone_idx.get(
                            other_stats.expected_dropoff_zone, zj
                        )
                        repo_value = (
                            other_stats.expected_fare
                            - reposition_cost
                            + V[dropoff_zi_other, final_t]
                        )
                    else:
                        repo_value = other_stats.expected_fare * 0.5 - reposition_cost

                    if repo_value > best_value:
                        best_value = repo_value
                        best_action = other_zone

                V[zi, t] = best_value
                policy_arr[zi, t] = best_action

        # Convert to dicts
        value_table = {}
        policy_dict = {}
        for zi, zone in enumerate(self.zones):
            value_table[zone] = {t: float(V[zi, t]) for t in range(n_t)}
            policy_dict[zone] = {t: int(policy_arr[zi, t]) for t in range(n_t)}

        logger.info(f"DP solved: {n_zones} zones x {n_t} time slots")
        return DPResult(
            value_table=value_table,
            policy=policy_dict,
            time_slot_minutes=self.time_slot_minutes,
        )

    def get_action(self, result: DPResult, zone: int, time_slot: int) -> int:
        """Look up the optimal action from a solved DP result.

        Args:
            result: Solved DPResult.
            zone: Current zone ID.
            time_slot: Current time slot index.

        Returns:
            Zone to reposition to, or -1 to stay.
        """
        return result.policy.get(zone, {}).get(time_slot, -1)
