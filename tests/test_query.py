"""Tests for query helper functions."""

import pytest

from nyc_taxi_strategy.query import _slot_to_time, _time_to_slot


class TestTimeToSlot:
    def test_shift_start_is_slot_zero(self):
        assert _time_to_slot("06:00", "06:00") == 0

    def test_one_slot_in(self):
        assert _time_to_slot("06:30", "06:00") == 1

    def test_two_hours_in(self):
        assert _time_to_slot("08:00", "06:00") == 4

    def test_before_shift_raises(self):
        with pytest.raises(ValueError):
            _time_to_slot("05:00", "06:00")


class TestSlotToTime:
    def test_slot_zero(self):
        assert _slot_to_time(0, "06:00") == "06:00"

    def test_slot_one(self):
        assert _slot_to_time(1, "06:00") == "06:30"

    def test_slot_four(self):
        assert _slot_to_time(4, "06:00") == "08:00"

    def test_crosses_midnight(self):
        assert _slot_to_time(48, "06:00") == "30:00"

    def test_roundtrip(self):
        for slot in range(24):
            time_str = _slot_to_time(slot, "06:00")
            assert _time_to_slot(time_str, "06:00") == slot
