"""Tests for simulation run helper functions."""

from nyc_taxi_strategy.simulation.run import _shift_slots, _shift_start_hour


class TestShiftSlots:
    def test_twelve_hour_shift(self):
        assert _shift_slots("06:00", "18:00") == 24

    def test_eight_hour_shift(self):
        assert _shift_slots("08:00", "16:00") == 16

    def test_single_slot(self):
        assert _shift_slots("10:00", "10:30") == 1

    def test_overnight_shift(self):
        assert _shift_slots("00:00", "24:00") == 48


class TestShiftStartHour:
    def test_six_am(self):
        assert _shift_start_hour("06:00") == 6

    def test_midnight(self):
        assert _shift_start_hour("00:00") == 0

    def test_noon(self):
        assert _shift_start_hour("12:30") == 12
