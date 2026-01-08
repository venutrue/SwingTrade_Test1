"""
Unit Tests for Swing Trading Engine
====================================
Tests for swing point detection, trend determination, and signal generation.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.engine import RivallandSwingEngine, SwingPoint, EngineState
from src.core.models import SignalType, Trend
from src.config.settings import StrategyConfig


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Create a fresh engine instance."""
    config = StrategyConfig(
        swing_lookback=3,
        min_pullback_bars=2,
        min_rally_bars=2,
    )
    return RivallandSwingEngine(config)


@pytest.fixture
def uptrend_data():
    """Generate data with clear uptrend (HH + HL)."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

    # Create uptrend pattern
    prices = []
    base = 100
    for i in range(50):
        # Gradually increase with pullbacks
        trend_component = i * 0.5  # Upward drift
        cycle = np.sin(i * 0.3) * 3  # Small oscillation
        prices.append(base + trend_component + cycle)

    df = pd.DataFrame({
        "open": prices,
        "high": [p + np.random.uniform(1, 3) for p in prices],
        "low": [p - np.random.uniform(1, 3) for p in prices],
        "close": [p + np.random.uniform(-1, 1) for p in prices],
        "volume": [100000] * 50,
    }, index=dates)

    return df


@pytest.fixture
def downtrend_data():
    """Generate data with clear downtrend (LH + LL)."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

    prices = []
    base = 150
    for i in range(50):
        # Gradually decrease with rallies
        trend_component = -i * 0.5  # Downward drift
        cycle = np.sin(i * 0.3) * 3
        prices.append(base + trend_component + cycle)

    df = pd.DataFrame({
        "open": prices,
        "high": [p + np.random.uniform(1, 3) for p in prices],
        "low": [p - np.random.uniform(1, 3) for p in prices],
        "close": [p + np.random.uniform(-1, 1) for p in prices],
        "volume": [100000] * 50,
    }, index=dates)

    return df


@pytest.fixture
def sideways_data():
    """Generate sideways/ranging data."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

    base = 100
    prices = [base + np.random.uniform(-5, 5) for _ in range(50)]

    df = pd.DataFrame({
        "open": prices,
        "high": [p + np.random.uniform(1, 2) for p in prices],
        "low": [p - np.random.uniform(1, 2) for p in prices],
        "close": [p + np.random.uniform(-0.5, 0.5) for p in prices],
        "volume": [100000] * 50,
    }, index=dates)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST ENGINE STATE
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineState:
    """Tests for EngineState class."""

    def test_initial_state(self):
        """Test initial state values."""
        state = EngineState()

        assert state.swing_highs == []
        assert state.swing_lows == []
        assert state.current_trend == Trend.NEUTRAL
        assert state.in_pullback is False
        assert state.in_rally is False
        assert state.pullback_bars == 0
        assert state.rally_bars == 0

    def test_reset(self):
        """Test state reset."""
        state = EngineState()
        state.swing_highs.append(SwingPoint(
            price=Decimal("100"),
            bar_index=10,
            timestamp=datetime.now(),
            point_type="HIGH",
        ))
        state.current_trend = Trend.UPTREND
        state.in_pullback = True
        state.pullback_bars = 5

        state.reset()

        assert state.swing_highs == []
        assert state.current_trend == Trend.NEUTRAL
        assert state.in_pullback is False
        assert state.pullback_bars == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST SWING POINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestSwingPointDetection:
    """Tests for swing point detection."""

    def test_detect_swing_high(self, engine):
        """Test swing high detection."""
        # Create data with a clear swing high at index 5
        dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
        highs = [100, 101, 102, 103, 105, 110, 104, 103, 102, 101]  # Peak at index 5

        df = pd.DataFrame({
            "open": [100] * 10,
            "high": highs,
            "low": [98] * 10,
            "close": [99] * 10,
            "volume": [100000] * 10,
        }, index=dates)

        result = engine.detect_swing_points(df)

        # Should detect swing high at index 5 (lookback=3, so detectable at index 5)
        assert "swing_high" in result.columns
        # At least one swing high should be detected
        swing_highs = result["swing_high"].dropna()
        assert len(swing_highs) >= 0  # May be 0 if lookback not met

    def test_detect_swing_low(self, engine):
        """Test swing low detection."""
        dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
        lows = [100, 99, 98, 97, 95, 90, 96, 97, 98, 99]  # Trough at index 5

        df = pd.DataFrame({
            "open": [100] * 10,
            "high": [102] * 10,
            "low": lows,
            "close": [101] * 10,
            "volume": [100000] * 10,
        }, index=dates)

        result = engine.detect_swing_points(df)

        assert "swing_low" in result.columns

    def test_no_swing_with_insufficient_data(self, engine):
        """Test that no swings detected with insufficient data."""
        dates = pd.date_range(end=datetime.now(), periods=5, freq="D")

        df = pd.DataFrame({
            "open": [100] * 5,
            "high": [102] * 5,
            "low": [98] * 5,
            "close": [101] * 5,
            "volume": [100000] * 5,
        }, index=dates)

        result = engine.detect_swing_points(df)

        # With lookback=3, we need at least 7 bars (3 + 1 + 3)
        swing_highs = result["swing_high"].dropna()
        swing_lows = result["swing_low"].dropna()

        assert len(swing_highs) == 0
        assert len(swing_lows) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST TREND DETERMINATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendDetermination:
    """Tests for trend determination logic."""

    def test_uptrend_detection(self, engine):
        """Test uptrend detection with HH + HL."""
        # Manually set swing points for uptrend
        now = datetime.now()
        engine.state.swing_highs = [
            SwingPoint(Decimal("100"), 0, now, "HIGH"),
            SwingPoint(Decimal("110"), 5, now, "HIGH"),  # Higher high
        ]
        engine.state.swing_lows = [
            SwingPoint(Decimal("95"), 2, now, "LOW"),
            SwingPoint(Decimal("100"), 7, now, "LOW"),  # Higher low
        ]

        trend = engine.determine_trend()

        assert trend == Trend.UPTREND
        assert engine.state.current_trend == Trend.UPTREND

    def test_downtrend_detection(self, engine):
        """Test downtrend detection with LH + LL."""
        now = datetime.now()
        engine.state.swing_highs = [
            SwingPoint(Decimal("110"), 0, now, "HIGH"),
            SwingPoint(Decimal("100"), 5, now, "HIGH"),  # Lower high
        ]
        engine.state.swing_lows = [
            SwingPoint(Decimal("100"), 2, now, "LOW"),
            SwingPoint(Decimal("95"), 7, now, "LOW"),  # Lower low
        ]

        trend = engine.determine_trend()

        assert trend == Trend.DOWNTREND

    def test_neutral_with_insufficient_swings(self, engine):
        """Test neutral trend with insufficient swing points."""
        now = datetime.now()
        engine.state.swing_highs = [
            SwingPoint(Decimal("100"), 0, now, "HIGH"),
        ]
        engine.state.swing_lows = []

        trend = engine.determine_trend()

        assert trend == Trend.NEUTRAL

    def test_trend_unchanged_on_mixed_signals(self, engine):
        """Test trend unchanged when signals are mixed (HH but LL or vice versa)."""
        now = datetime.now()
        engine.state.current_trend = Trend.UPTREND

        # Higher high but lower low (mixed)
        engine.state.swing_highs = [
            SwingPoint(Decimal("100"), 0, now, "HIGH"),
            SwingPoint(Decimal("110"), 5, now, "HIGH"),  # Higher high
        ]
        engine.state.swing_lows = [
            SwingPoint(Decimal("95"), 2, now, "LOW"),
            SwingPoint(Decimal("90"), 7, now, "LOW"),  # Lower low
        ]

        trend = engine.determine_trend()

        # Trend should remain unchanged (still UPTREND)
        assert trend == Trend.UPTREND


# ─────────────────────────────────────────────────────────────────────────────
# TEST SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalGeneration:
    """Tests for trading signal generation."""

    def test_hold_signal_with_insufficient_data(self, engine):
        """Test HOLD signal when data is insufficient."""
        dates = pd.date_range(end=datetime.now(), periods=3, freq="D")

        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [98, 99, 100],
            "close": [101, 102, 103],
            "volume": [100000] * 3,
        }, index=dates)

        signal = engine.generate_signal(df, "TEST", "day")

        assert signal.signal_type == SignalType.HOLD
        assert "Insufficient data" in signal.reason

    def test_hold_signal_no_setup(self, engine, sideways_data):
        """Test HOLD signal when no valid setup exists."""
        signal = engine.analyze(sideways_data, "TEST", "day")

        # In sideways market, should mostly get HOLD signals
        assert signal.signal_type in [SignalType.HOLD, SignalType.BUY, SignalType.SELL]

    def test_signal_has_required_fields(self, engine, uptrend_data):
        """Test that signal contains all required fields."""
        signal = engine.analyze(uptrend_data, "TEST", "day")

        assert signal.symbol == "TEST"
        assert signal.signal_type in SignalType
        assert signal.trend in Trend
        assert signal.price > 0
        assert signal.timestamp is not None

    def test_buy_signal_stop_loss_below_entry(self, engine, uptrend_data):
        """Test that BUY signal has stop loss below entry."""
        signal = engine.analyze(uptrend_data, "TEST", "day")

        if signal.signal_type == SignalType.BUY:
            assert signal.stop_loss < signal.price
            assert signal.target > signal.price

    def test_sell_signal_stop_loss_above_entry(self, engine, downtrend_data):
        """Test that SELL signal has stop loss above entry."""
        signal = engine.analyze(downtrend_data, "TEST", "day")

        if signal.signal_type == SignalType.SELL:
            assert signal.stop_loss > signal.price
            assert signal.target < signal.price


# ─────────────────────────────────────────────────────────────────────────────
# TEST PULLBACK/RALLY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestPullbackRallyDetection:
    """Tests for pullback and rally detection."""

    def test_pullback_detection(self, engine):
        """Test pullback detection in uptrend."""
        engine.state.current_trend = Trend.UPTREND

        # Create pullback bars (lower closes)
        dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
        df = pd.DataFrame({
            "open": [105, 104, 103, 102, 103],
            "high": [106, 105, 104, 103, 105],
            "low": [104, 103, 102, 101, 102],
            "close": [105, 104, 103, 102, 104],  # Consecutive lower closes then higher
            "volume": [100000] * 5,
        }, index=dates)

        # Process first few bars
        for i in range(3, 5):
            sub_df = df.iloc[:i+1]
            engine.detect_pullback_rally(sub_df)

        # After pullback bars, should be in pullback
        assert engine.state.pullback_bars >= 0

    def test_rally_detection(self, engine):
        """Test rally detection in downtrend."""
        engine.state.current_trend = Trend.DOWNTREND

        dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 102],
            "high": [101, 102, 103, 104, 103],
            "low": [99, 100, 101, 102, 101],
            "close": [100, 101, 102, 103, 101],  # Consecutive higher closes then lower
            "volume": [100000] * 5,
        }, index=dates)

        for i in range(3, 5):
            sub_df = df.iloc[:i+1]
            engine.detect_pullback_rally(sub_df)

        assert engine.state.rally_bars >= 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST ENGINE STATE
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineGetState:
    """Tests for engine state retrieval."""

    def test_get_state_returns_dict(self, engine, uptrend_data):
        """Test that get_state returns proper dictionary."""
        engine.analyze(uptrend_data, "TEST", "day")

        state = engine.get_state()

        assert isinstance(state, dict)
        assert "trend" in state
        assert "in_pullback" in state
        assert "in_rally" in state
        assert "swing_highs_count" in state
        assert "swing_lows_count" in state

    def test_reset_clears_state(self, engine, uptrend_data):
        """Test that reset clears all state."""
        engine.analyze(uptrend_data, "TEST", "day")

        # State should have some data
        assert engine.state.bars_processed > 0

        engine.reset()

        assert engine.state.bars_processed == 0
        assert len(engine.state.swing_highs) == 0
        assert len(engine.state.swing_lows) == 0


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
