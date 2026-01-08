"""
Unit Tests for Risk Management System
======================================
Tests for position sizing, risk validation, circuit breakers, and kill switch.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.risk.risk_manager import (
    PositionSizer, RiskValidator, RiskCheckResult,
    CircuitBreaker, CircuitBreakerStatus
)
from src.core.models import Signal, SignalType, Trend, DailyPnL, Position, PositionStatus, Side
from src.config.settings import Settings, StrategyConfig, RiskConfig


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.risk = MagicMock(spec=RiskConfig)
    settings.risk.max_risk_per_trade_percent = Decimal("1.0")
    settings.risk.max_position_size_percent = Decimal("10.0")
    settings.risk.max_open_positions = 5
    settings.risk.min_risk_reward_ratio = Decimal("2.0")
    settings.risk.max_daily_loss_percent = Decimal("3.0")
    settings.risk.max_daily_trades = 10
    settings.risk.max_sector_exposure_percent = Decimal("30.0")
    settings.risk.max_single_stock_percent = Decimal("15.0")
    settings.risk.consecutive_loss_limit = 3
    settings.risk.circuit_breaker_cooldown_minutes = 30
    return settings


@pytest.fixture
def position_sizer(mock_settings):
    """Create a PositionSizer instance."""
    return PositionSizer(mock_settings)


@pytest.fixture
def risk_validator(mock_settings):
    """Create a RiskValidator instance."""
    return RiskValidator(mock_settings)


@pytest.fixture
def circuit_breaker(mock_settings):
    """Create a CircuitBreaker instance."""
    return CircuitBreaker(mock_settings)


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return Signal(
        symbol="TEST",
        signal_type=SignalType.BUY,
        trend=Trend.UPTREND,
        price=Decimal("100.00"),
        stop_loss=Decimal("95.00"),
        target=Decimal("115.00"),
        confidence=Decimal("0.8"),
        reason="Test signal",
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_daily_pnl():
    """Create sample daily P&L."""
    return DailyPnL(
        date=datetime.now().date(),
        realized_pnl=Decimal("-1000"),
        unrealized_pnl=Decimal("0"),
        net_pnl=Decimal("-1000"),
        fees=Decimal("50"),
        trades_count=3,
        winning_trades=1,
        losing_trades=2,
        max_drawdown=Decimal("-1500"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST POSITION SIZER
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_calculate_risk_based_size_basic(self, position_sizer):
        """Test basic risk-based position sizing."""
        capital = Decimal("100000")
        entry_price = Decimal("100")
        stop_loss = Decimal("95")

        size = position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        # Risk = 1% of 100000 = 1000
        # Risk per share = 100 - 95 = 5
        # Theoretical size = 1000 / 5 = 200
        # But max_position_size_percent = 10% = 10000 value = 100 shares
        # So actual size is capped at 100
        assert size == 100

    def test_calculate_risk_based_size_with_custom_risk(self, position_sizer):
        """Test position sizing with custom risk percentage."""
        capital = Decimal("100000")
        entry_price = Decimal("100")
        stop_loss = Decimal("90")  # 10% risk per share

        size = position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=Decimal("2.0"),  # 2% risk
        )

        # Risk = 2% of 100000 = 2000
        # Risk per share = 100 - 90 = 10
        # Theoretical size = 2000 / 10 = 200
        # But max_position_size_percent = 10% = 10000 value = 100 shares
        # So actual size is capped at 100
        assert size == 100

    def test_calculate_risk_based_size_zero_risk(self, position_sizer):
        """Test that zero risk per share returns 0."""
        capital = Decimal("100000")
        entry_price = Decimal("100")
        stop_loss = Decimal("100")  # Same as entry = zero risk

        size = position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        assert size == 0

    def test_calculate_risk_based_size_respects_max_position(self, position_sizer):
        """Test that position size respects max position limit."""
        capital = Decimal("100000")
        entry_price = Decimal("10")  # Cheap stock
        stop_loss = Decimal("9.90")  # Small risk per share

        size = position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        # Max position = 10% of 100000 = 10000 value
        # Max shares = 10000 / 10 = 1000
        assert size <= 1000

    def test_calculate_risk_based_size_respects_affordable(self, position_sizer):
        """Test that position size respects capital available."""
        capital = Decimal("1000")  # Small capital
        entry_price = Decimal("100")
        stop_loss = Decimal("99")  # Small risk per share

        size = position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        # Max affordable = 1000 / 100 = 10
        assert size <= 10

    def test_calculate_volatility_adjusted_size(self, position_sizer):
        """Test volatility-adjusted position sizing."""
        capital = Decimal("100000")
        entry_price = Decimal("100")
        atr = Decimal("5")  # ATR of 5

        size = position_sizer.calculate_volatility_adjusted_size(
            capital=capital,
            entry_price=entry_price,
            atr=atr,
            atr_multiplier=Decimal("2.0"),
        )

        # Risk = 1% of 100000 = 1000
        # Volatility stop = 5 * 2 = 10
        # Expected size = 1000 / 10 = 100
        assert size == 100

    def test_calculate_volatility_adjusted_size_zero_atr(self, position_sizer):
        """Test that zero ATR returns 0."""
        capital = Decimal("100000")
        entry_price = Decimal("100")
        atr = Decimal("0")

        size = position_sizer.calculate_volatility_adjusted_size(
            capital=capital,
            entry_price=entry_price,
            atr=atr,
        )

        assert size == 0

    def test_calculate_kelly_size_basic(self, position_sizer):
        """Test Kelly Criterion position sizing."""
        capital = Decimal("100000")
        entry_price = Decimal("100")
        win_rate = Decimal("0.6")  # 60% win rate
        avg_win = Decimal("200")
        avg_loss = Decimal("100")

        size = position_sizer.calculate_kelly_size(
            capital=capital,
            entry_price=entry_price,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        # Kelly formula: f* = (bp - q) / b
        # b = 200/100 = 2, p = 0.6, q = 0.4
        # f* = (2*0.6 - 0.4) / 2 = 0.8 / 2 = 0.4
        # Quarter Kelly = 0.4 * 0.25 = 0.1 (10%)
        # Position value = 100000 * 0.1 = 10000
        # Size = 10000 / 100 = 100
        assert size > 0
        assert size <= 100  # Capped by max position

    def test_calculate_kelly_size_zero_loss(self, position_sizer):
        """Test Kelly with zero average loss returns 0."""
        capital = Decimal("100000")
        entry_price = Decimal("100")

        size = position_sizer.calculate_kelly_size(
            capital=capital,
            entry_price=entry_price,
            win_rate=Decimal("0.6"),
            avg_win=Decimal("200"),
            avg_loss=Decimal("0"),
        )

        assert size == 0

    def test_adjust_for_correlation_no_positions(self, position_sizer):
        """Test correlation adjustment with no existing positions."""
        position_size = 100

        adjusted = position_sizer.adjust_for_correlation(
            position_size=position_size,
            symbol="TEST",
            existing_positions=[],
            correlation_matrix=None,
        )

        assert adjusted == position_size

    def test_adjust_for_correlation_with_positions(self, position_sizer):
        """Test correlation adjustment with existing correlated positions."""
        position_size = 100

        # Create mock positions
        pos1 = MagicMock(spec=Position)
        pos1.symbol = "CORR1"

        correlation_matrix = {
            "TEST": {"CORR1": 0.8}  # 80% correlation
        }

        adjusted = position_sizer.adjust_for_correlation(
            position_size=position_size,
            symbol="TEST",
            existing_positions=[pos1],
            correlation_matrix=correlation_matrix,
        )

        # High correlation should reduce position size
        assert adjusted < position_size
        assert adjusted >= 50  # Min 50% due to reduction cap


# ─────────────────────────────────────────────────────────────────────────────
# TEST RISK VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskValidator:
    """Tests for RiskValidator class."""

    @pytest.mark.asyncio
    async def test_validate_trade_passes_all_checks(self, risk_validator, sample_signal):
        """Test that valid trade passes all checks."""
        result = await risk_validator.validate_trade(
            signal=sample_signal,
            capital=Decimal("100000"),
            position_count=0,
            daily_pnl=None,
            existing_positions=None,
        )

        assert result.passed is True
        assert "All risk checks passed" in result.reason

    @pytest.mark.asyncio
    async def test_check_max_positions_fails_at_limit(self, risk_validator):
        """Test max positions check fails at limit."""
        result = await risk_validator._check_max_positions(position_count=5)

        assert result.passed is False
        assert "Max positions" in result.reason
        assert result.risk_score == 100

    @pytest.mark.asyncio
    async def test_check_max_positions_passes_below_limit(self, risk_validator):
        """Test max positions check passes below limit."""
        result = await risk_validator._check_max_positions(position_count=3)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_risk_reward_ratio_fails_below_minimum(self, risk_validator, sample_signal):
        """Test R:R check fails below minimum."""
        bad_signal = sample_signal
        bad_signal.target = Decimal("105")  # Only 1:1 R:R

        result = await risk_validator._check_risk_reward_ratio(bad_signal)

        assert result.passed is False
        assert "R:R ratio" in result.reason

    @pytest.mark.asyncio
    async def test_check_risk_reward_ratio_passes_above_minimum(self, risk_validator, sample_signal):
        """Test R:R check passes above minimum (3:1 ratio)."""
        result = await risk_validator._check_risk_reward_ratio(sample_signal)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_daily_loss_limit_fails_at_limit(self, risk_validator, sample_daily_pnl):
        """Test daily loss limit check fails at limit."""
        capital = Decimal("100000")
        # Simulate 3% loss (at limit)
        sample_daily_pnl.net_pnl = Decimal("-3000")

        result = await risk_validator._check_daily_loss_limit(sample_daily_pnl, capital)

        assert result.passed is False
        assert "Daily loss limit" in result.reason

    @pytest.mark.asyncio
    async def test_check_daily_loss_limit_passes_below_limit(self, risk_validator, sample_daily_pnl):
        """Test daily loss limit check passes below limit."""
        capital = Decimal("100000")
        sample_daily_pnl.net_pnl = Decimal("-1000")  # 1% loss

        result = await risk_validator._check_daily_loss_limit(sample_daily_pnl, capital)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_daily_trade_limit_fails_at_limit(self, risk_validator, sample_daily_pnl):
        """Test daily trade limit check fails at limit."""
        sample_daily_pnl.trades_count = 10

        result = await risk_validator._check_daily_trade_limit(sample_daily_pnl)

        assert result.passed is False
        assert "Daily trade limit" in result.reason

    @pytest.mark.asyncio
    async def test_check_capital_available_fails_insufficient(self, risk_validator, sample_signal):
        """Test capital check fails with insufficient funds."""
        result = await risk_validator._check_capital_available(
            signal=sample_signal,
            capital=Decimal("50"),  # Can't even buy 1 share at 100
        )

        assert result.passed is False
        assert "Insufficient capital" in result.reason

    @pytest.mark.asyncio
    async def test_check_consecutive_losses(self, risk_validator, sample_daily_pnl):
        """Test consecutive loss check."""
        sample_daily_pnl.winning_trades = 0
        sample_daily_pnl.losing_trades = 3

        result = await risk_validator._check_consecutive_losses(sample_daily_pnl)

        assert result.passed is False
        assert "Consecutive loss limit" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# TEST RISK CHECK RESULT
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskCheckResult:
    """Tests for RiskCheckResult class."""

    def test_passed_result_is_truthy(self):
        """Test that passed result evaluates to True."""
        result = RiskCheckResult(passed=True, reason="OK")
        assert bool(result) is True

    def test_failed_result_is_falsy(self):
        """Test that failed result evaluates to False."""
        result = RiskCheckResult(passed=False, reason="Failed")
        assert bool(result) is False

    def test_repr_shows_status(self):
        """Test repr shows pass/fail status."""
        passed = RiskCheckResult(passed=True, reason="OK")
        failed = RiskCheckResult(passed=False, reason="Bad")

        assert "PASS" in repr(passed)
        assert "FAIL" in repr(failed)


# ─────────────────────────────────────────────────────────────────────────────
# TEST CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_active(self, circuit_breaker):
        """Test circuit breaker starts in ACTIVE state."""
        assert circuit_breaker.status == CircuitBreakerStatus.ACTIVE
        assert circuit_breaker.is_trading_allowed() is True

    @pytest.mark.asyncio
    async def test_trigger_on_daily_loss_exceeded(self, circuit_breaker, sample_daily_pnl):
        """Test circuit breaker triggers on daily loss exceeded."""
        capital = Decimal("100000")
        sample_daily_pnl.net_pnl = Decimal("-5000")  # 5% loss, exceeds 3% limit

        with patch.object(circuit_breaker, '_trigger', new_callable=AsyncMock) as mock_trigger:
            triggered, reason = await circuit_breaker.check_and_trigger(
                sample_daily_pnl, capital
            )

            mock_trigger.assert_called_once()
            assert triggered is True

    @pytest.mark.asyncio
    async def test_no_trigger_within_limits(self, circuit_breaker, sample_daily_pnl):
        """Test circuit breaker doesn't trigger within limits."""
        capital = Decimal("100000")
        sample_daily_pnl.net_pnl = Decimal("-1000")  # 1% loss
        sample_daily_pnl.max_drawdown = Decimal("-500")

        triggered, reason = await circuit_breaker.check_and_trigger(
            sample_daily_pnl, capital
        )

        assert triggered is False
        assert circuit_breaker.status == CircuitBreakerStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_locked_state_always_triggered(self, circuit_breaker, sample_daily_pnl):
        """Test locked circuit breaker always returns triggered."""
        capital = Decimal("100000")
        sample_daily_pnl.net_pnl = Decimal("1000")  # Profitable day

        # Lock the breaker
        await circuit_breaker.lock("Manual lock")

        triggered, reason = await circuit_breaker.check_and_trigger(
            sample_daily_pnl, capital
        )

        assert triggered is True
        assert "manually locked" in reason

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, circuit_breaker):
        """Test reset clears circuit breaker state."""
        # Set some state
        circuit_breaker.status = CircuitBreakerStatus.COOLDOWN
        circuit_breaker.triggered_at = datetime.now()
        circuit_breaker.trigger_reason = "Test"

        await circuit_breaker.reset()

        assert circuit_breaker.status == CircuitBreakerStatus.ACTIVE
        assert circuit_breaker.triggered_at is None
        assert circuit_breaker.trigger_reason is None

    @pytest.mark.asyncio
    async def test_cannot_reset_locked(self, circuit_breaker):
        """Test cannot reset locked circuit breaker."""
        await circuit_breaker.lock("Locked")
        await circuit_breaker.reset()

        # Should still be locked
        assert circuit_breaker.status == CircuitBreakerStatus.LOCKED

    @pytest.mark.asyncio
    async def test_unlock_from_locked(self, circuit_breaker):
        """Test unlock from locked state."""
        await circuit_breaker.lock("Locked")
        await circuit_breaker.unlock()

        assert circuit_breaker.status == CircuitBreakerStatus.ACTIVE
        assert circuit_breaker.is_trading_allowed() is True

    def test_get_status_returns_dict(self, circuit_breaker):
        """Test get_status returns proper dictionary."""
        status = circuit_breaker.get_status()

        assert isinstance(status, dict)
        assert "status" in status
        assert "trading_allowed" in status
        assert status["status"] == "ACTIVE"
        assert status["trading_allowed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
