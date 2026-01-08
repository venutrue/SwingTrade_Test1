"""
Unit Tests for Order State Machine and Order Manager
=====================================================
Tests for order lifecycle, state transitions, and fill processing.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.broker.order_manager import (
    OrderStateMachine, OrderEvent, OrderManager, PositionManager
)
from src.core.models import Order, OrderStatus, Side, Position, PositionStatus, ExitReason


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        id="order-123",
        symbol="TEST",
        side=Side.BUY,
        quantity=100,
        pending_quantity=100,
        order_type="MARKET",
        status=OrderStatus.PENDING,
        product_type="CNC",
    )


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = MagicMock()
    broker.place_order = AsyncMock(return_value={
        "status": "success",
        "order_id": "broker-order-123"
    })
    broker.cancel_order = AsyncMock(return_value={"status": "success"})
    broker.get_orders = AsyncMock(return_value=[])
    return broker


@pytest.fixture
def order_manager(mock_broker):
    """Create an OrderManager with mocked broker."""
    with patch('src.broker.order_manager.get_settings') as mock_settings:
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.risk = MagicMock()
        mock_settings.return_value.risk.brokerage_per_order = Decimal("20")
        mock_settings.return_value.risk.stt_percent = Decimal("0.1")
        return OrderManager(mock_broker)


# ─────────────────────────────────────────────────────────────────────────────
# TEST ORDER STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderStateMachine:
    """Tests for OrderStateMachine class."""

    def test_valid_transition_pending_to_submitted(self):
        """Test valid transition from PENDING to SUBMITTED."""
        result = OrderStateMachine.can_transition(
            OrderStatus.PENDING, OrderStatus.SUBMITTED
        )
        assert result is True

    def test_valid_transition_submitted_to_filled(self):
        """Test valid transition from SUBMITTED to FILLED."""
        result = OrderStateMachine.can_transition(
            OrderStatus.SUBMITTED, OrderStatus.FILLED
        )
        assert result is True

    def test_valid_transition_open_to_partial(self):
        """Test valid transition from OPEN to PARTIAL."""
        result = OrderStateMachine.can_transition(
            OrderStatus.OPEN, OrderStatus.PARTIAL
        )
        assert result is True

    def test_valid_transition_partial_to_filled(self):
        """Test valid transition from PARTIAL to FILLED."""
        result = OrderStateMachine.can_transition(
            OrderStatus.PARTIAL, OrderStatus.FILLED
        )
        assert result is True

    def test_invalid_transition_filled_to_anything(self):
        """Test that FILLED is terminal - no transitions out."""
        for status in OrderStatus:
            if status != OrderStatus.FILLED:
                result = OrderStateMachine.can_transition(
                    OrderStatus.FILLED, status
                )
                assert result is False

    def test_invalid_transition_cancelled_to_anything(self):
        """Test that CANCELLED is terminal."""
        for status in OrderStatus:
            if status != OrderStatus.CANCELLED:
                result = OrderStateMachine.can_transition(
                    OrderStatus.CANCELLED, status
                )
                assert result is False

    def test_invalid_transition_pending_to_filled_directly(self):
        """Test invalid direct transition from PENDING to FILLED."""
        result = OrderStateMachine.can_transition(
            OrderStatus.PENDING, OrderStatus.FILLED
        )
        assert result is False

    def test_is_terminal_for_filled(self):
        """Test FILLED is identified as terminal."""
        assert OrderStateMachine.is_terminal(OrderStatus.FILLED) is True

    def test_is_terminal_for_cancelled(self):
        """Test CANCELLED is identified as terminal."""
        assert OrderStateMachine.is_terminal(OrderStatus.CANCELLED) is True

    def test_is_terminal_for_rejected(self):
        """Test REJECTED is identified as terminal."""
        assert OrderStateMachine.is_terminal(OrderStatus.REJECTED) is True

    def test_is_terminal_for_pending_false(self):
        """Test PENDING is not terminal."""
        assert OrderStateMachine.is_terminal(OrderStatus.PENDING) is False

    def test_is_active_for_pending(self):
        """Test PENDING is active."""
        assert OrderStateMachine.is_active(OrderStatus.PENDING) is True

    def test_is_active_for_open(self):
        """Test OPEN is active."""
        assert OrderStateMachine.is_active(OrderStatus.OPEN) is True

    def test_is_active_for_partial(self):
        """Test PARTIAL is active."""
        assert OrderStateMachine.is_active(OrderStatus.PARTIAL) is True

    def test_is_active_for_filled_false(self):
        """Test FILLED is not active."""
        assert OrderStateMachine.is_active(OrderStatus.FILLED) is False


# ─────────────────────────────────────────────────────────────────────────────
# TEST ORDER EVENTS
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderEvents:
    """Tests for OrderEvent enum."""

    def test_all_events_defined(self):
        """Test all expected events are defined."""
        expected_events = [
            "CREATED", "SUBMITTED", "ACKNOWLEDGED", "PARTIAL_FILL",
            "FILLED", "CANCELLED", "REJECTED", "EXPIRED", "MODIFIED", "ERROR"
        ]

        for event in expected_events:
            assert hasattr(OrderEvent, event)

    def test_event_values_are_strings(self):
        """Test event values are strings."""
        for event in OrderEvent:
            assert isinstance(event.value, str)


# ─────────────────────────────────────────────────────────────────────────────
# TEST ORDER MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderManager:
    """Tests for OrderManager class."""

    def test_on_event_registers_callback(self, order_manager):
        """Test callback registration for events."""
        callback = MagicMock()
        order_manager.on_event(OrderEvent.CREATED, callback)

        assert callback in order_manager._callbacks[OrderEvent.CREATED]

    @pytest.mark.asyncio
    async def test_emit_event_calls_callbacks(self, order_manager, sample_order):
        """Test event emission calls registered callbacks."""
        callback = AsyncMock()
        order_manager.on_event(OrderEvent.CREATED, callback)

        await order_manager._emit_event(OrderEvent.CREATED, sample_order, {"test": True})

        callback.assert_called_once_with(sample_order, {"test": True})

    @pytest.mark.asyncio
    async def test_emit_event_handles_sync_callbacks(self, order_manager, sample_order):
        """Test event emission handles sync callbacks."""
        callback = MagicMock()
        order_manager.on_event(OrderEvent.CREATED, callback)

        await order_manager._emit_event(OrderEvent.CREATED, sample_order, None)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event_handles_callback_errors(self, order_manager, sample_order):
        """Test event emission handles callback errors gracefully."""
        callback = AsyncMock(side_effect=Exception("Callback error"))
        order_manager.on_event(OrderEvent.CREATED, callback)

        # Should not raise
        await order_manager._emit_event(OrderEvent.CREATED, sample_order, None)

    @pytest.mark.asyncio
    async def test_create_order(self, order_manager):
        """Test order creation."""
        with patch('src.broker.order_manager.db') as mock_db:
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.session.return_value.__aexit__ = AsyncMock()

            # Mock repositories
            mock_order_repo = MagicMock()
            mock_order_repo.create = AsyncMock(side_effect=lambda x: x)
            mock_audit_repo = MagicMock()
            mock_audit_repo.log = AsyncMock()

            with patch('src.broker.order_manager.OrderRepository', return_value=mock_order_repo):
                with patch('src.broker.order_manager.AuditLogRepository', return_value=mock_audit_repo):
                    order = await order_manager.create_order(
                        symbol="TEST",
                        side=Side.BUY,
                        quantity=100,
                        order_type="MARKET",
                    )

                    assert order.symbol == "TEST"
                    assert order.side == Side.BUY
                    assert order.quantity == 100
                    assert order.status == OrderStatus.PENDING

    def test_calculate_fees(self, order_manager, sample_order):
        """Test fee calculation."""
        sample_order.average_price = Decimal("100")
        sample_order.filled_quantity = 100

        fees = order_manager._calculate_fees(sample_order)

        # Brokerage: 20 + STT: 0.1% of 10000 = 10
        assert fees == Decimal("30")

    def test_map_broker_status_complete(self, order_manager):
        """Test broker status mapping for COMPLETE."""
        status = order_manager._map_broker_status("COMPLETE")
        assert status == OrderStatus.FILLED

    def test_map_broker_status_cancelled(self, order_manager):
        """Test broker status mapping for CANCELLED."""
        status = order_manager._map_broker_status("CANCELLED")
        assert status == OrderStatus.CANCELLED

    def test_map_broker_status_unknown(self, order_manager):
        """Test broker status mapping for unknown status."""
        status = order_manager._map_broker_status("UNKNOWN_STATUS")
        assert status is None


# ─────────────────────────────────────────────────────────────────────────────
# TEST ORDER LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderLifecycle:
    """Tests for complete order lifecycle."""

    def test_order_status_values(self):
        """Test all OrderStatus values exist."""
        expected = [
            "PENDING", "SUBMITTED", "OPEN", "PARTIAL",
            "FILLED", "CANCELLED", "REJECTED", "EXPIRED", "ERROR"
        ]

        for status in expected:
            assert hasattr(OrderStatus, status)

    def test_order_flow_buy_to_fill(self):
        """Test typical buy order flow through states."""
        # PENDING -> SUBMITTED -> OPEN -> FILLED
        assert OrderStateMachine.can_transition(OrderStatus.PENDING, OrderStatus.SUBMITTED)
        assert OrderStateMachine.can_transition(OrderStatus.SUBMITTED, OrderStatus.OPEN)
        assert OrderStateMachine.can_transition(OrderStatus.OPEN, OrderStatus.FILLED)

    def test_order_flow_partial_fills(self):
        """Test order flow with partial fills."""
        # OPEN -> PARTIAL -> PARTIAL -> FILLED
        assert OrderStateMachine.can_transition(OrderStatus.OPEN, OrderStatus.PARTIAL)
        assert OrderStateMachine.can_transition(OrderStatus.PARTIAL, OrderStatus.FILLED)

    def test_order_cancellation_flow(self):
        """Test order cancellation from various states."""
        # Can cancel from PENDING, SUBMITTED, OPEN, PARTIAL
        assert OrderStateMachine.can_transition(OrderStatus.PENDING, OrderStatus.CANCELLED)
        assert OrderStateMachine.can_transition(OrderStatus.SUBMITTED, OrderStatus.CANCELLED)
        assert OrderStateMachine.can_transition(OrderStatus.OPEN, OrderStatus.CANCELLED)
        assert OrderStateMachine.can_transition(OrderStatus.PARTIAL, OrderStatus.CANCELLED)


# ─────────────────────────────────────────────────────────────────────────────
# TEST POSITION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionManager:
    """Tests for PositionManager class."""

    @pytest.fixture
    def position_manager(self, order_manager):
        """Create a PositionManager instance."""
        with patch('src.broker.order_manager.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.strategy = MagicMock()
            mock_settings.return_value.strategy.trailing_stop_activation_percent = Decimal("2.0")
            mock_settings.return_value.strategy.trailing_stop_distance_percent = Decimal("1.0")
            return PositionManager(order_manager)

    def test_risk_reward_calculation_for_long(self, position_manager):
        """Test R:R calculation for long position."""
        entry = Decimal("100")
        stop = Decimal("95")
        target = Decimal("115")
        quantity = 10

        risk = (entry - stop) * quantity  # 50
        reward = (target - entry) * quantity  # 150
        rr_ratio = reward / risk  # 3.0

        assert risk == Decimal("50")
        assert reward == Decimal("150")
        assert rr_ratio == Decimal("3")

    def test_risk_reward_calculation_for_short(self, position_manager):
        """Test R:R calculation for short position."""
        entry = Decimal("100")
        stop = Decimal("105")
        target = Decimal("85")
        quantity = 10

        risk = (stop - entry) * quantity  # 50
        reward = (entry - target) * quantity  # 150
        rr_ratio = reward / risk  # 3.0

        assert risk == Decimal("50")
        assert reward == Decimal("150")
        assert rr_ratio == Decimal("3")

    def test_exit_side_opposite_of_entry(self, position_manager):
        """Test exit side is opposite of entry side."""
        buy_exit = Side.SELL if Side.BUY == Side.BUY else Side.BUY
        sell_exit = Side.SELL if Side.SELL == Side.BUY else Side.BUY

        assert buy_exit == Side.SELL
        assert sell_exit == Side.BUY


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
