"""
Order State Machine and Execution Manager
==========================================
Production-grade order management with state machine, fill tracking,
and broker reconciliation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, Callable, List, Awaitable
from enum import Enum
import logging
import uuid

from src.core.models import (
    Order, Position, OrderStatus, PositionStatus, Side, ExitReason, AlertLevel
)
from src.core.database import (
    db, OrderRepository, PositionRepository, AuditLogRepository
)
from src.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ORDER STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

class OrderStateMachine:
    """
    Finite state machine for order lifecycle management.

    Valid state transitions:
    PENDING -> SUBMITTED -> OPEN -> PARTIAL -> FILLED
                        -> CANCELLED
                        -> REJECTED
                        -> EXPIRED
                        -> ERROR
    """

    # Define valid transitions
    VALID_TRANSITIONS: Dict[OrderStatus, List[OrderStatus]] = {
        OrderStatus.PENDING: [
            OrderStatus.SUBMITTED,
            OrderStatus.CANCELLED,
            OrderStatus.ERROR,
        ],
        OrderStatus.SUBMITTED: [
            OrderStatus.OPEN,
            OrderStatus.FILLED,  # Market orders can fill immediately
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.ERROR,
        ],
        OrderStatus.OPEN: [
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        ],
        OrderStatus.PARTIAL: [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.ERROR,
        ],
        # Terminal states - no transitions out
        OrderStatus.FILLED: [],
        OrderStatus.CANCELLED: [],
        OrderStatus.REJECTED: [],
        OrderStatus.EXPIRED: [],
        OrderStatus.ERROR: [],
    }

    @classmethod
    def can_transition(cls, from_status: OrderStatus, to_status: OrderStatus) -> bool:
        """Check if a state transition is valid."""
        valid_targets = cls.VALID_TRANSITIONS.get(from_status, [])
        return to_status in valid_targets

    @classmethod
    def is_terminal(cls, status: OrderStatus) -> bool:
        """Check if status is a terminal (final) state."""
        return len(cls.VALID_TRANSITIONS.get(status, [])) == 0

    @classmethod
    def is_active(cls, status: OrderStatus) -> bool:
        """Check if order is still active (can receive updates)."""
        return status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIAL,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ORDER EVENT TYPES
# ─────────────────────────────────────────────────────────────────────────────

class OrderEvent(str, Enum):
    """Events that can occur on orders."""
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    MODIFIED = "MODIFIED"
    ERROR = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# ORDER MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class OrderManager:
    """
    Production-grade order management with:
    - State machine validation
    - Fill tracking
    - Position management
    - Broker reconciliation
    - Event callbacks
    """

    def __init__(self, broker: "BaseBroker"):
        self.broker = broker
        self.settings = get_settings()
        self._callbacks: Dict[OrderEvent, List[Callable]] = {event: [] for event in OrderEvent}
        self._pending_orders: Dict[str, Order] = {}
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._running = False

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT SYSTEM
    # ─────────────────────────────────────────────────────────────────────────

    def on_event(self, event: OrderEvent, callback: Callable) -> None:
        """Register a callback for an order event."""
        self._callbacks[event].append(callback)

    async def _emit_event(
        self,
        event: OrderEvent,
        order: Order,
        data: Optional[Dict] = None
    ) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, data)
                else:
                    callback(order, data)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER CREATION
    # ─────────────────────────────────────────────────────────────────────────

    async def create_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[Decimal] = None,
        trigger_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        target: Optional[Decimal] = None,
        product_type: str = "CNC",
        position_id: Optional[str] = None,
        signal_id: Optional[str] = None,
        tags: Optional[Dict] = None,
    ) -> Order:
        """
        Create a new order with validation.

        Returns the order in PENDING state.
        """
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            pending_quantity=quantity,
            order_type=order_type,
            price=price,
            trigger_price=trigger_price,
            stoploss_value=stop_loss,
            target_value=target,
            product_type=product_type,
            position_id=position_id,
            signal_id=signal_id,
            status=OrderStatus.PENDING,
            tags=tags or {},
        )

        async with db.session() as session:
            repo = OrderRepository(session)
            order = await repo.create(order)

            # Audit log
            audit_repo = AuditLogRepository(session)
            await audit_repo.log(
                event_type="ORDER_CREATED",
                description=f"Order created: {symbol} {side.value} {quantity}",
                entity_type="order",
                entity_id=order.id,
                new_value=order.to_dict(),
                level=AlertLevel.INFO,
            )

        await self._emit_event(OrderEvent.CREATED, order)
        logger.info(f"Order created: {order}")

        return order

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER SUBMISSION
    # ─────────────────────────────────────────────────────────────────────────

    async def submit_order(self, order_id: str) -> Order:
        """
        Submit an order to the broker.

        Transitions: PENDING -> SUBMITTED
        """
        async with db.session() as session:
            repo = OrderRepository(session)
            order = await repo.get_by_id(order_id)

            if not order:
                raise ValueError(f"Order {order_id} not found")

            if order.status != OrderStatus.PENDING:
                raise ValueError(f"Cannot submit order in {order.status} state")

            # Submit to broker
            try:
                broker_response = await self.broker.place_order(
                    symbol=order.symbol,
                    side=order.side.value if isinstance(order.side, Side) else order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    price=float(order.price) if order.price else None,
                    trigger_price=float(order.trigger_price) if order.trigger_price else None,
                    product_type=order.product_type,
                )

                if broker_response.get("status") == "success":
                    order.broker_order_id = broker_response.get("order_id")
                    order.status = OrderStatus.SUBMITTED
                    order.submitted_at = datetime.now()
                    order.broker_response = broker_response

                    # Track pending order for fill monitoring
                    self._pending_orders[order.id] = order

                    logger.info(f"Order submitted: {order.id} -> broker_id: {order.broker_order_id}")
                else:
                    order.status = OrderStatus.REJECTED
                    order.status_message = broker_response.get("error", "Unknown error")
                    order.broker_response = broker_response

                    logger.warning(f"Order rejected by broker: {order.id} - {order.status_message}")

            except Exception as e:
                order.status = OrderStatus.ERROR
                order.status_message = str(e)
                logger.error(f"Order submission error: {order.id} - {e}")

            await session.flush()

            # Audit log
            audit_repo = AuditLogRepository(session)
            await audit_repo.log(
                event_type="ORDER_SUBMITTED" if order.status == OrderStatus.SUBMITTED else "ORDER_REJECTED",
                description=f"Order {order.status.value}: {order.symbol}",
                entity_type="order",
                entity_id=order.id,
                new_value=order.to_dict(),
                level=AlertLevel.INFO if order.status == OrderStatus.SUBMITTED else AlertLevel.WARNING,
            )

        event = OrderEvent.SUBMITTED if order.status == OrderStatus.SUBMITTED else OrderEvent.REJECTED
        await self._emit_event(event, order)

        return order

    # ─────────────────────────────────────────────────────────────────────────
    # FILL PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    async def process_fill(
        self,
        broker_order_id: str,
        filled_quantity: int,
        average_price: Decimal,
        is_complete: bool = False,
    ) -> Optional[Order]:
        """
        Process a fill update from the broker.

        Transitions: SUBMITTED/OPEN -> PARTIAL or FILLED
        """
        async with db.session() as session:
            repo = OrderRepository(session)
            order = await repo.get_by_broker_id(broker_order_id)

            if not order:
                logger.warning(f"Fill received for unknown order: {broker_order_id}")
                return None

            old_status = order.status
            old_filled = order.filled_quantity

            # Validate state transition
            if not OrderStateMachine.is_active(order.status):
                logger.warning(f"Fill received for terminal order: {order.id} [{order.status}]")
                return order

            # Update fill info
            order.filled_quantity = filled_quantity
            order.pending_quantity = order.quantity - filled_quantity
            order.average_price = average_price

            # Determine new status
            if is_complete or filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                # Remove from pending tracking
                self._pending_orders.pop(order.id, None)
            elif filled_quantity > old_filled:
                order.status = OrderStatus.PARTIAL

            await session.flush()

            # Update position if exists
            if order.position_id:
                await self._update_position_on_fill(session, order)

            # Audit log
            audit_repo = AuditLogRepository(session)
            await audit_repo.log(
                event_type="ORDER_FILL",
                description=f"Fill: {filled_quantity}/{order.quantity} @ {average_price}",
                entity_type="order",
                entity_id=order.id,
                old_value={"status": old_status.value, "filled": old_filled},
                new_value=order.to_dict(),
                level=AlertLevel.INFO,
            )

        # Emit events
        if order.status == OrderStatus.FILLED:
            await self._emit_event(OrderEvent.FILLED, order, {"average_price": average_price})
            logger.info(f"Order filled: {order.id} @ {average_price}")
        else:
            await self._emit_event(OrderEvent.PARTIAL_FILL, order, {"filled": filled_quantity})
            logger.info(f"Partial fill: {order.id} - {filled_quantity}/{order.quantity}")

        return order

    async def _update_position_on_fill(self, session, order: Order) -> None:
        """Update position when order is filled."""
        pos_repo = PositionRepository(session)
        position = await pos_repo.get_by_id(order.position_id)

        if not position:
            return

        # Entry fill
        if position.status == PositionStatus.PENDING_ENTRY and order.status == OrderStatus.FILLED:
            position.status = PositionStatus.OPEN
            position.entry_price = order.average_price
            position.entry_value = order.average_price * order.filled_quantity
            await session.flush()

        # Exit fill
        elif position.status == PositionStatus.PENDING_EXIT and order.status == OrderStatus.FILLED:
            # Determine exit reason from tags
            exit_reason = ExitReason.MANUAL
            if order.tags:
                if order.tags.get("exit_type") == "stop_loss":
                    exit_reason = ExitReason.STOP_LOSS_HIT
                elif order.tags.get("exit_type") == "target":
                    exit_reason = ExitReason.TARGET_HIT
                elif order.tags.get("exit_type") == "trailing":
                    exit_reason = ExitReason.TRAILING_STOP

            await pos_repo.close_position(
                position_id=position.id,
                exit_price=order.average_price,
                exit_reason=exit_reason,
                fees=self._calculate_fees(order),
            )

    def _calculate_fees(self, order: Order) -> Decimal:
        """Calculate trading fees for an order."""
        settings = self.settings
        value = (order.average_price or order.price or Decimal(0)) * order.filled_quantity

        brokerage = settings.risk.brokerage_per_order
        stt = value * (settings.risk.stt_percent / 100)

        return brokerage + stt

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER CANCELLATION
    # ─────────────────────────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str, reason: str = "User requested") -> Order:
        """
        Cancel an active order.

        Transitions: PENDING/SUBMITTED/OPEN/PARTIAL -> CANCELLED
        """
        async with db.session() as session:
            repo = OrderRepository(session)
            order = await repo.get_by_id(order_id)

            if not order:
                raise ValueError(f"Order {order_id} not found")

            if not OrderStateMachine.is_active(order.status):
                raise ValueError(f"Cannot cancel order in {order.status} state")

            old_status = order.status

            # Cancel with broker if submitted
            if order.broker_order_id:
                try:
                    await self.broker.cancel_order(order.broker_order_id)
                except Exception as e:
                    logger.error(f"Broker cancel failed: {e}")
                    # Continue with local cancellation anyway

            order.status = OrderStatus.CANCELLED
            order.status_message = reason
            order.cancelled_quantity = order.pending_quantity

            # Remove from pending tracking
            self._pending_orders.pop(order.id, None)

            await session.flush()

            # Audit log
            audit_repo = AuditLogRepository(session)
            await audit_repo.log(
                event_type="ORDER_CANCELLED",
                description=f"Order cancelled: {reason}",
                entity_type="order",
                entity_id=order.id,
                old_value={"status": old_status.value},
                new_value=order.to_dict(),
                level=AlertLevel.INFO,
            )

        await self._emit_event(OrderEvent.CANCELLED, order, {"reason": reason})
        logger.info(f"Order cancelled: {order.id} - {reason}")

        return order

    # ─────────────────────────────────────────────────────────────────────────
    # BROKER RECONCILIATION
    # ─────────────────────────────────────────────────────────────────────────

    async def reconcile_with_broker(self) -> Dict[str, Any]:
        """
        Reconcile local order state with broker state.

        Should be called:
        - On startup
        - Periodically during trading hours
        - After connection recovery
        """
        results = {
            "checked": 0,
            "updated": 0,
            "mismatches": [],
            "errors": [],
        }

        try:
            # Get broker's view of open orders
            broker_orders = await self.broker.get_orders()
            broker_order_map = {o["order_id"]: o for o in broker_orders}

            async with db.session() as session:
                repo = OrderRepository(session)
                active_orders = await repo.get_active_orders()

                for order in active_orders:
                    results["checked"] += 1

                    if not order.broker_order_id:
                        continue

                    broker_order = broker_order_map.get(order.broker_order_id)

                    if not broker_order:
                        # Order not found at broker - might be filled/cancelled
                        logger.warning(f"Order {order.id} not found at broker")
                        results["mismatches"].append({
                            "order_id": order.id,
                            "issue": "not_found_at_broker"
                        })
                        continue

                    # Check for status mismatch
                    broker_status = self._map_broker_status(broker_order.get("status", ""))
                    if broker_status and broker_status != order.status:
                        old_status = order.status
                        order.status = broker_status

                        # Update fill info if available
                        if broker_order.get("filled_quantity"):
                            order.filled_quantity = broker_order["filled_quantity"]
                        if broker_order.get("average_price"):
                            order.average_price = Decimal(str(broker_order["average_price"]))

                        results["updated"] += 1
                        logger.info(f"Order {order.id} reconciled: {old_status} -> {broker_status}")

                await session.commit()

        except Exception as e:
            logger.error(f"Reconciliation error: {e}")
            results["errors"].append(str(e))

        return results

    def _map_broker_status(self, broker_status: str) -> Optional[OrderStatus]:
        """Map broker status string to OrderStatus enum."""
        status_map = {
            "PENDING": OrderStatus.PENDING,
            "OPEN": OrderStatus.OPEN,
            "TRIGGER PENDING": OrderStatus.OPEN,
            "COMPLETE": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return status_map.get(broker_status.upper())

    # ─────────────────────────────────────────────────────────────────────────
    # BACKGROUND RECONCILIATION TASK
    # ─────────────────────────────────────────────────────────────────────────

    async def start_reconciliation_loop(self, interval_seconds: int = 30) -> None:
        """Start background reconciliation loop."""
        if self._running:
            return

        self._running = True
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop(interval_seconds)
        )
        logger.info(f"Started reconciliation loop (interval: {interval_seconds}s)")

    async def stop_reconciliation_loop(self) -> None:
        """Stop background reconciliation loop."""
        self._running = False
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
            self._reconciliation_task = None
        logger.info("Stopped reconciliation loop")

    async def _reconciliation_loop(self, interval: int) -> None:
        """Background loop for periodic reconciliation."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._running:
                    results = await self.reconcile_with_broker()
                    if results["updated"] > 0 or results["mismatches"]:
                        logger.info(f"Reconciliation results: {results}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation loop error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders."""
        async with db.session() as session:
            repo = OrderRepository(session)
            return await repo.get_active_orders(symbol)

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        async with db.session() as session:
            repo = OrderRepository(session)
            return await repo.get_by_id(order_id)

    async def get_today_orders(self) -> List[Order]:
        """Get all orders created today."""
        async with db.session() as session:
            repo = OrderRepository(session)
            return await repo.get_today_orders()


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class PositionManager:
    """
    Position lifecycle management with entry/exit handling.
    """

    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.settings = get_settings()

    async def open_position(
        self,
        symbol: str,
        side: Side,
        entry_price: Decimal,
        stop_loss: Decimal,
        target: Decimal,
        quantity: int,
        signal_id: Optional[str] = None,
        trend: Optional[str] = None,
    ) -> Position:
        """
        Open a new position by creating entry order.

        Returns position in PENDING_ENTRY state.
        """
        # Calculate risk metrics
        if side == Side.BUY:
            risk_amount = (entry_price - stop_loss) * quantity
            reward_amount = (target - entry_price) * quantity
        else:
            risk_amount = (stop_loss - entry_price) * quantity
            reward_amount = (entry_price - target) * quantity

        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else Decimal(0)

        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            remaining_quantity=quantity,
            entry_price=entry_price,
            entry_value=entry_price * quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            initial_stop_loss=stop_loss,
            target=target,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            status=PositionStatus.PENDING_ENTRY,
            signal_id=signal_id,
            trend_at_entry=trend,
        )

        async with db.session() as session:
            pos_repo = PositionRepository(session)
            position = await pos_repo.create(position)

            # Audit log
            audit_repo = AuditLogRepository(session)
            await audit_repo.log(
                event_type="POSITION_OPENED",
                description=f"Position opened: {symbol} {side.value} {quantity}",
                entity_type="position",
                entity_id=position.id,
                new_value=position.to_dict(),
                level=AlertLevel.INFO,
            )

        # Create and submit entry order
        order = await self.order_manager.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="LIMIT",
            price=entry_price,
            stop_loss=stop_loss,
            target=target,
            position_id=position.id,
            signal_id=signal_id,
            tags={"entry_type": "signal"},
        )

        await self.order_manager.submit_order(order.id)

        logger.info(f"Position opened: {position}")
        return position

    async def close_position(
        self,
        position_id: str,
        exit_reason: ExitReason,
        price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """
        Close a position by creating exit order.
        """
        async with db.session() as session:
            pos_repo = PositionRepository(session)
            position = await pos_repo.get_by_id(position_id)

            if not position:
                raise ValueError(f"Position {position_id} not found")

            if position.status != PositionStatus.OPEN:
                raise ValueError(f"Cannot close position in {position.status} state")

            # Update status
            position.status = PositionStatus.PENDING_EXIT
            await session.flush()

        # Determine exit side (opposite of entry)
        exit_side = Side.SELL if position.side == Side.BUY else Side.BUY

        # Create exit order
        order = await self.order_manager.create_order(
            symbol=position.symbol,
            side=exit_side,
            quantity=position.remaining_quantity,
            order_type="MARKET" if price is None else "LIMIT",
            price=price,
            position_id=position.id,
            tags={"exit_type": exit_reason.value.lower()},
        )

        await self.order_manager.submit_order(order.id)

        logger.info(f"Position exit initiated: {position_id} - {exit_reason.value}")
        return order

    async def update_trailing_stop(
        self,
        position_id: str,
        current_price: Decimal
    ) -> Optional[Position]:
        """
        Update trailing stop loss for a position.
        """
        settings = self.settings

        async with db.session() as session:
            pos_repo = PositionRepository(session)
            position = await pos_repo.get_by_id(position_id)

            if not position or position.status != PositionStatus.OPEN:
                return None

            # Check if trailing stop should be activated
            activation_percent = settings.strategy.trailing_stop_activation_percent
            trailing_distance = settings.strategy.trailing_stop_distance_percent

            if position.side == Side.BUY:
                # For long positions
                profit_percent = ((current_price - position.entry_price) / position.entry_price) * 100

                if profit_percent >= float(activation_percent):
                    new_stop = current_price * (1 - float(trailing_distance) / 100)

                    if new_stop > position.stop_loss:
                        await pos_repo.update_stop_loss(
                            position_id=position_id,
                            new_stop_loss=new_stop,
                            is_trailing=True
                        )
                        logger.info(f"Trailing stop updated: {position_id} SL -> {new_stop}")
            else:
                # For short positions
                profit_percent = ((position.entry_price - current_price) / position.entry_price) * 100

                if profit_percent >= float(activation_percent):
                    new_stop = current_price * (1 + float(trailing_distance) / 100)

                    if new_stop < position.stop_loss:
                        await pos_repo.update_stop_loss(
                            position_id=position_id,
                            new_stop_loss=new_stop,
                            is_trailing=True
                        )
                        logger.info(f"Trailing stop updated: {position_id} SL -> {new_stop}")

            return position

    async def check_stop_loss_target(
        self,
        position_id: str,
        current_price: Decimal
    ) -> Optional[str]:
        """
        Check if stop loss or target is hit.

        Returns: "stop_loss", "target", or None
        """
        async with db.session() as session:
            pos_repo = PositionRepository(session)
            position = await pos_repo.get_by_id(position_id)

            if not position or position.status != PositionStatus.OPEN:
                return None

            if position.should_trigger_stop_loss(current_price):
                return "stop_loss"
            elif position.should_trigger_target(current_price):
                return "target"

            return None

    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        async with db.session() as session:
            pos_repo = PositionRepository(session)
            return await pos_repo.get_open_positions()

    async def get_position_count(self) -> int:
        """Get count of open positions."""
        async with db.session() as session:
            pos_repo = PositionRepository(session)
            return await pos_repo.get_position_count()

    async def flatten_all_positions(self, reason: ExitReason) -> List[Order]:
        """
        Close all open positions immediately.

        Used for kill switch / emergency exit.
        """
        orders = []
        positions = await self.get_open_positions()

        for position in positions:
            if position.status == PositionStatus.OPEN:
                try:
                    order = await self.close_position(position.id, reason)
                    if order:
                        orders.append(order)
                except Exception as e:
                    logger.error(f"Failed to close position {position.id}: {e}")

        logger.warning(f"Flattened {len(orders)} positions - Reason: {reason.value}")
        return orders
