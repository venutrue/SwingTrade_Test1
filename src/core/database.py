"""
Database Connection and Repository Layer
=========================================
Async database management with connection pooling and transaction support.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, TypeVar, Generic, Type
import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from src.core.models import (
    Base, Order, Position, Signal, DailyPnL, AuditLog, MarketHoliday,
    OrderStatus, PositionStatus, Side, ExitReason, AlertLevel, Trend, SignalType
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Base)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE ENGINE & SESSION
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Async database connection manager with connection pooling.
    """

    _instance: Optional["Database"] = None
    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def __new__(cls) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        """Initialize database engine and create tables."""
        if self._engine is not None:
            return

        settings = get_settings()

        self._engine = create_async_engine(
            settings.database.url,
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,  # Verify connections before use
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")

    @asynccontextmanager
    async def session(self):
        """Get a database session with automatic cleanup."""
        if self._session_factory is None:
            await self.initialize()

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def transaction(self):
        """Get a session with explicit transaction control."""
        if self._session_factory is None:
            await self.initialize()

        session = self._session_factory()
        try:
            async with session.begin():
                yield session
        except Exception as e:
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            await session.close()


# Global database instance
db = Database()


# ─────────────────────────────────────────────────────────────────────────────
# ORDER REPOSITORY
# ─────────────────────────────────────────────────────────────────────────────

class OrderRepository:
    """
    Repository for Order CRUD operations.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, order: Order) -> Order:
        """Create a new order."""
        self.session.add(order)
        await self.session.flush()
        await self.session.refresh(order)
        return order

    async def get_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        result = await self.session.execute(
            select(Order).where(Order.id == order_id)
        )
        return result.scalar_one_or_none()

    async def get_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        """Get order by broker order ID."""
        result = await self.session.execute(
            select(Order).where(Order.broker_order_id == broker_order_id)
        )
        return result.scalar_one_or_none()

    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active (non-terminal) orders."""
        query = select(Order).where(
            Order.status.in_([
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.OPEN,
                OrderStatus.PARTIAL,
            ])
        )
        if symbol:
            query = query.where(Order.symbol == symbol)
        query = query.order_by(Order.created_at.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_orders_by_position(self, position_id: str) -> List[Order]:
        """Get all orders for a position."""
        result = await self.session.execute(
            select(Order)
            .where(Order.position_id == position_id)
            .order_by(Order.created_at)
        )
        return list(result.scalars().all())

    async def get_today_orders(self) -> List[Order]:
        """Get all orders created today."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.session.execute(
            select(Order)
            .where(Order.created_at >= today_start)
            .order_by(Order.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        order_id: str,
        status: OrderStatus,
        message: Optional[str] = None,
        **kwargs
    ) -> Optional[Order]:
        """Update order status and related fields."""
        order = await self.get_by_id(order_id)
        if not order:
            return None

        order.status = status
        if message:
            order.status_message = message

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(order, key):
                setattr(order, key, value)

        # Set timestamp based on status
        if status == OrderStatus.SUBMITTED:
            order.submitted_at = datetime.now()
        elif status == OrderStatus.FILLED:
            order.filled_at = datetime.now()

        await self.session.flush()
        return order

    async def update_fill(
        self,
        order_id: str,
        filled_quantity: int,
        average_price: Decimal,
        is_complete: bool = False
    ) -> Optional[Order]:
        """Update order with fill information."""
        order = await self.get_by_id(order_id)
        if not order:
            return None

        order.filled_quantity = filled_quantity
        order.average_price = average_price
        order.pending_quantity = order.quantity - filled_quantity

        if is_complete or filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
        elif filled_quantity > 0:
            order.status = OrderStatus.PARTIAL

        await self.session.flush()
        return order


# ─────────────────────────────────────────────────────────────────────────────
# POSITION REPOSITORY
# ─────────────────────────────────────────────────────────────────────────────

class PositionRepository:
    """
    Repository for Position CRUD operations.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, position: Position) -> Position:
        """Create a new position."""
        self.session.add(position)
        await self.session.flush()
        await self.session.refresh(position)
        return position

    async def get_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID with orders."""
        result = await self.session.execute(
            select(Position)
            .options(selectinload(Position.orders))
            .where(Position.id == position_id)
        )
        return result.scalar_one_or_none()

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions."""
        query = select(Position).where(
            Position.status.in_([
                PositionStatus.PENDING_ENTRY,
                PositionStatus.OPEN,
                PositionStatus.PENDING_EXIT,
            ])
        )
        if symbol:
            query = query.where(Position.symbol == symbol)
        query = query.order_by(Position.entry_time.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_position_count(self) -> int:
        """Get count of open positions."""
        result = await self.session.execute(
            select(func.count(Position.id)).where(
                Position.status.in_([
                    PositionStatus.OPEN,
                    PositionStatus.PENDING_EXIT,
                ])
            )
        )
        return result.scalar() or 0

    async def update_status(
        self,
        position_id: str,
        status: PositionStatus,
        **kwargs
    ) -> Optional[Position]:
        """Update position status."""
        position = await self.get_by_id(position_id)
        if not position:
            return None

        position.status = status

        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)

        await self.session.flush()
        return position

    async def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        exit_reason: ExitReason,
        fees: Decimal = Decimal("0"),
    ) -> Optional[Position]:
        """Close a position with P&L calculation."""
        position = await self.get_by_id(position_id)
        if not position:
            return None

        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = exit_reason
        position.status = PositionStatus.CLOSED
        position.remaining_quantity = 0
        position.fees = fees

        # Calculate P&L
        if position.side == Side.BUY:
            position.realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            position.realized_pnl = (position.entry_price - exit_price) * position.quantity

        position.net_pnl = position.realized_pnl - fees
        position.exit_value = exit_price * position.quantity

        if position.entry_price > 0:
            if position.side == Side.BUY:
                position.return_percent = ((exit_price - position.entry_price) / position.entry_price) * 100
            else:
                position.return_percent = ((position.entry_price - exit_price) / position.entry_price) * 100

        await self.session.flush()
        return position

    async def update_unrealized_pnl(
        self,
        position_id: str,
        current_price: Decimal
    ) -> Optional[Position]:
        """Update unrealized P&L for an open position."""
        position = await self.get_by_id(position_id)
        if not position or position.status != PositionStatus.OPEN:
            return None

        position.unrealized_pnl = position.calculate_unrealized_pnl(current_price)

        # Update high/low tracking for trailing stop
        if position.highest_price is None or current_price > position.highest_price:
            position.highest_price = current_price
        if position.lowest_price is None or current_price < position.lowest_price:
            position.lowest_price = current_price

        await self.session.flush()
        return position

    async def update_stop_loss(
        self,
        position_id: str,
        new_stop_loss: Decimal,
        is_trailing: bool = False
    ) -> Optional[Position]:
        """Update stop loss (for trailing stops)."""
        position = await self.get_by_id(position_id)
        if not position:
            return None

        position.stop_loss = new_stop_loss
        if is_trailing:
            position.trailing_stop_active = True

        await self.session.flush()
        return position

    async def get_closed_positions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> List[Position]:
        """Get closed positions within date range."""
        query = select(Position).where(Position.status == PositionStatus.CLOSED)

        if start_date:
            query = query.where(Position.exit_time >= start_date)
        if end_date:
            query = query.where(Position.exit_time <= end_date)
        if symbol:
            query = query.where(Position.symbol == symbol)

        query = query.order_by(Position.exit_time.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL REPOSITORY
# ─────────────────────────────────────────────────────────────────────────────

class SignalRepository:
    """
    Repository for Signal CRUD operations.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, signal: Signal) -> Signal:
        """Create a new signal."""
        self.session.add(signal)
        await self.session.flush()
        await self.session.refresh(signal)
        return signal

    async def get_by_id(self, signal_id: str) -> Optional[Signal]:
        """Get signal by ID."""
        result = await self.session.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        return result.scalar_one_or_none()

    async def get_recent_signals(
        self,
        symbol: str,
        hours: int = 24
    ) -> List[Signal]:
        """Get recent signals for a symbol."""
        cutoff = datetime.now() - timedelta(hours=hours)
        result = await self.session.execute(
            select(Signal)
            .where(and_(
                Signal.symbol == symbol,
                Signal.timestamp >= cutoff
            ))
            .order_by(Signal.timestamp.desc())
        )
        return list(result.scalars().all())

    async def mark_executed(
        self,
        signal_id: str,
        position_id: str
    ) -> Optional[Signal]:
        """Mark signal as executed."""
        signal = await self.get_by_id(signal_id)
        if not signal:
            return None

        signal.executed = True
        signal.position_id = position_id
        await self.session.flush()
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# DAILY P&L REPOSITORY
# ─────────────────────────────────────────────────────────────────────────────

class DailyPnLRepository:
    """
    Repository for DailyPnL tracking.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_today(self, starting_capital: Decimal) -> DailyPnL:
        """Get or create today's P&L record."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        result = await self.session.execute(
            select(DailyPnL).where(DailyPnL.date == today)
        )
        daily_pnl = result.scalar_one_or_none()

        if not daily_pnl:
            daily_pnl = DailyPnL(
                date=today,
                starting_capital=starting_capital,
            )
            self.session.add(daily_pnl)
            await self.session.flush()
            await self.session.refresh(daily_pnl)

        return daily_pnl

    async def update_pnl(
        self,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        fees: Decimal,
        trades_count: int,
        winning_trades: int,
        losing_trades: int,
    ) -> DailyPnL:
        """Update today's P&L."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        result = await self.session.execute(
            select(DailyPnL).where(DailyPnL.date == today)
        )
        daily_pnl = result.scalar_one_or_none()

        if daily_pnl:
            daily_pnl.realized_pnl = realized_pnl
            daily_pnl.unrealized_pnl = unrealized_pnl
            daily_pnl.fees = fees
            daily_pnl.net_pnl = realized_pnl + unrealized_pnl - fees
            daily_pnl.trades_count = trades_count
            daily_pnl.winning_trades = winning_trades
            daily_pnl.losing_trades = losing_trades

            # Track max drawdown
            if daily_pnl.net_pnl < daily_pnl.max_drawdown:
                daily_pnl.max_drawdown = daily_pnl.net_pnl
            if daily_pnl.net_pnl > daily_pnl.max_runup:
                daily_pnl.max_runup = daily_pnl.net_pnl

            await self.session.flush()

        return daily_pnl

    async def trigger_circuit_breaker(self) -> DailyPnL:
        """Mark circuit breaker as triggered for today."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        result = await self.session.execute(
            select(DailyPnL).where(DailyPnL.date == today)
        )
        daily_pnl = result.scalar_one_or_none()

        if daily_pnl:
            daily_pnl.circuit_breaker_triggered = True
            daily_pnl.circuit_breaker_time = datetime.now()
            await self.session.flush()

        return daily_pnl

    async def get_weekly_pnl(self) -> List[DailyPnL]:
        """Get P&L for the last 7 days."""
        week_ago = datetime.now() - timedelta(days=7)
        result = await self.session.execute(
            select(DailyPnL)
            .where(DailyPnL.date >= week_ago)
            .order_by(DailyPnL.date.desc())
        )
        return list(result.scalars().all())

    async def get_monthly_pnl(self) -> List[DailyPnL]:
        """Get P&L for the last 30 days."""
        month_ago = datetime.now() - timedelta(days=30)
        result = await self.session.execute(
            select(DailyPnL)
            .where(DailyPnL.date >= month_ago)
            .order_by(DailyPnL.date.desc())
        )
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG REPOSITORY
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogRepository:
    """
    Repository for AuditLog operations.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(
        self,
        event_type: str,
        description: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        old_value: Optional[Dict] = None,
        new_value: Optional[Dict] = None,
        level: AlertLevel = AlertLevel.INFO,
        extra_data: Optional[Dict] = None,
    ) -> AuditLog:
        """Create an audit log entry."""
        entry = AuditLog(
            event_type=event_type,
            description=description,
            entity_type=entity_type,
            entity_id=entity_id,
            old_value=old_value,
            new_value=new_value,
            level=level,
            extra_data=extra_data,
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_recent(
        self,
        event_type: Optional[str] = None,
        hours: int = 24,
        level: Optional[AlertLevel] = None,
    ) -> List[AuditLog]:
        """Get recent audit log entries."""
        cutoff = datetime.now() - timedelta(hours=hours)
        query = select(AuditLog).where(AuditLog.timestamp >= cutoff)

        if event_type:
            query = query.where(AuditLog.event_type == event_type)
        if level:
            query = query.where(AuditLog.level == level)

        query = query.order_by(AuditLog.timestamp.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOLIDAY REPOSITORY
# ─────────────────────────────────────────────────────────────────────────────

class MarketHolidayRepository:
    """
    Repository for MarketHoliday operations.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def is_holiday(self, check_date: date, exchange: str = "NSE") -> bool:
        """Check if a date is a market holiday."""
        result = await self.session.execute(
            select(MarketHoliday).where(
                and_(
                    func.date(MarketHoliday.date) == check_date,
                    MarketHoliday.exchange == exchange
                )
            )
        )
        return result.scalar_one_or_none() is not None

    async def get_holidays(
        self,
        year: int,
        exchange: str = "NSE"
    ) -> List[MarketHoliday]:
        """Get all holidays for a year."""
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)

        result = await self.session.execute(
            select(MarketHoliday).where(
                and_(
                    MarketHoliday.date >= start,
                    MarketHoliday.date <= end,
                    MarketHoliday.exchange == exchange
                )
            ).order_by(MarketHoliday.date)
        )
        return list(result.scalars().all())

    async def add_holiday(
        self,
        date: datetime,
        exchange: str,
        name: str,
        is_full_day: bool = True,
        early_close_time: Optional[str] = None
    ) -> MarketHoliday:
        """Add a new holiday."""
        holiday = MarketHoliday(
            date=date,
            exchange=exchange,
            name=name,
            is_full_day=is_full_day,
            early_close_time=early_close_time
        )
        self.session.add(holiday)
        await self.session.flush()
        return holiday
