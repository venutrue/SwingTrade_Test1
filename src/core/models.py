"""
Database Models for Prop Trading System
========================================
SQLAlchemy models for persistent storage of orders, positions, and audit data.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, JSON,
    ForeignKey, Index, Enum as SQLEnum, Text, Numeric, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class OrderStatus(str, Enum):
    """Order lifecycle states."""
    PENDING = "PENDING"           # Created, not yet sent to broker
    SUBMITTED = "SUBMITTED"       # Sent to broker, awaiting acknowledgment
    OPEN = "OPEN"                 # Acknowledged by broker, in order book
    PARTIAL = "PARTIAL"           # Partially filled
    FILLED = "FILLED"             # Completely filled
    CANCELLED = "CANCELLED"       # Cancelled by user or system
    REJECTED = "REJECTED"         # Rejected by broker/exchange
    EXPIRED = "EXPIRED"           # Order expired (GTD/GTT)
    ERROR = "ERROR"               # System error


class PositionStatus(str, Enum):
    """Position lifecycle states."""
    PENDING_ENTRY = "PENDING_ENTRY"    # Order placed, awaiting fill
    OPEN = "OPEN"                       # Position is active
    PENDING_EXIT = "PENDING_EXIT"       # Exit order placed
    CLOSED = "CLOSED"                   # Position fully closed


class Side(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"


class Trend(str, Enum):
    """Market trend direction."""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    NEUTRAL = "NEUTRAL"


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


class ExitReason(str, Enum):
    """Reason for position exit."""
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"
    SIGNAL = "SIGNAL"
    EOD_SQUARE_OFF = "EOD_SQUARE_OFF"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    KILL_SWITCH = "KILL_SWITCH"
    ERROR = "ERROR"


class AlertLevel(str, Enum):
    """Notification alert levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# BASE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ORDER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Order(Base):
    """
    Complete order tracking with full lifecycle management.

    This is the source of truth for all order states - always reconcile
    with broker state on startup and periodically.
    """
    __tablename__ = "orders"

    # Primary identification
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(50), index=True, nullable=True)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Symbol and exchange
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")

    # Order details
    side: Mapped[str] = mapped_column(SQLEnum(Side))
    order_type: Mapped[str] = mapped_column(String(20))  # MARKET, LIMIT, SL, SL-M
    product_type: Mapped[str] = mapped_column(String(20))  # CNC, MIS, NRML
    variety: Mapped[str] = mapped_column(String(20), default="regular")  # regular, bo, co, amo

    # Quantities
    quantity: Mapped[int] = mapped_column(Integer)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    pending_quantity: Mapped[int] = mapped_column(Integer, default=0)
    cancelled_quantity: Mapped[int] = mapped_column(Integer, default=0)

    # Prices
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)  # Limit price
    trigger_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)  # SL trigger
    average_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)  # Fill price

    # Bracket order specifics
    stoploss_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)
    target_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)
    trailing_stoploss: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)

    # Status tracking
    status: Mapped[str] = mapped_column(SQLEnum(OrderStatus), default=OrderStatus.PENDING, index=True)
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    position_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("positions.id"), nullable=True)
    position: Mapped[Optional["Position"]] = relationship("Position", back_populates="orders")

    # Audit trail
    signal_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    strategy_name: Mapped[str] = mapped_column(String(100), default="rivalland_swing")

    # Metadata
    tags: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    broker_response: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_orders_symbol_status", "symbol", "status"),
        Index("ix_orders_created_at_status", "created_at", "status"),
    )

    def __repr__(self) -> str:
        return f"<Order {self.id[:8]} {self.symbol} {self.side.value} {self.quantity}@{self.price} [{self.status.value}]>"

    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be modified/cancelled)."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIAL)

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR)

    @property
    def fill_rate(self) -> float:
        """Get fill rate as percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, Side) else self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "price": float(self.price) if self.price else None,
            "average_price": float(self.average_price) if self.average_price else None,
            "status": self.status.value if isinstance(self.status, OrderStatus) else self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Position(Base):
    """
    Position tracking with P&L calculation and exit management.

    A position represents an open trade from entry to exit.
    """
    __tablename__ = "positions"

    # Primary identification
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Symbol and exchange
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")

    # Position details
    side: Mapped[str] = mapped_column(SQLEnum(Side))
    quantity: Mapped[int] = mapped_column(Integer)
    remaining_quantity: Mapped[int] = mapped_column(Integer)

    # Entry details
    entry_price: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    entry_value: Mapped[Decimal] = mapped_column(Numeric(20, 4))  # entry_price * quantity
    entry_time: Mapped[datetime] = mapped_column(DateTime, index=True)

    # Exit details
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)
    exit_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 4), nullable=True)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(SQLEnum(ExitReason), nullable=True)

    # Risk management levels
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    initial_stop_loss: Mapped[Decimal] = mapped_column(Numeric(15, 4))  # Original SL
    target: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    trailing_stop_active: Mapped[bool] = mapped_column(Boolean, default=False)
    highest_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)  # For trailing
    lowest_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4), nullable=True)

    # P&L tracking
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    fees: Mapped[Decimal] = mapped_column(Numeric(15, 4), default=0)
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)

    # Risk metrics
    risk_amount: Mapped[Decimal] = mapped_column(Numeric(15, 4))  # Entry - SL * qty
    reward_amount: Mapped[Decimal] = mapped_column(Numeric(15, 4))  # Target - Entry * qty
    risk_reward_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 4))
    return_percent: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=0)

    # Status
    status: Mapped[str] = mapped_column(SQLEnum(PositionStatus), default=PositionStatus.PENDING_ENTRY, index=True)

    # Signal reference
    signal_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    trend_at_entry: Mapped[Optional[str]] = mapped_column(SQLEnum(Trend), nullable=True)

    # Strategy
    strategy_name: Mapped[str] = mapped_column(String(100), default="rivalland_swing")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="position")

    # Metadata
    tags: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_positions_symbol_status", "symbol", "status"),
        Index("ix_positions_entry_time", "entry_time"),
    )

    def __repr__(self) -> str:
        return f"<Position {self.id[:8]} {self.symbol} {self.side.value} {self.quantity} [{self.status.value}]>"

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN, PositionStatus.PENDING_EXIT)

    @property
    def is_long(self) -> bool:
        return self.side == Side.BUY

    @property
    def is_short(self) -> bool:
        return self.side == Side.SELL

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L at current price."""
        if self.is_long:
            return (current_price - self.entry_price) * self.remaining_quantity
        else:
            return (self.entry_price - current_price) * self.remaining_quantity

    def calculate_return_percent(self, current_price: Decimal) -> Decimal:
        """Calculate return percentage."""
        if self.entry_price == 0:
            return Decimal(0)
        if self.is_long:
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100

    def should_trigger_stop_loss(self, current_price: Decimal) -> bool:
        """Check if stop loss should be triggered."""
        if self.is_long:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss

    def should_trigger_target(self, current_price: Decimal) -> bool:
        """Check if target should be triggered."""
        if self.is_long:
            return current_price >= self.target
        else:
            return current_price <= self.target

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, Side) else self.side,
            "quantity": self.quantity,
            "remaining_quantity": self.remaining_quantity,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "stop_loss": float(self.stop_loss),
            "target": float(self.target),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "status": self.status.value if isinstance(self.status, PositionStatus) else self.status,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Signal(Base):
    """
    Trading signal generated by the strategy engine.

    Signals are recorded even if not executed for analysis.
    """
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Symbol
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")

    # Signal details
    signal_type: Mapped[str] = mapped_column(SQLEnum(SignalType))
    trend: Mapped[str] = mapped_column(SQLEnum(Trend))

    # Price levels
    price: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    swing_high: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    swing_low: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(15, 4))
    target: Mapped[Decimal] = mapped_column(Numeric(15, 4))

    # Signal quality
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=0)
    strength: Mapped[int] = mapped_column(Integer, default=1)  # 1-5 scale

    # Context
    pullback_bars: Mapped[int] = mapped_column(Integer, default=0)
    rally_bars: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str] = mapped_column(Text)

    # Execution status
    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    position_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Strategy
    strategy_name: Mapped[str] = mapped_column(String(100), default="rivalland_swing")
    timeframe: Mapped[str] = mapped_column(String(20), default="day")

    # Metadata
    market_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_signals_symbol_timestamp", "symbol", "timestamp"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAILY P&L MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DailyPnL(Base):
    """
    Daily P&L tracking for risk management circuit breakers.
    """
    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, unique=True, index=True)

    # Starting capital for the day
    starting_capital: Mapped[Decimal] = mapped_column(Numeric(20, 4))

    # P&L components
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    fees: Mapped[Decimal] = mapped_column(Numeric(15, 4), default=0)
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)

    # Trade statistics
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)

    # Risk metrics
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    max_runup: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)

    # Circuit breaker status
    circuit_breaker_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    circuit_breaker_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Comprehensive audit log for compliance and debugging.
    """
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Event identification
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    event_subtype: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Entity reference
    entity_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # order, position, signal
    entity_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Event details
    description: Mapped[str] = mapped_column(Text)
    old_value: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    new_value: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Context
    user_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Alert level
    level: Mapped[str] = mapped_column(SQLEnum(AlertLevel), default=AlertLevel.INFO)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)

    # Extra data
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_audit_event_timestamp", "event_type", "timestamp"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOLIDAY MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MarketHoliday(Base):
    """
    Market holiday calendar for NSE/BSE.
    """
    __tablename__ = "market_holidays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    exchange: Mapped[str] = mapped_column(String(10))
    name: Mapped[str] = mapped_column(String(100))
    is_full_day: Mapped[bool] = mapped_column(Boolean, default=True)
    early_close_time: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    __table_args__ = (
        UniqueConstraint("date", "exchange", name="uq_holiday_date_exchange"),
    )
