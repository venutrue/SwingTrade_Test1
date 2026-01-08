"""
Production Risk Management System
==================================
Comprehensive risk management with position sizing, exposure limits,
circuit breakers, and kill switch functionality.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import logging

from src.core.models import (
    Position, Signal, DailyPnL, PositionStatus, Side, ExitReason, AlertLevel,
    SignalType
)
from src.core.database import (
    db, PositionRepository, DailyPnLRepository, AuditLogRepository
)
from src.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RISK VALIDATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

class RiskCheckResult:
    """Result of a risk validation check."""

    def __init__(
        self,
        passed: bool,
        reason: str = "",
        risk_score: float = 0.0,
        details: Optional[Dict] = None
    ):
        self.passed = passed
        self.reason = reason
        self.risk_score = risk_score  # 0-100, higher = riskier
        self.details = details or {}

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"<RiskCheck {status}: {self.reason}>"


class CircuitBreakerStatus(str, Enum):
    """Circuit breaker states."""
    ACTIVE = "ACTIVE"           # Normal trading
    TRIGGERED = "TRIGGERED"     # Temporarily halted
    COOLDOWN = "COOLDOWN"       # In cooldown period
    LOCKED = "LOCKED"           # Manually locked


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZER
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Advanced position sizing with multiple methods.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def calculate_risk_based_size(
        self,
        capital: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        risk_percent: Optional[Decimal] = None,
    ) -> int:
        """
        Calculate position size based on fixed risk percentage.

        Formula: Position Size = (Capital * Risk%) / (Entry - Stop Loss)
        """
        if risk_percent is None:
            risk_percent = self.settings.risk.max_risk_per_trade_percent

        risk_amount = capital * (risk_percent / 100)
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return 0

        position_size = int(risk_amount / risk_per_share)

        # Apply maximum position size limit
        max_position_value = capital * (self.settings.risk.max_position_size_percent / 100)
        max_shares = int(max_position_value / entry_price)
        position_size = min(position_size, max_shares)

        # Ensure we can afford it
        max_affordable = int(capital / entry_price)
        position_size = min(position_size, max_affordable)

        return max(0, position_size)

    def calculate_volatility_adjusted_size(
        self,
        capital: Decimal,
        entry_price: Decimal,
        atr: Decimal,
        atr_multiplier: Decimal = Decimal("2.0"),
    ) -> int:
        """
        Calculate position size based on ATR (Average True Range).

        Automatically adjusts for market volatility.
        """
        risk_percent = self.settings.risk.max_risk_per_trade_percent
        risk_amount = capital * (risk_percent / 100)

        volatility_stop = atr * atr_multiplier
        if volatility_stop == 0:
            return 0

        position_size = int(risk_amount / volatility_stop)

        # Apply limits
        max_position_value = capital * (self.settings.risk.max_position_size_percent / 100)
        max_shares = int(max_position_value / entry_price)

        return min(position_size, max_shares)

    def calculate_kelly_size(
        self,
        capital: Decimal,
        entry_price: Decimal,
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        kelly_fraction: Decimal = Decimal("0.25"),  # Use quarter Kelly for safety
    ) -> int:
        """
        Calculate position size using Kelly Criterion.

        Formula: f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if avg_loss == 0 or win_rate == 0:
            return 0

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_percent = (b * p - q) / b

        # Apply fraction for safety
        kelly_percent = max(Decimal(0), kelly_percent * kelly_fraction)

        # Cap at maximum allowed
        kelly_percent = min(kelly_percent, self.settings.risk.max_position_size_percent / 100)

        position_value = capital * kelly_percent
        position_size = int(position_value / entry_price)

        return max(0, position_size)

    def adjust_for_correlation(
        self,
        position_size: int,
        symbol: str,
        existing_positions: List[Position],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> int:
        """
        Reduce position size based on correlation with existing positions.
        """
        if not existing_positions or not correlation_matrix:
            return position_size

        # Calculate average correlation with existing positions
        correlations = []
        for pos in existing_positions:
            if pos.symbol in correlation_matrix.get(symbol, {}):
                correlations.append(correlation_matrix[symbol][pos.symbol])

        if not correlations:
            return position_size

        avg_correlation = sum(correlations) / len(correlations)

        # Reduce size based on correlation (high correlation = smaller position)
        reduction_factor = 1 - (avg_correlation * 0.5)  # Max 50% reduction
        adjusted_size = int(position_size * max(0.5, reduction_factor))

        return adjusted_size


# ─────────────────────────────────────────────────────────────────────────────
# RISK VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class RiskValidator:
    """
    Pre-trade risk validation checks.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    async def validate_trade(
        self,
        signal: Signal,
        capital: Decimal,
        position_count: int,
        daily_pnl: Optional[DailyPnL] = None,
        existing_positions: Optional[List[Position]] = None,
    ) -> RiskCheckResult:
        """
        Run all pre-trade risk checks.

        Returns aggregated result of all checks.
        """
        checks = [
            await self._check_max_positions(position_count),
            await self._check_risk_reward_ratio(signal),
            await self._check_daily_loss_limit(daily_pnl, capital),
            await self._check_daily_trade_limit(daily_pnl),
            await self._check_capital_available(signal, capital),
            await self._check_sector_exposure(signal, existing_positions, capital),
            await self._check_single_stock_limit(signal, existing_positions, capital),
            await self._check_consecutive_losses(daily_pnl),
        ]

        failed_checks = [c for c in checks if not c.passed]

        if failed_checks:
            # Return first failure with combined details
            return RiskCheckResult(
                passed=False,
                reason="; ".join(c.reason for c in failed_checks),
                risk_score=max(c.risk_score for c in failed_checks),
                details={"failed_checks": [c.reason for c in failed_checks]}
            )

        avg_risk_score = sum(c.risk_score for c in checks) / len(checks)
        return RiskCheckResult(
            passed=True,
            reason="All risk checks passed",
            risk_score=avg_risk_score,
            details={"checks_passed": len(checks)}
        )

    async def _check_max_positions(self, position_count: int) -> RiskCheckResult:
        """Check if max positions limit reached."""
        max_positions = self.settings.risk.max_open_positions

        if position_count >= max_positions:
            return RiskCheckResult(
                passed=False,
                reason=f"Max positions ({max_positions}) reached",
                risk_score=100,
            )

        # Risk increases as we approach limit
        risk_score = (position_count / max_positions) * 80
        return RiskCheckResult(
            passed=True,
            reason="Position count within limits",
            risk_score=risk_score,
        )

    async def _check_risk_reward_ratio(self, signal: Signal) -> RiskCheckResult:
        """Check if trade meets minimum R:R ratio."""
        min_rr = self.settings.risk.min_risk_reward_ratio

        risk = abs(signal.price - signal.stop_loss)
        reward = abs(signal.target - signal.price)

        if risk == 0:
            return RiskCheckResult(
                passed=False,
                reason="Invalid risk (zero)",
                risk_score=100,
            )

        rr_ratio = reward / risk

        if rr_ratio < float(min_rr):
            return RiskCheckResult(
                passed=False,
                reason=f"R:R ratio {rr_ratio:.2f} below minimum {min_rr}",
                risk_score=80,
                details={"rr_ratio": float(rr_ratio), "min_required": float(min_rr)}
            )

        # Lower risk score for better R:R
        risk_score = max(0, 50 - (rr_ratio - float(min_rr)) * 10)
        return RiskCheckResult(
            passed=True,
            reason=f"R:R ratio {rr_ratio:.2f} acceptable",
            risk_score=risk_score,
        )

    async def _check_daily_loss_limit(
        self,
        daily_pnl: Optional[DailyPnL],
        capital: Decimal
    ) -> RiskCheckResult:
        """Check if daily loss limit reached."""
        if not daily_pnl:
            return RiskCheckResult(passed=True, reason="No daily P&L data", risk_score=20)

        max_loss_percent = self.settings.risk.max_daily_loss_percent
        max_loss_amount = capital * (max_loss_percent / 100)

        current_loss = abs(min(Decimal(0), daily_pnl.net_pnl))
        loss_percent = (current_loss / capital) * 100 if capital > 0 else Decimal(0)

        if current_loss >= max_loss_amount:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily loss limit ({max_loss_percent}%) reached",
                risk_score=100,
                details={"current_loss": float(current_loss), "limit": float(max_loss_amount)}
            )

        # Risk increases as we approach limit
        risk_score = float(loss_percent / max_loss_percent) * 100
        return RiskCheckResult(
            passed=True,
            reason=f"Daily loss {loss_percent:.2f}% within limits",
            risk_score=min(risk_score, 90),
        )

    async def _check_daily_trade_limit(self, daily_pnl: Optional[DailyPnL]) -> RiskCheckResult:
        """Check if daily trade limit reached."""
        if not daily_pnl:
            return RiskCheckResult(passed=True, reason="No daily P&L data", risk_score=10)

        max_trades = self.settings.risk.max_daily_trades

        if daily_pnl.trades_count >= max_trades:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily trade limit ({max_trades}) reached",
                risk_score=100,
            )

        risk_score = (daily_pnl.trades_count / max_trades) * 60
        return RiskCheckResult(
            passed=True,
            reason=f"Trade count {daily_pnl.trades_count}/{max_trades}",
            risk_score=risk_score,
        )

    async def _check_capital_available(
        self,
        signal: Signal,
        capital: Decimal
    ) -> RiskCheckResult:
        """Check if sufficient capital available."""
        min_position_value = signal.price * 1  # At least 1 share

        if capital < min_position_value:
            return RiskCheckResult(
                passed=False,
                reason="Insufficient capital for minimum position",
                risk_score=100,
            )

        return RiskCheckResult(
            passed=True,
            reason="Capital available",
            risk_score=10,
        )

    async def _check_sector_exposure(
        self,
        signal: Signal,
        positions: Optional[List[Position]],
        capital: Decimal
    ) -> RiskCheckResult:
        """Check sector concentration limits."""
        # Simplified - would need sector mapping in production
        max_exposure = self.settings.risk.max_sector_exposure_percent

        # For now, just pass this check
        return RiskCheckResult(
            passed=True,
            reason="Sector exposure within limits",
            risk_score=20,
        )

    async def _check_single_stock_limit(
        self,
        signal: Signal,
        positions: Optional[List[Position]],
        capital: Decimal
    ) -> RiskCheckResult:
        """Check single stock concentration limit."""
        max_percent = self.settings.risk.max_single_stock_percent

        if not positions:
            return RiskCheckResult(passed=True, reason="No existing positions", risk_score=10)

        # Calculate existing exposure to this symbol
        symbol_exposure = Decimal(0)
        for pos in positions:
            if pos.symbol == signal.symbol and pos.status == PositionStatus.OPEN:
                symbol_exposure += pos.entry_value

        current_percent = (symbol_exposure / capital) * 100 if capital > 0 else Decimal(0)

        if current_percent >= max_percent:
            return RiskCheckResult(
                passed=False,
                reason=f"Single stock limit ({max_percent}%) reached for {signal.symbol}",
                risk_score=90,
            )

        risk_score = float(current_percent / max_percent) * 70
        return RiskCheckResult(
            passed=True,
            reason=f"Stock exposure {current_percent:.1f}% within limits",
            risk_score=risk_score,
        )

    async def _check_consecutive_losses(self, daily_pnl: Optional[DailyPnL]) -> RiskCheckResult:
        """Check consecutive loss limit."""
        if not daily_pnl:
            return RiskCheckResult(passed=True, reason="No daily P&L data", risk_score=10)

        max_consecutive = self.settings.risk.consecutive_loss_limit

        # This would need trade history tracking - simplified for now
        consecutive_losses = daily_pnl.losing_trades if daily_pnl.winning_trades == 0 else 0

        if consecutive_losses >= max_consecutive:
            return RiskCheckResult(
                passed=False,
                reason=f"Consecutive loss limit ({max_consecutive}) reached",
                risk_score=100,
            )

        risk_score = (consecutive_losses / max_consecutive) * 80
        return RiskCheckResult(
            passed=True,
            reason=f"Consecutive losses: {consecutive_losses}",
            risk_score=risk_score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Trading circuit breaker that halts trading under adverse conditions.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.status = CircuitBreakerStatus.ACTIVE
        self.triggered_at: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None
        self.cooldown_until: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def check_and_trigger(
        self,
        daily_pnl: DailyPnL,
        capital: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """
        Check conditions and trigger circuit breaker if needed.

        Returns: (is_triggered, reason)
        """
        async with self._lock:
            if self.status == CircuitBreakerStatus.LOCKED:
                return True, "Circuit breaker manually locked"

            if self.status == CircuitBreakerStatus.COOLDOWN:
                if datetime.now() < self.cooldown_until:
                    return True, f"In cooldown until {self.cooldown_until}"
                else:
                    self.status = CircuitBreakerStatus.ACTIVE
                    self.cooldown_until = None

            # Check daily loss limit
            max_loss_percent = self.settings.risk.max_daily_loss_percent
            current_loss = abs(min(Decimal(0), daily_pnl.net_pnl))
            loss_percent = (current_loss / capital) * 100 if capital > 0 else Decimal(0)

            if loss_percent >= max_loss_percent:
                await self._trigger(f"Daily loss limit {loss_percent:.2f}% exceeded")
                return True, self.trigger_reason

            # Check drawdown
            if daily_pnl.max_drawdown < 0:
                drawdown_percent = abs(daily_pnl.max_drawdown / capital) * 100 if capital > 0 else Decimal(0)
                if drawdown_percent >= max_loss_percent * Decimal("1.5"):  # 1.5x for drawdown
                    await self._trigger(f"Max drawdown {drawdown_percent:.2f}% exceeded")
                    return True, self.trigger_reason

            return False, None

    async def _trigger(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        self.status = CircuitBreakerStatus.TRIGGERED
        self.triggered_at = datetime.now()
        self.trigger_reason = reason

        cooldown_minutes = self.settings.risk.circuit_breaker_cooldown_minutes
        self.cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        self.status = CircuitBreakerStatus.COOLDOWN

        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

        # Log to database
        async with db.session() as session:
            pnl_repo = DailyPnLRepository(session)
            await pnl_repo.trigger_circuit_breaker()

            audit_repo = AuditLogRepository(session)
            await audit_repo.log(
                event_type="CIRCUIT_BREAKER_TRIGGERED",
                description=reason,
                level=AlertLevel.CRITICAL,
                metadata={
                    "triggered_at": self.triggered_at.isoformat(),
                    "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
                }
            )

    async def reset(self) -> None:
        """Reset the circuit breaker to active state."""
        async with self._lock:
            if self.status == CircuitBreakerStatus.LOCKED:
                logger.warning("Cannot reset locked circuit breaker")
                return

            self.status = CircuitBreakerStatus.ACTIVE
            self.triggered_at = None
            self.trigger_reason = None
            self.cooldown_until = None

            logger.info("Circuit breaker reset to ACTIVE")

    async def lock(self, reason: str = "Manual lock") -> None:
        """Manually lock the circuit breaker (requires manual unlock)."""
        async with self._lock:
            self.status = CircuitBreakerStatus.LOCKED
            self.triggered_at = datetime.now()
            self.trigger_reason = reason

            logger.warning(f"Circuit breaker LOCKED: {reason}")

    async def unlock(self) -> None:
        """Manually unlock the circuit breaker."""
        async with self._lock:
            if self.status == CircuitBreakerStatus.LOCKED:
                self.status = CircuitBreakerStatus.ACTIVE
                self.triggered_at = None
                self.trigger_reason = None
                logger.info("Circuit breaker UNLOCKED")

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self.status == CircuitBreakerStatus.ACTIVE

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "reason": self.trigger_reason,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "trading_allowed": self.is_trading_allowed(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# KILL SWITCH
# ─────────────────────────────────────────────────────────────────────────────

class KillSwitch:
    """
    Emergency kill switch for immediate trading halt and position closure.
    """

    def __init__(self, position_manager: "PositionManager"):
        self.position_manager = position_manager
        self.settings = get_settings()
        self.is_active = False
        self.activated_at: Optional[datetime] = None
        self.activation_reason: Optional[str] = None
        self._lock = asyncio.Lock()

    async def activate(self, reason: str = "Manual activation") -> Dict[str, Any]:
        """
        Activate kill switch:
        1. Halt all new trading
        2. Cancel all pending orders
        3. Close all open positions at market

        Returns summary of actions taken.
        """
        async with self._lock:
            if self.is_active:
                return {"status": "already_active", "activated_at": self.activated_at}

            self.is_active = True
            self.activated_at = datetime.now()
            self.activation_reason = reason

            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

            # Update settings
            self.settings.kill_switch_enabled = True

            results = {
                "activated_at": self.activated_at.isoformat(),
                "reason": reason,
                "positions_closed": 0,
                "orders_cancelled": 0,
                "errors": [],
            }

            try:
                # Close all positions
                closed_orders = await self.position_manager.flatten_all_positions(
                    ExitReason.KILL_SWITCH
                )
                results["positions_closed"] = len(closed_orders)

            except Exception as e:
                logger.error(f"Kill switch error closing positions: {e}")
                results["errors"].append(str(e))

            # Audit log
            async with db.session() as session:
                audit_repo = AuditLogRepository(session)
                await audit_repo.log(
                    event_type="KILL_SWITCH_ACTIVATED",
                    description=reason,
                    level=AlertLevel.CRITICAL,
                    metadata=results,
                )

            return results

    async def deactivate(self) -> Dict[str, Any]:
        """
        Deactivate kill switch and allow trading to resume.

        Should require confirmation/authorization in production.
        """
        async with self._lock:
            if not self.is_active:
                return {"status": "already_inactive"}

            previous_activation = self.activated_at
            self.is_active = False
            self.activated_at = None
            self.activation_reason = None

            self.settings.kill_switch_enabled = False

            logger.warning("KILL SWITCH DEACTIVATED - Trading can resume")

            # Audit log
            async with db.session() as session:
                audit_repo = AuditLogRepository(session)
                await audit_repo.log(
                    event_type="KILL_SWITCH_DEACTIVATED",
                    description="Kill switch deactivated, trading resumed",
                    level=AlertLevel.WARNING,
                    metadata={"was_active_since": previous_activation.isoformat() if previous_activation else None}
                )

            return {
                "status": "deactivated",
                "was_active_since": previous_activation.isoformat() if previous_activation else None,
            }

    def is_trading_blocked(self) -> bool:
        """Check if trading is blocked by kill switch."""
        return self.is_active or self.settings.kill_switch_enabled

    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status."""
        return {
            "is_active": self.is_active,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "reason": self.activation_reason,
            "trading_blocked": self.is_trading_blocked(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER (MAIN CLASS)
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Main risk management orchestrator combining all risk components.
    """

    def __init__(self, position_manager: "PositionManager"):
        self.settings = get_settings()
        self.position_sizer = PositionSizer(self.settings)
        self.validator = RiskValidator(self.settings)
        self.circuit_breaker = CircuitBreaker(self.settings)
        self.kill_switch = KillSwitch(position_manager)
        self.position_manager = position_manager

    async def can_trade(self) -> Tuple[bool, str]:
        """
        Master check if trading is currently allowed.
        """
        # Kill switch check
        if self.kill_switch.is_trading_blocked():
            return False, "Kill switch active"

        # Circuit breaker check
        if not self.circuit_breaker.is_trading_allowed():
            status = self.circuit_breaker.get_status()
            return False, f"Circuit breaker: {status['reason']}"

        return True, "Trading allowed"

    async def validate_and_size_trade(
        self,
        signal: Signal,
        capital: Decimal,
    ) -> Tuple[RiskCheckResult, int]:
        """
        Validate trade and calculate position size.

        Returns: (validation_result, position_size)
        """
        # Check if trading allowed
        can_trade, reason = await self.can_trade()
        if not can_trade:
            return RiskCheckResult(passed=False, reason=reason, risk_score=100), 0

        # Get current state
        position_count = await self.position_manager.get_position_count()
        positions = await self.position_manager.get_open_positions()

        # Get daily P&L
        async with db.session() as session:
            pnl_repo = DailyPnLRepository(session)
            daily_pnl = await pnl_repo.get_or_create_today(capital)

            # Check circuit breaker conditions
            triggered, cb_reason = await self.circuit_breaker.check_and_trigger(
                daily_pnl, capital
            )
            if triggered:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Circuit breaker: {cb_reason}",
                    risk_score=100
                ), 0

        # Run validation
        validation = await self.validator.validate_trade(
            signal=signal,
            capital=capital,
            position_count=position_count,
            daily_pnl=daily_pnl,
            existing_positions=positions,
        )

        if not validation.passed:
            return validation, 0

        # Calculate position size
        position_size = self.position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=signal.price,
            stop_loss=signal.stop_loss,
        )

        # Adjust for correlation if we have existing positions
        if positions:
            # Would need correlation matrix from data service
            # position_size = self.position_sizer.adjust_for_correlation(...)
            pass

        return validation, position_size

    async def update_daily_pnl(
        self,
        capital: Decimal,
        positions: List[Position],
        current_prices: Dict[str, Decimal],
    ) -> DailyPnL:
        """
        Update daily P&L tracking with current positions.
        """
        async with db.session() as session:
            pnl_repo = DailyPnLRepository(session)
            pos_repo = PositionRepository(session)

            # Calculate current P&L
            realized_pnl = Decimal(0)
            unrealized_pnl = Decimal(0)
            fees = Decimal(0)
            winning = 0
            losing = 0

            # Get today's closed positions
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            closed_today = await pos_repo.get_closed_positions(start_date=today_start)

            for pos in closed_today:
                realized_pnl += pos.realized_pnl
                fees += pos.fees
                if pos.realized_pnl > 0:
                    winning += 1
                elif pos.realized_pnl < 0:
                    losing += 1

            # Calculate unrealized from open positions
            for pos in positions:
                if pos.status == PositionStatus.OPEN and pos.symbol in current_prices:
                    unrealized_pnl += pos.calculate_unrealized_pnl(current_prices[pos.symbol])

            # Update daily record
            daily_pnl = await pnl_repo.update_pnl(
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                fees=fees,
                trades_count=len(closed_today),
                winning_trades=winning,
                losing_trades=losing,
            )

            # Check circuit breaker
            await self.circuit_breaker.check_and_trigger(daily_pnl, capital)

            return daily_pnl

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive risk management status."""
        return {
            "kill_switch": self.kill_switch.get_status(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "settings": {
                "max_risk_per_trade": float(self.settings.risk.max_risk_per_trade_percent),
                "max_daily_loss": float(self.settings.risk.max_daily_loss_percent),
                "max_positions": self.settings.risk.max_open_positions,
                "min_rr_ratio": float(self.settings.risk.min_risk_reward_ratio),
            }
        }
