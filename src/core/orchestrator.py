"""
Trading Orchestrator
====================
Main async trading engine that coordinates all components.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import logging
import signal

from src.config.settings import get_settings, Settings, TradingMode
from src.core.database import db
from src.core.engine import RivallandSwingEngine
from src.core.models import Signal, SignalType, PositionStatus, ExitReason
from src.broker.base import BaseBroker, ConnectionState
from src.broker.kite import KiteConnectBroker, PaperTradingBroker
from src.broker.order_manager import OrderManager, PositionManager
from src.risk.risk_manager import RiskManager
from src.utils.market_hours import MarketCalendar, MarketSessionScheduler
from src.monitoring.notifications import NotificationManager, NotificationType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRADING BOT ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class TradingOrchestrator:
    """
    Main trading orchestrator that coordinates:
    - Broker connection and data fetching
    - Signal generation via swing engine
    - Order and position management
    - Risk management and circuit breakers
    - Notifications and monitoring

    This is the central brain of the prop trading system.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Core components
        self.broker: Optional[BaseBroker] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.engine = RivallandSwingEngine()
        self.calendar = MarketCalendar()
        self.scheduler = MarketSessionScheduler(self.calendar)
        self.notifications = NotificationManager()

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._analysis_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None

        # Caches
        self._last_analysis_time: Dict[str, datetime] = {}
        self._current_prices: Dict[str, Decimal] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """
        Initialize all components.

        Returns True if initialization successful.
        """
        logger.info("Initializing Trading Orchestrator...")

        try:
            # Initialize database
            await db.initialize()
            logger.info("Database initialized")

            # Initialize market calendar
            await self.calendar.initialize()
            logger.info("Market calendar initialized")

            # Initialize broker based on trading mode
            if self.settings.trading_mode == TradingMode.PAPER:
                balance_data = await self._get_initial_capital()
                self.broker = PaperTradingBroker(
                    initial_capital=balance_data.get("available_cash", 100000)
                )
                logger.info("Paper trading broker initialized")
            else:
                self.broker = KiteConnectBroker()
                logger.info("Kite Connect broker initialized")

            # Connect to broker
            if not await self.broker.connect():
                logger.error("Failed to connect to broker")
                return False

            # Initialize order and position managers
            self.order_manager = OrderManager(self.broker)
            self.position_manager = PositionManager(self.order_manager)

            # Initialize risk manager
            self.risk_manager = RiskManager(self.position_manager)

            # Set up event handlers
            self._setup_event_handlers()

            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()

            logger.info("Trading Orchestrator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def _get_initial_capital(self) -> Dict[str, float]:
        """Get initial capital (from config or broker)."""
        # For paper trading, use config value
        return {"available_cash": 1000000.0}

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for order/position events."""
        # Order events
        from src.broker.order_manager import OrderEvent

        self.order_manager.on_event(
            OrderEvent.FILLED,
            self._on_order_filled
        )
        self.order_manager.on_event(
            OrderEvent.REJECTED,
            self._on_order_rejected
        )

        # Broker connection events
        self.broker.on_connected(self._on_broker_connected)
        self.broker.on_disconnected(self._on_broker_disconnected)

        # Market session events
        self.scheduler.on_event("trading_start", self._on_trading_start)
        self.scheduler.on_event("trading_end", self._on_trading_end)

    def _setup_signal_handlers(self) -> None:
        """Set up OS signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.shutdown())
                )
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_order_filled(self, order, data) -> None:
        """Handle order fill event."""
        logger.info(f"Order filled: {order.id}")
        await self.notifications.notify_order_filled(order)

    async def _on_order_rejected(self, order, data) -> None:
        """Handle order rejection."""
        logger.warning(f"Order rejected: {order.id} - {order.status_message}")
        await self.notifications.notify_error(
            f"Order rejected: {order.status_message}",
            {"order_id": order.id, "symbol": order.symbol}
        )

    async def _on_broker_connected(self) -> None:
        """Handle broker connection."""
        logger.info("Broker connected")
        await self.notifications.notify(
            NotificationType.CONNECTION_STATUS,
            "Broker connected successfully"
        )

    async def _on_broker_disconnected(self) -> None:
        """Handle broker disconnection."""
        logger.warning("Broker disconnected")
        await self.notifications.notify(
            NotificationType.CONNECTION_STATUS,
            "Broker disconnected - attempting reconnection",
            level="WARNING"
        )

    async def _on_trading_start(self) -> None:
        """Handle trading session start."""
        logger.info("Trading session started")
        await self.notifications.notify(
            NotificationType.CONNECTION_STATUS,
            "Trading session started"
        )

    async def _on_trading_end(self) -> None:
        """Handle trading session end."""
        logger.info("Trading session ended")

        # Send daily summary
        await self._send_daily_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRADING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main trading loop.

        Runs until shutdown signal received.
        """
        if not await self.initialize():
            logger.error("Failed to initialize - exiting")
            return

        self._running = True
        logger.info("Starting trading loop...")

        try:
            # Start background tasks
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Start session scheduler
            await self.scheduler.start()

            # Start order reconciliation
            await self.order_manager.start_reconciliation_loop()

            # Wait for shutdown
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await self.notifications.notify_error(f"Trading loop error: {e}")

        finally:
            await self.shutdown()

    async def _analysis_loop(self) -> None:
        """
        Background task for periodic analysis.
        """
        analysis_interval = 60  # Seconds between analysis cycles

        while self._running:
            try:
                # Check if we can trade
                can_trade, reason = await self.calendar.can_trade_now()

                if not can_trade:
                    logger.debug(f"Not trading: {reason}")
                    await asyncio.sleep(60)
                    continue

                # Check risk manager
                trade_allowed, risk_reason = await self.risk_manager.can_trade()
                if not trade_allowed:
                    logger.warning(f"Trading blocked: {risk_reason}")
                    await asyncio.sleep(60)
                    continue

                # Analyze each symbol
                for symbol in self.settings.symbols:
                    await self._analyze_symbol(symbol)

                await asyncio.sleep(analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(60)

    async def _analyze_symbol(self, symbol: str) -> None:
        """
        Analyze a single symbol and execute signals.
        """
        try:
            # Fetch latest data
            df = await self.broker.get_historical_data(
                symbol=symbol,
                interval=self.settings.timeframe,
                from_date=datetime.now() - timedelta(days=self.settings.data.default_lookback_days),
                to_date=datetime.now(),
            )

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return

            # Reset engine and run analysis
            self.engine.reset()
            signal = self.engine.analyze(df, symbol, self.settings.timeframe)

            # Update price cache
            self._current_prices[symbol] = signal.price

            # Execute signal if actionable
            if signal.signal_type != SignalType.HOLD:
                await self._execute_signal(signal)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    async def _execute_signal(self, signal: Signal) -> None:
        """
        Execute a trading signal after risk validation.
        """
        logger.info(f"Processing signal: {signal.signal_type.value} {signal.symbol}")

        # Notify about signal
        await self.notifications.notify_signal(signal)

        # Get current capital
        balance = await self.broker.get_balance()
        capital = Decimal(str(balance.get("available_cash", 0)))

        if capital <= 0:
            logger.warning("No capital available")
            return

        # Validate and size trade
        validation, position_size = await self.risk_manager.validate_and_size_trade(
            signal=signal,
            capital=capital,
        )

        if not validation.passed:
            logger.info(f"Trade rejected: {validation.reason}")
            return

        if position_size < 1:
            logger.info("Position size too small")
            return

        # Open position
        try:
            from src.core.models import Side, Trend

            side = Side.BUY if signal.signal_type == SignalType.BUY else Side.SELL

            position = await self.position_manager.open_position(
                symbol=signal.symbol,
                side=side,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                target=signal.target,
                quantity=position_size,
                signal_id=signal.id if hasattr(signal, 'id') else None,
                trend=signal.trend.value if hasattr(signal.trend, 'value') else str(signal.trend),
            )

            logger.info(f"Position opened: {position.id}")

            await self.notifications.notify(
                NotificationType.POSITION_OPENED,
                f"Position opened: {signal.symbol}",
                {
                    "Side": side.value,
                    "Quantity": position_size,
                    "Entry": f"₹{signal.price:,.2f}",
                    "Stop Loss": f"₹{signal.stop_loss:,.2f}",
                    "Target": f"₹{signal.target:,.2f}",
                }
            )

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            await self.notifications.notify_error(f"Position open failed: {e}")

    async def _monitoring_loop(self) -> None:
        """
        Background task for position monitoring.
        """
        monitor_interval = 30  # Seconds

        while self._running:
            try:
                # Get open positions
                positions = await self.position_manager.get_open_positions()

                for position in positions:
                    if position.status != PositionStatus.OPEN:
                        continue

                    # Get current price
                    current_price = self._current_prices.get(position.symbol)
                    if current_price is None:
                        quotes = await self.broker.get_quote([position.symbol])
                        if quotes:
                            key = f"NSE:{position.symbol}"
                            if key in quotes:
                                current_price = Decimal(str(quotes[key].get("last_price", 0)))

                    if current_price:
                        # Check stop loss / target
                        trigger = await self.position_manager.check_stop_loss_target(
                            position.id, current_price
                        )

                        if trigger == "stop_loss":
                            await self.position_manager.close_position(
                                position.id, ExitReason.STOP_LOSS_HIT
                            )
                            await self.notifications.notify_position_closed(
                                position, "STOP_LOSS_HIT"
                            )

                        elif trigger == "target":
                            await self.position_manager.close_position(
                                position.id, ExitReason.TARGET_HIT
                            )
                            await self.notifications.notify_position_closed(
                                position, "TARGET_HIT"
                            )

                        else:
                            # Update trailing stop
                            if self.settings.strategy.enable_trailing_stop:
                                await self.position_manager.update_trailing_stop(
                                    position.id, current_price
                                )

                # Update daily P&L
                balance = await self.broker.get_balance()
                capital = Decimal(str(balance.get("available_cash", 0)))
                await self.risk_manager.update_daily_pnl(
                    capital=capital,
                    positions=positions,
                    current_prices=self._current_prices,
                )

                await asyncio.sleep(monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(monitor_interval)

    # ─────────────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Graceful shutdown of all components.
        """
        if not self._running:
            return

        logger.info("Initiating shutdown...")
        self._running = False

        # Cancel background tasks
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop scheduler
        await self.scheduler.stop()

        # Stop reconciliation
        if self.order_manager:
            await self.order_manager.stop_reconciliation_loop()

        # Disconnect broker
        if self.broker:
            await self.broker.disconnect()

        # Close database
        await db.close()

        # Signal shutdown complete
        self._shutdown_event.set()

        logger.info("Shutdown complete")

    async def _send_daily_summary(self) -> None:
        """Send end-of-day summary."""
        try:
            from src.core.database import DailyPnLRepository

            async with db.session() as session:
                pnl_repo = DailyPnLRepository(session)
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

                daily_pnl = await pnl_repo.get_or_create_today(Decimal("0"))

                positions = await self.position_manager.get_open_positions()

                await self.notifications.send_daily_summary(
                    pnl=float(daily_pnl.net_pnl),
                    trades=daily_pnl.trades_count,
                    win_rate=float(daily_pnl.winning_trades / daily_pnl.trades_count * 100)
                        if daily_pnl.trades_count > 0 else 0,
                    positions=len(positions),
                )

        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # MANUAL CONTROLS
    # ─────────────────────────────────────────────────────────────────────────

    async def activate_kill_switch(self, reason: str = "Manual activation") -> Dict:
        """Activate kill switch."""
        return await self.risk_manager.kill_switch.activate(reason)

    async def deactivate_kill_switch(self) -> Dict:
        """Deactivate kill switch."""
        return await self.risk_manager.kill_switch.deactivate()

    async def flatten_all(self) -> List:
        """Close all positions immediately."""
        return await self.position_manager.flatten_all_positions(ExitReason.MANUAL)

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            "running": self._running,
            "broker_connected": self.broker.is_connected() if self.broker else False,
            "trading_mode": self.settings.trading_mode.value,
            "risk_status": self.risk_manager.get_status() if self.risk_manager else {},
            "market_session": self.calendar.get_session_info(),
            "engine_state": self.engine.get_state(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    orchestrator = TradingOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
