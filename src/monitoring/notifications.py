"""
Notification and Alerting System
=================================
Multi-channel notifications for trading events.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import aiohttp

from src.core.models import Order, Position, Signal, AlertLevel
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION TYPES
# ─────────────────────────────────────────────────────────────────────────────

class NotificationType(str, Enum):
    SIGNAL = "SIGNAL"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TARGET_HIT = "TARGET_HIT"
    TRAILING_STOP = "TRAILING_STOP"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    KILL_SWITCH = "KILL_SWITCH"
    ERROR = "ERROR"
    DAILY_SUMMARY = "DAILY_SUMMARY"
    CONNECTION_STATUS = "CONNECTION_STATUS"


# ─────────────────────────────────────────────────────────────────────────────
# BASE NOTIFIER
# ─────────────────────────────────────────────────────────────────────────────

class BaseNotifier:
    """Base class for notification channels."""

    async def send(
        self,
        notification_type: NotificationType,
        message: str,
        data: Optional[Dict] = None,
        level: AlertLevel = AlertLevel.INFO,
    ) -> bool:
        """Send a notification. Returns True if successful."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM NOTIFIER
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier(BaseNotifier):
    """Telegram notification channel."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(
        self,
        notification_type: NotificationType,
        message: str,
        data: Optional[Dict] = None,
        level: AlertLevel = AlertLevel.INFO,
    ) -> bool:
        """Send message via Telegram."""
        try:
            # Format message with emoji based on type
            emoji_map = {
                NotificationType.SIGNAL: "📊",
                NotificationType.ORDER_PLACED: "📝",
                NotificationType.ORDER_FILLED: "✅",
                NotificationType.ORDER_REJECTED: "❌",
                NotificationType.POSITION_OPENED: "🚀",
                NotificationType.POSITION_CLOSED: "🏁",
                NotificationType.STOP_LOSS_HIT: "🛑",
                NotificationType.TARGET_HIT: "🎯",
                NotificationType.TRAILING_STOP: "📈",
                NotificationType.CIRCUIT_BREAKER: "⚠️",
                NotificationType.KILL_SWITCH: "🚨",
                NotificationType.ERROR: "❗",
                NotificationType.DAILY_SUMMARY: "📋",
                NotificationType.CONNECTION_STATUS: "🔗",
            }

            level_indicator = {
                AlertLevel.INFO: "",
                AlertLevel.WARNING: "⚠️ ",
                AlertLevel.ERROR: "❗ ",
                AlertLevel.CRITICAL: "🚨 ",
            }

            emoji = emoji_map.get(notification_type, "📌")
            level_prefix = level_indicator.get(level, "")

            formatted_message = (
                f"{emoji} *{notification_type.value}*\n"
                f"{level_prefix}{message}"
            )

            if data:
                details = "\n".join(f"• {k}: {v}" for k, v in data.items())
                formatted_message += f"\n\n```\n{details}\n```"

            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message += f"\n\n_{timestamp}_"

            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": formatted_message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                }

                async with session.post(
                    f"{self.api_url}/sendMessage",
                    json=payload
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram send failed: {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Telegram notification error: {e}")
            return False

    async def send_photo(
        self,
        photo_path: str,
        caption: str = "",
    ) -> bool:
        """Send a photo (e.g., equity curve chart)."""
        try:
            async with aiohttp.ClientSession() as session:
                with open(photo_path, 'rb') as photo:
                    data = aiohttp.FormData()
                    data.add_field('chat_id', self.chat_id)
                    data.add_field('photo', photo)
                    if caption:
                        data.add_field('caption', caption)

                    async with session.post(
                        f"{self.api_url}/sendPhoto",
                        data=data
                    ) as response:
                        return response.status == 200

        except Exception as e:
            logger.error(f"Telegram photo send error: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE NOTIFIER (For development/logging)
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleNotifier(BaseNotifier):
    """Console/log notification channel."""

    async def send(
        self,
        notification_type: NotificationType,
        message: str,
        data: Optional[Dict] = None,
        level: AlertLevel = AlertLevel.INFO,
    ) -> bool:
        """Log notification to console."""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }.get(level, logging.INFO)

        log_message = f"[{notification_type.value}] {message}"
        if data:
            log_message += f" | {data}"

        logger.log(log_level, log_message)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class NotificationManager:
    """
    Central notification manager that routes to multiple channels.
    """

    def __init__(self):
        self.settings = get_settings()
        self.notifiers: List[BaseNotifier] = []
        self._setup_notifiers()

    def _setup_notifiers(self) -> None:
        """Initialize configured notification channels."""
        # Always add console notifier
        self.notifiers.append(ConsoleNotifier())

        # Telegram
        if self.settings.notification.telegram_enabled:
            token = self.settings.notification.telegram_bot_token
            chat_id = self.settings.notification.telegram_chat_id
            if token and chat_id:
                self.notifiers.append(TelegramNotifier(token, chat_id))
                logger.info("Telegram notifications enabled")

    async def notify(
        self,
        notification_type: NotificationType,
        message: str,
        data: Optional[Dict] = None,
        level: AlertLevel = AlertLevel.INFO,
    ) -> None:
        """Send notification to all configured channels."""
        # Check if this notification type should be sent
        if not self._should_notify(notification_type):
            return

        tasks = [
            notifier.send(notification_type, message, data, level)
            for notifier in self.notifiers
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    def _should_notify(self, notification_type: NotificationType) -> bool:
        """Check if notification type is enabled."""
        settings = self.settings.notification

        type_settings = {
            NotificationType.SIGNAL: settings.notify_on_signal,
            NotificationType.ORDER_FILLED: settings.notify_on_fill,
            NotificationType.STOP_LOSS_HIT: settings.notify_on_stop_loss,
            NotificationType.TARGET_HIT: settings.notify_on_target,
            NotificationType.ERROR: settings.notify_on_error,
            NotificationType.CIRCUIT_BREAKER: settings.notify_on_circuit_breaker,
            NotificationType.KILL_SWITCH: True,  # Always notify on kill switch
        }

        return type_settings.get(notification_type, True)

    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def notify_signal(self, signal: Signal) -> None:
        """Notify about a trading signal."""
        await self.notify(
            NotificationType.SIGNAL,
            f"{signal.signal_type.value} signal for {signal.symbol}",
            {
                "Price": f"₹{signal.price:,.2f}",
                "Stop Loss": f"₹{signal.stop_loss:,.2f}",
                "Target": f"₹{signal.target:,.2f}",
                "Trend": signal.trend.value,
                "Confidence": f"{float(signal.confidence) * 100:.0f}%",
            },
        )

    async def notify_order_filled(self, order: Order) -> None:
        """Notify about order fill."""
        await self.notify(
            NotificationType.ORDER_FILLED,
            f"Order filled: {order.symbol}",
            {
                "Side": order.side.value if hasattr(order.side, 'value') else order.side,
                "Quantity": order.filled_quantity,
                "Price": f"₹{float(order.average_price):,.2f}" if order.average_price else "N/A",
            },
        )

    async def notify_position_closed(
        self,
        position: Position,
        exit_reason: str
    ) -> None:
        """Notify about position closure."""
        pnl_emoji = "📈" if position.realized_pnl > 0 else "📉"

        level = AlertLevel.INFO
        if exit_reason in ("STOP_LOSS_HIT", "CIRCUIT_BREAKER", "KILL_SWITCH"):
            level = AlertLevel.WARNING

        notification_type = {
            "STOP_LOSS_HIT": NotificationType.STOP_LOSS_HIT,
            "TARGET_HIT": NotificationType.TARGET_HIT,
            "TRAILING_STOP": NotificationType.TRAILING_STOP,
        }.get(exit_reason, NotificationType.POSITION_CLOSED)

        await self.notify(
            notification_type,
            f"{pnl_emoji} Position closed: {position.symbol}",
            {
                "Side": position.side.value if hasattr(position.side, 'value') else position.side,
                "Entry": f"₹{float(position.entry_price):,.2f}",
                "Exit": f"₹{float(position.exit_price):,.2f}" if position.exit_price else "N/A",
                "P&L": f"₹{float(position.realized_pnl):,.2f}",
                "Return": f"{float(position.return_percent):.2f}%",
                "Reason": exit_reason,
            },
            level=level,
        )

    async def notify_circuit_breaker(self, reason: str) -> None:
        """Notify about circuit breaker trigger."""
        await self.notify(
            NotificationType.CIRCUIT_BREAKER,
            f"Circuit breaker triggered: {reason}",
            level=AlertLevel.CRITICAL,
        )

    async def notify_kill_switch(self, reason: str, results: Dict) -> None:
        """Notify about kill switch activation."""
        await self.notify(
            NotificationType.KILL_SWITCH,
            f"KILL SWITCH ACTIVATED: {reason}",
            {
                "Positions Closed": results.get("positions_closed", 0),
                "Orders Cancelled": results.get("orders_cancelled", 0),
            },
            level=AlertLevel.CRITICAL,
        )

    async def notify_error(self, error: str, context: Optional[Dict] = None) -> None:
        """Notify about an error."""
        await self.notify(
            NotificationType.ERROR,
            error,
            data=context,
            level=AlertLevel.ERROR,
        )

    async def send_daily_summary(
        self,
        pnl: float,
        trades: int,
        win_rate: float,
        positions: int,
    ) -> None:
        """Send end-of-day summary."""
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        await self.notify(
            NotificationType.DAILY_SUMMARY,
            f"{pnl_emoji} Daily Trading Summary",
            {
                "P&L": f"₹{pnl:,.2f}",
                "Trades": trades,
                "Win Rate": f"{win_rate:.1f}%",
                "Open Positions": positions,
            },
        )


# Global instance
notification_manager = NotificationManager()
