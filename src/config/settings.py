"""
Prop-Level Configuration Management
====================================
Centralized, validated, environment-aware configuration with runtime reloading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PAPER = "paper"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    MCX = "MCX"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class ProductType(str, Enum):
    CNC = "CNC"      # Delivery
    MIS = "MIS"      # Intraday
    NRML = "NRML"    # F&O


# ─────────────────────────────────────────────────────────────────────────────
# SUB-CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

class BrokerConfig(BaseModel):
    """Zerodha Kite Connect Configuration"""
    api_key: str = Field(..., min_length=8)
    api_secret: str = Field(..., min_length=8)
    access_token: Optional[str] = None
    request_token: Optional[str] = None

    # Connection settings
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1)
    connection_timeout: float = Field(default=30.0, ge=5.0)
    request_timeout: float = Field(default=10.0, ge=1.0)

    # Rate limiting
    max_requests_per_second: int = Field(default=3, ge=1, le=10)

    model_config = SettingsConfigDict(extra="forbid")


class RiskConfig(BaseModel):
    """Risk Management Configuration"""
    # Per-trade limits
    max_risk_per_trade_percent: Decimal = Field(default=Decimal("1.0"), ge=0.1, le=5.0)
    min_risk_reward_ratio: Decimal = Field(default=Decimal("2.0"), ge=1.0)
    max_position_size_percent: Decimal = Field(default=Decimal("20.0"), ge=1.0, le=50.0)

    # Portfolio limits
    max_open_positions: int = Field(default=5, ge=1, le=20)
    max_portfolio_risk_percent: Decimal = Field(default=Decimal("5.0"), ge=1.0, le=20.0)
    max_sector_exposure_percent: Decimal = Field(default=Decimal("30.0"), ge=10.0, le=50.0)
    max_single_stock_percent: Decimal = Field(default=Decimal("25.0"), ge=5.0, le=40.0)

    # Daily limits (CRITICAL for prop trading)
    max_daily_loss_percent: Decimal = Field(default=Decimal("2.0"), ge=0.5, le=10.0)
    max_daily_trades: int = Field(default=20, ge=1, le=100)
    max_daily_volume: Decimal = Field(default=Decimal("500000"), ge=10000)

    # Weekly limits
    max_weekly_loss_percent: Decimal = Field(default=Decimal("5.0"), ge=1.0, le=15.0)

    # Monthly limits
    max_monthly_loss_percent: Decimal = Field(default=Decimal("10.0"), ge=2.0, le=25.0)

    # Circuit breakers
    consecutive_loss_limit: int = Field(default=3, ge=2, le=10)
    circuit_breaker_cooldown_minutes: int = Field(default=30, ge=5, le=120)

    # Slippage and fees (for calculations)
    expected_slippage_percent: Decimal = Field(default=Decimal("0.05"), ge=0.0, le=1.0)
    brokerage_per_order: Decimal = Field(default=Decimal("20.0"), ge=0.0)
    stt_percent: Decimal = Field(default=Decimal("0.1"), ge=0.0)  # Securities Transaction Tax

    model_config = SettingsConfigDict(extra="forbid")


class StrategyConfig(BaseModel):
    """Rivalland Swing Trading Strategy Configuration"""
    swing_lookback: int = Field(default=3, ge=2, le=10)
    min_pullback_bars: int = Field(default=2, ge=1, le=5)
    min_rally_bars: int = Field(default=2, ge=1, le=5)

    # Signal filters
    require_volume_confirmation: bool = Field(default=True)
    volume_threshold_multiplier: Decimal = Field(default=Decimal("1.2"), ge=1.0, le=3.0)

    # Stop loss placement
    stop_loss_buffer_percent: Decimal = Field(default=Decimal("0.5"), ge=0.1, le=2.0)

    # Trailing stop
    enable_trailing_stop: bool = Field(default=True)
    trailing_stop_activation_percent: Decimal = Field(default=Decimal("1.5"), ge=0.5, le=5.0)
    trailing_stop_distance_percent: Decimal = Field(default=Decimal("1.0"), ge=0.3, le=3.0)

    # Time filters
    min_bars_between_signals: int = Field(default=3, ge=1, le=10)

    model_config = SettingsConfigDict(extra="forbid")


class MarketConfig(BaseModel):
    """Market Hours and Trading Calendar Configuration"""
    exchange: Exchange = Field(default=Exchange.NSE)

    # Market hours (IST)
    market_open_time: str = Field(default="09:15")
    market_close_time: str = Field(default="15:30")
    pre_market_start: str = Field(default="09:00")
    pre_market_end: str = Field(default="09:08")

    # Trading window (avoid first/last minutes)
    trading_start_buffer_minutes: int = Field(default=15, ge=0, le=60)
    trading_end_buffer_minutes: int = Field(default=15, ge=0, le=60)

    # Special sessions
    allow_pre_market: bool = Field(default=False)
    allow_post_market: bool = Field(default=False)

    # Timezone
    timezone: str = Field(default="Asia/Kolkata")

    model_config = SettingsConfigDict(extra="forbid")


class DataConfig(BaseModel):
    """Data Management Configuration"""
    # Historical data
    default_lookback_days: int = Field(default=100, ge=30, le=365)

    # Real-time data
    tick_buffer_size: int = Field(default=1000, ge=100, le=10000)
    candle_aggregation_seconds: int = Field(default=60, ge=1, le=3600)

    # Data quality
    max_data_age_seconds: int = Field(default=5, ge=1, le=60)
    gap_fill_method: str = Field(default="forward")  # forward, interpolate, skip

    # Caching
    enable_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)

    model_config = SettingsConfigDict(extra="forbid")


class NotificationConfig(BaseModel):
    """Notification Configuration"""
    # Telegram
    telegram_enabled: bool = Field(default=False)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Email
    email_enabled: bool = Field(default=False)
    smtp_host: Optional[str] = None
    smtp_port: int = Field(default=587)
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = Field(default_factory=list)

    # Alert levels
    notify_on_signal: bool = Field(default=True)
    notify_on_fill: bool = Field(default=True)
    notify_on_stop_loss: bool = Field(default=True)
    notify_on_target: bool = Field(default=True)
    notify_on_error: bool = Field(default=True)
    notify_on_circuit_breaker: bool = Field(default=True)

    model_config = SettingsConfigDict(extra="forbid")


class DatabaseConfig(BaseModel):
    """Database Configuration"""
    url: str = Field(default="sqlite+aiosqlite:///./data/trading.db")
    echo: bool = Field(default=False)
    pool_size: int = Field(default=5, ge=1, le=20)
    max_overflow: int = Field(default=10, ge=0, le=50)

    # Backup
    enable_backup: bool = Field(default=True)
    backup_interval_hours: int = Field(default=24, ge=1, le=168)
    backup_retention_days: int = Field(default=30, ge=7, le=365)

    model_config = SettingsConfigDict(extra="forbid")


class LoggingConfig(BaseModel):
    """Logging Configuration"""
    level: str = Field(default="INFO")
    format: str = Field(default="json")  # json, console

    # File logging
    log_dir: str = Field(default="./logs")
    max_file_size_mb: int = Field(default=100, ge=10, le=1000)
    backup_count: int = Field(default=10, ge=1, le=50)

    # Audit logging (separate from operational logs)
    audit_log_enabled: bool = Field(default=True)
    audit_log_file: str = Field(default="./logs/audit.log")

    model_config = SettingsConfigDict(extra="forbid")


class MonitoringConfig(BaseModel):
    """Monitoring and Metrics Configuration"""
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)

    # Health checks
    health_check_interval_seconds: int = Field(default=30, ge=10, le=300)

    # Performance tracking
    track_latency: bool = Field(default=True)
    track_order_flow: bool = Field(default=True)

    model_config = SettingsConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SETTINGS CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Master configuration for the Prop Trading System.

    Configuration Priority:
    1. Environment variables (highest)
    2. .env file
    3. Default values (lowest)
    """

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    debug: bool = Field(default=False)

    # Application
    app_name: str = Field(default="RivallandPropTrader")
    version: str = Field(default="2.0.0")

    # Symbols to trade
    symbols: List[str] = Field(default_factory=lambda: ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"])
    timeframe: str = Field(default="day")  # day, 60minute, 15minute, 5minute

    # Sub-configurations
    broker: BrokerConfig = Field(default_factory=lambda: BrokerConfig(
        api_key=os.getenv("KITE_API_KEY", ""),
        api_secret=os.getenv("KITE_API_SECRET", "")
    ))
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Kill switch
    kill_switch_enabled: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Ensure production has required security settings."""
        if self.environment == Environment.PRODUCTION:
            if not self.broker.api_key or not self.broker.api_secret:
                raise ValueError("Production requires valid broker credentials")
            if self.trading_mode == TradingMode.BACKTEST:
                raise ValueError("Cannot run backtest mode in production environment")
            if not self.notification.telegram_enabled and not self.notification.email_enabled:
                raise ValueError("Production requires at least one notification channel")
        return self

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one symbol must be configured")
        return [s.upper().strip() for s in v]

    def is_live_trading(self) -> bool:
        """Check if this is live trading with real money."""
        return (
            self.trading_mode == TradingMode.LIVE and
            self.environment == Environment.PRODUCTION
        )

    def get_effective_risk_percent(self) -> Decimal:
        """Get risk percent adjusted for environment."""
        if self.environment == Environment.DEVELOPMENT:
            return min(self.risk.max_risk_per_trade_percent, Decimal("0.5"))
        return self.risk.max_risk_per_trade_percent


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON SETTINGS INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload of settings (for runtime reconfiguration)."""
    global _settings
    _settings = Settings()
    return _settings
