"""
Rivalland Swing Trading Engine
==============================
Production-grade swing trading signal generation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Tuple, Dict, Any
import logging

import numpy as np
import pandas as pd

from src.core.models import Signal, Trend, SignalType
from src.config.settings import get_settings, StrategyConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """Represents a swing high or low point."""
    price: Decimal
    bar_index: int
    timestamp: datetime
    point_type: str  # 'HIGH' or 'LOW'
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": float(self.price),
            "bar_index": self.bar_index,
            "timestamp": self.timestamp.isoformat(),
            "point_type": self.point_type,
            "confirmed": self.confirmed,
        }


@dataclass
class EngineState:
    """Tracks internal engine state."""
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    current_trend: Trend = Trend.NEUTRAL
    in_pullback: bool = False
    in_rally: bool = False
    pullback_bars: int = 0
    rally_bars: int = 0
    last_signal_bar: int = -100
    bars_processed: int = 0

    def reset(self) -> None:
        """Reset state for new analysis."""
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.current_trend = Trend.NEUTRAL
        self.in_pullback = False
        self.in_rally = False
        self.pullback_bars = 0
        self.rally_bars = 0
        self.last_signal_bar = -100
        self.bars_processed = 0


# ─────────────────────────────────────────────────────────────────────────────
# RIVALLAND SWING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RivallandSwingEngine:
    """
    Production implementation of Marc Rivalland's Swing Trading methodology.

    Features:
    - Accurate swing point detection using pivot logic
    - HH/HL and LH/LL trend identification
    - Pullback/rally phase tracking
    - Volume confirmation filtering
    - Signal quality scoring
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or get_settings().strategy
        self.state = EngineState()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def reset(self) -> None:
        """Reset engine state."""
        self.state.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # SWING POINT DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and lows using pivot point logic.

        A swing high is confirmed when the high is higher than
        `lookback` bars on both sides.
        """
        lookback = self.config.swing_lookback
        df = df.copy()

        df['swing_high'] = np.nan
        df['swing_low'] = np.nan

        highs = df['high'].values
        lows = df['low'].values

        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                df.iloc[i, df.columns.get_loc('swing_high')] = highs[i]

            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                df.iloc[i, df.columns.get_loc('swing_low')] = lows[i]

        return df

    def update_swing_points(self, df: pd.DataFrame) -> None:
        """Update internal swing point tracking."""
        # Extract and store swing highs
        swing_high_mask = df['swing_high'].notna()
        for idx in df[swing_high_mask].index:
            bar_index = df.index.get_loc(idx)
            point = SwingPoint(
                price=Decimal(str(df.loc[idx, 'swing_high'])),
                bar_index=bar_index,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                point_type='HIGH',
                confirmed=True,
            )
            self.state.swing_highs.append(point)

        # Extract and store swing lows
        swing_low_mask = df['swing_low'].notna()
        for idx in df[swing_low_mask].index:
            bar_index = df.index.get_loc(idx)
            point = SwingPoint(
                price=Decimal(str(df.loc[idx, 'swing_low'])),
                bar_index=bar_index,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                point_type='LOW',
                confirmed=True,
            )
            self.state.swing_lows.append(point)

        # Keep only recent swing points
        self.state.swing_highs = self.state.swing_highs[-10:]
        self.state.swing_lows = self.state.swing_lows[-10:]

    # ─────────────────────────────────────────────────────────────────────────
    # TREND DETERMINATION
    # ─────────────────────────────────────────────────────────────────────────

    def determine_trend(self) -> Trend:
        """
        Determine trend based on Rivalland's HH/HL and LH/LL pattern.

        Uptrend: Higher Highs AND Higher Lows
        Downtrend: Lower Highs AND Lower Lows
        """
        if len(self.state.swing_highs) < 2 or len(self.state.swing_lows) < 2:
            return Trend.NEUTRAL

        last_high = self.state.swing_highs[-1].price
        prev_high = self.state.swing_highs[-2].price
        last_low = self.state.swing_lows[-1].price
        prev_low = self.state.swing_lows[-2].price

        higher_high = last_high > prev_high
        higher_low = last_low > prev_low
        lower_high = last_high < prev_high
        lower_low = last_low < prev_low

        if higher_high and higher_low:
            self.state.current_trend = Trend.UPTREND
        elif lower_high and lower_low:
            self.state.current_trend = Trend.DOWNTREND
        # Mixed signals - trend continues

        return self.state.current_trend

    # ─────────────────────────────────────────────────────────────────────────
    # PULLBACK/RALLY DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_pullback_rally(self, df: pd.DataFrame) -> Tuple[bool, int, bool, int]:
        """
        Detect pullback in uptrend or rally in downtrend.

        Pullback: Lower close or lower low in an uptrend
        Rally: Higher close or higher high in a downtrend
        """
        if len(df) < 3:
            return False, 0, False, 0

        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Pullback detection (price retracing in uptrend)
        is_pullback_bar = (
            current['close'] < previous['close'] or
            current['low'] < previous['low']
        )

        # Rally detection (price retracing in downtrend)
        is_rally_bar = (
            current['close'] > previous['close'] or
            current['high'] > previous['high']
        )

        if self.state.current_trend == Trend.UPTREND:
            if is_pullback_bar:
                self.state.in_pullback = True
                self.state.pullback_bars += 1
                self.state.in_rally = False
                self.state.rally_bars = 0
            elif current['high'] > previous['high'] and self.state.in_pullback:
                # Pullback potentially ending
                if self.state.pullback_bars >= self.config.min_pullback_bars:
                    pass  # Signal generation will handle this
                self.state.in_pullback = False
                self.state.pullback_bars = 0

        elif self.state.current_trend == Trend.DOWNTREND:
            if is_rally_bar:
                self.state.in_rally = True
                self.state.rally_bars += 1
                self.state.in_pullback = False
                self.state.pullback_bars = 0
            elif current['low'] < previous['low'] and self.state.in_rally:
                # Rally potentially ending
                if self.state.rally_bars >= self.config.min_rally_bars:
                    pass  # Signal generation will handle this
                self.state.in_rally = False
                self.state.rally_bars = 0

        return (
            self.state.in_pullback,
            self.state.pullback_bars,
            self.state.in_rally,
            self.state.rally_bars,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "day"
    ) -> Signal:
        """
        Generate trading signal based on current market state.

        Buy conditions:
        1. Uptrend (HH + HL)
        2. In pullback phase
        3. Minimum pullback bars met
        4. Current bar breaks above previous high
        5. Bullish candle (close > open)
        6. Optional: Volume confirmation

        Sell conditions (inverse for downtrend)
        """
        current_bar = len(df) - 1
        self.state.bars_processed = current_bar

        if len(df) < 5:
            return self._create_hold_signal(df, symbol, timeframe, "Insufficient data")

        current = df.iloc[-1]
        previous = df.iloc[-2]

        last_swing_high = (
            self.state.swing_highs[-1].price
            if self.state.swing_highs
            else Decimal(str(current['high']))
        )
        last_swing_low = (
            self.state.swing_lows[-1].price
            if self.state.swing_lows
            else Decimal(str(current['low']))
        )

        # Check minimum bars between signals
        bars_since_signal = current_bar - self.state.last_signal_bar
        if bars_since_signal < self.config.min_bars_between_signals:
            return self._create_hold_signal(
                df, symbol, timeframe,
                f"Too soon after last signal ({bars_since_signal} bars)"
            )

        # Volume confirmation
        volume_ok = True
        if self.config.require_volume_confirmation and 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = current['volume']
            volume_ok = current_volume > avg_volume * float(self.config.volume_threshold_multiplier)

        # ─────────────────────────────────────────────────────────────────────
        # BUY SIGNAL CONDITIONS
        # ─────────────────────────────────────────────────────────────────────
        buy_conditions = (
            self.state.current_trend == Trend.UPTREND and
            self.state.in_pullback and
            self.state.pullback_bars >= self.config.min_pullback_bars and
            current['high'] > previous['high'] and
            current['close'] > current['open']  # Bullish candle
        )

        if buy_conditions and volume_ok:
            entry_price = Decimal(str(current['close']))
            buffer = entry_price * (self.config.stop_loss_buffer_percent / 100)
            stop_loss = last_swing_low - buffer
            risk = entry_price - stop_loss
            target = entry_price + (risk * self.settings.risk.min_risk_reward_ratio)

            # Calculate signal strength
            strength = self._calculate_signal_strength(
                df, "BUY", volume_ok,
                self.state.pullback_bars,
                last_swing_low
            )

            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                trend=Trend.UPTREND,
                price=entry_price,
                swing_high=last_swing_high,
                swing_low=last_swing_low,
                stop_loss=stop_loss,
                target=target,
                confidence=Decimal(str(strength / 5)),  # Normalize to 0-1
                strength=strength,
                pullback_bars=self.state.pullback_bars,
                rally_bars=0,
                reason=f"Uptrend pullback buy after {self.state.pullback_bars} bars",
                timestamp=datetime.now(),
                timeframe=timeframe,
            )

            self.state.last_signal_bar = current_bar
            self.logger.info(f"BUY signal generated: {symbol} @ {entry_price}")
            return signal

        # ─────────────────────────────────────────────────────────────────────
        # SELL SIGNAL CONDITIONS
        # ─────────────────────────────────────────────────────────────────────
        sell_conditions = (
            self.state.current_trend == Trend.DOWNTREND and
            self.state.in_rally and
            self.state.rally_bars >= self.config.min_rally_bars and
            current['low'] < previous['low'] and
            current['close'] < current['open']  # Bearish candle
        )

        if sell_conditions and volume_ok:
            entry_price = Decimal(str(current['close']))
            buffer = entry_price * (self.config.stop_loss_buffer_percent / 100)
            stop_loss = last_swing_high + buffer
            risk = stop_loss - entry_price
            target = entry_price - (risk * self.settings.risk.min_risk_reward_ratio)

            strength = self._calculate_signal_strength(
                df, "SELL", volume_ok,
                self.state.rally_bars,
                last_swing_high
            )

            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                trend=Trend.DOWNTREND,
                price=entry_price,
                swing_high=last_swing_high,
                swing_low=last_swing_low,
                stop_loss=stop_loss,
                target=target,
                confidence=Decimal(str(strength / 5)),
                strength=strength,
                pullback_bars=0,
                rally_bars=self.state.rally_bars,
                reason=f"Downtrend rally sell after {self.state.rally_bars} bars",
                timestamp=datetime.now(),
                timeframe=timeframe,
            )

            self.state.last_signal_bar = current_bar
            self.logger.info(f"SELL signal generated: {symbol} @ {entry_price}")
            return signal

        return self._create_hold_signal(df, symbol, timeframe, "No valid setup")

    def _create_hold_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        reason: str
    ) -> Signal:
        """Create a HOLD signal."""
        current = df.iloc[-1]

        last_swing_high = (
            self.state.swing_highs[-1].price
            if self.state.swing_highs
            else Decimal(str(current['high']))
        )
        last_swing_low = (
            self.state.swing_lows[-1].price
            if self.state.swing_lows
            else Decimal(str(current['low']))
        )

        return Signal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            trend=self.state.current_trend,
            price=Decimal(str(current['close'])),
            swing_high=last_swing_high,
            swing_low=last_swing_low,
            stop_loss=Decimal(0),
            target=Decimal(0),
            confidence=Decimal(0),
            strength=0,
            pullback_bars=self.state.pullback_bars,
            rally_bars=self.state.rally_bars,
            reason=reason,
            timestamp=datetime.now(),
            timeframe=timeframe,
        )

    def _calculate_signal_strength(
        self,
        df: pd.DataFrame,
        signal_type: str,
        volume_confirmed: bool,
        retracement_bars: int,
        reference_swing: Decimal,
    ) -> int:
        """
        Calculate signal strength score (1-5).

        Factors:
        - Volume confirmation
        - Retracement depth
        - Retracement duration
        - Trend strength
        """
        strength = 1

        # Volume confirmation (+1)
        if volume_confirmed:
            strength += 1

        # Retracement duration (+1 for optimal 2-4 bars)
        if 2 <= retracement_bars <= 4:
            strength += 1

        # Trend alignment (+1 if multiple swings confirm)
        if len(self.state.swing_highs) >= 3 and len(self.state.swing_lows) >= 3:
            strength += 1

        # Price near swing point (+1 if within 2% of swing)
        current_price = Decimal(str(df.iloc[-1]['close']))
        distance_percent = abs(current_price - reference_swing) / reference_swing * 100
        if distance_percent < 2:
            strength += 1

        return min(strength, 5)

    @property
    def settings(self):
        """Get global settings."""
        return get_settings()

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ANALYSIS PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "day"
    ) -> Signal:
        """
        Main analysis pipeline.

        Steps:
        1. Detect swing points
        2. Update internal state
        3. Determine trend
        4. Detect pullback/rally
        5. Generate signal
        """
        # Step 1: Detect swing points
        df = self.detect_swing_points(df)

        # Step 2: Update internal state
        self.update_swing_points(df)

        # Step 3: Determine trend
        self.determine_trend()

        # Step 4: Detect pullback/rally
        self.detect_pullback_rally(df)

        # Step 5: Generate signal
        signal = self.generate_signal(df, symbol, timeframe)

        self.logger.debug(
            f"Analysis complete - {symbol}: "
            f"Trend={self.state.current_trend.value}, "
            f"Signal={signal.signal_type.value}"
        )

        return signal

    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        return {
            "trend": self.state.current_trend.value,
            "in_pullback": self.state.in_pullback,
            "pullback_bars": self.state.pullback_bars,
            "in_rally": self.state.in_rally,
            "rally_bars": self.state.rally_bars,
            "swing_highs_count": len(self.state.swing_highs),
            "swing_lows_count": len(self.state.swing_lows),
            "last_swing_high": float(self.state.swing_highs[-1].price) if self.state.swing_highs else None,
            "last_swing_low": float(self.state.swing_lows[-1].price) if self.state.swing_lows else None,
            "bars_processed": self.state.bars_processed,
        }
