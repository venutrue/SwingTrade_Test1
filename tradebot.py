"""
Marc Rivalland Swing Trading Bot
================================
A Python implementation for automated swing trading using Rivalland's methodology.
Designed for integration with Zerodha Kite Connect API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import logging

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class Trend(Enum):
    UPTREND = 1
    DOWNTREND = -1
    NEUTRAL = 0

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class SwingPoint:
    price: float
    bar_index: int
    timestamp: datetime
    point_type: str  # 'HIGH' or 'LOW'

@dataclass
class TradeSignal:
    signal_type: SignalType
    price: float
    timestamp: datetime
    trend: Trend
    swing_high: float
    swing_low: float
    stop_loss: float
    target: float
    confidence: float = 0.0
    reason: str = ""

@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: int
    side: str  # 'BUY' or 'SELL'
    stop_loss: float
    target: float
    entry_time: datetime
    order_id: str = ""

@dataclass
class BotConfig:
    swing_lookback: int = 3
    min_pullback_bars: int = 2
    risk_percent: float = 1.0  # Risk 1% per trade
    risk_reward_ratio: float = 2.0
    max_positions: int = 3
    trailing_stop_percent: float = 0.5
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "day"  # 'day', '60minute', '15minute'


# ─────────────────────────────────────────────────────────────────────────────
# CORE SWING TRADING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RivallandSwingEngine:
    """
    Core engine implementing Marc Rivalland's Swing Trading methodology.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.current_trend = Trend.NEUTRAL
        self.pullback_bars = 0
        self.rally_bars = 0
        self.in_pullback = False
        self.in_rally = False
    
    def detect_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and lows using pivot point logic.
        """
        lookback = self.config.swing_lookback
        df = df.copy()
        
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or \
                   df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                df.loc[df.index[i], 'swing_high'] = df['high'].iloc[i]
            
            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or \
                   df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                df.loc[df.index[i], 'swing_low'] = df['low'].iloc[i]
        
        return df
    
    def update_swing_points(self, df: pd.DataFrame) -> None:
        """
        Update internal swing point tracking from detected points.
        """
        # Extract swing highs
        swing_high_mask = df['swing_high'].notna()
        for idx in df[swing_high_mask].index:
            row = df.loc[idx]
            point = SwingPoint(
                price=row['swing_high'],
                bar_index=df.index.get_loc(idx),
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                point_type='HIGH'
            )
            self.swing_highs.append(point)
        
        # Extract swing lows
        swing_low_mask = df['swing_low'].notna()
        for idx in df[swing_low_mask].index:
            row = df.loc[idx]
            point = SwingPoint(
                price=row['swing_low'],
                bar_index=df.index.get_loc(idx),
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                point_type='LOW'
            )
            self.swing_lows.append(point)
        
        # Keep only recent swing points (last 10)
        self.swing_highs = self.swing_highs[-10:]
        self.swing_lows = self.swing_lows[-10:]
    
    def determine_trend(self) -> Trend:
        """
        Determine trend based on Rivalland's HH/HL and LH/LL pattern.
        """
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return Trend.NEUTRAL
        
        last_high = self.swing_highs[-1].price
        prev_high = self.swing_highs[-2].price
        last_low = self.swing_lows[-1].price
        prev_low = self.swing_lows[-2].price
        
        higher_high = last_high > prev_high
        higher_low = last_low > prev_low
        lower_high = last_high < prev_high
        lower_low = last_low < prev_low
        
        if higher_high and higher_low:
            self.current_trend = Trend.UPTREND
        elif lower_high and lower_low:
            self.current_trend = Trend.DOWNTREND
        # Trend remains unchanged if mixed signals
        
        return self.current_trend
    
    def detect_pullback_rally(self, df: pd.DataFrame) -> Tuple[bool, int, bool, int]:
        """
        Detect pullback in uptrend or rally in downtrend.
        Returns: (in_pullback, pullback_bars, in_rally, rally_bars)
        """
        if len(df) < 3:
            return False, 0, False, 0
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Pullback detection (in uptrend)
        is_pullback_bar = current['close'] < previous['close'] or current['low'] < previous['low']
        
        # Rally detection (in downtrend)  
        is_rally_bar = current['close'] > previous['close'] or current['high'] > previous['high']
        
        if self.current_trend == Trend.UPTREND:
            if is_pullback_bar:
                self.in_pullback = True
                self.pullback_bars += 1
                self.in_rally = False
                self.rally_bars = 0
            elif current['high'] > previous['high'] and self.in_pullback:
                if self.pullback_bars >= self.config.min_pullback_bars:
                    # Pullback complete, potential buy signal
                    pass
                self.in_pullback = False
                self.pullback_bars = 0
        
        elif self.current_trend == Trend.DOWNTREND:
            if is_rally_bar:
                self.in_rally = True
                self.rally_bars += 1
                self.in_pullback = False
                self.pullback_bars = 0
            elif current['low'] < previous['low'] and self.in_rally:
                if self.rally_bars >= self.config.min_pullback_bars:
                    # Rally complete, potential sell signal
                    pass
                self.in_rally = False
                self.rally_bars = 0
        
        return self.in_pullback, self.pullback_bars, self.in_rally, self.rally_bars
    
    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """
        Generate trading signal based on current market state.
        """
        if len(df) < 5:
            return TradeSignal(
                signal_type=SignalType.HOLD,
                price=df['close'].iloc[-1],
                timestamp=datetime.now(),
                trend=self.current_trend,
                swing_high=0, swing_low=0,
                stop_loss=0, target=0,
                reason="Insufficient data"
            )
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        last_swing_high = self.swing_highs[-1].price if self.swing_highs else current['high']
        last_swing_low = self.swing_lows[-1].price if self.swing_lows else current['low']
        
        # Buy signal conditions
        buy_signal = (
            self.current_trend == Trend.UPTREND and
            self.in_pullback and
            self.pullback_bars >= self.config.min_pullback_bars and
            current['high'] > previous['high'] and
            current['close'] > current['open']  # Bullish candle
        )
        
        # Sell signal conditions
        sell_signal = (
            self.current_trend == Trend.DOWNTREND and
            self.in_rally and
            self.rally_bars >= self.config.min_pullback_bars and
            current['low'] < previous['low'] and
            current['close'] < current['open']  # Bearish candle
        )
        
        if buy_signal:
            stop_loss = last_swing_low * 0.995  # Slightly below swing low
            risk = current['close'] - stop_loss
            target = current['close'] + (risk * self.config.risk_reward_ratio)
            
            return TradeSignal(
                signal_type=SignalType.BUY,
                price=current['close'],
                timestamp=datetime.now(),
                trend=self.current_trend,
                swing_high=last_swing_high,
                swing_low=last_swing_low,
                stop_loss=stop_loss,
                target=target,
                confidence=0.7,
                reason=f"Uptrend pullback buy after {self.pullback_bars} bars"
            )
        
        elif sell_signal:
            stop_loss = last_swing_high * 1.005  # Slightly above swing high
            risk = stop_loss - current['close']
            target = current['close'] - (risk * self.config.risk_reward_ratio)
            
            return TradeSignal(
                signal_type=SignalType.SELL,
                price=current['close'],
                timestamp=datetime.now(),
                trend=self.current_trend,
                swing_high=last_swing_high,
                swing_low=last_swing_low,
                stop_loss=stop_loss,
                target=target,
                confidence=0.7,
                reason=f"Downtrend rally sell after {self.rally_bars} bars"
            )
        
        return TradeSignal(
            signal_type=SignalType.HOLD,
            price=current['close'],
            timestamp=datetime.now(),
            trend=self.current_trend,
            swing_high=last_swing_high,
            swing_low=last_swing_low,
            stop_loss=0, target=0,
            reason="No valid setup"
        )
    
    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        """
        Main analysis pipeline - run all steps and return signal.
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
        signal = self.generate_signal(df)
        
        self.logger.info(f"Analysis complete - Trend: {self.current_trend.name}, Signal: {signal.signal_type.value}")
        
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# ZERODHA KITE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class KiteConnectBroker:
    """
    Broker interface for Zerodha Kite Connect API.
    """
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self, request_token: str = None) -> bool:
        """
        Initialize Kite Connect session.
        """
        try:
            from kiteconnect import KiteConnect
            
            self.kite = KiteConnect(api_key=self.api_key)
            
            if request_token and not self.access_token:
                data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                self.access_token = data["access_token"]
            
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                self.logger.info("Kite Connect session established")
                return True
            else:
                login_url = self.kite.login_url()
                self.logger.info(f"Login required. Visit: {login_url}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def get_historical_data(
        self, 
        symbol: str, 
        interval: str = "day",
        days: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE', 'INFY')
            interval: Candle interval ('minute', '5minute', '15minute', '60minute', 'day')
            days: Number of days of history
        """
        try:
            instrument_token = self._get_instrument_token(symbol)
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def _get_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """
        Get instrument token for a symbol.
        """
        instruments = self.kite.instruments(exchange)
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
        raise ValueError(f"Symbol {symbol} not found on {exchange}")
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = None,
        stop_loss: float = None,
        target: float = None,
        exchange: str = "NSE"
    ) -> dict:
        """
        Place an order with optional bracket order (SL + Target).
        """
        try:
            if stop_loss and target:
                # Place bracket order
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_BO,
                    exchange=exchange,
                    tradingsymbol=symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL,
                    quantity=quantity,
                    order_type=self.kite.ORDER_TYPE_LIMIT if price else self.kite.ORDER_TYPE_MARKET,
                    price=price,
                    stoploss=abs(price - stop_loss) if price else abs(stop_loss),
                    squareoff=abs(target - price) if price else abs(target),
                    product=self.kite.PRODUCT_MIS  # Intraday
                )
            else:
                # Regular order
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL,
                    quantity=quantity,
                    order_type=self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT,
                    price=price,
                    product=self.kite.PRODUCT_CNC  # Delivery
                )
            
            self.logger.info(f"Order placed: {order_id}")
            return {"order_id": order_id, "status": "success"}
            
        except Exception as e:
            self.logger.error(f"Order failed: {e}")
            return {"order_id": None, "status": "failed", "error": str(e)}
    
    def get_positions(self) -> List[dict]:
        """Get current positions."""
        try:
            return self.kite.positions()['net']
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return []
    
    def get_balance(self) -> float:
        """Get available margin/balance."""
        try:
            margins = self.kite.margins()
            return margins['equity']['available']['live_balance']
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Position sizing and risk management.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float
    ) -> int:
        """
        Calculate position size based on risk percentage.
        """
        risk_amount = capital * (self.config.risk_percent / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = int(risk_amount / risk_per_share)
        
        # Ensure position doesn't exceed capital
        max_affordable = int(capital / entry_price)
        position_size = min(position_size, max_affordable)
        
        self.logger.info(f"Position size: {position_size} shares (Risk: ₹{risk_amount:.2f})")
        
        return position_size
    
    def validate_trade(
        self,
        signal: TradeSignal,
        current_positions: int,
        capital: float
    ) -> Tuple[bool, str]:
        """
        Validate if trade should be taken based on risk rules.
        """
        # Check max positions
        if current_positions >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"
        
        # Check risk/reward ratio
        if signal.stop_loss and signal.target:
            risk = abs(signal.price - signal.stop_loss)
            reward = abs(signal.target - signal.price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.config.risk_reward_ratio:
                return False, f"R:R ratio {rr_ratio:.2f} below minimum {self.config.risk_reward_ratio}"
        
        # Check if we have enough capital
        position_size = self.calculate_position_size(capital, signal.price, signal.stop_loss)
        if position_size < 1:
            return False, "Insufficient capital for minimum position"
        
        return True, "Trade validated"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRADING BOT
# ─────────────────────────────────────────────────────────────────────────────

class RivallandTradingBot:
    """
    Main trading bot orchestrator.
    """
    
    def __init__(
        self,
        config: BotConfig,
        api_key: str,
        api_secret: str,
        access_token: str = None
    ):
        self.config = config
        self.engine = RivallandSwingEngine(config)
        self.broker = KiteConnectBroker(api_key, api_secret, access_token)
        self.risk_manager = RiskManager(config)
        self.positions: List[Position] = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def run_analysis(self, symbol: str) -> TradeSignal:
        """
        Run analysis for a single symbol.
        """
        self.logger.info(f"Analyzing {symbol}...")
        
        # Fetch data
        df = self.broker.get_historical_data(
            symbol=symbol,
            interval=self.config.timeframe,
            days=100
        )
        
        if df.empty:
            self.logger.warning(f"No data for {symbol}")
            return None
        
        # Run swing analysis
        signal = self.engine.analyze(df)
        signal.reason = f"{symbol}: {signal.reason}"
        
        return signal
    
    def execute_signal(self, symbol: str, signal: TradeSignal) -> bool:
        """
        Execute a trading signal.
        """
        if signal.signal_type == SignalType.HOLD:
            return False
        
        # Get current state
        balance = self.broker.get_balance()
        current_positions = len(self.broker.get_positions())
        
        # Validate trade
        is_valid, reason = self.risk_manager.validate_trade(
            signal, current_positions, balance
        )
        
        if not is_valid:
            self.logger.info(f"Trade rejected: {reason}")
            return False
        
        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(
            balance, signal.price, signal.stop_loss
        )
        
        if quantity < 1:
            self.logger.warning("Position size too small")
            return False
        
        # Place order
        side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
        result = self.broker.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=signal.price,
            stop_loss=signal.stop_loss,
            target=signal.target
        )
        
        if result['status'] == 'success':
            position = Position(
                symbol=symbol,
                entry_price=signal.price,
                quantity=quantity,
                side=side,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now(),
                order_id=result['order_id']
            )
            self.positions.append(position)
            self.logger.info(f"Position opened: {position}")
            return True
        
        return False
    
    def run(self):
        """
        Main bot loop - analyze all symbols and execute signals.
        """
        if not self.broker.connect():
            self.logger.error("Failed to connect to broker")
            return
        
        self.logger.info(f"Bot started. Monitoring {len(self.config.symbols)} symbols")
        
        for symbol in self.config.symbols:
            try:
                signal = self.run_analysis(symbol)
                
                if signal and signal.signal_type != SignalType.HOLD:
                    self.logger.info(f"Signal: {signal}")
                    self.execute_signal(symbol, signal)
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        self.logger.info("Analysis complete")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Simple backtesting engine for strategy validation.
    """
    
    def __init__(self, config: BotConfig, initial_capital: float = 100000):
        self.config = config
        self.initial_capital = initial_capital
        self.engine = RivallandSwingEngine(config)
        self.risk_manager = RiskManager(config)
    
    def run(self, df: pd.DataFrame) -> dict:
        """
        Run backtest on historical data.
        """
        capital = self.initial_capital
        trades = []
        position = None
        
        # Need enough bars for lookback
        start_idx = self.config.swing_lookback * 2 + 10
        
        for i in range(start_idx, len(df)):
            # Get data up to current bar
            current_df = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            
            # Reset engine state for clean analysis
            self.engine = RivallandSwingEngine(self.config)
            signal = self.engine.analyze(current_df)
            
            # Check for exit if in position
            if position:
                # Check stop loss
                if position['side'] == 'BUY' and current_bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    capital += pnl
                    trades.append({**position, 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': 'stop_loss'})
                    position = None
                    
                # Check target
                elif position['side'] == 'BUY' and current_bar['high'] >= position['target']:
                    exit_price = position['target']
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    capital += pnl
                    trades.append({**position, 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': 'target'})
                    position = None
                    
                # Similar for SELL positions
                elif position['side'] == 'SELL' and current_bar['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                    capital += pnl
                    trades.append({**position, 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': 'stop_loss'})
                    position = None
                    
                elif position['side'] == 'SELL' and current_bar['low'] <= position['target']:
                    exit_price = position['target']
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                    capital += pnl
                    trades.append({**position, 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': 'target'})
                    position = None
            
            # Check for entry if no position
            if not position and signal.signal_type != SignalType.HOLD:
                quantity = self.risk_manager.calculate_position_size(
                    capital, signal.price, signal.stop_loss
                )
                
                if quantity > 0:
                    position = {
                        'entry_price': signal.price,
                        'quantity': quantity,
                        'side': signal.signal_type.value,
                        'stop_loss': signal.stop_loss,
                        'target': signal.target,
                        'entry_idx': i
                    }
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        total_pnl = sum(t['pnl'] for t in trades)
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'trades': trades
        }


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    config = BotConfig(
        swing_lookback=3,
        min_pullback_bars=2,
        risk_percent=1.0,
        risk_reward_ratio=2.0,
        max_positions=3,
        symbols=['RELIANCE', 'INFY', 'TCS', 'HDFCBANK', 'ICICIBANK'],
        timeframe='day'
    )
    
    # For live trading (requires Kite Connect credentials)
    """
    bot = RivallandTradingBot(
        config=config,
        api_key='your_api_key',
        api_secret='your_api_secret',
        access_token='your_access_token'  # Optional, can authenticate via request_token
    )
    bot.run()
    """
    
    # For backtesting with sample data
    print("Running backtest with sample data...")
    
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Simulate price data with trend
    price = 100
    data = []
    for _ in range(200):
        open_p = price
        high = price + np.random.uniform(0, 3)
        low = price - np.random.uniform(0, 3)
        close = price + np.random.uniform(-2, 2)
        price = close + np.random.uniform(-1, 1)  # Random walk with drift
        data.append([open_p, high, low, close, np.random.randint(100000, 500000)])
    
    sample_df = pd.DataFrame(data, index=dates, columns=['open', 'high', 'low', 'close', 'volume'])
    
    backtester = Backtester(config, initial_capital=100000)
    results = backtester.run(sample_df)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
    print(f"Final Capital:   ₹{results['final_capital']:,.2f}")
    print(f"Total Return:    {results['total_return']:.2f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Win Rate:        {results['win_rate']:.1f}%")
    print(f"Total P&L:       ₹{results['total_pnl']:,.2f}")
```

## Project Structure

For a production setup, organize your project like this:
```
rivalland_bot/
├── config/
│   ├── settings.py          # Configuration management
│   └── credentials.py        # API keys (gitignore this)
├── core/
│   ├── engine.py             # RivallandSwingEngine
│   ├── signals.py            # Signal generation logic
│   └── indicators.py         # Technical indicators
├── broker/
│   ├── base.py               # Abstract broker interface
│   ├── kite.py               # Zerodha Kite implementation
│   └── paper.py              # Paper trading for testing
├── risk/
│   ├── position_sizing.py    # Position sizing logic
│   └── manager.py            # Risk management rules
├── backtest/
│   ├── engine.py             # Backtesting engine
│   └── reports.py            # Performance reporting
├── utils/
│   ├── logger.py             # Logging configuration
│   └── notifications.py      # Telegram/Email alerts
├── main.py                   # Entry point
├── scheduler.py              # Scheduled execution
└── requirements.txt
```

## Key Dependencies
```
kiteconnect>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
schedule>=1.2.0
requests>=2.31.0
