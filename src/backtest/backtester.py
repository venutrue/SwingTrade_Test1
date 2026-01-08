"""
Production Backtesting Engine
=============================
Comprehensive backtesting with slippage, commissions, and realistic simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd

from src.core.engine import RivallandSwingEngine
from src.core.models import SignalType, Side, Trend
from src.config.settings import get_settings, Settings
from src.risk.risk_manager import PositionSizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST TRADE RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Record of a completed backtest trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    side: str
    quantity: int
    entry_price: Decimal
    exit_price: Optional[Decimal]
    stop_loss: Decimal
    target: Decimal
    pnl: Decimal = Decimal(0)
    return_percent: Decimal = Decimal(0)
    exit_reason: str = ""
    slippage_cost: Decimal = Decimal(0)
    commission_cost: Decimal = Decimal(0)
    holding_bars: int = 0
    max_favorable_excursion: Decimal = Decimal(0)  # Best unrealized P&L
    max_adverse_excursion: Decimal = Decimal(0)   # Worst unrealized P&L

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "stop_loss": float(self.stop_loss),
            "target": float(self.target),
            "pnl": float(self.pnl),
            "return_percent": float(self.return_percent),
            "exit_reason": self.exit_reason,
            "slippage_cost": float(self.slippage_cost),
            "commission_cost": float(self.commission_cost),
            "holding_bars": self.holding_bars,
            "mfe": float(self.max_favorable_excursion),
            "mae": float(self.max_adverse_excursion),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    """Comprehensive backtest results and statistics."""
    # Basic info
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_bars: int

    # Capital
    initial_capital: Decimal
    final_capital: Decimal
    peak_capital: Decimal
    trough_capital: Decimal

    # Returns
    total_return: Decimal
    total_return_percent: Decimal
    cagr: Decimal  # Compound Annual Growth Rate

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal  # Gross profit / Gross loss
    expectancy: Decimal  # Expected value per trade

    # Risk metrics
    max_drawdown: Decimal
    max_drawdown_percent: Decimal
    max_drawdown_duration: int  # In bars
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal  # CAGR / Max Drawdown

    # Costs
    total_slippage: Decimal
    total_commissions: Decimal
    total_costs: Decimal

    # Trade details
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Decimal] = field(default_factory=list)

    # Additional stats
    avg_holding_period: Decimal = Decimal(0)
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_mfe: Decimal = Decimal(0)  # Average max favorable excursion
    avg_mae: Decimal = Decimal(0)  # Average max adverse excursion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "bars": self.total_bars,
            },
            "capital": {
                "initial": float(self.initial_capital),
                "final": float(self.final_capital),
                "peak": float(self.peak_capital),
                "trough": float(self.trough_capital),
            },
            "returns": {
                "total": float(self.total_return),
                "percent": float(self.total_return_percent),
                "cagr": float(self.cagr),
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "win_rate": float(self.win_rate),
                "avg_win": float(self.avg_win),
                "avg_loss": float(self.avg_loss),
                "profit_factor": float(self.profit_factor),
                "expectancy": float(self.expectancy),
                "avg_holding_period": float(self.avg_holding_period),
            },
            "risk": {
                "max_drawdown": float(self.max_drawdown),
                "max_drawdown_percent": float(self.max_drawdown_percent),
                "max_drawdown_duration": self.max_drawdown_duration,
                "sharpe_ratio": float(self.sharpe_ratio),
                "sortino_ratio": float(self.sortino_ratio),
                "calmar_ratio": float(self.calmar_ratio),
            },
            "costs": {
                "slippage": float(self.total_slippage),
                "commissions": float(self.total_commissions),
                "total": float(self.total_costs),
            },
            "streaks": {
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
            },
        }

    def print_summary(self) -> str:
        """Generate a printable summary."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Symbol: {self.symbol} | Timeframe: {self.timeframe}",
            f"Period: {self.start_date.date()} to {self.end_date.date()} ({self.total_bars} bars)",
            "-" * 60,
            "CAPITAL",
            f"  Initial:     ₹{self.initial_capital:>15,.2f}",
            f"  Final:       ₹{self.final_capital:>15,.2f}",
            f"  Peak:        ₹{self.peak_capital:>15,.2f}",
            "-" * 60,
            "RETURNS",
            f"  Total Return: ₹{self.total_return:>14,.2f} ({self.total_return_percent:>6.2f}%)",
            f"  CAGR:         {self.cagr:>6.2f}%",
            "-" * 60,
            "TRADE STATISTICS",
            f"  Total Trades:     {self.total_trades:>6}",
            f"  Winning Trades:   {self.winning_trades:>6} ({self.win_rate:.1f}%)",
            f"  Losing Trades:    {self.losing_trades:>6}",
            f"  Avg Win:         ₹{self.avg_win:>10,.2f}",
            f"  Avg Loss:        ₹{self.avg_loss:>10,.2f}",
            f"  Profit Factor:    {self.profit_factor:>6.2f}",
            f"  Expectancy:      ₹{self.expectancy:>10,.2f}",
            "-" * 60,
            "RISK METRICS",
            f"  Max Drawdown:    ₹{self.max_drawdown:>10,.2f} ({self.max_drawdown_percent:.2f}%)",
            f"  Sharpe Ratio:     {self.sharpe_ratio:>6.2f}",
            f"  Sortino Ratio:    {self.sortino_ratio:>6.2f}",
            f"  Calmar Ratio:     {self.calmar_ratio:>6.2f}",
            "-" * 60,
            "COSTS",
            f"  Slippage:        ₹{self.total_slippage:>10,.2f}",
            f"  Commissions:     ₹{self.total_commissions:>10,.2f}",
            f"  Total Costs:     ₹{self.total_costs:>10,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Production backtesting engine with realistic simulation.

    Features:
    - Configurable slippage models
    - Commission/fee calculation
    - Position sizing using risk management
    - Trailing stop simulation
    - Detailed trade metrics (MFE/MAE)
    - Comprehensive performance statistics
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        settings: Optional[Settings] = None,
    ):
        self.initial_capital = initial_capital
        self.settings = settings or get_settings()
        self.position_sizer = PositionSizer(self.settings)

        # Slippage and commission settings
        self.slippage_percent = float(self.settings.risk.expected_slippage_percent)
        self.commission_per_order = float(self.settings.risk.brokerage_per_order)
        self.stt_percent = float(self.settings.risk.stt_percent)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "day",
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data (must have open, high, low, close, volume)
            symbol: Trading symbol
            timeframe: Candle timeframe

        Returns:
            BacktestResults with comprehensive statistics
        """
        # Validate data
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must have columns: {required_columns}")

        df = df.copy()
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Initialize engine
        engine = RivallandSwingEngine()

        # State tracking
        capital = self.initial_capital
        trades: List[BacktestTrade] = []
        equity_curve: List[Decimal] = [capital]
        position: Optional[Dict] = None

        # Statistics
        peak_capital = capital
        trough_capital = capital
        max_drawdown = Decimal(0)
        max_drawdown_bars = 0
        current_drawdown_bars = 0

        # Need enough bars for lookback
        start_idx = self.settings.strategy.swing_lookback * 2 + 10

        logger.info(f"Starting backtest for {symbol} with {len(df)} bars")

        for i in range(start_idx, len(df)):
            # Get data up to current bar
            current_df = df.iloc[:i + 1].copy()
            current_bar = df.iloc[i]
            current_date = df.index[i]

            # Reset engine for clean analysis (prevents look-ahead)
            engine.reset()

            # Run analysis
            signal = engine.analyze(current_df, symbol, timeframe)

            # Check existing position for exit
            if position is not None:
                exit_price, exit_reason = self._check_exit(
                    position, current_bar, current_date
                )

                if exit_price is not None:
                    # Close position
                    trade = self._close_position(
                        position, exit_price, exit_reason,
                        current_date, i - position['entry_idx']
                    )
                    trades.append(trade)
                    capital += trade.pnl
                    position = None

                    logger.debug(f"Trade closed: {trade.exit_reason} P&L: {trade.pnl}")

                else:
                    # Update MFE/MAE
                    self._update_excursions(position, current_bar)

            # Check for new entry (only if no position)
            if position is None and signal.signal_type != SignalType.HOLD:
                # Calculate position size
                quantity = self.position_sizer.calculate_risk_based_size(
                    capital=capital,
                    entry_price=signal.price,
                    stop_loss=signal.stop_loss,
                )

                if quantity > 0:
                    # Apply entry slippage
                    slippage = self._calculate_slippage(
                        signal.price,
                        "BUY" if signal.signal_type == SignalType.BUY else "SELL"
                    )
                    entry_price = signal.price + slippage

                    # Calculate commission
                    commission = self._calculate_commission(entry_price * quantity)

                    position = {
                        'side': signal.signal_type.value,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'stop_loss': signal.stop_loss,
                        'target': signal.target,
                        'entry_idx': i,
                        'entry_date': current_date,
                        'slippage': abs(slippage) * quantity,
                        'commission': commission,
                        'mfe': Decimal(0),
                        'mae': Decimal(0),
                        'highest_price': entry_price,
                        'lowest_price': entry_price,
                    }

                    capital -= commission  # Entry commission

                    logger.debug(f"Position opened: {signal.signal_type.value} {quantity}@{entry_price}")

            # Update equity curve
            current_equity = capital
            if position:
                unrealized = self._calculate_unrealized(position, current_bar)
                current_equity += unrealized

            equity_curve.append(current_equity)

            # Track peak/trough for drawdown
            if current_equity > peak_capital:
                peak_capital = current_equity
                current_drawdown_bars = 0
            else:
                current_drawdown_bars += 1
                drawdown = peak_capital - current_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_bars = current_drawdown_bars

            if current_equity < trough_capital:
                trough_capital = current_equity

        # Close any remaining position at last bar
        if position is not None:
            last_bar = df.iloc[-1]
            exit_price = Decimal(str(last_bar['close']))
            trade = self._close_position(
                position, exit_price, "END_OF_DATA",
                df.index[-1], len(df) - 1 - position['entry_idx']
            )
            trades.append(trade)
            capital += trade.pnl

        # Calculate final statistics
        results = self._calculate_results(
            trades=trades,
            equity_curve=equity_curve,
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[start_idx],
            end_date=df.index[-1],
            total_bars=len(df) - start_idx,
            initial_capital=self.initial_capital,
            final_capital=capital,
            peak_capital=peak_capital,
            trough_capital=trough_capital,
            max_drawdown=max_drawdown,
            max_drawdown_bars=max_drawdown_bars,
        )

        logger.info(f"Backtest complete: {len(trades)} trades, {results.total_return_percent:.2f}% return")

        return results

    def _check_exit(
        self,
        position: Dict,
        current_bar: pd.Series,
        current_date: datetime,
    ) -> Tuple[Optional[Decimal], str]:
        """Check if position should be exited."""
        high = Decimal(str(current_bar['high']))
        low = Decimal(str(current_bar['low']))

        if position['side'] == 'BUY':
            # Check stop loss
            if low <= position['stop_loss']:
                return position['stop_loss'], "STOP_LOSS"

            # Check target
            if high >= position['target']:
                return position['target'], "TARGET"

        else:  # SELL
            # Check stop loss
            if high >= position['stop_loss']:
                return position['stop_loss'], "STOP_LOSS"

            # Check target
            if low <= position['target']:
                return position['target'], "TARGET"

        return None, ""

    def _update_excursions(self, position: Dict, current_bar: pd.Series) -> None:
        """Update Maximum Favorable/Adverse Excursion."""
        high = Decimal(str(current_bar['high']))
        low = Decimal(str(current_bar['low']))

        if position['side'] == 'BUY':
            # MFE is highest high - entry
            if high > position['highest_price']:
                position['highest_price'] = high
                position['mfe'] = max(
                    position['mfe'],
                    (high - position['entry_price']) * position['quantity']
                )

            # MAE is entry - lowest low
            if low < position['lowest_price']:
                position['lowest_price'] = low
                position['mae'] = max(
                    position['mae'],
                    (position['entry_price'] - low) * position['quantity']
                )
        else:
            # For shorts, inverse
            if low < position['lowest_price']:
                position['lowest_price'] = low
                position['mfe'] = max(
                    position['mfe'],
                    (position['entry_price'] - low) * position['quantity']
                )

            if high > position['highest_price']:
                position['highest_price'] = high
                position['mae'] = max(
                    position['mae'],
                    (high - position['entry_price']) * position['quantity']
                )

    def _close_position(
        self,
        position: Dict,
        exit_price: Decimal,
        exit_reason: str,
        exit_date: datetime,
        holding_bars: int,
    ) -> BacktestTrade:
        """Close a position and create trade record."""
        # Apply exit slippage
        slippage = self._calculate_slippage(
            exit_price,
            "SELL" if position['side'] == 'BUY' else "BUY"
        )
        exit_price_with_slippage = exit_price + slippage

        # Calculate P&L
        if position['side'] == 'BUY':
            gross_pnl = (exit_price_with_slippage - position['entry_price']) * position['quantity']
        else:
            gross_pnl = (position['entry_price'] - exit_price_with_slippage) * position['quantity']

        # Exit commission
        exit_commission = self._calculate_commission(exit_price_with_slippage * position['quantity'])
        total_commission = position['commission'] + exit_commission
        total_slippage = position['slippage'] + abs(slippage) * position['quantity']

        net_pnl = gross_pnl - total_commission

        # Calculate return percent
        entry_value = position['entry_price'] * position['quantity']
        return_percent = (net_pnl / entry_value) * 100 if entry_value > 0 else Decimal(0)

        return BacktestTrade(
            entry_date=position['entry_date'],
            exit_date=exit_date,
            symbol="",  # Will be set by results
            side=position['side'],
            quantity=position['quantity'],
            entry_price=position['entry_price'],
            exit_price=exit_price_with_slippage,
            stop_loss=position['stop_loss'],
            target=position['target'],
            pnl=net_pnl,
            return_percent=return_percent,
            exit_reason=exit_reason,
            slippage_cost=total_slippage,
            commission_cost=total_commission,
            holding_bars=holding_bars,
            max_favorable_excursion=position['mfe'],
            max_adverse_excursion=position['mae'],
        )

    def _calculate_unrealized(self, position: Dict, current_bar: pd.Series) -> Decimal:
        """Calculate unrealized P&L."""
        current_price = Decimal(str(current_bar['close']))

        if position['side'] == 'BUY':
            return (current_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - current_price) * position['quantity']

    def _calculate_slippage(self, price: Decimal, side: str) -> Decimal:
        """Calculate slippage based on side."""
        slippage_amount = price * Decimal(str(self.slippage_percent / 100))

        if side == "BUY":
            return slippage_amount  # Pay more
        else:
            return -slippage_amount  # Receive less

    def _calculate_commission(self, trade_value: Decimal) -> Decimal:
        """Calculate commission and taxes."""
        brokerage = Decimal(str(self.commission_per_order))
        stt = trade_value * Decimal(str(self.stt_percent / 100))
        return brokerage + stt

    def _calculate_results(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[Decimal],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        total_bars: int,
        initial_capital: Decimal,
        final_capital: Decimal,
        peak_capital: Decimal,
        trough_capital: Decimal,
        max_drawdown: Decimal,
        max_drawdown_bars: int,
    ) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        # Trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = Decimal(win_count / total_trades * 100) if total_trades > 0 else Decimal(0)

        # Average win/loss
        avg_win = (
            sum(t.pnl for t in winning_trades) / win_count
            if win_count > 0 else Decimal(0)
        )
        avg_loss = (
            abs(sum(t.pnl for t in losing_trades)) / loss_count
            if loss_count > 0 else Decimal(0)
        )

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal(0)

        # Expectancy
        expectancy = (
            (Decimal(win_count / total_trades) * avg_win) -
            (Decimal(loss_count / total_trades) * avg_loss)
            if total_trades > 0 else Decimal(0)
        )

        # Returns
        total_return = final_capital - initial_capital
        total_return_percent = (total_return / initial_capital) * 100

        # CAGR
        years = (end_date - start_date).days / 365.25
        if years > 0 and final_capital > 0:
            cagr = (pow(float(final_capital / initial_capital), 1 / years) - 1) * 100
        else:
            cagr = 0
        cagr = Decimal(str(cagr))

        # Max drawdown percent
        max_dd_percent = (max_drawdown / peak_capital) * 100 if peak_capital > 0 else Decimal(0)

        # Risk ratios
        returns = [float(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
        returns_array = np.array(returns)

        # Sharpe (assuming 6% risk-free rate annually, ~0.023% daily)
        if len(returns) > 0:
            rf_daily = 0.06 / 252
            excess_returns = returns_array - rf_daily
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        else:
            sharpe = 0

        # Sortino (downside deviation only)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino = np.mean(returns_array) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0

        # Calmar ratio
        calmar = float(cagr / max_dd_percent) if max_dd_percent > 0 else 0

        # Costs
        total_slippage = sum(t.slippage_cost for t in trades)
        total_commissions = sum(t.commission_cost for t in trades)

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            elif trade.pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)

        # Average holding period
        avg_holding = (
            sum(t.holding_bars for t in trades) / total_trades
            if total_trades > 0 else 0
        )

        # MFE/MAE averages
        avg_mfe = (
            sum(t.max_favorable_excursion for t in trades) / total_trades
            if total_trades > 0 else Decimal(0)
        )
        avg_mae = (
            sum(t.max_adverse_excursion for t in trades) / total_trades
            if total_trades > 0 else Decimal(0)
        )

        # Update trades with symbol
        for trade in trades:
            trade.symbol = symbol

        return BacktestResults(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_bars=total_bars,
            initial_capital=initial_capital,
            final_capital=final_capital,
            peak_capital=peak_capital,
            trough_capital=trough_capital,
            total_return=total_return,
            total_return_percent=total_return_percent,
            cagr=cagr,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_dd_percent,
            max_drawdown_duration=max_drawdown_bars,
            sharpe_ratio=Decimal(str(sharpe)),
            sortino_ratio=Decimal(str(sortino)),
            calmar_ratio=Decimal(str(calmar)),
            total_slippage=total_slippage,
            total_commissions=total_commissions,
            total_costs=total_slippage + total_commissions,
            trades=trades,
            equity_curve=equity_curve,
            avg_holding_period=Decimal(str(avg_holding)),
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloSimulator:
    """
    Monte Carlo simulation for robustness testing.
    """

    def __init__(self, trades: List[BacktestTrade], initial_capital: Decimal):
        self.trades = trades
        self.initial_capital = initial_capital

    def run(
        self,
        num_simulations: int = 1000,
        num_trades: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation by randomizing trade order.

        Returns distribution of outcomes.
        """
        if num_trades is None:
            num_trades = len(self.trades)

        trade_pnls = [float(t.pnl) for t in self.trades]
        final_capitals = []

        for _ in range(num_simulations):
            # Random sample with replacement
            sampled_pnls = np.random.choice(trade_pnls, size=num_trades, replace=True)
            final_capital = float(self.initial_capital) + np.sum(sampled_pnls)
            final_capitals.append(final_capital)

        final_capitals = np.array(final_capitals)

        return {
            "mean_final_capital": float(np.mean(final_capitals)),
            "median_final_capital": float(np.median(final_capitals)),
            "std_dev": float(np.std(final_capitals)),
            "percentile_5": float(np.percentile(final_capitals, 5)),
            "percentile_25": float(np.percentile(final_capitals, 25)),
            "percentile_75": float(np.percentile(final_capitals, 75)),
            "percentile_95": float(np.percentile(final_capitals, 95)),
            "probability_of_profit": float(np.mean(final_capitals > float(self.initial_capital))),
            "probability_of_ruin": float(np.mean(final_capitals <= 0)),
            "worst_case": float(np.min(final_capitals)),
            "best_case": float(np.max(final_capitals)),
        }
