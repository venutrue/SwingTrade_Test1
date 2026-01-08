"""
Unit Tests for Backtesting Engine
==================================
Tests for backtest execution, trade recording, and results calculation.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtest.backtester import (
    Backtester, BacktestTrade, BacktestResults, MonteCarloSimulator
)
from src.config.settings import Settings


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Generate realistic price movement
    prices = [100]
    for _ in range(99):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'volume': [np.random.randint(100000, 500000) for _ in range(100)],
    }, index=dates)

    return df


@pytest.fixture
def uptrend_data():
    """Generate clear uptrend data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create consistent uptrend
    base = 100
    prices = [base + i * 0.5 + np.random.uniform(-0.5, 0.5) for i in range(100)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(1, 3) for p in prices],
        'low': [p - np.random.uniform(0.5, 1.5) for p in prices],
        'close': [p + np.random.uniform(-0.5, 1) for p in prices],
        'volume': [100000] * 100,
    }, index=dates)

    return df


@pytest.fixture
def downtrend_data():
    """Generate clear downtrend data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create consistent downtrend
    base = 150
    prices = [base - i * 0.5 + np.random.uniform(-0.5, 0.5) for i in range(100)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(0.5, 1.5) for p in prices],
        'low': [p - np.random.uniform(1, 3) for p in prices],
        'close': [p + np.random.uniform(-1, 0.5) for p in prices],
        'volume': [100000] * 100,
    }, index=dates)

    return df


@pytest.fixture
def backtester():
    """Create a Backtester instance."""
    return Backtester(initial_capital=Decimal("100000"))


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    base_date = datetime(2023, 1, 1)
    trades = [
        BacktestTrade(
            entry_date=base_date,
            exit_date=base_date + timedelta(days=5),
            symbol="TEST",
            side="BUY",
            quantity=100,
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            stop_loss=Decimal("95"),
            target=Decimal("115"),
            pnl=Decimal("1000"),
            return_percent=Decimal("10"),
            exit_reason="TARGET",
            holding_bars=5,
        ),
        BacktestTrade(
            entry_date=base_date + timedelta(days=10),
            exit_date=base_date + timedelta(days=15),
            symbol="TEST",
            side="BUY",
            quantity=100,
            entry_price=Decimal("105"),
            exit_price=Decimal("100"),
            stop_loss=Decimal("100"),
            target=Decimal("115"),
            pnl=Decimal("-500"),
            return_percent=Decimal("-4.76"),
            exit_reason="STOP_LOSS",
            holding_bars=5,
        ),
        BacktestTrade(
            entry_date=base_date + timedelta(days=20),
            exit_date=base_date + timedelta(days=25),
            symbol="TEST",
            side="BUY",
            quantity=100,
            entry_price=Decimal("102"),
            exit_price=Decimal("112"),
            stop_loss=Decimal("97"),
            target=Decimal("117"),
            pnl=Decimal("1000"),
            return_percent=Decimal("9.8"),
            exit_reason="TARGET",
            holding_bars=5,
        ),
    ]
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# TEST BACKTEST TRADE
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestTrade:
    """Tests for BacktestTrade dataclass."""

    def test_trade_to_dict(self, sample_trades):
        """Test trade serialization to dictionary."""
        trade = sample_trades[0]
        trade_dict = trade.to_dict()

        assert trade_dict["symbol"] == "TEST"
        assert trade_dict["side"] == "BUY"
        assert trade_dict["quantity"] == 100
        assert trade_dict["entry_price"] == 100.0
        assert trade_dict["exit_price"] == 110.0
        assert trade_dict["pnl"] == 1000.0
        assert trade_dict["exit_reason"] == "TARGET"

    def test_winning_trade_positive_pnl(self, sample_trades):
        """Test winning trade has positive P&L."""
        winning_trade = sample_trades[0]
        assert winning_trade.pnl > 0

    def test_losing_trade_negative_pnl(self, sample_trades):
        """Test losing trade has negative P&L."""
        losing_trade = sample_trades[1]
        assert losing_trade.pnl < 0

    def test_trade_has_required_fields(self, sample_trades):
        """Test trade has all required fields."""
        trade = sample_trades[0]

        assert trade.entry_date is not None
        assert trade.exit_date is not None
        assert trade.symbol is not None
        assert trade.side is not None
        assert trade.quantity > 0
        assert trade.entry_price > 0
        assert trade.exit_price is not None
        assert trade.stop_loss > 0
        assert trade.target > 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestResults:
    """Tests for BacktestResults dataclass."""

    @pytest.fixture
    def sample_results(self, sample_trades):
        """Create sample backtest results."""
        return BacktestResults(
            symbol="TEST",
            timeframe="day",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 4, 1),
            total_bars=90,
            initial_capital=Decimal("100000"),
            final_capital=Decimal("101500"),
            peak_capital=Decimal("102000"),
            trough_capital=Decimal("99500"),
            total_return=Decimal("1500"),
            total_return_percent=Decimal("1.5"),
            cagr=Decimal("6.1"),
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=Decimal("66.67"),
            avg_win=Decimal("1000"),
            avg_loss=Decimal("500"),
            profit_factor=Decimal("4.0"),
            expectancy=Decimal("500"),
            max_drawdown=Decimal("2000"),
            max_drawdown_percent=Decimal("1.96"),
            max_drawdown_duration=10,
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("3.1"),
            total_slippage=Decimal("50"),
            total_commissions=Decimal("100"),
            total_costs=Decimal("150"),
            trades=sample_trades,
            equity_curve=[Decimal("100000"), Decimal("101000"), Decimal("100500"), Decimal("101500")],
        )

    def test_results_to_dict(self, sample_results):
        """Test results serialization to dictionary."""
        results_dict = sample_results.to_dict()

        assert results_dict["symbol"] == "TEST"
        assert results_dict["timeframe"] == "day"
        assert "capital" in results_dict
        assert "returns" in results_dict
        assert "trades" in results_dict
        assert "risk" in results_dict
        assert "costs" in results_dict

    def test_print_summary(self, sample_results):
        """Test print summary generates readable output."""
        summary = sample_results.print_summary()

        assert "BACKTEST RESULTS" in summary
        assert "TEST" in summary
        assert "CAPITAL" in summary
        assert "RETURNS" in summary
        assert "TRADE STATISTICS" in summary
        assert "RISK METRICS" in summary

    def test_win_rate_calculation(self, sample_results):
        """Test win rate is calculated correctly."""
        expected_win_rate = (2 / 3) * 100
        assert abs(float(sample_results.win_rate) - expected_win_rate) < 0.1

    def test_profit_factor(self, sample_results):
        """Test profit factor is gross profit / gross loss."""
        # 2 wins @ 1000 = 2000 profit
        # 1 loss @ 500 = 500 loss
        # Profit factor = 2000 / 500 = 4.0
        assert sample_results.profit_factor == Decimal("4.0")


# ─────────────────────────────────────────────────────────────────────────────
# TEST BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktester:
    """Tests for Backtester class."""

    def test_backtester_initialization(self, backtester):
        """Test backtester initializes correctly."""
        assert backtester.initial_capital == Decimal("100000")
        assert backtester.slippage_percent >= 0
        assert backtester.commission_per_order >= 0

    def test_run_requires_ohlcv_columns(self, backtester):
        """Test run raises error without required columns."""
        df = pd.DataFrame({
            'price': [100, 101, 102],
        })

        with pytest.raises(ValueError) as exc_info:
            backtester.run(df, "TEST")

        assert "must have columns" in str(exc_info.value)

    def test_run_adds_volume_if_missing(self, backtester):
        """Test run adds volume column if missing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 50,
        }, index=dates)

        # Should not raise - volume will be added
        results = backtester.run(df, "TEST")
        assert results is not None

    def test_run_returns_results(self, backtester, sample_ohlcv_data):
        """Test run returns BacktestResults."""
        results = backtester.run(sample_ohlcv_data, "TEST", "day")

        assert isinstance(results, BacktestResults)
        assert results.symbol == "TEST"
        assert results.timeframe == "day"

    def test_run_tracks_equity_curve(self, backtester, sample_ohlcv_data):
        """Test run tracks equity curve."""
        results = backtester.run(sample_ohlcv_data, "TEST")

        assert len(results.equity_curve) > 0
        assert results.equity_curve[0] == backtester.initial_capital

    def test_slippage_calculation_buy(self, backtester):
        """Test slippage for buy order (pay more)."""
        price = Decimal("100")
        slippage = backtester._calculate_slippage(price, "BUY")

        assert slippage >= 0  # Buy pays more

    def test_slippage_calculation_sell(self, backtester):
        """Test slippage for sell order (receive less)."""
        price = Decimal("100")
        slippage = backtester._calculate_slippage(price, "SELL")

        assert slippage <= 0  # Sell receives less

    def test_commission_calculation(self, backtester):
        """Test commission includes brokerage and STT."""
        trade_value = Decimal("10000")
        commission = backtester._calculate_commission(trade_value)

        assert commission > 0

    def test_check_exit_stop_loss_long(self, backtester):
        """Test stop loss detection for long position."""
        position = {
            'side': 'BUY',
            'stop_loss': Decimal("95"),
            'target': Decimal("110"),
        }
        current_bar = pd.Series({
            'high': 98,
            'low': 94,  # Hits stop loss
        })

        exit_price, reason = backtester._check_exit(position, current_bar, datetime.now())

        assert exit_price == Decimal("95")
        assert reason == "STOP_LOSS"

    def test_check_exit_target_long(self, backtester):
        """Test target detection for long position."""
        position = {
            'side': 'BUY',
            'stop_loss': Decimal("95"),
            'target': Decimal("110"),
        }
        current_bar = pd.Series({
            'high': 112,  # Hits target
            'low': 108,
        })

        exit_price, reason = backtester._check_exit(position, current_bar, datetime.now())

        assert exit_price == Decimal("110")
        assert reason == "TARGET"

    def test_check_exit_stop_loss_short(self, backtester):
        """Test stop loss detection for short position."""
        position = {
            'side': 'SELL',
            'stop_loss': Decimal("105"),
            'target': Decimal("90"),
        }
        current_bar = pd.Series({
            'high': 106,  # Hits stop loss
            'low': 98,
        })

        exit_price, reason = backtester._check_exit(position, current_bar, datetime.now())

        assert exit_price == Decimal("105")
        assert reason == "STOP_LOSS"

    def test_check_exit_no_exit(self, backtester):
        """Test no exit when price stays within range."""
        position = {
            'side': 'BUY',
            'stop_loss': Decimal("95"),
            'target': Decimal("110"),
        }
        current_bar = pd.Series({
            'high': 105,
            'low': 98,
        })

        exit_price, reason = backtester._check_exit(position, current_bar, datetime.now())

        assert exit_price is None
        assert reason == ""

    def test_calculate_unrealized_long(self, backtester):
        """Test unrealized P&L for long position."""
        position = {
            'side': 'BUY',
            'entry_price': Decimal("100"),
            'quantity': 10,
        }
        current_bar = pd.Series({'close': 110})

        unrealized = backtester._calculate_unrealized(position, current_bar)

        assert unrealized == Decimal("100")  # 10 * (110 - 100)

    def test_calculate_unrealized_short(self, backtester):
        """Test unrealized P&L for short position."""
        position = {
            'side': 'SELL',
            'entry_price': Decimal("100"),
            'quantity': 10,
        }
        current_bar = pd.Series({'close': 95})

        unrealized = backtester._calculate_unrealized(position, current_bar)

        assert unrealized == Decimal("50")  # 10 * (100 - 95)


# ─────────────────────────────────────────────────────────────────────────────
# TEST MONTE CARLO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    @pytest.fixture
    def monte_carlo(self, sample_trades):
        """Create a MonteCarloSimulator instance."""
        return MonteCarloSimulator(
            trades=sample_trades,
            initial_capital=Decimal("100000")
        )

    def test_run_returns_statistics(self, monte_carlo):
        """Test run returns expected statistics."""
        results = monte_carlo.run(num_simulations=100)

        assert "mean_final_capital" in results
        assert "median_final_capital" in results
        assert "percentile_5" in results
        assert "percentile_95" in results
        assert "probability_of_profit" in results
        assert "probability_of_ruin" in results
        assert "worst_case" in results
        assert "best_case" in results

    def test_run_with_custom_num_trades(self, monte_carlo):
        """Test run with custom number of trades."""
        results = monte_carlo.run(num_simulations=50, num_trades=10)

        assert results is not None
        assert results["mean_final_capital"] > 0

    def test_probability_of_profit_range(self, monte_carlo):
        """Test probability of profit is in valid range."""
        results = monte_carlo.run(num_simulations=100)

        assert 0 <= results["probability_of_profit"] <= 1

    def test_percentiles_in_order(self, monte_carlo):
        """Test percentiles are in ascending order."""
        results = monte_carlo.run(num_simulations=100)

        assert results["percentile_5"] <= results["percentile_25"]
        assert results["percentile_25"] <= results["median_final_capital"]
        assert results["median_final_capital"] <= results["percentile_75"]
        assert results["percentile_75"] <= results["percentile_95"]

    def test_worst_best_case_bounds(self, monte_carlo):
        """Test worst/best case are bounds of distribution."""
        results = monte_carlo.run(num_simulations=100)

        assert results["worst_case"] <= results["percentile_5"]
        assert results["best_case"] >= results["percentile_95"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST EXCURSION TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class TestExcursionTracking:
    """Tests for MFE/MAE (Maximum Favorable/Adverse Excursion) tracking."""

    def test_update_excursions_long_mfe(self, backtester):
        """Test MFE updates for long position."""
        position = {
            'side': 'BUY',
            'entry_price': Decimal("100"),
            'quantity': 10,
            'highest_price': Decimal("100"),
            'lowest_price': Decimal("100"),
            'mfe': Decimal("0"),
            'mae': Decimal("0"),
        }

        # Price goes up
        current_bar = pd.Series({'high': 110, 'low': 99})
        backtester._update_excursions(position, current_bar)

        assert position['highest_price'] == Decimal("110")
        assert position['mfe'] == Decimal("100")  # (110 - 100) * 10

    def test_update_excursions_long_mae(self, backtester):
        """Test MAE updates for long position."""
        position = {
            'side': 'BUY',
            'entry_price': Decimal("100"),
            'quantity': 10,
            'highest_price': Decimal("100"),
            'lowest_price': Decimal("100"),
            'mfe': Decimal("0"),
            'mae': Decimal("0"),
        }

        # Price goes down
        current_bar = pd.Series({'high': 101, 'low': 95})
        backtester._update_excursions(position, current_bar)

        assert position['lowest_price'] == Decimal("95")
        assert position['mae'] == Decimal("50")  # (100 - 95) * 10

    def test_update_excursions_short_mfe(self, backtester):
        """Test MFE updates for short position."""
        position = {
            'side': 'SELL',
            'entry_price': Decimal("100"),
            'quantity': 10,
            'highest_price': Decimal("100"),
            'lowest_price': Decimal("100"),
            'mfe': Decimal("0"),
            'mae': Decimal("0"),
        }

        # Price goes down (favorable for short)
        current_bar = pd.Series({'high': 101, 'low': 90})
        backtester._update_excursions(position, current_bar)

        assert position['lowest_price'] == Decimal("90")
        assert position['mfe'] == Decimal("100")  # (100 - 90) * 10


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
