#!/usr/bin/env python3
"""
Rivalland Prop Trading System - Main Entry Point
=================================================

Usage:
    python main.py live          # Run live trading
    python main.py paper         # Run paper trading
    python main.py backtest      # Run backtest
    python main.py status        # Show system status

Environment variables:
    KITE_API_KEY        - Zerodha Kite API key
    KITE_API_SECRET     - Zerodha Kite API secret
    KITE_ACCESS_TOKEN   - Zerodha access token (optional)
    TRADING_MODE        - 'live', 'paper', or 'backtest'
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings, reload_settings, TradingMode, Environment
from src.core.orchestrator import TradingOrchestrator
from src.core.engine import RivallandSwingEngine
from src.backtest.backtester import Backtester, MonteCarloSimulator


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
            ),
        ],
    )

    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

async def run_live() -> None:
    """Run live trading."""
    settings = get_settings()

    if settings.trading_mode != TradingMode.LIVE:
        settings.trading_mode = TradingMode.LIVE
        settings.environment = Environment.PRODUCTION

    print("=" * 60)
    print("RIVALLAND PROP TRADING SYSTEM - LIVE MODE")
    print("=" * 60)
    print(f"Symbols: {', '.join(settings.symbols)}")
    print(f"Timeframe: {settings.timeframe}")
    print(f"Max Risk/Trade: {settings.risk.max_risk_per_trade_percent}%")
    print(f"Max Positions: {settings.risk.max_open_positions}")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This is LIVE trading with REAL money!")
    print("Press Ctrl+C to stop")
    print()

    orchestrator = TradingOrchestrator(settings)
    await orchestrator.run()


async def run_paper() -> None:
    """Run paper trading."""
    settings = get_settings()
    settings.trading_mode = TradingMode.PAPER
    settings.environment = Environment.PAPER

    print("=" * 60)
    print("RIVALLAND PROP TRADING SYSTEM - PAPER MODE")
    print("=" * 60)
    print(f"Symbols: {', '.join(settings.symbols)}")
    print(f"Timeframe: {settings.timeframe}")
    print(f"Initial Capital: ₹1,000,000")
    print("=" * 60)
    print()
    print("📝 Running in PAPER mode - No real trades")
    print("Press Ctrl+C to stop")
    print()

    orchestrator = TradingOrchestrator(settings)
    await orchestrator.run()


def run_backtest(
    symbol: str = "SAMPLE",
    days: int = 365,
    initial_capital: float = 100000,
) -> None:
    """Run backtest with sample or real data."""
    print("=" * 60)
    print("RIVALLAND PROP TRADING SYSTEM - BACKTEST")
    print("=" * 60)

    # Generate sample data if no real data available
    print(f"Generating {days} days of sample data...")

    np.random.seed(42)
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days,
        freq='D'
    )

    # Generate realistic price series with trend and volatility
    price = 100.0
    trend = 0.0005  # Slight upward bias
    volatility = 0.02

    data = []
    for i in range(days):
        # Random walk with trend
        returns = np.random.normal(trend, volatility)
        price *= (1 + returns)

        open_p = price * (1 + np.random.uniform(-0.005, 0.005))
        high = max(open_p, price) * (1 + np.random.uniform(0, 0.015))
        low = min(open_p, price) * (1 - np.random.uniform(0, 0.015))
        close = price
        volume = np.random.randint(100000, 1000000)

        data.append([open_p, high, low, close, volume])

    df = pd.DataFrame(
        data,
        index=dates,
        columns=['open', 'high', 'low', 'close', 'volume']
    )

    print(f"Data generated: {len(df)} bars")
    print(f"Price range: ₹{df['low'].min():.2f} - ₹{df['high'].max():.2f}")
    print()

    # Run backtest
    print("Running backtest...")
    backtester = Backtester(initial_capital=Decimal(str(initial_capital)))
    results = backtester.run(df, symbol, "day")

    # Print results
    print(results.print_summary())

    # Run Monte Carlo simulation
    if results.total_trades >= 10:
        print("\nRunning Monte Carlo simulation (1000 iterations)...")
        mc = MonteCarloSimulator(results.trades, results.initial_capital)
        mc_results = mc.run(num_simulations=1000)

        print("\nMONTE CARLO RESULTS")
        print("-" * 40)
        print(f"Mean Final Capital:    ₹{mc_results['mean_final_capital']:,.2f}")
        print(f"Median Final Capital:  ₹{mc_results['median_final_capital']:,.2f}")
        print(f"5th Percentile:        ₹{mc_results['percentile_5']:,.2f}")
        print(f"95th Percentile:       ₹{mc_results['percentile_95']:,.2f}")
        print(f"Probability of Profit: {mc_results['probability_of_profit']*100:.1f}%")
        print(f"Worst Case:            ₹{mc_results['worst_case']:,.2f}")
        print(f"Best Case:             ₹{mc_results['best_case']:,.2f}")

    # Save results
    results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


def show_status() -> None:
    """Show current system status."""
    settings = get_settings()

    print("=" * 60)
    print("RIVALLAND PROP TRADING SYSTEM - STATUS")
    print("=" * 60)
    print()
    print("Configuration")
    print("-" * 40)
    print(f"Environment:     {settings.environment.value}")
    print(f"Trading Mode:    {settings.trading_mode.value}")
    print(f"Symbols:         {', '.join(settings.symbols)}")
    print(f"Timeframe:       {settings.timeframe}")
    print()
    print("Risk Settings")
    print("-" * 40)
    print(f"Max Risk/Trade:  {settings.risk.max_risk_per_trade_percent}%")
    print(f"Max Daily Loss:  {settings.risk.max_daily_loss_percent}%")
    print(f"Max Positions:   {settings.risk.max_open_positions}")
    print(f"Min R:R Ratio:   {settings.risk.min_risk_reward_ratio}")
    print()
    print("Strategy Settings")
    print("-" * 40)
    print(f"Swing Lookback:  {settings.strategy.swing_lookback} bars")
    print(f"Min Pullback:    {settings.strategy.min_pullback_bars} bars")
    print(f"Trailing Stop:   {'Enabled' if settings.strategy.enable_trailing_stop else 'Disabled'}")
    print()
    print("Broker")
    print("-" * 40)
    print(f"API Key:         {'*' * 8 + settings.broker.api_key[-4:] if settings.broker.api_key else 'Not set'}")
    print(f"Access Token:    {'Set' if settings.broker.access_token else 'Not set'}")
    print()
    print("Notifications")
    print("-" * 40)
    print(f"Telegram:        {'Enabled' if settings.notification.telegram_enabled else 'Disabled'}")
    print(f"Email:           {'Enabled' if settings.notification.email_enabled else 'Disabled'}")


def analyze_symbol(symbol: str) -> None:
    """Run single symbol analysis."""
    print(f"Analyzing {symbol}...")

    # Generate sample data
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

    price = 100.0
    data = []
    for _ in range(100):
        open_p = price
        high = price * (1 + np.random.uniform(0, 0.03))
        low = price * (1 - np.random.uniform(0, 0.03))
        close = price + np.random.uniform(-2, 2)
        price = close
        volume = np.random.randint(100000, 500000)
        data.append([open_p, high, low, close, volume])

    df = pd.DataFrame(data, index=dates, columns=['open', 'high', 'low', 'close', 'volume'])

    # Run analysis
    engine = RivallandSwingEngine()
    signal = engine.analyze(df, symbol, "day")

    print()
    print("=" * 50)
    print(f"ANALYSIS RESULTS: {symbol}")
    print("=" * 50)
    print(f"Signal:      {signal.signal_type.value}")
    print(f"Trend:       {signal.trend.value}")
    print(f"Price:       ₹{signal.price:.2f}")

    if signal.signal_type.value != "HOLD":
        print(f"Stop Loss:   ₹{signal.stop_loss:.2f}")
        print(f"Target:      ₹{signal.target:.2f}")
        print(f"Confidence:  {float(signal.confidence) * 100:.0f}%")

    print(f"Reason:      {signal.reason}")
    print()

    state = engine.get_state()
    print("Engine State:")
    print(f"  Swing Highs:   {state['swing_highs_count']}")
    print(f"  Swing Lows:    {state['swing_lows_count']}")
    print(f"  In Pullback:   {state['in_pullback']} ({state['pullback_bars']} bars)")
    print(f"  In Rally:      {state['in_rally']} ({state['rally_bars']} bars)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rivalland Prop Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Live trading
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Paper trading
    paper_parser = subparsers.add_parser("paper", help="Run paper trading")
    paper_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbol", default="SAMPLE", help="Symbol to backtest")
    backtest_parser.add_argument("--days", type=int, default=365, help="Number of days")
    backtest_parser.add_argument("--capital", type=float, default=100000, help="Initial capital")

    # Status
    subparsers.add_parser("status", help="Show system status")

    # Analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a symbol")
    analyze_parser.add_argument("symbol", help="Symbol to analyze")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if getattr(args, "debug", False) else "INFO"
    setup_logging(log_level)

    # Execute command
    if args.command == "live":
        asyncio.run(run_live())

    elif args.command == "paper":
        asyncio.run(run_paper())

    elif args.command == "backtest":
        run_backtest(
            symbol=args.symbol,
            days=args.days,
            initial_capital=args.capital
        )

    elif args.command == "status":
        show_status()

    elif args.command == "analyze":
        analyze_symbol(args.symbol)

    else:
        parser.print_help()
        print("\n📊 Rivalland Prop Trading System v2.0")
        print("Use one of the commands above to get started.")


if __name__ == "__main__":
    main()
