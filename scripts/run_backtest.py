#!/usr/bin/env python3
"""
Production Backtest Runner
==========================
Downloads real market data and runs comprehensive backtests.

Usage:
    python scripts/run_backtest.py --market us --years 5
    python scripts/run_backtest.py --market india --symbols RELIANCE,TCS
    python scripts/run_backtest.py --symbol AAPL --years 10
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchanges.base import ExchangeCode
from src.data.providers.yahoo import (
    YahooFinanceProvider, US_BLUE_CHIPS, INDIA_BLUE_CHIPS,
    get_stock_data, download_historical_data
)
from src.backtest.backtester import Backtester, BacktestResults, MonteCarloSimulator
from src.config.settings import get_settings


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class BacktestRunner:
    """
    Runs backtests on real market data.
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        market: str = "us",
    ):
        self.initial_capital = initial_capital
        self.market = market.lower()
        self.exchange = ExchangeCode.NYSE if market == "us" else ExchangeCode.NSE
        self.results: Dict[str, BacktestResults] = {}

    async def download_data(
        self,
        symbols: List[str],
        years: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """Download historical data for symbols."""
        logger.info(f"Downloading {years} years of data for {len(symbols)} symbols...")

        end = datetime.now()
        start = end - timedelta(days=years * 365)

        data = await download_historical_data(
            symbols=symbols,
            start=start,
            end=end,
            interval="1day",
            exchange=self.exchange,
        )

        logger.info(f"Downloaded data for {len(data)} symbols")

        # Log data summary
        for symbol, df in data.items():
            if not df.empty:
                logger.info(
                    f"  {symbol}: {len(df)} bars, "
                    f"{df.index[0].date()} to {df.index[-1].date()}, "
                    f"Price: {df['close'].iloc[-1]:.2f}"
                )

        return data

    def run_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Optional[BacktestResults]:
        """Run backtest on a single symbol."""
        if df.empty or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        logger.info(f"Running backtest for {symbol}...")

        backtester = Backtester(initial_capital=self.initial_capital)
        results = backtester.run(df, symbol, "day")

        self.results[symbol] = results
        return results

    async def run_all(
        self,
        symbols: List[str],
        years: int = 5,
    ) -> Dict[str, BacktestResults]:
        """Run backtests for all symbols."""
        # Download data
        data = await self.download_data(symbols, years)

        # Run backtests
        for symbol, df in data.items():
            self.run_backtest(symbol, df)

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive backtest report."""
        if not self.results:
            return "No backtest results available."

        lines = [
            "=" * 80,
            "COMPREHENSIVE BACKTEST REPORT",
            f"Market: {self.market.upper()} | Initial Capital: ${self.initial_capital:,.2f}",
            "=" * 80,
            "",
        ]

        # Summary table
        lines.append("SYMBOL PERFORMANCE SUMMARY")
        lines.append("-" * 80)
        lines.append(
            f"{'Symbol':<10} {'Trades':>7} {'Win%':>7} {'Return%':>10} "
            f"{'Sharpe':>8} {'MaxDD%':>8} {'PF':>6}"
        )
        lines.append("-" * 80)

        total_return = Decimal(0)
        total_trades = 0
        total_wins = 0

        for symbol, r in sorted(
            self.results.items(),
            key=lambda x: x[1].total_return_percent,
            reverse=True
        ):
            lines.append(
                f"{symbol:<10} {r.total_trades:>7} {r.win_rate:>6.1f}% "
                f"{r.total_return_percent:>9.2f}% {r.sharpe_ratio:>8.2f} "
                f"{r.max_drawdown_percent:>7.2f}% {r.profit_factor:>6.2f}"
            )
            total_return += r.total_return_percent
            total_trades += r.total_trades
            total_wins += r.winning_trades

        lines.append("-" * 80)

        # Aggregate statistics
        avg_return = total_return / len(self.results) if self.results else Decimal(0)
        avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        lines.extend([
            "",
            "AGGREGATE STATISTICS",
            "-" * 40,
            f"Symbols Tested:        {len(self.results)}",
            f"Total Trades:          {total_trades}",
            f"Average Return:        {avg_return:.2f}%",
            f"Average Win Rate:      {avg_win_rate:.1f}%",
            "",
        ])

        # Best and worst performers
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].total_return_percent,
            reverse=True
        )

        if sorted_results:
            best = sorted_results[0]
            worst = sorted_results[-1]

            lines.extend([
                "BEST PERFORMER",
                "-" * 40,
                f"Symbol: {best[0]}",
                f"Return: {best[1].total_return_percent:.2f}%",
                f"Trades: {best[1].total_trades} (Win Rate: {best[1].win_rate:.1f}%)",
                f"Sharpe: {best[1].sharpe_ratio:.2f}",
                "",
                "WORST PERFORMER",
                "-" * 40,
                f"Symbol: {worst[0]}",
                f"Return: {worst[1].total_return_percent:.2f}%",
                f"Trades: {worst[1].total_trades} (Win Rate: {worst[1].win_rate:.1f}%)",
                f"Sharpe: {worst[1].sharpe_ratio:.2f}",
                "",
            ])

        # Strategy verdict
        profitable_count = sum(1 for r in self.results.values() if r.total_return_percent > 0)
        profitable_pct = profitable_count / len(self.results) * 100 if self.results else 0

        lines.extend([
            "=" * 80,
            "STRATEGY VERDICT",
            "=" * 80,
            f"Profitable Symbols: {profitable_count}/{len(self.results)} ({profitable_pct:.1f}%)",
            f"Average Return: {avg_return:.2f}%",
            "",
        ])

        # Verdict based on results
        if avg_return > 10 and profitable_pct > 60:
            verdict = "✅ POSITIVE EXPECTANCY - Strategy shows promise"
        elif avg_return > 0 and profitable_pct > 50:
            verdict = "⚠️ MARGINAL - Strategy needs optimization"
        else:
            verdict = "❌ NEGATIVE EXPECTANCY - Strategy not profitable"

        lines.append(verdict)
        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_results(self, output_dir: str = "backtest_results") -> str:
        """Save results to files."""
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary JSON
        summary = {
            "timestamp": timestamp,
            "market": self.market,
            "initial_capital": float(self.initial_capital),
            "symbols": list(self.results.keys()),
            "results": {
                symbol: {
                    "total_return_percent": float(r.total_return_percent),
                    "total_trades": r.total_trades,
                    "win_rate": float(r.win_rate),
                    "sharpe_ratio": float(r.sharpe_ratio),
                    "max_drawdown_percent": float(r.max_drawdown_percent),
                    "profit_factor": float(r.profit_factor),
                }
                for symbol, r in self.results.items()
            }
        }

        summary_file = Path(output_dir) / f"backtest_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed report
        report_file = Path(output_dir) / f"backtest_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(self.generate_report())

        # Save individual symbol results
        for symbol, r in self.results.items():
            symbol_file = Path(output_dir) / f"{symbol}_{timestamp}.json"
            with open(symbol_file, "w") as f:
                json.dump(r.to_dict(), f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}/")
        return str(summary_file)


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo_analysis(
    results: Dict[str, BacktestResults],
    num_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation on aggregated results.
    """
    # Combine all trades
    all_trades = []
    total_capital = Decimal(0)

    for r in results.values():
        all_trades.extend(r.trades)
        total_capital = max(total_capital, r.initial_capital)

    if len(all_trades) < 10:
        return {"error": "Insufficient trades for Monte Carlo simulation"}

    simulator = MonteCarloSimulator(all_trades, total_capital)
    return simulator.run(num_simulations=num_simulations)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Run production backtests")

    parser.add_argument(
        "--market",
        choices=["us", "india"],
        default="us",
        help="Market to backtest (us or india)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: blue chips)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Single symbol to backtest"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of historical data (default: 5)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo simulation"
    )

    args = parser.parse_args()

    # Determine symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        # Use default blue chips
        symbols = US_BLUE_CHIPS[:10] if args.market == "us" else INDIA_BLUE_CHIPS[:10]

    print("=" * 60)
    print("RIVALLAND SWING TRADING - PRODUCTION BACKTEST")
    print("=" * 60)
    print(f"Market:    {args.market.upper()}")
    print(f"Symbols:   {', '.join(symbols)}")
    print(f"Years:     {args.years}")
    print(f"Capital:   ${args.capital:,.2f}")
    print("=" * 60)
    print()

    # Run backtest
    runner = BacktestRunner(
        initial_capital=Decimal(str(args.capital)),
        market=args.market,
    )

    results = await runner.run_all(symbols, args.years)

    # Print report
    print()
    print(runner.generate_report())

    # Monte Carlo simulation
    if args.monte_carlo and results:
        print()
        print("MONTE CARLO SIMULATION (1000 iterations)")
        print("-" * 40)

        mc_results = run_monte_carlo_analysis(results)

        if "error" not in mc_results:
            print(f"Mean Final Capital:    ${mc_results['mean_final_capital']:,.2f}")
            print(f"Median Final Capital:  ${mc_results['median_final_capital']:,.2f}")
            print(f"5th Percentile:        ${mc_results['percentile_5']:,.2f}")
            print(f"95th Percentile:       ${mc_results['percentile_95']:,.2f}")
            print(f"Probability of Profit: {mc_results['probability_of_profit']*100:.1f}%")
            print(f"Worst Case:            ${mc_results['worst_case']:,.2f}")
            print(f"Best Case:             ${mc_results['best_case']:,.2f}")
        else:
            print(f"Error: {mc_results['error']}")

    # Save results
    if results:
        summary_file = runner.save_results(args.output)
        print()
        print(f"Results saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
