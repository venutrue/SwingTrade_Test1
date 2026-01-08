# Rivalland Prop Trading System - Operational Runbook

## Table of Contents
1. [System Overview](#system-overview)
2. [Pre-Flight Checklist](#pre-flight-checklist)
3. [Starting the System](#starting-the-system)
4. [Daily Operations](#daily-operations)
5. [Monitoring](#monitoring)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Emergency Procedures](#emergency-procedures)
8. [Exchange-Specific Notes](#exchange-specific-notes)

---

## System Overview

### Components
- **Trading Engine**: Rivalland Swing methodology implementation
- **Order Manager**: State machine for order lifecycle (PENDING -> SUBMITTED -> OPEN -> FILLED)
- **Risk Manager**: Position sizing, circuit breakers, kill switch
- **Broker Connectors**: Zerodha Kite (NSE/BSE), Alpaca (NYSE/NASDAQ)
- **Data Providers**: Yahoo Finance (free), Polygon (premium)
- **Market Calendar**: Exchange-specific trading hours and holidays

### Architecture
```
main.py
    -> TradingOrchestrator
        -> RivallandSwingEngine (signal generation)
        -> OrderManager (order lifecycle)
        -> RiskManager (position sizing, limits)
        -> Broker (Kite/Alpaca)
        -> MarketCalendar (trading hours)
```

---

## Pre-Flight Checklist

### Before Going Live

- [ ] **Environment Configuration**
  - `.env` file has correct API keys
  - `ENVIRONMENT=production` only when ready
  - `TRADING_MODE=paper` for testing

- [ ] **Broker Connection**
  - API key and secret configured
  - Access token valid (Kite requires daily login)
  - Funds available in trading account

- [ ] **Risk Settings Verified**
  - `MAX_RISK_PER_TRADE=1.0` (1% per trade)
  - `MAX_DAILY_LOSS=3.0` (3% daily limit)
  - `MAX_POSITIONS=5` (concurrent positions)

- [ ] **Test Suite Passed**
  ```bash
  python -m pytest tests/ -v
  ```
  All 154 tests should pass.

- [ ] **Paper Trading Period Complete**
  - Minimum 3-6 months of paper trading
  - Positive expectancy verified on historical data
  - Strategy parameters validated

---

## Starting the System

### Paper Trading Mode
```bash
# Start paper trading
python main.py paper

# With debug logging
python main.py paper --debug
```

### Live Trading Mode
```bash
# CAUTION: Uses real money
python main.py live
```

### Backtest Mode
```bash
# Run backtest with sample data
python main.py backtest

# Custom parameters
python main.py backtest --symbol AAPL --days 365 --capital 100000
```

### Check System Status
```bash
python main.py status
```

---

## Daily Operations

### Morning Routine (Before Market Open)

1. **Verify System Health**
   ```bash
   python main.py status
   ```

2. **Check Broker Connection**
   - For Kite: Generate fresh access token via login
   - For Alpaca: Token is persistent

3. **Review Previous Day's Trades**
   - Check `logs/trading_YYYYMMDD.log`
   - Verify all positions closed properly

4. **Verify Market Calendar**
   - Check for holidays (no trading)
   - Special market hours (early close days)

### During Trading Hours

- **Monitor logs**: `tail -f logs/trading_$(date +%Y%m%d).log`
- **Watch for circuit breaker triggers**
- **Monitor open positions via broker dashboard**

### End of Day Routine

1. **Verify all positions status**
2. **Review day's P&L**
3. **Check for any ERROR logs**
4. **Backup trading database**

---

## Monitoring

### Key Metrics to Watch

| Metric | Warning Level | Critical Level |
|--------|---------------|----------------|
| Daily Loss | 2% | 3% (circuit breaker) |
| Max Drawdown | 5% | 10% |
| Consecutive Losses | 3 | 5 (kill switch) |
| Open Positions | 4 | 5 (max) |

### Log Locations
- Trading logs: `logs/trading_YYYYMMDD.log`
- Error logs: `logs/error_YYYYMMDD.log`
- Backtest results: `backtest_results_*.json`

### Alert Channels
- Telegram notifications (if configured)
- Email alerts for critical events

---

## Troubleshooting Guide

### Common Issues

#### 1. "Order rejected by broker"
**Symptoms**: Orders failing to submit
**Causes**:
- Insufficient funds
- Market closed
- Invalid symbol
- Position limits reached

**Resolution**:
1. Check available margin/funds
2. Verify market hours
3. Confirm symbol is tradeable
4. Check position count

#### 2. "Circuit breaker triggered"
**Symptoms**: Trading halted automatically
**Causes**:
- Daily loss limit reached (3%)
- Max drawdown exceeded

**Resolution**:
1. Wait for cooldown period (default: 30 minutes)
2. Review what caused the losses
3. Manually reset if appropriate:
   ```python
   # In Python console
   from src.risk.risk_manager import CircuitBreaker
   await circuit_breaker.reset()
   ```

#### 3. "Connection lost to broker"
**Symptoms**: No order updates, stale data
**Causes**:
- Network issues
- Broker API downtime
- Token expired

**Resolution**:
1. Check internet connectivity
2. Verify broker status page
3. For Kite: Re-authenticate with fresh token
4. Restart the application

#### 4. "Kill switch activated"
**Symptoms**: All trading stopped, positions closed
**Causes**:
- Manual activation
- Critical system error
- Extreme market conditions

**Resolution**:
1. Review activation reason in logs
2. Ensure market conditions are normal
3. Deactivate only when safe:
   ```python
   await kill_switch.deactivate()
   ```

#### 5. "No signals generated"
**Symptoms**: System running but no trades
**Causes**:
- Market in ranging mode (no clear trend)
- Insufficient historical data
- Symbol not trading

**Resolution**:
1. Check engine state: `engine.get_state()`
2. Verify data feed is active
3. Review trend detection thresholds

### Database Issues

#### Reset Trading Database
```bash
# CAUTION: Deletes all trading history
rm trading.db
python main.py status  # Recreates database
```

#### Backup Database
```bash
cp trading.db trading_backup_$(date +%Y%m%d).db
```

---

## Emergency Procedures

### Immediate Position Close (Kill Switch)

**When to use**: Emergency market conditions, system malfunction

```python
# Via code
from src.risk.risk_manager import KillSwitch
results = await kill_switch.activate("Emergency market conditions")
print(f"Closed {results['positions_closed']} positions")
```

**Or manually via broker**:
1. Log into broker dashboard
2. Navigate to positions
3. Close all positions manually

### Manual Order Cancellation

```bash
# Via broker CLI (if available)
# Or log into broker web interface and cancel pending orders
```

### System Shutdown

1. **Graceful shutdown**: Press Ctrl+C
2. **Verify positions**: Check broker dashboard
3. **Cancel pending orders**: Via broker interface
4. **Document reason**: Log why shutdown was needed

---

## Exchange-Specific Notes

### NSE/BSE (India) - Zerodha Kite

**Trading Hours**: 9:15 AM - 3:30 PM IST (Mon-Fri)

**Holidays 2025**:
- Jan 26 (Republic Day)
- Mar 14 (Holi)
- Apr 18 (Good Friday)
- Aug 15 (Independence Day)
- Oct 2 (Gandhi Jayanti)
- Oct 22 (Diwali)
- Dec 25 (Christmas)

**Authentication**:
- Daily token refresh required
- Login via Kite Connect web flow
- Store access token securely

**Fees**:
- Brokerage: Rs 20 per order (Equity delivery: Free)
- STT: 0.1% on sell value
- Exchange charges: ~0.003%

### NYSE/NASDAQ (US) - Alpaca

**Trading Hours**: 9:30 AM - 4:00 PM ET (Mon-Fri)

**Holidays 2025**:
- Jan 1 (New Year)
- Jan 20 (MLK Day)
- Feb 17 (Presidents Day)
- Apr 18 (Good Friday)
- May 26 (Memorial Day)
- Jul 4 (Independence Day)
- Sep 1 (Labor Day)
- Nov 27 (Thanksgiving)
- Dec 25 (Christmas)

**Authentication**:
- API keys are persistent (no daily refresh)
- Paper trading available with separate keys

**Fees** (Alpaca):
- Commission: $0
- SEC fee: 0.0008% on sell
- FINRA fee: 0.000119% on sell

---

## Contact Information

### Emergency Contacts
- Broker Support (Zerodha): support@zerodha.com
- Broker Support (Alpaca): support@alpaca.markets
- System Administrator: [Your contact]

### Useful Links
- [Zerodha Kite Status](https://status.zerodha.com)
- [Alpaca Status](https://status.alpaca.markets)
- [NSE Holidays](https://www.nseindia.com/national-stock-exchange/holidays-clearing-settlement)
- [NYSE Calendar](https://www.nyse.com/markets/hours-calendars)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025 | Multi-exchange support, comprehensive testing |
| 1.0.0 | 2024 | Initial release (NSE only) |
