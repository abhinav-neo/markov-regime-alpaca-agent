"""
Multi-Asset Markov Regime Trading Agent
Alpaca Paper Trading (alpaca-py)
Target: Aggressive exponential growth with controlled risk-of-ruin
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, CryptoBarsRequest, OptionChainRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed

# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent.log")]
)
logger = logging.getLogger("MarkovAgent")

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = True  # Hard-coded for safety

# Universe
STOCK_SYMBOLS = ["SPY", "QQQ", "IWM"]          # Liquid large-caps for regime proxy
CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD"]
OPTION_UNDERLYINGS = ["SPY"]                   # Options only when premium is attractive

INITIAL_CAPITAL = 100.0                        # Simulation unit
MAX_RISK_PER_TRADE = 0.08                      # Hard cap even on aggressive Kelly
STOP_LOSS_ATR_MULT = 1.8                       # Tight stop protocol
LOOKBACK_BARS = 60                             # For regime estimation

# ---------------------------------------------------------------------------
# Markov Regime Model (Discrete-Time)
# ---------------------------------------------------------------------------
REGIME_NAMES = [
    "LowVol_Bull",   # 0
    "HighVol_Bull",  # 1
    "LowVol_Bear",   # 2
    "HighVol_Bear",  # 3
]

class MarkovRegimeModel:
    """
    Discrete-time Markov Chain for market regime switching.
    States inferred from rolling return sign + realized volatility quantile.
    Transition matrix is estimated online from observed state sequence.
    """
    def __init__(self, n_states: int = 4, alpha: float = 0.05):
        self.n_states = n_states
        self.alpha = alpha  # Laplace smoothing
        self.transition_counts = np.ones((n_states, n_states)) * alpha
        self.current_state: int = 0
        self.state_history: deque = deque(maxlen=500)
        self.return_buffer: deque = deque(maxlen=LOOKBACK_BARS)
        self.vol_buffer: deque = deque(maxlen=LOOKBACK_BARS)

    def _classify_state(self, ret: float, vol: float, vol_threshold: float) -> int:
        is_bull = ret > 0
        is_high_vol = vol > vol_threshold
        if is_bull and not is_high_vol:
            return 0
        elif is_bull and is_high_vol:
            return 1
        elif not is_bull and not is_high_vol:
            return 2
        else:
            return 3

    def update(self, latest_return: float, latest_vol: float) -> int:
        """Update buffers, classify new state, update transition matrix."""
        self.return_buffer.append(latest_return)
        self.vol_buffer.append(latest_vol)

        if len(self.vol_buffer) < 20:
            return self.current_state

        vol_threshold = np.percentile(list(self.vol_buffer), 60)
        new_state = self._classify_state(latest_return, latest_vol, vol_threshold)

        # Update transition counts
        if self.state_history:
            prev = self.state_history[-1]
            self.transition_counts[prev, new_state] += 1

        self.state_history.append(new_state)
        self.current_state = new_state
        return new_state

    @property
    def transition_matrix(self) -> np.ndarray:
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        return self.transition_counts / row_sums

    def next_state_probabilities(self) -> np.ndarray:
        return self.transition_matrix[self.current_state]

    def expected_edge(self) -> float:
        """Simple directional bias derived from transition probabilities."""
        probs = self.next_state_probabilities()
        # Bullish mass - bearish mass
        bull_mass = probs[0] + probs[1]
        bear_mass = probs[2] + probs[3]
        return bull_mass - bear_mass

# ---------------------------------------------------------------------------
# Dynamic Fractional Kelly
# ---------------------------------------------------------------------------
class KellySizer:
    """
    Fractional Kelly with aggressive bias for micro/paper capital,
    hard-capped and protected by stop-loss protocol.
    """
    def __init__(self, fraction: float = 0.35, max_fraction: float = 0.5):
        self.fraction = fraction          # Fractional Kelly (aggressive but not full)
        self.max_fraction = max_fraction

    def size(self, edge: float, win_prob: float, payoff_ratio: float,
             equity: float, stop_distance_pct: float) -> float:
        """
        edge ~ (p*(b+1) - 1) classic Kelly form.
        We map regime edge into a conservative p and b.
        """
        if edge <= 0.05 or win_prob <= 0.5:
            return 0.0

        # Classic Kelly: f* = (p*b - q) / b
        q = 1 - win_prob
        kelly_full = (win_prob * payoff_ratio - q) / payoff_ratio
        kelly_full = max(0.0, kelly_full)

        f = min(self.fraction * kelly_full, self.max_fraction)
        f = min(f, MAX_RISK_PER_TRADE)

        # Risk-based notional: risk only stop_distance_pct of position
        if stop_distance_pct <= 0:
            return 0.0
        notional = (f * equity) / stop_distance_pct
        return max(0.0, notional)

# ---------------------------------------------------------------------------
# Alpaca Integration Layer
# ---------------------------------------------------------------------------
class AlpacaManager:
    def __init__(self):
        self.trading = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        self.stock_data = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.crypto_data = CryptoHistoricalDataClient()  # Keys optional for crypto hist
        self.option_data = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
        self.account = None
        self._refresh_account()

    def _refresh_account(self):
        try:
            self.account = self.trading.get_account()
            logger.info(f"Equity: ${float(self.account.equity):.2f} | Buying Power: ${float(self.account.buying_power):.2f}")
        except Exception as e:
            logger.error(f"Account refresh failed: {e}")

    def get_equity(self) -> float:
        self._refresh_account()
        return float(self.account.equity) if self.account else INITIAL_CAPITAL

    def fetch_bars(self, symbol: str, asset_class: str = "stock", limit: int = LOOKBACK_BARS) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=10)
            if asset_class == "stock":
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    limit=limit
                )
                bars = self.stock_data.get_stock_bars(req)
            else:
                req = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    limit=limit
                )
                bars = self.crypto_data.get_crypto_bars(req)
            df = bars.df
            if df.empty:
                return None
            return df.droplevel(0) if isinstance(df.index, pd.MultiIndex) else df
        except Exception as e:
            logger.error(f"Bar fetch failed for {symbol}: {e}")
            return None

    def get_option_greeks_placeholder(self, underlying: str) -> Optional[Dict]:
        """Placeholder: pull snapshot / chain and extract Delta/Gamma when premium is optimal."""
        try:
            # Example: nearest liquid chain
            chain_req = OptionChainRequest(underlying_symbol=underlying, limit=20)
            chain = self.option_data.get_option_chain(chain_req)
            # In production: filter by moneyness, DTE, IV rank, then pull specific contract greeks
            # via OptionSnapshotRequest
            logger.info(f"Option chain retrieved for {underlying} (greeks placeholder active)")
            return {"status": "placeholder", "underlying": underlying}
        except Exception as e:
            logger.warning(f"Option Greeks fetch skipped: {e}")
            return None

    def submit_order(self, symbol: str, notional: float, side: OrderSide,
                     asset_class: str = "stock", order_type: str = "market") -> Optional[str]:
        try:
            if notional < 1.0:
                logger.info(f"Notional too small for {symbol}, skipping")
                return None

            if order_type == "market":
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side=side,
                    time_in_force=TimeInForce.DAY if asset_class == "stock" else TimeInForce.GTC
                )
            else:
                # Limit example (price would come from latest quote)
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=0.0  # Replace with real limit
                )

            order = self.trading.submit_order(order_data=order_data)
            logger.info(f"Order submitted: {side.value} {symbol} notional=${notional:.2f} id={order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None

    def close_position(self, symbol: str):
        try:
            self.trading.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
        except Exception as e:
            logger.error(f"Close failed {symbol}: {e}")

# ---------------------------------------------------------------------------
# Signal & Risk Engine
# ---------------------------------------------------------------------------
class SignalEngine:
    def __init__(self, markov: MarkovRegimeModel, kelly: KellySizer):
        self.markov = markov
        self.kelly = kelly

    def generate(self, symbol: str, df: pd.DataFrame, equity: float) -> Optional[Dict]:
        if df is None or len(df) < 30:
            return None

        closes = df["close"].astype(float)
        returns = closes.pct_change().dropna()
        latest_ret = returns.iloc[-1]
        realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(390)  # annualized-ish minute

        state = self.markov.update(latest_ret, realized_vol)
        edge = self.markov.expected_edge()
        probs = self.markov.next_state_probabilities()

        # Regime-conditioned win probability & payoff (simple mapping)
        if state in (0, 1):  # Bullish regimes
            win_prob = 0.55 + 0.15 * edge
            payoff = 1.8
            side = OrderSide.BUY
        else:
            win_prob = 0.55 + 0.15 * abs(edge)
            payoff = 1.6
            side = OrderSide.SELL

        # ATR-like stop distance
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        atr = (high - low).rolling(14).mean().iloc[-1]
        stop_pct = (atr / closes.iloc[-1]) * STOP_LOSS_ATR_MULT
        stop_pct = max(0.008, min(stop_pct, 0.05))  # 0.8%–5% tight band

        notional = self.kelly.size(edge, win_prob, payoff, equity, stop_pct)

        if notional < 5.0:
            return None

        return {
            "symbol": symbol,
            "side": side,
            "notional": notional,
            "state": REGIME_NAMES[state],
            "edge": edge,
            "stop_pct": stop_pct,
            "probs": probs.tolist()
        }

# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------
class TradingAgent:
    def __init__(self):
        self.alpaca = AlpacaManager()
        self.markov = MarkovRegimeModel()
        self.kelly = KellySizer(fraction=0.40, max_fraction=0.55)  # Aggressive for paper
        self.signals = SignalEngine(self.markov, self.kelly)
        self.positions: Dict[str, Dict] = {}
        self.running = True

    def _manage_stops(self, symbol: str, current_price: float):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        entry = pos["entry"]
        stop = pos["stop"]
        side = pos["side"]

        hit = (side == OrderSide.BUY and current_price <= stop) or \
              (side == OrderSide.SELL and current_price >= stop)
        if hit:
            logger.warning(f"STOP HIT {symbol} @ {current_price:.2f}")
            self.alpaca.close_position(symbol)
            del self.positions[symbol]

    def run_polling_loop(self, interval_sec: int = 60):
        """Clean polling loop with full error isolation."""
        logger.info("Starting Markov Regime Agent (Paper)")
        while self.running:
            try:
                equity = self.alpaca.get_equity()
                logger.info(f"Current equity: ${equity:.2f} | Regime: {REGIME_NAMES[self.markov.current_state]}")

                # Multi-asset scan
                for sym in STOCK_SYMBOLS:
                    df = self.alpaca.fetch_bars(sym, "stock")
                    if df is not None and not df.empty:
                        price = float(df["close"].iloc[-1])
                        self._manage_stops(sym, price)
                        sig = self.signals.generate(sym, df, equity)
                        if sig and sym not in self.positions:
                            oid = self.alpaca.submit_order(sig["symbol"], sig["notional"], sig["side"], "stock")
                            if oid:
                                stop_price = price * (1 - sig["stop_pct"]) if sig["side"] == OrderSide.BUY else price * (1 + sig["stop_pct"])
                                self.positions[sym] = {
                                    "entry": price,
                                    "stop": stop_price,
                                    "side": sig["side"],
                                    "notional": sig["notional"]
                                }

                for sym in CRYPTO_SYMBOLS:
                    df = self.alpaca.fetch_bars(sym, "crypto")
                    if df is not None and not df.empty:
                        price = float(df["close"].iloc[-1])
                        self._manage_stops(sym, price)
                        sig = self.signals.generate(sym, df, equity)
                        if sig and sym not in self.positions:
                            oid = self.alpaca.submit_order(sig["symbol"], sig["notional"], sig["side"], "crypto")
                            if oid:
                                stop_price = price * (1 - sig["stop_pct"]) if sig["side"] == OrderSide.BUY else price * (1 + sig["stop_pct"])
                                self.positions[sym] = {
                                    "entry": price,
                                    "stop": stop_price,
                                    "side": sig["side"],
                                    "notional": sig["notional"]
                                }

                # Options Greeks placeholder (scale into contracts when IV/premium optimal)
                for und in OPTION_UNDERLYINGS:
                    self.alpaca.get_option_greeks_placeholder(und)

                time.sleep(interval_sec)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                self.running = False
            except Exception as e:
                logger.exception(f"Loop error (isolated): {e}")
                time.sleep(15)

    # Optional: async stream skeleton (can be extended)
    async def _bar_handler(self, bar):
        logger.debug(f"Stream bar: {bar.symbol} {bar.close}")

    def start_streams(self):
        """Skeleton for live streams (run in separate task or process)."""
        stock_stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)
        stock_stream.subscribe_bars(self._bar_handler, *STOCK_SYMBOLS)
        # stock_stream.run()  # Blocking; prefer asyncio.gather in production

if __name__ == "__main__":
    agent = TradingAgent()
    try:
        agent.run_polling_loop(interval_sec=45)
    finally:
        logger.info("Agent stopped")
