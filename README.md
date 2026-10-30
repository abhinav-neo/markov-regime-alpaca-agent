# Markov Regime Alpaca Agent

Multi-asset Markov regime switching algorithmic trading agent for **Alpaca paper trading**.

Supports stocks, crypto, and options (Greeks placeholder). Uses a discrete-time Markov Chain for regime detection (Low/High Vol × Bull/Bear) and fractional Kelly criterion for aggressive but protected position sizing.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your **paper** Alpaca keys:
   ```env
   ALPACA_API_KEY=your_paper_key
   ALPACA_SECRET_KEY=your_paper_secret
   ```

3. Run:
   ```bash
   python markov_agent.py
   ```

## Notes

- Hard-coded to paper trading.
- Tight stop-loss protocol and risk caps to limit risk-of-ruin.
- Polling loop (streams skeleton included for extension).
- Always paper-trade and validate thoroughly before any live use.

**Disclaimer:** This is a research scaffold, not financial advice. Past/paper performance does not guarantee future results.
