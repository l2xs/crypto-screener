"""
Binance USDT-M Futures Screener
--------------------------------
Purpose: narrow ~200+ Binance futures pairs down to a handful of
candidates that already show multi-indicator convergence, so you only
run your detailed AI scalping prompt on pairs worth analyzing.

Pipeline:
  1. Pull all USDT-M futures tickers (24h volume, price change).
  2. Filter by a minimum liquidity floor (avoid thin/illiquid pairs).
  3. Pull 1h klines for each surviving pair.
  4. Compute RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA(9,21,200).
  5. Score each pair by how many of your prompt's conditions it already
     satisfies (trend / momentum / volatility alignment).
  6. Print the top N candidates, ranked by score, ready to paste into
     your detailed AI prompt.

Requirements:
  pip install requests pandas numpy
  (indicators are computed with plain pandas/numpy math, no pandas-ta/numba
  needed, so there's nothing here that requires a compiler on Windows)

Usage:
  python screener.py                # default: top 5 candidates
  python screener.py --top 10
  python screener.py --min-volume 50000000   # min 24h quote volume in USDT
"""

import argparse
import time
import sys
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://fapi.binance.com"


def get_all_tickers():
    """24h ticker stats for every USDT-M futures pair."""
    url = f"{BASE_URL}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")].copy()
    df["quoteVolume"] = df["quoteVolume"].astype(float)
    df["priceChangePercent"] = df["priceChangePercent"].astype(float)
    df["lastPrice"] = df["lastPrice"].astype(float)
    return df[["symbol", "quoteVolume", "priceChangePercent", "lastPrice"]]


def get_klines(symbol, interval="1h", limit=250):
    """Historical candles for one symbol. limit=250 gives enough
    history for a stable 200 EMA."""
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def _ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(df, length=14):
    """Wilder's ATR: average true range over `length` periods."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _swing_levels(df, lookback=20):
    """Nearest recent swing high/low, excluding the current (last) candle,
    used as an alternative SL reference alongside ATR."""
    window = df.iloc[-(lookback + 1):-1]
    return window["high"].max(), window["low"].min()


def compute_indicators(df):
    """
    Computes EMA(9/21/200), RSI(14), MACD(12,26,9), and Bollinger Bands(20,2)
    using plain pandas/numpy math only. No pandas-ta / numba dependency,
    so there's nothing here that needs to be compiled on install.
    """
    df["ema9"] = _ema(df["close"], 9)
    df["ema21"] = _ema(df["close"], 21)
    df["ema200"] = _ema(df["close"], 200)

    df["rsi14"] = _rsi(df["close"], 14)

    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["atr14"] = _atr(df, 14)
    return df


def score_pair(df, macd_lookback=3, bb_zone_pct=15.0):
    """
    Mirrors the convergence logic in your AI prompt:
      - Trend: price vs 200 EMA
      - Momentum: RSI extreme + MACD momentum state
      - Volatility: price vs Bollinger Bands (zone, not just exact touch)
      - Volume: spike vs 20-period average

    macd_lookback: MACD counts as "converging" if it's currently on the
        bullish/bearish side of its signal line, and separately notes
        whether it crossed within the last N candles. This matches how
        an LLM reading "MACD crossover" loosely tends to interpret it
        (a current momentum state) rather than an exact-instant event.
    bb_zone_pct: price counts as "near" a band if it's within this % of
        the band range from the edge, not only when it touches/exceeds
        the band exactly.
    Returns (direction, score 0-4, notes list).
    """
    if len(df) < 210 or df["ema200"].isna().iloc[-1]:
        return None, 0, ["insufficient history"]

    last = df.iloc[-1]
    notes = []
    long_pts = 0
    short_pts = 0

    # Trend
    if last["close"] > last["ema200"]:
        long_pts += 1
        notes.append("price>200EMA (uptrend)")
    elif last["close"] < last["ema200"]:
        short_pts += 1
        notes.append("price<200EMA (downtrend)")

    # Momentum: RSI
    if last["rsi14"] < 30:
        long_pts += 1
        notes.append(f"RSI oversold ({last['rsi14']:.1f})")
    elif last["rsi14"] > 70:
        short_pts += 1
        notes.append(f"RSI overbought ({last['rsi14']:.1f})")

    # Momentum: MACD - current state (above/below signal) rather than
    # requiring the cross to happen on this exact candle. Still notes
    # whether a fresh cross happened recently, since that's a stronger signal.
    window = df.iloc[-(macd_lookback + 1):]
    crossed_up_recently = ((window["macd"].shift(1) < window["macd_signal"].shift(1)) &
                            (window["macd"] > window["macd_signal"])).any()
    crossed_down_recently = ((window["macd"].shift(1) > window["macd_signal"].shift(1)) &
                              (window["macd"] < window["macd_signal"])).any()
    if last["macd"] > last["macd_signal"]:
        long_pts += 1
        notes.append("MACD bullish" + (" (fresh cross)" if crossed_up_recently else " (holding above signal)"))
    elif last["macd"] < last["macd_signal"]:
        short_pts += 1
        notes.append("MACD bearish" + (" (fresh cross)" if crossed_down_recently else " (holding below signal)"))

    # Volatility: Bollinger Band zone (near the band, not just touching it)
    band_range = last["bb_upper"] - last["bb_lower"]
    if pd.notna(band_range) and band_range > 0:
        pct_from_bottom = (last["close"] - last["bb_lower"]) / band_range * 100
        if pct_from_bottom <= bb_zone_pct:
            long_pts += 1
            notes.append(f"price near lower BB ({pct_from_bottom:.0f}% of band)")
        elif pct_from_bottom >= 100 - bb_zone_pct:
            short_pts += 1
            notes.append(f"price near upper BB ({pct_from_bottom:.0f}% of band)")

    # Volume confirmation (not a directional point on its own,
    # but flags whether a breakout has real participation behind it)
    vol_spike = last["volume"] > 1.5 * last["vol_avg20"] if pd.notna(last["vol_avg20"]) else False
    if vol_spike:
        notes.append("volume spike vs 20-period avg")

    if long_pts >= short_pts and long_pts > 0:
        return "Long", long_pts, notes
    elif short_pts > 0:
        return "Short", short_pts, notes
    else:
        return None, 0, notes


def build_trade_setup(df, direction, sl_atr_mult=1.5, tp_atr_mult=2.5, min_rr=1.5, swing_lookback=20):
    """
    Computes entry/SL/TP/R:R directly from the data - no AI/LLM involved in
    the arithmetic. Mirrors the rules in your 1000-char prompt:
      SL = sl_atr_mult x ATR(14) from entry, or nearest swing high/low if tighter
      TP = tp_atr_mult x ATR(14) from entry, or nearest key support/resistance
    Returns a dict with the full setup, or None if R:R falls below min_rr.
    """
    last = df.iloc[-1]
    entry = last["close"]
    atr = last["atr14"]
    if pd.isna(atr) or atr <= 0:
        return None

    swing_high, swing_low = _swing_levels(df, lookback=swing_lookback)

    if direction == "Long":
        atr_sl = entry - sl_atr_mult * atr
        atr_tp = entry + tp_atr_mult * atr
        # "tighter" SL = whichever is closer to entry (smaller risk), but
        # still below entry
        candidates_sl = [x for x in [atr_sl, swing_low] if x < entry]
        sl = max(candidates_sl) if candidates_sl else atr_sl
        tp = atr_tp  # key resistance isn't reliably derivable from candles alone; ATR-based
    else:  # Short
        atr_sl = entry + sl_atr_mult * atr
        atr_tp = entry - tp_atr_mult * atr
        candidates_sl = [x for x in [atr_sl, swing_high] if x > entry]
        sl = min(candidates_sl) if candidates_sl else atr_sl
        tp = atr_tp

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < min_rr:
        return None

    return {
        "entry": entry, "sl": sl, "tp": tp, "rr": rr, "atr": atr,
    }


def run_screen(min_volume, top_n, interval, sleep_between, min_score=3, diagnostic=False,
                signal_mode=False, sl_atr_mult=1.5, tp_atr_mult=2.5, min_rr=1.5):
    print(f"Fetching tickers from Binance USDT-M Futures...")
    tickers = get_all_tickers()
    liquid = tickers[tickers["quoteVolume"] >= min_volume].sort_values(
        "quoteVolume", ascending=False
    )
    print(f"{len(liquid)} pairs pass the {min_volume:,.0f} USDT 24h-volume floor "
          f"(out of {len(tickers)} total).\n")

    all_scored = []
    dataframes = {}  # keep each pair's df around for signal_mode's setup calc
    for i, row in enumerate(liquid.itertuples(), start=1):
        symbol = row.symbol
        try:
            df = get_klines(symbol, interval=interval)
            df = compute_indicators(df)
            direction, score, notes = score_pair(df)
            if direction:
                all_scored.append({
                    "symbol": symbol,
                    "direction": direction,
                    "score": score,
                    "notes": "; ".join(notes),
                    "last_price": df["close"].iloc[-1],
                    "24h_quote_volume": row.quoteVolume,
                    "24h_change_%": row.priceChangePercent,
                })
                dataframes[symbol] = df
        except Exception as e:
            print(f"  [skip] {symbol}: {e}", file=sys.stderr)

        if sleep_between:
            time.sleep(sleep_between)  # stay under Binance rate limits

    if not all_scored:
        print("No pairs currently meet 3+ indicator convergence. NO SIGNAL universe-wide.")
        return

    scored_df = pd.DataFrame(all_scored).sort_values("score", ascending=False)
    results = scored_df[scored_df["score"] >= min_score]

    if signal_mode:
        _print_signals(results, dataframes, top_n, min_score, sl_atr_mult, tp_atr_mult, min_rr)
        return

    if diagnostic:
        print(f"DIAGNOSTIC MODE: showing top {top_n} pairs by score, regardless of threshold.\n"
              f"(This lets you see how close pairs are, and sanity-check any signal your AI prompt reports.)\n")
        out = scored_df.head(top_n)
    elif len(results) == 0:
        print(f"No pairs currently meet the {min_score}+ indicator threshold.\n"
              f"Closest pairs (run with --diagnostic to always see this):\n")
        out = scored_df.head(top_n)
    else:
        out = results.head(top_n)
        print(f"Top {len(out)} candidate(s) meeting {min_score}+ indicators for your detailed prompt:\n")

    for r in out.itertuples():
        print(f"{r.symbol}  |  {r.direction}  |  score {r.score}/4  |  "
              f"price {r.last_price}  |  24h vol {r._6:,.0f} USDT  |  24h chg {r._7}%")
        print(f"   conditions: {r.notes}\n")

    if not diagnostic and len(results) > 0:
        print("Paste one of these symbols into your detailed AI scalping prompt.")


def _print_signals(results, dataframes, top_n, min_score, sl_atr_mult, tp_atr_mult, min_rr):
    """
    Prints a fully computed trade setup - Direction/Entry/TP/SL/R:R -
    directly from the data, matching your 1000-char prompt's output format.
    No AI involved in the arithmetic; a language model can't miscompute
    what code already computed.
    """
    printed = 0
    for r in results.head(top_n).itertuples():
        df = dataframes[r.symbol]
        setup = build_trade_setup(df, r.direction, sl_atr_mult, tp_atr_mult, min_rr)
        if setup is None:
            print(f"{r.symbol}: {r.score}/4 indicators aligned, but R:R fell below "
                  f"1:{min_rr} -> NO SIGNAL (R:R below threshold)\n")
            continue
        # Simple, transparent confidence proxy: indicator score + R:R quality.
        # Not a probability - just a deterministic way to rank/compare setups.
        confidence = min(100, round((r.score / 4) * 60 + min(setup['rr'] / 3, 1) * 40))
        print(f"{r.symbol}")
        print(f"- Direction: {r.direction}")
        print(f"- Market entry: {setup['entry']:.6g}")
        print(f"- Take Profit (TP): {setup['tp']:.6g}")
        print(f"- Stop Loss (SL): {setup['sl']:.6g}")
        print(f"- Risk/Reward: 1:{setup['rr']:.2f}")
        print(f"- Confidence Score: {confidence}%")
        print(f"- Core Catalyst: {r.notes}")
        print()
        printed += 1

    if printed == 0:
        print("No pairs produced a valid signal (either <3 indicators aligned, "
              "or R:R fell below threshold on all candidates).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen Binance futures pairs for indicator convergence.")
    parser.add_argument("--min-volume", type=float, default=50_000_000,
                         help="Minimum 24h quote volume in USDT (liquidity floor). Default 50M.")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to show.")
    parser.add_argument("--interval", type=str, default="1h", help="Kline interval (matches your prompt: 1h).")
    parser.add_argument("--sleep", type=float, default=0.1,
                         help="Seconds to sleep between API calls (avoid rate limits).")
    parser.add_argument("--min-score", type=int, default=3,
                         help="Minimum indicators that must align (out of 4) to count as a signal. Default 3.")
    parser.add_argument("--diagnostic", action="store_true",
                         help="Always show the top-scoring pairs, even below the threshold, "
                              "so you can see how close things are and verify AI-reported signals.")
    parser.add_argument("--signal", action="store_true",
                         help="Compute and print a full trade setup (Direction/Entry/TP/SL/R:R) "
                              "directly from the data - no AI needed for the numbers. "
                              "Matches the output format of your 1000-char prompt.")
    parser.add_argument("--sl-atr-mult", type=float, default=1.5,
                         help="Stop-loss distance as a multiple of ATR(14). Default 1.5.")
    parser.add_argument("--tp-atr-mult", type=float, default=2.5,
                         help="Take-profit distance as a multiple of ATR(14). Default 2.5.")
    parser.add_argument("--min-rr", type=float, default=1.5,
                         help="Minimum acceptable risk:reward ratio. Setups below this are rejected. Default 1.5.")
    args = parser.parse_args()

    run_screen(args.min_volume, args.top, args.interval, args.sleep, args.min_score, args.diagnostic,
               args.signal, args.sl_atr_mult, args.tp_atr_mult, args.min_rr)

