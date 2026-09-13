"""
gold_multi_timeframe_predictor.py
A live website that continuously analyzes gold on 15-minute, 1-hour, and
4-hour timeframes and projects the next N_HORIZONS candles on each -
custom-built charts in a Topstep-inspired dark theme.

Pipeline per timeframe:
  1. Live OHLC from Twelve Data (a real, sanctioned free API) + daily macro
     context (DXY, 10Y yield, VIX) when available.
  2. ~24 features: returns/lags, moving averages, RSI/MACD/ADX/Stochastic,
     Bollinger %B, session/seasonality (hour/day cyclical encoding), a
     mean-reversion z-score, price-action market structure (confirmed
     break-of-structure trend, not naive new-high/new-low), and distance to
     the nearest supply/demand zone.
  3. A 3-model ensemble per horizon - XGBoost, Ridge, Random Forest - each
     weighted by its own walk-forward validation accuracy, not a fixed split.
  4. Real, data-estimated uncertainty bands via XGBoost quantile regression
     (p10/p90), falling back to an ATR heuristic if that ever fails.
  5. Cross-timeframe confluence checking, plus an alert system that fires
     only when accuracy, model agreement, AND risk:reward all clear
     configurable bars together.

Run locally with:
    streamlit run gold_multi_timeframe_predictor.py
Or deploy for free at share.streamlit.io - push this file + requirements.txt
to a GitHub repo, connect it there, add a TWELVEDATA_API_KEY secret, done.

⚠️ HONEST FRAMING, STATED ONCE: "AI" here means real, walk-forward-validated
models on real historical data - not magic, and not a guarantee. Check the
accuracy panel under each chart before trusting any of it; on a liquid,
efficiently-traded instrument like gold, expect that number only modestly
above 50% - the normal, honest result for this class of model, not a bug.
More features and models make this more rigorous, not more certain.
Educational tool, not financial advice.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="Gold Predictor", layout="wide", initial_sidebar_state="expanded")

TIMEFRAMES = [
    {"label": "15 Minute", "key": "15m", "td_interval": "15min"},
    {"label": "1 Hour",    "key": "1h",  "td_interval": "1h"},
    {"label": "4 Hour",    "key": "4h",  "td_interval": "4h"},
]
N_HORIZONS = 8

# Chart candle colors - kept as the universal trading convention (teal-green
# up / red down), independent of site branding below.
TV_UP        = "#26A69A"
TV_DOWN      = "#EF5350"
TV_UP_GHOST  = "rgba(38,166,154,0.35)"
TV_DOWN_GHOST= "rgba(239,83,80,0.35)"

# Site theme, styled after Topstep's prop-trading aesthetic: near-black
# background, one bold lime accent, confident big-stat typography. (Built
# from Topstep's actual page structure and stated brand direction, not a
# pixel-for-pixel clone - treat this as "in that style," not an exact match.)
TS_BG        = "#0A0A0B"
TS_PANEL_BG  = "#141416"
TS_PANEL_BG2 = "#1B1B1E"
TS_BORDER    = "#2A2A2E"
TS_TEXT      = "#E8E8EA"
TS_MUTED     = "#8A8A90"
TS_ACCENT    = "#C6FF3D"
TS_ACCENT_DK = "#8FCC00"

# Reuse the old names so the rest of the file (chart function etc.) still works
TV_BG, TV_PANEL_BG, TV_BORDER, TV_TEXT, TV_MUTED = TS_BG, TS_PANEL_BG, TS_BORDER, TS_TEXT, TS_MUTED


# =============================================================================
# STYLE (custom CSS - Topstep-inspired: near-black + bold lime accent + big
# confident stat typography, real cards instead of default Streamlit chrome)
# =============================================================================
st.markdown(f"""
<style>
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stApp {{ background-color: {TS_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {TS_PANEL_BG}; border-right: 1px solid {TS_BORDER}; }}
    h1, h2, h3 {{ color: #FFFFFF !important; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; font-weight: 800 !important; letter-spacing: -0.5px; }}
    p, span, label, .stMarkdown {{ color: {TS_TEXT}; }}

    div[data-testid="stMetric"] {{
        background-color: {TS_PANEL_BG}; border: 1px solid {TS_BORDER}; border-radius: 10px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {TS_MUTED}; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
    div[data-testid="stMetricValue"] {{ color: #FFFFFF; font-weight: 800; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {TS_PANEL_BG}; border: 1px solid {TS_BORDER} !important; border-radius: 14px;
    }}

    .badge {{ display:inline-block; padding: 5px 14px; border-radius: 20px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; }}
    .badge-buy {{ background: rgba(198,255,61,0.15); color: {TS_ACCENT}; border: 1px solid {TS_ACCENT}; }}
    .badge-sell {{ background: rgba(239,83,80,0.15); color: {TV_DOWN}; border: 1px solid {TV_DOWN}; }}
    .badge-neutral {{ background: rgba(138,138,144,0.15); color: {TS_MUTED}; border: 1px solid {TS_MUTED}; }}

    .hero {{
        background: linear-gradient(135deg, {TS_PANEL_BG2} 0%, {TS_BG} 100%);
        border: 1px solid {TS_BORDER}; border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;
    }}
    .hero-eyebrow {{ color: {TS_ACCENT}; font-weight: 800; font-size: 12px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }}
    .hero-title {{ color: #FFFFFF; font-size: 34px; font-weight: 900; letter-spacing: -1px; line-height: 1.1; margin-bottom: 6px; }}
    .hero-sub {{ color: {TS_MUTED}; font-size: 14px; max-width: 640px; }}

    .stat-row {{ display:flex; gap: 14px; flex-wrap: wrap; margin-top: 18px; }}
    .stat-box {{ background: {TS_BG}; border: 1px solid {TS_BORDER}; border-radius: 10px; padding: 12px 18px; min-width: 130px; }}
    .stat-num {{ color: {TS_ACCENT}; font-size: 22px; font-weight: 900; }}
    .stat-label {{ color: {TS_MUTED}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}

    .ticker-row {{ display:flex; align-items:baseline; gap:14px; }}
    .ticker-price {{ font-size: 32px; font-weight: 900; color: #FFFFFF; }}
    .ticker-label {{ font-size: 13px; color: {TS_MUTED}; text-transform: uppercase; letter-spacing: 1px; }}

    .stButton>button {{ background: {TS_ACCENT}; color: #0A0A0B; font-weight: 800; border: none; border-radius: 8px; }}
    .stButton>button:hover {{ background: {TS_ACCENT_DK}; color: #0A0A0B; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FETCHING (Twelve Data - a real, sanctioned free API, not scraping)
# =============================================================================
TD_BASE_URL = "https://api.twelvedata.com/time_series"


def _get_api_key():
    key = st.secrets.get("TWELVEDATA_API_KEY", None)
    if not key:
        raise RuntimeError(
            "No Twelve Data API key found. Get a free one at twelvedata.com, then add it "
            "to your Streamlit app's Settings → Secrets as: TWELVEDATA_API_KEY = \"your-key-here\""
        )
    return key


def _twelvedata_request(symbol, interval, outputsize=500, attempts=3):
    api_key = _get_api_key()
    last_err = None
    for attempt in range(attempts):
        try:
            resp = requests.get(TD_BASE_URL, params={
                "symbol": symbol, "interval": interval, "outputsize": outputsize,
                "apikey": api_key, "timezone": "UTC",
            }, timeout=15)
            data = resp.json()

            if isinstance(data, dict) and data.get("status") == "error":
                msg = data.get("message", str(data))
                if "credit" in msg.lower() or "limit" in msg.lower() or resp.status_code == 429:
                    last_err = f"Twelve Data rate limit hit: {msg}"
                    if attempt < attempts - 1:
                        time.sleep(8 * (attempt + 1))
                        continue
                raise RuntimeError(f"Twelve Data error: {msg}")

            values = data.get("values")
            if not values:
                raise RuntimeError(f"Twelve Data returned no values for {symbol} ({interval}): {data}")

            df = pd.DataFrame(values)
            # Explicit timezone="UTC" above means these timestamps are UTC,
            # not exchange-local or otherwise ambiguous - localize them as
            # such so any downstream hour-of-day/session-window logic (e.g.
            # kill zones) converts correctly instead of silently assuming a
            # timezone that was never actually confirmed.
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
            df = df.set_index("datetime").sort_index()
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            return df[["Open", "High", "Low", "Close", "Volume"]]

        except RuntimeError:
            raise
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {symbol} ({interval}) after {attempts} attempts. Last error: {last_err}")


@st.cache_data(ttl=600, show_spinner=False)  # ~10 min: fast enough to feel live, safely within Twelve Data's 800/day free-tier budget (see sidebar note)
def fetch_ohlc(symbol, interval):
    return _twelvedata_request(symbol, interval, outputsize=500)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_daily():
    # Best-effort: macro symbol coverage on the free tier can vary, so each
    # one fails independently and gracefully - the app runs fine on price
    # data alone if none of these come through.
    symbols = {"dxy": "DXY", "yield_10y": "US10Y", "vix": "VIX"}
    frames = {}
    for name, sym in symbols.items():
        try:
            df = _twelvedata_request(sym, "1day", outputsize=500, attempts=1)
            frames[name] = df["Close"].rename(name)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1).ffill().dropna(how="all")


def merge_macro_onto_intraday(intraday_df, macro_daily_df):
    if macro_daily_df.empty:
        return intraday_df
    macro2 = macro_daily_df.copy()
    macro2.index = pd.to_datetime(macro2.index.date)
    full_dates = pd.date_range(macro2.index.min(), macro2.index.max() + pd.Timedelta(days=5), freq="D")
    macro2 = macro2.reindex(full_dates).ffill()
    intraday_dates = pd.to_datetime(intraday_df.index.date)
    merged_macro = macro2.reindex(intraday_dates).ffill()
    merged_macro.index = intraday_df.index
    out = intraday_df.join(merged_macro)
    out[macro_daily_df.columns] = out[macro_daily_df.columns].ffill()
    return out


# =============================================================================
# FEATURES
# =============================================================================
def rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = series.ewm(span=fast, adjust=False).mean(), series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_adx(df, period=14):
    """Wilder's ADX - trend strength, independent of direction. Used so the
    model can weigh trend-following features more when there's an actual
    trend to follow, and less in choppy/range-bound conditions."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)


def stochastic_k(df, period=14):
    low_min, high_max = df["Low"].rolling(period).min(), df["High"].rolling(period).max()
    return (100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)).fillna(50)


def compute_market_structure(df, pivot_window=3):
    """
    Price-action market structure: the trend flips from bearish to bullish
    only when price CLOSES above the most recent confirmed pivot high, and
    from bullish to bearish only on a close below the most recent confirmed
    pivot low - not on every new local high/low. This is the standard
    "break of structure" formulation behind most price-action/supply-demand
    trading approaches: it deliberately ignores shallow pullbacks that never
    actually break prior structure, avoiding premature reversal calls.
    Returns a Series of +1 (bullish structure) / -1 (bearish) / 0 (undetermined).
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    is_ph, is_pl = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in range(pivot_window, n - pivot_window):
        window_h = highs[i-pivot_window:i+pivot_window+1]
        window_l = lows[i-pivot_window:i+pivot_window+1]
        if highs[i] == window_h.max():
            is_ph[i] = True
        if lows[i] == window_l.min():
            is_pl[i] = True

    trend_out = np.zeros(n)
    trend, swing_high, swing_low = 0, None, None
    for i in range(n):
        if is_ph[i]:
            swing_high = highs[i]
        if is_pl[i]:
            swing_low = lows[i]
        if trend <= 0 and swing_high is not None and closes[i] > swing_high:
            trend = 1
        elif trend >= 0 and swing_low is not None and closes[i] < swing_low:
            trend = -1
        trend_out[i] = trend
    return pd.Series(trend_out, index=df.index)


def compute_fvg(df):
    """
    Fair Value Gap (ICT): a 3-candle imbalance left behind by a strong
    displacement move. Bullish FVG when candle[i-2].High < candle[i].Low
    (no overlap - a gap the market often returns to fill); bearish FVG is
    the mirror. Definition cross-checked across multiple independent ICT
    sources (TheInnerCircleTraders.com, TrendSpider, InnerCircleTrader.net,
    WritoFinance) - all consistent on this exact 3-candle no-overlap rule.
    """
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if high[i-2] < low[i]:
            bullish[i] = True
        if low[i-2] > high[i]:
            bearish[i] = True
    return pd.Series(bullish, index=df.index), pd.Series(bearish, index=df.index)


def find_nearest_unfilled_fvg(df, bullish_fvg, bearish_fvg):
    """Most recent FVG that hasn't since been 'filled' by price trading back
    through the gap - ICT treats unfilled gaps as still-active price magnets."""
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    for i in range(n - 1, 1, -1):
        if bullish_fvg.iloc[i]:
            gap_low, gap_high = high[i-2], low[i]
            filled = (low[i+1:] <= gap_low).any() if i + 1 < n else False
            if not filled:
                return {"low": float(gap_low), "high": float(gap_high), "direction": "bullish"}
        if bearish_fvg.iloc[i]:
            gap_low, gap_high = high[i], low[i-2]
            filled = (high[i+1:] >= gap_high).any() if i + 1 < n else False
            if not filled:
                return {"low": float(gap_low), "high": float(gap_high), "direction": "bearish"}
    return None


def find_last_order_block(df, bullish_fvg, bearish_fvg):
    """
    ICT Order Block: the final opposite-colored candle immediately before a
    displacement move that leaves an FVG - a bullish OB is the last DOWN
    candle before a rally, a bearish OB is the last UP candle before a
    decline. The FVG requirement (confirming genuine displacement, not
    ordinary consolidation) is explicit in multiple cross-referenced
    sources, e.g. TradingStrategyGuides' identification guide.
    """
    opens, closes = df["Open"].values, df["Close"].values
    n = len(df)
    for i in range(n - 1, 1, -1):
        if bullish_fvg.iloc[i]:
            for j in range(i - 2, max(i - 6, -1), -1):
                if closes[j] < opens[j]:
                    return {"low": float(df["Low"].iloc[j]), "high": float(df["High"].iloc[j]), "direction": "bullish"}
        if bearish_fvg.iloc[i]:
            for j in range(i - 2, max(i - 6, -1), -1):
                if closes[j] > opens[j]:
                    return {"low": float(df["Low"].iloc[j]), "high": float(df["High"].iloc[j]), "direction": "bearish"}
    return None


def detect_liquidity_sweep(df, lookback=20):
    """
    ICT liquidity sweep / stop hunt: price briefly trades beyond a recent
    swing high/low (triggering resting stop orders resting there) then
    closes back within the prior range - a false breakout ICT treats as a
    precursor to a reversal in the opposite direction.
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    if n < lookback + 2:
        return None
    recent_high = highs[-lookback-1:-1].max()
    recent_low = lows[-lookback-1:-1].min()
    if highs[-1] > recent_high and closes[-1] < recent_high:
        return "bearish"
    if lows[-1] < recent_low and closes[-1] > recent_low:
        return "bullish"
    return None


def compute_killzone_flags(index_utc, asian_hours=(20, 22), london_hours=(2, 5), ny_hours=(7, 10)):
    """
    ICT kill zones: session windows (converted to US/Eastern, so Daylight
    Saving Time is handled correctly via proper timezone conversion rather
    than a fixed UTC offset that would drift wrong for half the year).
    HONESTLY FLAGGED: exact kill zone hours vary meaningfully across
    different ICT courses/sources (confirmed by cross-referencing several -
    some cite 7-9am ET for New York, others 7-10am, others 8:30-11am). The
    defaults here are the most commonly-cited windows across that
    cross-reference, not one single "official" ICT definition - adjust via
    the sidebar if you follow a specific model with different hours.
    """
    et_index = index_utc.tz_convert("America/New_York")
    hour = et_index.hour
    in_asian = (hour >= asian_hours[0]) & (hour < asian_hours[1])
    in_london = (hour >= london_hours[0]) & (hour < london_hours[1])
    in_ny = (hour >= ny_hours[0]) & (hour < ny_hours[1])
    return pd.Series(in_asian, index=index_utc), pd.Series(in_london, index=index_utc), pd.Series(in_ny, index=index_utc)


def compute_premium_discount(df, lookback=50):
    """
    ICT Premium/Discount: where current price sits within the most recent
    significant swing, as a 0-1 fraction (0 = at the swing low/deepest
    discount, 1 = at the swing high/richest premium, 0.618-0.79 = the
    "golden pocket" ICT treats as the optimal-entry discount/premium zone
    for longs/shorts respectively - cross-confirmed via Pineify's ICT guide).
    """
    swing_high = df["High"].rolling(lookback).max()
    swing_low = df["Low"].rolling(lookback).min()
    rng = (swing_high - swing_low).replace(0, np.nan)
    return ((df["Close"] - swing_low) / rng).fillna(0.5)


def detect_last_zone(df, atr_series, lookback=100, impulse_atr_mult=2.0, consolidation_atr_mult=0.6):
    """
    Supply/demand zone detection: scans backward for the most recent "base
    before breakout" pattern - a single bar with an unusually tight range
    (relative to ATR) immediately followed by a strong directional impulse
    over the next few bars. This mirrors the standard price-action practice
    of marking a zone at the last consolidation candle before a big move,
    on the theory that a retest of that zone may attract the same buyers/
    sellers who caused the original move. Returns None if nothing qualifies
    in the lookback window - not every chart has a clean recent zone, and
    forcing one where none exists would be worse than reporting none.
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    atr = atr_series.values
    n = len(df)
    start = max(5, n - lookback)
    for i in range(n - 4, start, -1):
        if atr[i] <= 0 or np.isnan(atr[i]):
            continue
        if (highs[i] - lows[i]) > consolidation_atr_mult * atr[i]:
            continue
        move = closes[i+3] - closes[i]
        if move > impulse_atr_mult * atr[i]:
            return {"low": float(lows[i]), "high": float(highs[i]), "direction": "demand"}
        if move < -impulse_atr_mult * atr[i]:
            return {"low": float(lows[i]), "high": float(highs[i]), "direction": "supply"}
    return None


def build_features(raw, use_macro, asian_hours=(20, 22), london_hours=(2, 5), ny_hours=(7, 10)):
    out = raw.copy()
    close = out["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out["ret_1"], out["ret_3"], out["ret_5"] = close.pct_change(1), close.pct_change(3), close.pct_change(5)
    for lag in [1, 2, 3, 5]:
        out[f"ret_1_lag{lag}"] = out["ret_1"].shift(lag)

    out["sma_10"], out["sma_50"] = close.rolling(10).mean(), close.rolling(50).mean()
    out["sma_ratio"] = out["sma_10"] / out["sma_50"] - 1
    out["rsi_14"] = rsi(close, 14)
    _, _, hist = macd(close)
    out["macd_hist"] = hist / close
    out["volatility_10"] = out["ret_1"].rolling(10).std()

    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    out["bb_pct_b"] = (close - (sma20 - 2*std20)) / ((sma20 + 2*std20) - (sma20 - 2*std20))
    out["high_low_range"] = (out["High"] - out["Low"]) / close

    # --- expanded feature set: trend strength, stochastic momentum, gaps,
    #     session/seasonality effects, and longer-horizon mean reversion ---
    out["adx_14"] = compute_adx(out, 14)
    out["stoch_k"] = stochastic_k(out, 14)
    out["gap"] = ((out["Open"] - close.shift(1)) / close.shift(1)).fillna(0)
    out["hour_sin"] = np.sin(2*np.pi*out.index.hour/24)
    out["hour_cos"] = np.cos(2*np.pi*out.index.hour/24)
    out["dow_sin"] = np.sin(2*np.pi*out.index.dayofweek/7)
    out["dow_cos"] = np.cos(2*np.pi*out.index.dayofweek/7)
    roll_mean50, roll_std50 = close.rolling(50).mean(), close.rolling(50).std()
    out["zscore_50"] = ((close - roll_mean50) / roll_std50.replace(0, np.nan)).fillna(0)

    # --- price-action market structure ---
    out["structure_trend"] = compute_market_structure(out, pivot_window=3)

    # --- ICT concepts: FVG, Order Blocks, liquidity sweeps, kill zones,
    #     premium/discount - cross-referenced against multiple independent
    #     ICT sources before implementing (see function docstrings) ---
    bullish_fvg, bearish_fvg = compute_fvg(out)
    out["bullish_fvg"] = bullish_fvg.astype(float)
    out["bearish_fvg"] = bearish_fvg.astype(float)
    nearest_fvg = find_nearest_unfilled_fvg(out, bullish_fvg, bearish_fvg)
    order_block = find_last_order_block(out, bullish_fvg, bearish_fvg)
    sweep = detect_liquidity_sweep(out, lookback=20)
    kh = {"asian": asian_hours, "london": london_hours, "ny": ny_hours}
    in_asian, in_london, in_ny = compute_killzone_flags(out.index, kh["asian"], kh["london"], kh["ny"])
    out["in_asian_kz"] = in_asian.astype(float)
    out["in_london_kz"] = in_london.astype(float)
    out["in_ny_kz"] = in_ny.astype(float)
    out["premium_discount"] = compute_premium_discount(out, lookback=50)

    atr_for_ict = (out["High"] - out["Low"]).rolling(14).mean()
    if nearest_fvg is not None:
        fvg_mid = (nearest_fvg["low"] + nearest_fvg["high"]) / 2
        out["fvg_dist_atr"] = (close - fvg_mid) / atr_for_ict.replace(0, np.nan)
    else:
        out["fvg_dist_atr"] = 0.0
    if order_block is not None:
        ob_mid = (order_block["low"] + order_block["high"]) / 2
        out["ob_dist_atr"] = (close - ob_mid) / atr_for_ict.replace(0, np.nan)
    else:
        out["ob_dist_atr"] = 0.0
    out["liquidity_sweep_signal"] = 1.0 if sweep == "bullish" else (-1.0 if sweep == "bearish" else 0.0)

    feature_cols = ["ret_1", "ret_3", "ret_5", "ret_1_lag1", "ret_1_lag2", "ret_1_lag3",
                     "ret_1_lag5", "sma_ratio", "rsi_14", "macd_hist", "volatility_10",
                     "bb_pct_b", "high_low_range", "adx_14", "stoch_k", "gap",
                     "hour_sin", "hour_cos", "dow_sin", "dow_cos", "zscore_50", "structure_trend",
                     "bullish_fvg", "bearish_fvg", "in_asian_kz", "in_london_kz", "in_ny_kz",
                     "premium_discount", "fvg_dist_atr", "ob_dist_atr", "liquidity_sweep_signal"]

    if use_macro and "dxy" in out.columns:
        out["dxy_ret_1"] = out["dxy"].pct_change(1)
        feature_cols.append("dxy_ret_1")
    if use_macro and "yield_10y" in out.columns:
        out["yield_change_1"] = out["yield_10y"].diff(1)
        feature_cols.append("yield_change_1")
    if use_macro and "vix" in out.columns:
        out["vix_change_1"] = out["vix"].pct_change(1)
        feature_cols.append("vix_change_1")

    out["atr"] = atr_for_ict

    # --- supply/demand zone: distance from current price to the most recent
    #     qualifying zone, normalized by ATR (na/0 when no zone was found) ---
    zone = detect_last_zone(out, out["atr"])
    if zone is not None:
        mid = (zone["low"] + zone["high"]) / 2
        out["zone_dist_atr"] = (close - mid) / out["atr"].replace(0, np.nan)
        out["zone_is_demand"] = 1.0 if zone["direction"] == "demand" else -1.0
    else:
        out["zone_dist_atr"] = 0.0
        out["zone_is_demand"] = 0.0
    feature_cols += ["zone_dist_atr", "zone_is_demand"]

    for h in range(1, N_HORIZONS+1):
        out[f"target_h{h}"] = close.shift(-h) / close - 1

    return out.dropna(), feature_cols, zone, nearest_fvg, order_block, sweep


# =============================================================================
# MODEL: a 3-model ensemble (XGBoost + Ridge + Random Forest) per horizon,
# weighted by which one actually validates better on held-out data, not a
# fixed split. Three genuinely different model families - boosted trees,
# regularized linear, bagged trees - tend to have less-correlated errors
# than three variants of the same model, which is where an ensemble's
# benefit, if any, comes from.
# =============================================================================
def train_fast_xgb(X, y):
    model = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          objective="reg:squarederror", random_state=42)
    model.fit(X, y)
    return model


def predict_xgb(model, X):
    return model.predict(X)


def train_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(Xs, y)
    return {"scaler": scaler, "model": model}


def predict_ridge(ridge_obj, X):
    return ridge_obj["model"].predict(ridge_obj["scaler"].transform(X))


def train_rf(X, y):
    model = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=5,
                                   n_jobs=-1, random_state=42)
    model.fit(X, y)
    return model


def predict_rf(model, X):
    return model.predict(X)


def train_quantile_xgb(X, y, alpha):
    """Single-quantile model (one alpha per model, not the multi-quantile-
    array form) - more verbose but far more certain to work identically to
    every other XGBRegressor call in this file, since it's the same sklearn
    wrapper interface with just a different objective/param."""
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          objective="reg:quantileerror", quantile_alpha=alpha,
                          random_state=42)
    model.fit(X, y)
    return model


def walk_forward_accuracy(X, y, train_fn, predict_fn, n_folds=5, min_train=150):
    n = len(X)
    fold_size = (n - min_train) // n_folds
    if fold_size <= 5:
        return None, None
    dir_accs, rmses = [], []
    for f in range(n_folds):
        train_end = min_train + f*fold_size
        test_end = n if f == n_folds-1 else train_end + fold_size
        model = train_fn(X[:train_end], y[:train_end])
        pred = predict_fn(model, X[train_end:test_end])
        actual = y[train_end:test_end]
        dir_accs.append(((pred > 0) == (actual > 0)).mean())
        rmses.append(np.sqrt(np.mean((pred-actual)**2)))
    return float(np.mean(dir_accs)), float(np.mean(rmses))


def walk_forward_accuracy_ensemble(X, y, weights, n_folds=5, min_train=150):
    """
    Same walk-forward procedure, but scoring the actual BLENDED ensemble
    prediction (using the given weights) rather than any single component
    model. This is the number that belongs next to a blended prediction -
    an individual model validating well doesn't guarantee the blend does
    too. Costs 3x the fits of the single-model version (one fold now trains
    all three models instead of one), so it's used for the headline metric
    only; per-horizon detail numbers use the cheaper single-model version
    and are labeled accordingly in the UI.
    """
    n = len(X)
    fold_size = (n - min_train) // n_folds
    if fold_size <= 5:
        return None, None
    dir_accs, rmses = [], []
    for f in range(n_folds):
        train_end = min_train + f*fold_size
        test_end = n if f == n_folds-1 else train_end + fold_size
        xgb_m = train_fast_xgb(X[:train_end], y[:train_end])
        ridge_m = train_ridge(X[:train_end], y[:train_end])
        rf_m = train_rf(X[:train_end], y[:train_end])
        pred = (weights["xgb"] * predict_xgb(xgb_m, X[train_end:test_end])
                + weights["ridge"] * predict_ridge(ridge_m, X[train_end:test_end])
                + weights["rf"] * predict_rf(rf_m, X[train_end:test_end]))
        actual = y[train_end:test_end]
        dir_accs.append(((pred > 0) == (actual > 0)).mean())
        rmses.append(np.sqrt(np.mean((pred-actual)**2)))
    return float(np.mean(dir_accs)), float(np.mean(rmses))


@st.cache_data(ttl=600, show_spinner=False)  # matches fetch_ohlc's cadence so the whole pipeline refreshes together
def analyze_timeframe(tf_key, symbol, use_macro, asian_hours=(20, 22), london_hours=(2, 5), ny_hours=(7, 10)):
    tf = next(t for t in TIMEFRAMES if t["key"] == tf_key)
    raw = fetch_ohlc(symbol, tf["td_interval"])
    if raw.empty:
        return {"error": f"Received data but it was empty after processing for {tf['label']}."}
    if len(raw) < 200:
        return {"error": f"Only got {len(raw)} bars for {tf['label']} (need 200+). Twelve Data's free tier may cap history depth for this interval."}

    macro = fetch_macro_daily() if use_macro else pd.DataFrame()
    raw = merge_macro_onto_intraday(raw, macro) if use_macro else raw

    feat_df, feature_cols, zone, nearest_fvg, order_block, sweep = build_features(
        raw, use_macro, asian_hours, london_hours, ny_hours)
    if len(feat_df) < 150:
        return {"error": f"Only {len(feat_df)} usable rows survived feature engineering for {tf['label']} (need 150+)."}

    X = feat_df[feature_cols].values
    last_row = feat_df[feature_cols].iloc[[-1]].values
    current_close = float(feat_df["Close"].iloc[-1])
    current_atr = float(feat_df["atr"].iloc[-1])

    # Validation-weighted 3-model ensemble: each model family gets weighted
    # by how well it actually validated on held-out data, not a fixed guess.
    # Three genuinely different model families (boosted trees, regularized
    # linear, bagged trees) tend to have less-correlated errors than three
    # variants of the same model - that's where an ensemble's benefit, if
    # any, comes from.
    y_h1 = feat_df["target_h1"].values
    # Individual-model RMSEs purely to derive ensemble weights (their
    # individual accuracies aren't what gets reported - see below).
    _, xgb_rmse_h1 = walk_forward_accuracy(X, y_h1, train_fast_xgb, predict_xgb)
    _, ridge_rmse_h1 = walk_forward_accuracy(X, y_h1, train_ridge, predict_ridge)
    _, rf_rmse_h1 = walk_forward_accuracy(X, y_h1, train_rf, predict_rf)

    rmses = {"xgb": xgb_rmse_h1, "ridge": ridge_rmse_h1, "rf": rf_rmse_h1}
    valid = {k: v for k, v in rmses.items() if v and v > 0}
    if valid:
        inv = {k: 1/v for k, v in valid.items()}
        total = sum(inv.values())
        weights = {k: (inv[k]/total if k in inv else 0.0) for k in ["xgb", "ridge", "rf"]}
    else:
        weights = {"xgb": 1.0, "ridge": 0.0, "rf": 0.0}

    # The headline accuracy: the BLENDED ensemble's own walk-forward
    # accuracy using these weights - not any single component's, since
    # that's what the displayed prediction actually is.
    ensemble_acc_h1, ensemble_rmse_h1 = walk_forward_accuracy_ensemble(X, y_h1, weights)

    horizon_preds, horizon_accs, horizon_bands, horizon_agree = [], {1: ensemble_acc_h1}, [], []
    for h in range(1, N_HORIZONS+1):
        y_h = feat_df[f"target_h{h}"].values
        xgb_model = train_fast_xgb(X, y_h)
        ridge_model = train_ridge(X, y_h)
        rf_model = train_rf(X, y_h)
        xgb_p = float(predict_xgb(xgb_model, last_row)[0])
        ridge_p = float(predict_ridge(ridge_model, last_row)[0])
        rf_p = float(predict_rf(rf_model, last_row)[0])
        combined = weights["xgb"]*xgb_p + weights["ridge"]*ridge_p + weights["rf"]*rf_p
        horizon_preds.append(combined)
        horizon_agree.append((xgb_p > 0) == (ridge_p > 0) == (rf_p > 0))
        if h > 1:
            # Cheaper XGBoost-component accuracy for the per-horizon detail
            # view (full ensemble walk-forward at every horizon would 3x
            # this loop's cost) - labeled as such in the UI, not presented
            # as the blended ensemble's accuracy.
            xgb_only_acc_h, _ = walk_forward_accuracy(X, y_h, train_fast_xgb, predict_xgb)
            horizon_accs[h] = xgb_only_acc_h

        # Real, data-estimated uncertainty band via quantile regression,
        # replacing a fixed ATR-multiplier guess. Wrapped defensively: if
        # this ever fails for any reason, fall back to the ATR heuristic
        # rather than breaking the app - reliability over sophistication.
        try:
            q10 = float(predict_xgb(train_quantile_xgb(X, y_h, 0.1), last_row)[0])
            q90 = float(predict_xgb(train_quantile_xgb(X, y_h, 0.9), last_row)[0])
            if q90 < q10:  # occasionally quantile crossing happens on noisy data; guard against it
                q10, q90 = q90, q10
            horizon_bands.append((q10, q90))
        except Exception:
            horizon_bands.append(None)  # signals build_ghost_candles to use the ATR fallback for this horizon

    # Risk:reward, in the same spirit as the video's "only take setups above
    # a minimum R:R" filter: risk is the distance to a sensible stop (the far
    # edge of the detected supply/demand zone when it aligns with the
    # predicted direction, otherwise a 1x-ATR fallback since that's the
    # standard convention with no specific zone to anchor to); reward is the
    # cumulative projected move up to each horizon.
    cumulative_move = np.cumsum(horizon_preds) * current_close
    total_direction = 1 if cumulative_move[-1] > 0 else -1
    zone_matches = zone is not None and (
        (zone["direction"] == "demand" and total_direction > 0) or
        (zone["direction"] == "supply" and total_direction < 0)
    )
    if zone_matches:
        stop_level = zone["low"] if zone["direction"] == "demand" else zone["high"]
        risk = abs(current_close - stop_level)
    else:
        risk = current_atr
    risk = risk if risk > 0 else current_atr

    return {
        "raw": raw, "current_close": current_close, "current_atr": current_atr,
        "horizon_preds": horizon_preds, "horizon_accs": horizon_accs, "horizon_bands": horizon_bands,
        "horizon_agree": horizon_agree, "n_folds": 5,
        "dir_acc_h1": ensemble_acc_h1, "rmse_h1": ensemble_rmse_h1,
        "ensemble_weights": weights, "last_bar_time": feat_df.index[-1],
        "zone": zone, "risk": risk, "cumulative_move": cumulative_move,
        "structure_trend": int(feat_df["structure_trend"].iloc[-1]),
        "nearest_fvg": nearest_fvg, "order_block": order_block, "liquidity_sweep": sweep,
        "in_killzone": {
            "asian": bool(feat_df["in_asian_kz"].iloc[-1]),
            "london": bool(feat_df["in_london_kz"].iloc[-1]),
            "ny": bool(feat_df["in_ny_kz"].iloc[-1]),
        },
        "premium_discount": float(feat_df["premium_discount"].iloc[-1]),
    }


def build_ghost_candles(current_close, current_atr, horizon_preds, horizon_bands=None):
    """
    Builds the chained ghost-candle sequence. When a real quantile-regression
    band (p10/p90) is available for a horizon, the wick uses that genuine,
    data-estimated range instead of the ATR-multiplier heuristic - real
    uncertainty from the model's own spread of outcomes, not a guessed
    multiplier. Falls back to the heuristic per-horizon whenever the
    quantile band wasn't available (e.g. training failed for that horizon).
    """
    candles, open_price = [], current_close
    for h, pred_return in enumerate(horizon_preds, start=1):
        close_price = open_price * (1 + pred_return)
        band = horizon_bands[h-1] if horizon_bands and h-1 < len(horizon_bands) else None

        if band is not None:
            q10, q90 = band
            band_low_price = open_price * (1 + q10)
            band_high_price = open_price * (1 + q90)
            high = max(open_price, close_price, band_high_price)
            low = min(open_price, close_price, band_low_price)
        else:
            wick = current_atr * (0.5 + 0.15*h)
            high = max(open_price, close_price) + wick*0.5
            low = min(open_price, close_price) - wick*0.5

        candles.append({"open": open_price, "close": close_price, "high": high, "low": low})
        open_price = close_price
    return candles


# =============================================================================
# CHART (custom-built, TradingView-styled)
# =============================================================================
def render_chart(hist_df, ghost_candles):
    hist = hist_df.tail(60)
    n_hist = len(hist)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=list(range(n_hist)), open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        increasing_line_color=TV_UP, decreasing_line_color=TV_DOWN,
        increasing_fillcolor=TV_UP, decreasing_fillcolor=TV_DOWN,
        name="History", showlegend=False,
    ))

    ghost_x = list(range(n_hist, n_hist + len(ghost_candles)))
    fig.add_trace(go.Candlestick(
        x=ghost_x,
        open=[c["open"] for c in ghost_candles], high=[c["high"] for c in ghost_candles],
        low=[c["low"] for c in ghost_candles], close=[c["close"] for c in ghost_candles],
        increasing_line_color=TV_UP, decreasing_line_color=TV_DOWN,
        increasing_fillcolor=TV_UP_GHOST, decreasing_fillcolor=TV_DOWN_GHOST,
        name="Ghost Prediction", showlegend=False,
        line=dict(width=1),
    ))

    fig.add_vline(x=n_hist - 0.5, line_dash="dash", line_color=TV_BORDER, line_width=1)
    fig.add_annotation(x=n_hist + len(ghost_candles)/2 - 0.5, y=max(c["high"] for c in ghost_candles),
                        text="GHOST PREDICTION", showarrow=False, font=dict(color=TS_ACCENT, size=10),
                        yshift=14)

    fig.update_layout(
        height=340, margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor=TV_PANEL_BG, plot_bgcolor=TV_PANEL_BG,
        font=dict(color=TV_TEXT, family="-apple-system, BlinkMacSystemFont, sans-serif"),
        xaxis=dict(showgrid=False, rangeslider_visible=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=TV_BORDER, zeroline=False, side="right"),
    )
    return fig


def signal_badge(net_bias_pct):
    if net_bias_pct > 0.05:
        return '<span class="badge badge-buy">BUY BIAS</span>'
    elif net_bias_pct < -0.05:
        return '<span class="badge badge-sell">SELL BIAS</span>'
    return '<span class="badge badge-neutral">NEUTRAL</span>'


# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    st.title("⚙️ Settings")
    ticker = st.selectbox("Symbol", ["XAU/USD", "XAG/USD"], index=0, help="XAU/USD = Gold Spot, XAG/USD = Silver Spot (Twelve Data symbols)")
    use_macro = st.checkbox("Use macro features (DXY, 10Y yield, VIX)", value=True)
    refresh_seconds = st.slider("Page refresh (seconds)", 15, 300, 60,
                                 help="How often the page itself redraws. Separate from how often new data is fetched (see note below) — this can be fast without using up API quota.")
    st.caption(
        "New data is actually fetched from Twelve Data every ~10 minutes per "
        "timeframe (cached in between). The free API tier allows 800 requests/day; "
        "this pacing uses roughly 576/day with margin to spare, while the page "
        "above can still refresh every 15-60s to feel live in between fetches."
    )
    st.markdown("---")
    alert_threshold = st.slider("Alert threshold: min. historical accuracy (%)", 50, 90, 65,
                                 help="Alerts fire only when this horizon's real walk-forward accuracy meets this bar (full ensemble accuracy at +1, XGBoost-component accuracy at +2 and beyond) AND all 3 ensemble models currently agree on direction. Historical accuracy for this kind of model on a liquid instrument typically lands in the 50-56% range — thresholds much above that may rarely or never trigger, which is expected, not broken.")
    min_rr = st.slider("Alert filter: min. risk:reward", 1.0, 5.0, 1.5, step=0.5,
                        help="Risk is the distance to the nearest supply/demand zone edge that aligns with the predicted direction (or 1x ATR if no aligned zone exists); reward is the cumulative projected move to that horizon. A higher bar means fewer, more selective alerts — the video this was inspired by used 2.5:1.")
    st.markdown("---")
    with st.expander("ICT Kill Zone hours (ET) — adjust if you follow a specific model"):
        st.caption(
            "Exact kill zone hours genuinely vary across ICT courses/sources — these are "
            "the most commonly-cited windows after cross-referencing several. Not one "
            "single 'official' definition; adjust to match whichever model you follow."
        )
        asian_start, asian_end = st.slider("Asian killzone (ET)", 0, 23, (20, 22))
        london_start, london_end = st.slider("London killzone (ET)", 0, 23, (2, 5))
        ny_start, ny_end = st.slider("New York killzone (ET)", 0, 23, (7, 10))
    st.markdown("---")
    st.caption(
        "⚠️ Educational tool, not financial advice. 'AI' means real gradient-boosted "
        "trees, regularized linear regression, and random forests — three genuinely "
        "different model families, ensembled and walk-forward validated on real "
        "data. Check the accuracy panel under each chart before trusting any "
        "prediction; more sophistication makes this more rigorous, not more certain."
    )

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds*1000, key="refresh")

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Live Market Analysis</div>
    <div class="hero-title">Gold Predictor</div>
    <div class="hero-sub">
        3-model ensemble (XGBoost, Ridge, Random Forest) with price-action market structure,
        supply/demand zones, and real quantile-regression uncertainty bands — cross-checked across
        three timeframes for confluence. More rigorous isn't more certain: the accuracy numbers are
        the real historical track record, not a marketing claim.
    </div>
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">{ticker}</div><div class="stat-label">Symbol</div></div>
        <div class="stat-box"><div class="stat-num">3</div><div class="stat-label">Timeframes</div></div>
        <div class="stat-box"><div class="stat-num">{N_HORIZONS}</div><div class="stat-label">Candles Ahead</div></div>
        <div class="stat-box"><div class="stat-num">{datetime.now().strftime('%H:%M:%S')}</div><div class="stat-label">Last Updated</div></div>
    </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# Pass 1: compute every timeframe first, so confluence can compare across them
results_by_tf = {}
with st.status("Analyzing all timeframes...", expanded=False) as status:
    for tf in TIMEFRAMES:
        try:
            status.update(label=f"Analyzing {tf['label']}...")
            result = analyze_timeframe(tf["key"], ticker, use_macro,
                                        (asian_start, asian_end), (london_start, london_end), (ny_start, ny_end))
            if "error" not in result:
                net_bias = sum(result["horizon_preds"]) * 100
                result["net_bias"] = net_bias
                result["direction"] = "UP" if net_bias > 0.05 else "DOWN" if net_bias < -0.05 else "NEUTRAL"
            results_by_tf[tf["key"]] = result
        except Exception as e:
            results_by_tf[tf["key"]] = {"error": f"{type(e).__name__}: {e}"}
    status.update(label="Done.", state="complete")

# Confluence: how many OTHER timeframes agree on direction with this one
for tf_key, result in results_by_tf.items():
    if "error" in result:
        continue
    others = [r["direction"] for k, r in results_by_tf.items() if k != tf_key and "error" not in r]
    agree = sum(1 for d in others if d == result["direction"])
    result["confluence"] = (agree, len(others))

# High-confidence alerts: fires only when THREE conditions all hold for a
# specific timeframe+horizon: (1) real walk-forward accuracy clears the
# threshold - the only number here with actual empirical backing; (2) all 3
# ensemble models currently agree on direction - models agreeing isn't proof
# of being right, but disagreement is a real reason for caution; (3) the
# risk:reward for that horizon clears the configured bar, using the nearest
# aligned supply/demand zone as the risk anchor when one exists.
alerts = []
for tf in TIMEFRAMES:
    result = results_by_tf.get(tf["key"], {})
    if "error" in result:
        continue
    risk = result.get("risk")
    cum_move = result.get("cumulative_move")
    for h in range(1, N_HORIZONS+1):
        acc = result["horizon_accs"].get(h)
        agree_h = result["horizon_agree"][h-1] if h-1 < len(result["horizon_agree"]) else False
        rr_h = abs(cum_move[h-1]) / risk if (risk and cum_move is not None and h-1 < len(cum_move)) else 0
        if acc is not None and acc >= alert_threshold/100 and agree_h and rr_h >= min_rr:
            direction = "UP" if result["horizon_preds"][h-1] > 0 else "DOWN"
            alerts.append({
                "timeframe": tf["label"], "horizon": h, "direction": direction,
                "accuracy": acc, "n_folds": result.get("n_folds", 5), "risk_reward": rr_h,
            })

if alerts:
    for a in alerts:
        st.toast(f"{a['timeframe']} +{a['horizon']}: {a['direction']} — {a['accuracy']*100:.0f}% acc, {a['risk_reward']:.1f}:1 R:R", icon="🔔")
    st.markdown('<div class="hero" style="border-color:' + TS_ACCENT + ';">', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-eyebrow">🔔 {len(alerts)} High-Confidence Signal{"s" if len(alerts)>1 else ""} (≥{alert_threshold}% + agreement + ≥{min_rr}:1 R:R)</div>', unsafe_allow_html=True)
    for a in alerts:
        badge = signal_badge(1 if a["direction"] == "UP" else -1)
        st.markdown(
            f'<div style="margin-top:8px;">{badge} <b>{a["timeframe"]}</b>, candle +{a["horizon"]} — '
            f'<span style="color:{TS_ACCENT};font-weight:800;">{a["accuracy"]*100:.0f}%</span> accuracy '
            f'over {a["n_folds"]} folds, {a["risk_reward"]:.1f}:1 R:R, all 3 models agree.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="hero-sub" style="margin-top:12px;">Even at this threshold, this is a historical hit rate on '
        f'past data, current model agreement, and a favorable risk:reward setup — not a guarantee for this '
        f'specific instance. Small-sample fold results can be noisy; treat repeated, stable alerts across '
        f'refreshes as more meaningful than a single one.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

cols = st.columns(len(TIMEFRAMES))

for col, tf in zip(cols, TIMEFRAMES):
    result = results_by_tf[tf["key"]]
    with col:
        with st.container(border=True):
            st.markdown(f"**{tf['label']}**")
            try:
                if "error" in result:
                    st.warning(result["error"])
                    continue

                bar_age_min = (pd.Timestamp.now(tz=result["last_bar_time"].tz) - result["last_bar_time"]).total_seconds() / 60
                freshness = "🟢" if bar_age_min < 30 else "🟡" if bar_age_min < 120 else "🔴"
                st.caption(f"{freshness} Last bar: {bar_age_min:.0f} min ago")

                ghost = build_ghost_candles(result["current_close"], result["current_atr"], result["horizon_preds"], result.get("horizon_bands"))
                fig = render_chart(result["raw"], ghost)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                badge_col, price_col = st.columns([1, 1])
                with badge_col:
                    st.markdown(signal_badge(result["net_bias"]), unsafe_allow_html=True)
                with price_col:
                    st.markdown(f"<div style='text-align:right; font-weight:700;'>{result['net_bias']:+.2f}%</div>", unsafe_allow_html=True)

                agree, total = result["confluence"]
                if total > 0:
                    conf_msg = f"{agree}/{total} other timeframes agree" if agree > 0 else f"0/{total} agree — conflicting signals across timeframes"
                    st.caption(f"🔗 Confluence: {conf_msg}")

                struct = result.get("structure_trend", 0)
                struct_label = "Bullish" if struct > 0 else "Bearish" if struct < 0 else "Undetermined"
                zone = result.get("zone")
                zone_msg = f", {zone['direction']} zone at {zone['low']:.2f}-{zone['high']:.2f}" if zone else ", no recent zone found"
                st.caption(f"📐 Structure: {struct_label}{zone_msg}")

                kz = result.get("in_killzone", {})
                active_kz = [name.upper() for name, active in kz.items() if active]
                kz_msg = " + ".join(active_kz) if active_kz else "none active"
                pd_val = result.get("premium_discount", 0.5)
                pd_label = "Premium" if pd_val > 0.618 else "Discount" if pd_val < 0.382 else "Equilibrium"
                st.caption(f"🕐 Kill zone: {kz_msg} · {pd_label} ({pd_val*100:.0f}%)")

                with st.expander("ICT concepts (FVG, Order Block, liquidity sweep)"):
                    fvg = result.get("nearest_fvg")
                    ob = result.get("order_block")
                    sweep = result.get("liquidity_sweep")
                    st.write(f"Nearest unfilled FVG: {fvg['direction']} at {fvg['low']:.2f}-{fvg['high']:.2f}" if fvg else "Nearest unfilled FVG: none found")
                    st.write(f"Last Order Block: {ob['direction']} at {ob['low']:.2f}-{ob['high']:.2f}" if ob else "Last Order Block: none found")
                    st.write(f"Liquidity sweep (last 20 bars): {sweep}" if sweep else "Liquidity sweep (last 20 bars): none detected")

                w = result["ensemble_weights"]
                st.caption(f"Ensemble: XGBoost {w['xgb']*100:.0f}% / Ridge {w['ridge']*100:.0f}% / Random Forest {w['rf']*100:.0f}% (weighted by validation accuracy)")

                with st.expander("Per-candle predictions & accuracy by horizon"):
                    st.caption("Candle +1's accuracy is the full blended ensemble's; +2 through "
                               f"+{N_HORIZONS}'s are the XGBoost component only (the cheaper "
                               "check run at every horizon) — both are real walk-forward numbers, "
                               "just not quite the same measurement.")
                    for h, pred in enumerate(result["horizon_preds"], start=1):
                        acc = result["horizon_accs"].get(h)
                        acc_label = "ensemble accuracy" if h == 1 else "XGBoost-component accuracy"
                        acc_str = f" — {acc*100:.1f}% {acc_label}" if acc is not None else ""
                        st.write(f"Ghost candle +{h}: {pred*100:+.3f}% ({'▲' if pred>0 else '▼'}){acc_str}")

                if result["dir_acc_h1"] is not None:
                    st.metric("Walk-forward accuracy (1-candle, full ensemble)", f"{result['dir_acc_h1']*100:.1f}%")
                    if result["dir_acc_h1"] < 0.53:
                        st.caption("Close to a coin flip — the expected, honest result here, not a bug.")
                else:
                    st.caption("Not enough history yet for a walk-forward check.")

            except Exception as e:
                st.error(f"Something failed while rendering {tf['label']}: {type(e).__name__}: {e}")
                st.caption("This is a real bug if you see it repeatedly — copy this exact message back so it can be fixed.")

st.markdown("---")
st.caption(
    "Deploy as an always-on website for free: push this file + requirements.txt "
    "(streamlit, requests, pandas, numpy, scikit-learn, xgboost, plotly, streamlit-autorefresh) "
    "to a GitHub repo, connect it at share.streamlit.io, and add your free Twelve Data API key "
    "under the app's Settings → Secrets as TWELVEDATA_API_KEY."
)
