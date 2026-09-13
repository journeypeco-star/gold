"""
gold_multi_timeframe_predictor.py
A live website that continuously analyzes gold on 15-minute, 1-hour, and
4-hour timeframes and projects the next 4 candles on each - custom-built
charts styled after TradingView's dark theme, powered by XGBoost trained
separately per horizon, plus real macro data (DXY, 10Y yield, VIX).

Run locally with:
    streamlit run gold_multi_timeframe_predictor.py
Or deploy for free at share.streamlit.io - push this file + requirements.txt
to a GitHub repo, connect it there, done.

⚠️ HONEST FRAMING, STATED ONCE: "AI" here means real gradient-boosted trees,
trained and walk-forward validated on real historical data - not magic, and
not a guarantee. Check the accuracy panel under each chart before trusting
any of it; on a liquid, efficiently-traded instrument like gold, expect that
number only modestly above 50% - the normal, honest result for this class of
model, not a bug. Educational tool, not financial advice.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
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
N_HORIZONS = 4
GOLD_SYMBOL = "XAU/USD"

# Chart candle colors - kept as the universal trading convention (teal-green
# up / red down), independent of site branding below.
TV_UP        = "#26A69A"
TV_DOWN      = "#EF5350"
TV_UP_GHOST  = "rgba(38,166,154,0.35)"
TV_DOWN_GHOST= "rgba(239,83,80,0.35)"
GOLD_ACCENT  = "#EBA23A"

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
                "symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": api_key,
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
            df["datetime"] = pd.to_datetime(df["datetime"])
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


@st.cache_data(ttl=900, show_spinner=False)
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


def build_features(raw, use_macro):
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

    feature_cols = ["ret_1", "ret_3", "ret_5", "ret_1_lag1", "ret_1_lag2", "ret_1_lag3",
                     "ret_1_lag5", "sma_ratio", "rsi_14", "macd_hist", "volatility_10",
                     "bb_pct_b", "high_low_range"]

    if use_macro and "dxy" in out.columns:
        out["dxy_ret_1"] = out["dxy"].pct_change(1)
        feature_cols.append("dxy_ret_1")
    if use_macro and "yield_10y" in out.columns:
        out["yield_change_1"] = out["yield_10y"].diff(1)
        feature_cols.append("yield_change_1")
    if use_macro and "vix" in out.columns:
        out["vix_change_1"] = out["vix"].pct_change(1)
        feature_cols.append("vix_change_1")

    out["atr"] = (out["High"] - out["Low"]).rolling(14).mean()

    for h in range(1, N_HORIZONS+1):
        out[f"target_h{h}"] = close.shift(-h) / close - 1

    return out.dropna(), feature_cols


# =============================================================================
# MODEL: an XGBoost + Ridge ensemble, weighted by which one actually
# validates better (not a fixed 50/50 guess) - two genuinely different model
# families tend to have less-correlated errors than two variants of the same
# model, which is where an ensemble's benefit, if any, comes from.
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


@st.cache_data(ttl=900, show_spinner=False)
def analyze_timeframe(tf_key, symbol, use_macro):
    tf = next(t for t in TIMEFRAMES if t["key"] == tf_key)
    raw = fetch_ohlc(symbol, tf["td_interval"])
    if raw.empty:
        return {"error": f"Received data but it was empty after processing for {tf['label']}."}
    if len(raw) < 200:
        return {"error": f"Only got {len(raw)} bars for {tf['label']} (need 200+). Twelve Data's free tier may cap history depth for this interval."}

    macro = fetch_macro_daily() if use_macro else pd.DataFrame()
    raw = merge_macro_onto_intraday(raw, macro) if use_macro else raw

    feat_df, feature_cols = build_features(raw, use_macro)
    if len(feat_df) < 150:
        return {"error": f"Only {len(feat_df)} usable rows survived feature engineering for {tf['label']} (need 150+)."}

    X = feat_df[feature_cols].values
    last_row = feat_df[feature_cols].iloc[[-1]].values
    current_close = float(feat_df["Close"].iloc[-1])
    current_atr = float(feat_df["atr"].iloc[-1])

    # Validation-weighted ensemble: figure out how much to trust each model
    # family using held-out accuracy on the nearest horizon, then reuse those
    # weights across all horizons (recomputing per-horizon would be 4x the
    # cost for a weighting that doesn't change much bar to bar).
    y_h1 = feat_df["target_h1"].values
    xgb_acc_h1, xgb_rmse_h1 = walk_forward_accuracy(X, y_h1, train_fast_xgb, predict_xgb)
    ridge_acc_h1, ridge_rmse_h1 = walk_forward_accuracy(X, y_h1, train_ridge, predict_ridge)

    if xgb_rmse_h1 and ridge_rmse_h1 and xgb_rmse_h1 > 0 and ridge_rmse_h1 > 0:
        inv_x, inv_r = 1/xgb_rmse_h1, 1/ridge_rmse_h1
        w_xgb, w_ridge = inv_x/(inv_x+inv_r), inv_r/(inv_x+inv_r)
    else:
        w_xgb, w_ridge = 1.0, 0.0

    horizon_preds = []
    horizon_accs = {1: xgb_acc_h1}
    for h in range(1, N_HORIZONS+1):
        y_h = feat_df[f"target_h{h}"].values
        xgb_model = train_fast_xgb(X, y_h)
        ridge_model = train_ridge(X, y_h)
        combined = w_xgb * float(predict_xgb(xgb_model, last_row)[0]) + w_ridge * float(predict_ridge(ridge_model, last_row)[0])
        horizon_preds.append(combined)
        if h > 1:
            acc_h, _ = walk_forward_accuracy(X, y_h, train_fast_xgb, predict_xgb)
            horizon_accs[h] = acc_h

    return {
        "raw": raw, "current_close": current_close, "current_atr": current_atr,
        "horizon_preds": horizon_preds, "horizon_accs": horizon_accs,
        "dir_acc_h1": xgb_acc_h1, "rmse_h1": xgb_rmse_h1,
        "ensemble_weights": {"xgb": w_xgb, "ridge": w_ridge},
    }


def build_ghost_candles(current_close, current_atr, horizon_preds):
    candles, open_price = [], current_close
    for h, pred_return in enumerate(horizon_preds, start=1):
        close_price = open_price * (1 + pred_return)
        wick = current_atr * (0.5 + 0.15*h)
        candles.append({
            "open": open_price, "close": close_price,
            "high": max(open_price, close_price) + wick*0.5,
            "low": min(open_price, close_price) - wick*0.5,
        })
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
                        text="GHOST PREDICTION", showarrow=False, font=dict(color=GOLD_ACCENT, size=10),
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
    refresh_seconds = st.slider("Auto-refresh (seconds)", 60, 1800, 300)
    st.markdown("---")
    st.caption(
        "⚠️ Educational tool, not financial advice. 'AI' means real gradient-boosted "
        "trees, walk-forward validated on real data — check the accuracy panel under "
        "each chart before trusting any prediction."
    )

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds*1000, key="refresh")

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Live Market Analysis</div>
    <div class="hero-title">Gold Predictor</div>
    <div class="hero-sub">
        Ensemble of XGBoost + Ridge regression, weighted by which one actually validates better —
        cross-checked across three timeframes for confluence, with full per-horizon accuracy shown
        below every chart. More rigorous isn't more certain: the accuracy numbers are the real
        historical track record, not a marketing claim.
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
            result = analyze_timeframe(tf["key"], ticker, use_macro)
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

                ghost = build_ghost_candles(result["current_close"], result["current_atr"], result["horizon_preds"])
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

                w = result["ensemble_weights"]
                st.caption(f"Ensemble: XGBoost {w['xgb']*100:.0f}% / Ridge {w['ridge']*100:.0f}% (weighted by validation accuracy)")

                with st.expander("Per-candle predictions & accuracy by horizon"):
                    for h, pred in enumerate(result["horizon_preds"], start=1):
                        acc = result["horizon_accs"].get(h)
                        acc_str = f" — {acc*100:.1f}% walk-forward accuracy" if acc is not None else ""
                        st.write(f"Ghost candle +{h}: {pred*100:+.3f}% ({'▲' if pred>0 else '▼'}){acc_str}")

                if result["dir_acc_h1"] is not None:
                    st.metric("Walk-forward accuracy (1-candle)", f"{result['dir_acc_h1']*100:.1f}%")
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
