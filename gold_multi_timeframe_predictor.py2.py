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
import streamlit as st
import yfinance as yf
from xgboost import XGBRegressor

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="Gold Predictor", layout="wide", initial_sidebar_state="expanded")

TIMEFRAMES = [
    {"label": "15 Minute", "key": "15m", "yf_interval": "15m", "yf_period": "59d"},
    {"label": "1 Hour",    "key": "1h",  "yf_interval": "60m", "yf_period": "729d"},
    {"label": "4 Hour",    "key": "4h",  "yf_interval": "60m", "yf_period": "729d", "resample": "4h"},
]
N_HORIZONS = 4

# TradingView's actual dark-theme palette, for real visual continuity
TV_BG        = "#131722"
TV_PANEL_BG  = "#1E222D"
TV_BORDER    = "#2A2E39"
TV_TEXT      = "#D1D4DC"
TV_MUTED     = "#787B86"
TV_UP        = "#26A69A"
TV_DOWN      = "#EF5350"
TV_UP_GHOST  = "rgba(38,166,154,0.35)"
TV_DOWN_GHOST= "rgba(239,83,80,0.35)"
GOLD_ACCENT  = "#EBA23A"


# =============================================================================
# STYLE (custom CSS - this is what makes it "our own" UI rather than default Streamlit)
# =============================================================================
st.markdown(f"""
<style>
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stApp {{ background-color: {TV_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {TV_PANEL_BG}; border-right: 1px solid {TV_BORDER}; }}
    h1, h2, h3 {{ color: #F1F1F1 !important; font-family: -apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, sans-serif; }}
    p, span, label, .stMarkdown {{ color: {TV_TEXT}; }}
    div[data-testid="stMetric"] {{
        background-color: {TV_PANEL_BG}; border: 1px solid {TV_BORDER}; border-radius: 8px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {TV_MUTED}; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {TV_PANEL_BG}; border: 1px solid {TV_BORDER} !important; border-radius: 10px;
    }}
    .badge {{
        display:inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 13px;
    }}
    .badge-buy {{ background: rgba(38,166,154,0.18); color: {TV_UP}; border: 1px solid {TV_UP}; }}
    .badge-sell {{ background: rgba(239,83,80,0.18); color: {TV_DOWN}; border: 1px solid {TV_DOWN}; }}
    .badge-neutral {{ background: rgba(120,123,134,0.18); color: {TV_MUTED}; border: 1px solid {TV_MUTED}; }}
    .ticker-row {{ display:flex; align-items:baseline; gap:14px; }}
    .ticker-price {{ font-size: 32px; font-weight: 700; color: #F1F1F1; }}
    .ticker-label {{ font-size: 13px; color: {TV_MUTED}; text-transform: uppercase; letter-spacing: 1px; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FETCHING (hardened against Yahoo's cloud-IP rate limiting)
# =============================================================================
def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download_with_retry(ticker, interval, period, attempts=3):
    last_err = None
    for attempt in range(attempts):
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False, timeout=15)
            df = _flatten_columns(df).dropna()
            if not df.empty:
                return df
            last_err = f"Yahoo returned an empty dataset for {ticker} ({interval}, {period}) - usually a rate limit."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {ticker} after {attempts} attempts. Last error: {last_err}")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlc(ticker, interval, period):
    return _download_with_retry(ticker, interval, period)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_daily():
    tickers = {"dxy": "DX-Y.NYB", "yield_10y": "^TNX", "vix": "^VIX"}
    frames = {}
    for name, tk in tickers.items():
        try:
            d = yf.download(tk, interval="1d", period="729d", progress=False, timeout=15)
            d = _flatten_columns(d)
            if not d.empty and "Close" in d.columns:
                col = d["Close"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                frames[name] = col.rename(name)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1).ffill().dropna(how="all")


def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


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
# MODEL
# =============================================================================
def train_fast_xgb(X, y):
    model = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          objective="reg:squarederror", random_state=42)
    model.fit(X, y)
    return model


def walk_forward_accuracy(X, y, n_folds=5, min_train=150):
    n = len(X)
    fold_size = (n - min_train) // n_folds
    if fold_size <= 5:
        return None, None
    dir_accs, rmses = [], []
    for f in range(n_folds):
        train_end = min_train + f*fold_size
        test_end = n if f == n_folds-1 else train_end + fold_size
        model = train_fast_xgb(X[:train_end], y[:train_end])
        pred = model.predict(X[train_end:test_end])
        actual = y[train_end:test_end]
        dir_accs.append(((pred > 0) == (actual > 0)).mean())
        rmses.append(np.sqrt(np.mean((pred-actual)**2)))
    return float(np.mean(dir_accs)), float(np.mean(rmses))


@st.cache_data(ttl=900, show_spinner=False)
def analyze_timeframe(tf_key, ticker, use_macro):
    tf = next(t for t in TIMEFRAMES if t["key"] == tf_key)
    raw = fetch_ohlc(ticker, tf["yf_interval"], tf["yf_period"])
    if tf.get("resample"):
        raw = resample_ohlc(raw, tf["resample"])
    if raw.empty:
        return {"error": f"Received data but it was empty after processing for {tf['label']}."}
    if len(raw) < 200:
        return {"error": f"Only got {len(raw)} bars for {tf['label']} (need 200+). Yahoo may have limited history this time."}

    macro = fetch_macro_daily() if use_macro else pd.DataFrame()
    raw = merge_macro_onto_intraday(raw, macro) if use_macro else raw

    feat_df, feature_cols = build_features(raw, use_macro)
    if len(feat_df) < 150:
        return {"error": f"Only {len(feat_df)} usable rows survived feature engineering for {tf['label']} (need 150+)."}

    X = feat_df[feature_cols].values
    last_row = feat_df[feature_cols].iloc[[-1]].values
    current_close = float(feat_df["Close"].iloc[-1])
    current_atr = float(feat_df["atr"].iloc[-1])

    horizon_preds = []
    for h in range(1, N_HORIZONS+1):
        y = feat_df[f"target_h{h}"].values
        model = train_fast_xgb(X, y)
        horizon_preds.append(float(model.predict(last_row)[0]))

    dir_acc, rmse = walk_forward_accuracy(X, feat_df["target_h1"].values)

    return {
        "raw": raw, "current_close": current_close, "current_atr": current_atr,
        "horizon_preds": horizon_preds, "dir_acc_h1": dir_acc, "rmse_h1": rmse,
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
    ticker = st.selectbox("Ticker", ["GC=F", "XAUUSD=X"], index=0)
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
<div class="ticker-row">
    <span style="font-size:28px;">🪙</span>
    <div>
        <div style="font-size:22px; font-weight:700; color:#F1F1F1;">Gold Predictor</div>
        <div class="ticker-label">{ticker} · Live · Updated {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.write("")

cols = st.columns(len(TIMEFRAMES))

for col, tf in zip(cols, TIMEFRAMES):
    with col:
        with st.container(border=True):
            st.markdown(f"**{tf['label']}**")
            try:
                result = analyze_timeframe(tf["key"], ticker, use_macro)
            except Exception as e:
                st.error(f"Couldn't load data: {e}")
                st.caption("Almost always Yahoo rate-limiting shared cloud IPs — a known issue, not your setup. Retries automatically next refresh.")
                continue

            if "error" in result:
                st.warning(result["error"])
                continue

            ghost = build_ghost_candles(result["current_close"], result["current_atr"], result["horizon_preds"])
            fig = render_chart(result["raw"], ghost)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            net_bias = sum(result["horizon_preds"]) * 100
            badge_col, price_col = st.columns([1, 1])
            with badge_col:
                st.markdown(signal_badge(net_bias), unsafe_allow_html=True)
            with price_col:
                st.markdown(f"<div style='text-align:right; font-weight:700;'>{net_bias:+.2f}%</div>", unsafe_allow_html=True)

            with st.expander("Per-candle predictions"):
                for h, pred in enumerate(result["horizon_preds"], start=1):
                    st.write(f"Ghost candle +{h}: {pred*100:+.3f}% ({'▲' if pred>0 else '▼'})")

            if result["dir_acc_h1"] is not None:
                st.metric("Walk-forward accuracy (1-candle)", f"{result['dir_acc_h1']*100:.1f}%")
                if result["dir_acc_h1"] < 0.53:
                    st.caption("Close to a coin flip — the expected, honest result here, not a bug.")
            else:
                st.caption("Not enough history yet for a walk-forward check.")

st.markdown("---")
st.caption(
    "Deploy as an always-on website for free: push this file + requirements.txt "
    "(streamlit, yfinance, pandas, numpy, scikit-learn, xgboost, plotly, streamlit-autorefresh) "
    "to a GitHub repo, then connect it at share.streamlit.io."
)
