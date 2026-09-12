"""
gold_multi_timeframe_predictor.py
A live website that continuously analyzes gold on 15-minute, 1-hour, and
4-hour timeframes and projects the next 5 candles on each, using gradient-
boosted trees (XGBoost) trained separately per horizon, plus real macro data
(DXY, 10Y yield, VIX).

Run locally with:
    streamlit run gold_multi_timeframe_predictor.py
Or deploy for free at share.streamlit.io (push this file + requirements to
GitHub, connect the repo) for a genuine always-on public URL - see the note
at the bottom of this file.

⚠️ HONEST FRAMING, STATED ONCE: "AI" here means real gradient-boosted trees,
trained and validated on real historical data - not magic, and not a
guarantee. The walk-forward accuracy panel exists specifically so you can
see the model's actual out-of-sample track record rather than take my word
for it. On a liquid, efficiently-traded instrument like gold, expect that
number to land only modestly above 50% - that's the normal, honest result
for this class of model, not a bug. Educational tool, not financial advice.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from xgboost import XGBRegressor

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="Gold Multi-Timeframe Predictor", layout="wide")

TIMEFRAMES = [
    {"label": "15 Minute", "key": "15m", "yf_interval": "15m", "yf_period": "59d"},
    {"label": "1 Hour",    "key": "1h",  "yf_interval": "60m", "yf_period": "729d"},
    {"label": "4 Hour",    "key": "4h",  "yf_interval": "60m", "yf_period": "729d", "resample": "4h"},
]
N_HORIZONS = 5


# =============================================================================
# DATA FETCHING
# =============================================================================
def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = _flatten_columns(df)
    return df.dropna()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro_daily():
    tickers = {"dxy": "DX-Y.NYB", "yield_10y": "^TNX", "vix": "^VIX"}
    frames = {}
    for name, tk in tickers.items():
        try:
            d = yf.download(tk, interval="1d", period="729d", progress=False)
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
    macro = pd.concat(frames.values(), axis=1)
    return macro.ffill().dropna(how="all")


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
# FEATURES (computed once per timeframe, reused across all 5 horizon models)
# =============================================================================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def build_features(raw, use_macro):
    out = raw.copy()
    close = out["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    for lag in [1, 2, 3, 5]:
        out[f"ret_1_lag{lag}"] = out["ret_1"].shift(lag)

    out["sma_10"] = close.rolling(10).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_ratio"] = out["sma_10"] / out["sma_50"] - 1
    out["rsi_14"] = rsi(close, 14)
    macd_line, sig_line, hist = macd(close)
    out["macd_hist"] = hist / close
    out["volatility_10"] = out["ret_1"].rolling(10).std()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
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

    atr = (out["High"] - out["Low"]).rolling(14).mean()
    out["atr"] = atr

    # multi-horizon targets: forward return h bars ahead, for h=1..N_HORIZONS
    for h in range(1, N_HORIZONS+1):
        out[f"target_h{h}"] = close.shift(-h) / close - 1

    out = out.dropna()
    return out, feature_cols


# =============================================================================
# MODEL
# =============================================================================
def train_fast_xgb(X, y):
    model = XGBRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42,
    )
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


# =============================================================================
# PER-TIMEFRAME PIPELINE
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def analyze_timeframe(tf_key, ticker, use_macro):
    tf = next(t for t in TIMEFRAMES if t["key"] == tf_key)
    raw = fetch_ohlc(ticker, tf["yf_interval"], tf["yf_period"])
    if tf.get("resample"):
        raw = resample_ohlc(raw, tf["resample"])
    if raw.empty or len(raw) < 200:
        return None

    macro = fetch_macro_daily() if use_macro else pd.DataFrame()
    raw = merge_macro_onto_intraday(raw, macro) if use_macro else raw

    feat_df, feature_cols = build_features(raw, use_macro)
    if len(feat_df) < 150:
        return None

    X = feat_df[feature_cols].values
    last_row = feat_df[feature_cols].iloc[[-1]].values
    current_close = float(feat_df["Close"].iloc[-1])
    current_atr = float(feat_df["atr"].iloc[-1])

    horizon_preds = []
    for h in range(1, N_HORIZONS+1):
        y = feat_df[f"target_h{h}"].values
        model = train_fast_xgb(X, y)
        pred_return = float(model.predict(last_row)[0])
        horizon_preds.append(pred_return)

    dir_acc, rmse = walk_forward_accuracy(X, feat_df["target_h1"].values)

    return {
        "raw": raw, "feat_df": feat_df, "current_close": current_close,
        "current_atr": current_atr, "horizon_preds": horizon_preds,
        "dir_acc_h1": dir_acc, "rmse_h1": rmse,
    }


def build_projected_candles(current_close, current_atr, horizon_preds):
    candles = []
    open_price = current_close
    for h, pred_return in enumerate(horizon_preds, start=1):
        close_price = open_price * (1 + pred_return)
        wick = current_atr * (0.5 + 0.15*h)  # uncertainty grows with horizon
        high = max(open_price, close_price) + wick*0.5
        low = min(open_price, close_price) - wick*0.5
        candles.append({"open": open_price, "high": high, "low": low, "close": close_price})
        open_price = close_price
    return candles


# =============================================================================
# UI
# =============================================================================
st.title("🪙 Gold Multi-Timeframe Predictor")
st.caption("Continuously analyzes 15m / 1h / 4h gold data and projects the next 5 candles on each, using XGBoost trained separately per horizon.")

st.info(
    "**How to read the projected candles:** direction and size come from a real model "
    "prediction for that specific horizon; the widening range further out reflects growing "
    "uncertainty, not decoration. Check the walk-forward accuracy under each chart before "
    "trusting any of it — that's the model's actual historical track record, not a marketing number.",
    icon="ℹ️",
)

with st.sidebar:
    st.title("Settings")
    ticker = st.selectbox("Ticker", ["GC=F", "XAUUSD=X"], index=0)
    use_macro = st.checkbox("Use macro features (DXY, 10Y yield, VIX)", value=True)
    refresh_seconds = st.slider("Auto-refresh (seconds)", 60, 1800, 300)
    st.markdown("---")
    st.caption("⚠️ Educational tool, not financial advice. Not a guarantee of future price action.")

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds*1000, key="refresh")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

cols = st.columns(len(TIMEFRAMES))

for col, tf in zip(cols, TIMEFRAMES):
    with col:
        st.subheader(tf["label"])
        try:
            result = analyze_timeframe(tf["key"], ticker, use_macro)
        except Exception as e:
            st.error(f"Couldn't load data: {e}")
            continue

        if result is None:
            st.warning("Not enough data returned for this timeframe yet.")
            continue

        projected = build_projected_candles(result["current_close"], result["current_atr"], result["horizon_preds"])

        hist = result["raw"].tail(60)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=list(range(len(hist))), open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="History",
            increasing_line_color="#2ECC71", decreasing_line_color="#E74C3C",
        ))

        proj_x = list(range(len(hist), len(hist)+N_HORIZONS))
        fig.add_trace(go.Candlestick(
            x=proj_x,
            open=[c["open"] for c in projected], high=[c["high"] for c in projected],
            low=[c["low"] for c in projected], close=[c["close"] for c in projected],
            name="Projected", increasing_line_color="rgba(46,204,113,0.45)",
            decreasing_line_color="rgba(231,76,60,0.45)",
        ))
        fig.add_vline(x=len(hist)-0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10), xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        last_pred = result["horizon_preds"][-1]
        overall_dir = "UP" if sum(result["horizon_preds"]) > 0 else "DOWN"
        st.metric("Net 5-candle bias", overall_dir, f"{sum(result['horizon_preds'])*100:+.2f}%")

        with st.expander("Per-candle predictions"):
            for h, pred in enumerate(result["horizon_preds"], start=1):
                st.write(f"Candle +{h}: {pred*100:+.3f}% ({'UP' if pred>0 else 'DOWN'})")

        if result["dir_acc_h1"] is not None:
            acc_color = "normal" if result["dir_acc_h1"] >= 0.53 else "inverse"
            st.metric("Walk-forward accuracy (1-candle horizon)", f"{result['dir_acc_h1']*100:.1f}%")
            if result["dir_acc_h1"] < 0.53:
                st.caption("Close to a coin flip — the expected, honest result here, not a bug.")
        else:
            st.caption("Not enough history for a walk-forward check on this timeframe yet.")

st.markdown("---")
st.caption(
    "Deploy this as a real always-on website for free: push this file + a requirements.txt "
    "(streamlit, yfinance, pandas, numpy, xgboost, plotly, streamlit-autorefresh) to a GitHub "
    "repo, then connect it at share.streamlit.io. No server management needed."
)
