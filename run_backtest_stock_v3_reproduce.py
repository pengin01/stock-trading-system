import pandas as pd
import yfinance as yf
import ta

# =========================
# PARAMETERS
# =========================
PERIOD = "5y"  # 軽く確認するため短め
RSI_MAX = 52
PULLBACK_PCT = 0.02
MIN_VALUE20 = 500_000_000

TICKERS = [
    "7203.T",
    "6758.T",
    "8306.T",
    "8035.T",
    "9984.T",
    "9432.T",
    "4063.T",
    "6861.T",
    "6501.T",
    "6098.T",
]

MA_DAYS = 25
MA_SLOPE_DAYS = 5
RSI_DAYS = 14


# =========================
# LOAD
# =========================
def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if any(x in price_names for x in level0):
            df.columns = df.columns.get_level_values(0)
        elif any(x in price_names for x in level1):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join([str(a), str(b)]) for a, b in df.columns]

    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"{ticker}: missing column {col}")

        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"{ticker}: download error: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        df = normalize_ohlcv(df, ticker)
    except Exception as e:
        print(f"{ticker}: normalize error: {e}")
        return pd.DataFrame()

    close = df["Close"]
    vol = df["Volume"]

    df["MA"] = close.rolling(MA_DAYS).mean()
    df["MA_SLOPE"] = df["MA"] - df["MA"].shift(MA_SLOPE_DAYS)
    df["RSI"] = ta.momentum.RSIIndicator(close, RSI_DAYS).rsi()
    df["VALUE20"] = (close * vol).rolling(20).mean()

    return df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MA",
            "MA_SLOPE",
            "RSI",
            "VALUE20",
        ]
    ).copy()


# =========================
# MAIN
# =========================
data = {t: load(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

counts = {
    "total": 0,
    "ma_ok": 0,
    "slope_ok": 0,
    "pullback_ok": 0,
    "rsi_ok": 0,
    "value_ok": 0,
    "final": 0,
}

examples = []

for t, df in data.items():
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        counts["total"] += 1

        close = float(row["Close"])
        prev_close = float(prev["Close"])

        cond_ma = close > float(row["MA"])
        cond_slope = float(row["MA_SLOPE"]) > 0
        cond_pullback = close <= prev_close * (1 - PULLBACK_PCT)
        cond_rsi = float(row["RSI"]) <= RSI_MAX
        cond_val = float(row["VALUE20"]) >= MIN_VALUE20

        if cond_ma:
            counts["ma_ok"] += 1
        if cond_slope:
            counts["slope_ok"] += 1
        if cond_pullback:
            counts["pullback_ok"] += 1
        if cond_rsi:
            counts["rsi_ok"] += 1
        if cond_val:
            counts["value_ok"] += 1

        if cond_ma and cond_slope and cond_pullback and cond_rsi and cond_val:
            counts["final"] += 1

            if len(examples) < 20:
                examples.append(
                    {
                        "ticker": t,
                        "date": df.index[i].strftime("%Y-%m-%d"),
                        "close": round(close, 2),
                        "prev_close": round(prev_close, 2),
                        "drop_pct": round((close / prev_close - 1) * 100, 2),
                        "ma": round(float(row["MA"]), 2),
                        "ma_slope": round(float(row["MA_SLOPE"]), 4),
                        "rsi": round(float(row["RSI"]), 2),
                        "value20": int(float(row["VALUE20"])),
                    }
                )

print("\n=== FILTER DEBUG ===")
for k, v in counts.items():
    print(f"{k:20}: {v}")

if examples:
    print("\n=== EXAMPLES (first 20) ===")
    print(pd.DataFrame(examples).to_string(index=False))
else:
    print("\nNo final matches found.")
