import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt

TICKERS = ["1306.T", "1321.T"]

INITIAL_CAPITAL = 20000
HOLD_DAYS = 4
PULLBACK = 0.004
MA_DAYS = 25
RSI_DAYS = 14


def load_data(t):
    df = yf.download(t, period="1y", progress=False, auto_adjust=False)

    if df is None or df.empty:
        return pd.DataFrame()

    # MultiIndex対策
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

    if "Close" not in df.columns or "Volume" not in df.columns:
        print(f"{t}: missing columns -> {list(df.columns)}")
        return pd.DataFrame()

    close = df["Close"]
    volume = df["Volume"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["Close"] = close
    df["Volume"] = volume
    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()

    return df.dropna()


data = {t: load_data(t) for t in TICKERS}

cash = INITIAL_CAPITAL
equity_curve = []
positions = []

dates = sorted(set().union(*[df.index for df in data.values() if not df.empty]))

for date in dates:
    # EXIT
    new = []
    for p in positions:
        df = data[p["ticker"]]
        if date not in df.index:
            new.append(p)
            continue

        bars = ((df.index > p["entry_date"]) & (df.index <= date)).sum()

        if bars >= HOLD_DAYS:
            price = float(df.loc[date, "Close"])
            cash += price * p["qty"]
        else:
            new.append(p)

    positions = new

    # ENTRY
    if not positions:
        cands = []
        for t, df in data.items():
            if df.empty or date not in df.index:
                continue

            i = df.index.get_loc(date)
            if i < 30:
                continue

            c = float(df["Close"].iloc[i])
            prev = float(df["Close"].iloc[i - 1])

            if c < float(df["MA"].iloc[i]):
                continue
            if c > prev * (1 - PULLBACK):
                continue

            cands.append((t, float(df["RSI"].iloc[i]), i))

        cands.sort(key=lambda x: x[1])

        if cands:
            t, rsi, i = cands[0]
            price = float(data[t]["Close"].iloc[i])
            qty = int(cash * 0.7 // price)

            if qty > 0:
                cash -= price * qty
                positions.append({"ticker": t, "entry_date": date, "qty": qty})

    # EQUITY
    val = 0.0
    for p in positions:
        df = data[p["ticker"]]
        if date in df.index:
            val += float(df.loc[date, "Close"]) * p["qty"]

    equity_curve.append({"date": date, "equity": cash + val})

eq = pd.DataFrame(equity_curve)
eq.to_csv("equity_curve_v97_etf.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(eq["date"], eq["equity"])
plt.title("ETF Backtest Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()
