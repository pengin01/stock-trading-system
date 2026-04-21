import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import ta


# =========================
# MAIN2 STRATEGY
# =========================
PERIOD = "10y"
INITIAL_CAPITAL = 300000

LOT_SIZE = 100
MAX_POSITIONS = 2
RISK_RATIO = 1.0

MA_DAYS = 25
MA_SLOPE_DAYS = 5

RSI_DAYS = 14
RSI_MAX = 50

PULLBACK_PCT = 0.02
MIN_VALUE20 = 500_000_000

TAKE_PROFIT_PCT = 0.09
STOP_LOSS_PCT = 0.02
HOLD_DAYS = 4

BUY_FEE_PCT = 0.0005
SELL_FEE_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

RESULT_DIR = "backtest_results_main2"
UNIVERSE_FILE = "jp_universe.csv"

USE_MARKET_FILTER = True
MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200


# =========================
# DATA CLASS
# =========================
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int


# =========================
# HELPERS
# =========================
def ensure_dir() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_universe() -> List[str]:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"{UNIVERSE_FILE} not found")

    df = pd.read_csv(UNIVERSE_FILE)
    raw = df.iloc[:, 0].astype(str).str.strip()

    tickers = []
    for x in raw:
        if x.endswith(".T"):
            tickers.append(x)
        else:
            tickers.append(f"{x}.T")

    return sorted(set(tickers))


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


def load_stock_data(ticker: str) -> pd.DataFrame:
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


def load_market() -> pd.DataFrame:
    if not USE_MARKET_FILTER:
        return pd.DataFrame()

    try:
        df = yf.download(
            MARKET_TICKER, period=PERIOD, progress=False, auto_adjust=False
        )
    except Exception as e:
        print(f"{MARKET_TICKER}: download error: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df = normalize_ohlcv(df, MARKET_TICKER)
    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()
    return df.dropna(subset=["Close", "MA200"]).copy()


def calc_max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


# =========================
# BACKTEST
# =========================
def run() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    universe = load_universe()
    data = {t: load_stock_data(t) for t in universe}
    data = {k: v for k, v in data.items() if not v.empty}

    print("universe count:", len(universe))
    print("loaded data count:", len(data))

    if not data:
        return pd.DataFrame(), pd.DataFrame(), []

    market = load_market()
    dates = sorted(set().union(*[df.index for df in data.values()]))
    print("dates count:", len(dates))

    cash = INITIAL_CAPITAL
    positions: List[Position] = []
    trades = []
    equity_rows = []

    for date in dates:
        # ===== EXIT =====
        new_positions: List[Position] = []

        for p in positions:
            df = data[p.ticker]
            if date not in df.index:
                new_positions.append(p)
                continue

            row = df.loc[date]

            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            tp = p.entry_price * (1 + TAKE_PROFIT_PCT)
            sl = p.entry_price * (1 - STOP_LOSS_PCT)

            days_held = (date - p.entry_date).days

            exit_price = None
            reason = None

            # 保守的に SL 優先
            if low <= sl and high >= tp:
                exit_price = sl
                reason = "sl_tp_same_day"
            elif low <= sl:
                exit_price = sl
                reason = "stop_loss"
            elif high >= tp:
                exit_price = tp
                reason = "take_profit"
            elif days_held >= HOLD_DAYS:
                exit_price = close
                reason = "time_exit"

            if exit_price is None:
                new_positions.append(p)
                continue

            exit_price *= 1 - SLIPPAGE_PCT
            proceeds = exit_price * p.qty
            fee = proceeds * SELL_FEE_PCT
            pnl = proceeds - fee - (p.entry_price * p.qty)

            cash += proceeds - fee

            trades.append(
                {
                    "ticker": p.ticker,
                    "entry_date": p.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": p.entry_price,
                    "exit_price": exit_price,
                    "qty": p.qty,
                    "return": exit_price / p.entry_price - 1.0,
                    "pnl": pnl,
                    "reason": reason,
                }
            )

        positions = new_positions

        # ===== ENTRY =====
        market_ok = True
        if USE_MARKET_FILTER:
            if market.empty or date not in market.index:
                market_ok = False
            else:
                m = market.loc[date]
                if float(m["Close"]) < float(m["MA200"]):
                    market_ok = False

        if market_ok and len(positions) < MAX_POSITIONS:
            candidates = []

            for t, df in data.items():
                if any(p.ticker == t for p in positions):
                    continue
                if date not in df.index:
                    continue

                i = df.index.get_loc(date)
                if isinstance(i, slice) or i < 1:
                    continue

                row = df.iloc[i]
                prev = df.iloc[i - 1]

                close = float(row["Close"])
                prev_close = float(prev["Close"])
                ma = float(row["MA"])
                ma_slope = float(row["MA_SLOPE"])
                rsi = float(row["RSI"])
                value20 = float(row["VALUE20"])

                if close > prev_close * (1 - PULLBACK_PCT):
                    continue
                if close < ma:
                    continue
                if ma_slope <= 0:
                    continue
                if rsi > RSI_MAX:
                    continue
                if value20 < MIN_VALUE20:
                    continue

                score = (RSI_MAX - rsi) + ((ma / close - 1.0) * 100.0)

                candidates.append(
                    {
                        "ticker": t,
                        "close": close,
                        "score": score,
                    }
                )

            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

            for c in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break

                price = float(c["close"]) * (1 + SLIPPAGE_PCT)
                qty = int((cash * RISK_RATIO / price) // LOT_SIZE) * LOT_SIZE

                if qty <= 0:
                    continue

                cost = price * qty
                fee = cost * BUY_FEE_PCT

                if cost + fee > cash:
                    continue

                cash -= cost + fee
                positions.append(Position(c["ticker"], date, price, qty))

        # ===== EQUITY =====
        pos_val = 0.0
        for p in positions:
            df = data[p.ticker]
            if date in df.index:
                pos_val += float(df.loc[date, "Close"]) * p.qty

        total_equity = cash + pos_val

        equity_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "cash": cash,
                "position_value": pos_val,
                "equity": total_equity,
                "position_count": len(positions),
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity_rows), sorted(data.keys())


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dir()

    trades, equity, loaded = run()

    trades_file = os.path.join(RESULT_DIR, "trades_main2.csv")
    equity_file = os.path.join(RESULT_DIR, "equity_main2.csv")
    summary_file = os.path.join(RESULT_DIR, "summary_main2.csv")

    trades.to_csv(trades_file, index=False)
    equity.to_csv(equity_file, index=False)

    if equity.empty or "equity" not in equity.columns:
        print("equity is empty")
        print(f"Saved: {trades_file}")
        print(f"Saved: {equity_file}")
        return

    final_equity = float(equity["equity"].iloc[-1])
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_dd = calc_max_drawdown(equity["equity"])
    trade_count = len(trades)

    if trade_count > 0:
        win_rate = float((trades["return"] > 0).mean())
        avg_return = float(trades["return"].mean())
        total_pnl = float(trades["pnl"].sum())
        best_trade = float(trades["return"].max())
        worst_trade = float(trades["return"].min())
        tp_count = int((trades["reason"] == "take_profit").sum())
        sl_count = int((trades["reason"] == "stop_loss").sum())
        time_exit_count = int((trades["reason"] == "time_exit").sum())
        same_day_count = int((trades["reason"] == "sl_tp_same_day").sum())
    else:
        win_rate = 0.0
        avg_return = 0.0
        total_pnl = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        tp_count = 0
        sl_count = 0
        time_exit_count = 0
        same_day_count = 0

    summary = pd.DataFrame(
        [
            {
                "period": PERIOD,
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": final_equity,
                "total_return": total_return,
                "max_drawdown": max_dd,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "total_pnl": total_pnl,
                "lot_size": LOT_SIZE,
                "max_positions": MAX_POSITIONS,
                "risk_ratio": RISK_RATIO,
                "ma_days": MA_DAYS,
                "ma_slope_days": MA_SLOPE_DAYS,
                "rsi_days": RSI_DAYS,
                "rsi_max": RSI_MAX,
                "pullback_pct": PULLBACK_PCT,
                "min_avg_value20": MIN_VALUE20,
                "tp_pct": TAKE_PROFIT_PCT,
                "sl_pct": STOP_LOSS_PCT,
                "hold_days": HOLD_DAYS,
                "buy_fee_pct": BUY_FEE_PCT,
                "sell_fee_pct": SELL_FEE_PCT,
                "slippage_pct": SLIPPAGE_PCT,
                "use_market_filter": USE_MARKET_FILTER,
                "market_ticker": MARKET_TICKER if USE_MARKET_FILTER else "",
                "market_ma_days": MARKET_MA_DAYS if USE_MARKET_FILTER else "",
                "tp_count": tp_count,
                "sl_count": sl_count,
                "time_exit_count": time_exit_count,
                "sl_tp_same_day_count": same_day_count,
                "universe_count": len(loaded),
                "universe_file": UNIVERSE_FILE,
            }
        ]
    )

    summary.to_csv(summary_file, index=False)

    print("\n=== MAIN2 RESULT ===")
    print(f"final            : {final_equity}")
    print(f"return           : {total_return}")
    print(f"max_dd           : {max_dd}")
    print(f"trade_count      : {trade_count}")
    print(f"win_rate         : {win_rate}")
    print(f"avg_return       : {avg_return}")
    print(f"tp_count         : {tp_count}")
    print(f"sl_count         : {sl_count}")
    print(f"time_exit_count  : {time_exit_count}")
    print(f"same_day_count   : {same_day_count}")
    print(f"universe_count   : {len(loaded)}")

    print(f"\nSaved: {trades_file}")
    print(f"Saved: {equity_file}")
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
