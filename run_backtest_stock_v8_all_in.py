import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import ta


# =========================
# PARAMETERS（全部入り版）
# =========================
PERIOD = "10y"
INITIAL_CAPITAL = 100000

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

RESULT_DIR = "backtest_results_stock_v8"
UNIVERSE_FILE = "jp_universe.csv"

# 市場フィルタ
USE_MARKET_FILTER = True
MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200

# 戦略ON/OFF
USE_EQUITY_FILTER = True
EQUITY_MA_DAYS = 30

# DD停止
USE_DD_STOP = False
MAX_DD_STOP = -0.05  # -5% を超えたら新規エントリー停止


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

    col = None
    for c in ["Ticker", "ticker", "code", "Code"]:
        if c in df.columns:
            col = c
            break

    if col is None:
        col = df.columns[0]

    tickers = df[col].astype(str).str.strip().replace("", pd.NA).dropna().tolist()

    fixed = []
    for t in tickers:
        if not t.endswith(".T"):
            t = f"{t}.T"
        fixed.append(t)

    return sorted(set(fixed))


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


def load_market_data() -> pd.DataFrame:
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
    print(f"Universe loaded: {len(universe)} tickers")

    data = {}
    for i, t in enumerate(universe, start=1):
        if i % 50 == 0:
            print(f"Loading {i}/{len(universe)} ...")
        df = load_stock_data(t)
        if not df.empty:
            data[t] = df

    if not data:
        raise RuntimeError("No stock data loaded")

    market = load_market_data()
    dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = INITIAL_CAPITAL
    positions: List[Position] = []
    trades = []
    equity_rows = []

    equity_history = []

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

        # ===== EQUITY（entry前時点）=====
        pos_val_before = 0.0
        for p in positions:
            df = data[p.ticker]
            if date in df.index:
                pos_val_before += float(df.loc[date, "Close"]) * p.qty

        total_equity_before = cash + pos_val_before
        equity_history.append(total_equity_before)

        # ===== ENTRY ON/OFF 判定 =====
        trading_enabled = True

        if USE_MARKET_FILTER:
            if market.empty or date not in market.index:
                trading_enabled = False
            else:
                m = market.loc[date]
                if float(m["Close"]) < float(m["MA200"]):
                    trading_enabled = False

        if USE_EQUITY_FILTER and len(equity_history) >= EQUITY_MA_DAYS:
            eq_ma = pd.Series(equity_history).rolling(EQUITY_MA_DAYS).mean().iloc[-1]
            if total_equity_before < float(eq_ma):
                trading_enabled = False

        if USE_DD_STOP:
            peak_equity = max(equity_history) if equity_history else total_equity_before
            current_dd = (
                total_equity_before / peak_equity - 1.0 if peak_equity != 0 else 0.0
            )
            if current_dd <= MAX_DD_STOP:
                trading_enabled = False
        else:
            current_dd = 0.0

        # ===== ENTRY =====
        if trading_enabled and len(positions) < MAX_POSITIONS:
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
                        "rsi": rsi,
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

        # ===== EQUITY（entry後の終値時点）=====
        pos_val_after = 0.0
        for p in positions:
            df = data[p.ticker]
            if date in df.index:
                pos_val_after += float(df.loc[date, "Close"]) * p.qty

        total_equity_after = cash + pos_val_after

        equity_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "cash": cash,
                "position_value": pos_val_after,
                "equity": total_equity_after,
                "position_count": len(positions),
                "trading_enabled": int(trading_enabled),
                "current_dd": current_dd,
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity_rows), sorted(data.keys())


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dir()

    print("=== BACKTEST STOCK V8 ALL-IN ===")
    print(f"period             : {PERIOD}")
    print(f"initial_capital    : {INITIAL_CAPITAL}")
    print(f"lot_size           : {LOT_SIZE}")
    print(f"max_positions      : {MAX_POSITIONS}")
    print(f"risk_ratio         : {RISK_RATIO}")
    print(f"ma_days            : {MA_DAYS}")
    print(f"ma_slope_days      : {MA_SLOPE_DAYS}")
    print(f"rsi_days           : {RSI_DAYS}")
    print(f"rsi_max            : {RSI_MAX}")
    print(f"pullback_pct       : {PULLBACK_PCT}")
    print(f"min_value20        : {MIN_VALUE20}")
    print(f"tp_pct             : {TAKE_PROFIT_PCT}")
    print(f"sl_pct             : {STOP_LOSS_PCT}")
    print(f"hold_days          : {HOLD_DAYS}")
    print(f"buy_fee_pct        : {BUY_FEE_PCT}")
    print(f"sell_fee_pct       : {SELL_FEE_PCT}")
    print(f"slippage_pct       : {SLIPPAGE_PCT}")
    print(f"use_market_filter  : {USE_MARKET_FILTER}")
    print(f"market_ticker      : {MARKET_TICKER}")
    print(f"market_ma_days     : {MARKET_MA_DAYS}")
    print(f"use_equity_filter  : {USE_EQUITY_FILTER}")
    print(f"equity_ma_days     : {EQUITY_MA_DAYS}")
    print(f"use_dd_stop        : {USE_DD_STOP}")
    print(f"max_dd_stop        : {MAX_DD_STOP}")
    print(f"universe_file      : {UNIVERSE_FILE}")

    trades, equity, loaded = run()

    final_equity = (
        float(equity["equity"].iloc[-1]) if not equity.empty else float(INITIAL_CAPITAL)
    )
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_dd = calc_max_drawdown(equity["equity"]) if not equity.empty else 0.0
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
                "use_equity_filter": USE_EQUITY_FILTER,
                "equity_ma_days": EQUITY_MA_DAYS if USE_EQUITY_FILTER else "",
                "use_dd_stop": USE_DD_STOP,
                "max_dd_stop": MAX_DD_STOP if USE_DD_STOP else "",
                "tp_count": tp_count,
                "sl_count": sl_count,
                "time_exit_count": time_exit_count,
                "sl_tp_same_day_count": same_day_count,
                "universe_count": len(loaded),
                "universe_file": UNIVERSE_FILE,
            }
        ]
    )

    trades_file = os.path.join(RESULT_DIR, "trades_v8.csv")
    equity_file = os.path.join(RESULT_DIR, "equity_v8.csv")
    summary_file = os.path.join(RESULT_DIR, "summary_v8.csv")

    trades.to_csv(trades_file, index=False)
    equity.to_csv(equity_file, index=False)
    summary.to_csv(summary_file, index=False)

    print("\n=== RESULT ===")
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
