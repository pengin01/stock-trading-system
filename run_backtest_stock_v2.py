import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import yfinance as yf
import ta


# =========================
# PARAMETERS
# =========================
PERIOD = "10y"
INITIAL_CAPITAL = 20000
RISK_RATIO = 0.7
MAX_POSITIONS = 1

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 48
PULLBACK_PCT = 0.03

TAKE_PROFIT_PCT = 0.09
STOP_LOSS_PCT = 0.025
HOLD_DAYS = 4

MIN_VALUE20 = 1_000_000_000

USE_MARKET_FILTER = False
MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200

RESULT_DIR = "backtest_results_stock_v2"

UNIVERSE = [
    "7203.T",  # トヨタ
    "6758.T",  # ソニーG
    "8306.T",  # 三菱UFJ
    "8035.T",  # 東京エレクトロン
    "9984.T",  # ソフトバンクG
    "9432.T",  # NTT
    "4063.T",  # 信越化学
    "6861.T",  # キーエンス
    "6501.T",  # 日立
    "6098.T",  # リクルート
]


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
def ensure_result_dir() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"{ticker}: download error: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        df = normalize_ohlcv(df)
    except Exception as e:
        print(f"{ticker}: normalize error: {e}")
        return pd.DataFrame()

    close = df["Close"]
    volume = df["Volume"]

    df["MA"] = close.rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, RSI_DAYS).rsi()
    df["VALUE20"] = (close * volume).rolling(20).mean()

    return df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume", "MA", "RSI", "VALUE20"]
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

    try:
        df = normalize_ohlcv(df)
    except Exception as e:
        print(f"{MARKET_TICKER}: normalize error: {e}")
        return pd.DataFrame()

    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()
    return df.dropna(subset=["Close", "MA200"]).copy()


def calc_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calc_bars_passed(
    df: pd.DataFrame, entry_date: pd.Timestamp, current_date: pd.Timestamp
) -> int:
    hist = df.loc[(df.index >= entry_date) & (df.index <= current_date)]
    return max(len(hist) - 1, 0)


def build_candidates(
    signal_date: pd.Timestamp,
    held_tickers: set,
    data_cache: Dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
) -> List[dict]:
    candidates = []

    if USE_MARKET_FILTER:
        if market_df.empty or signal_date not in market_df.index:
            return []
        if float(market_df.loc[signal_date, "Close"]) <= float(
            market_df.loc[signal_date, "MA200"]
        ):
            return []

    for ticker, df in data_cache.items():
        if ticker in held_tickers:
            continue
        if signal_date not in df.index:
            continue

        loc = df.index.get_loc(signal_date)
        if isinstance(loc, slice) or loc < 1:
            continue

        row = df.iloc[loc]
        prev = df.iloc[loc - 1]

        close = float(row["Close"])
        ma = float(row["MA"])
        rsi = float(row["RSI"])
        value20 = float(row["VALUE20"])
        prev_close = float(prev["Close"])

        if close < ma:
            continue
        if close > prev_close * (1 - PULLBACK_PCT):
            continue
        if rsi > RSI_MAX:
            continue
        if value20 < MIN_VALUE20:
            continue

        score = (ma / close - 1.0) + ((RSI_MAX - rsi) / 100.0)

        candidates.append(
            {
                "ticker": ticker,
                "date": signal_date,
                "close": close,
                "prev_close": prev_close,
                "ma": ma,
                "rsi": rsi,
                "value20": value20,
                "score": score,
            }
        )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates


# =========================
# BACKTEST
# =========================
def run_backtest() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cache = {ticker: load_data(ticker) for ticker in UNIVERSE}
    data_cache = {k: v for k, v in data_cache.items() if not v.empty}

    if not data_cache:
        raise RuntimeError("No stock data loaded")

    market_df = load_market_data()

    all_dates = sorted(set().union(*[df.index for df in data_cache.values()]))

    cash = INITIAL_CAPITAL
    positions: List[Position] = []
    trades = []
    equity_rows = []

    for signal_date in all_dates:
        # =========================
        # EXIT
        # =========================
        next_positions: List[Position] = []

        for p in positions:
            df = data_cache.get(p.ticker, pd.DataFrame())

            if df.empty or signal_date not in df.index:
                next_positions.append(p)
                continue

            row = df.loc[signal_date]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            tp_price = p.entry_price * (1.0 + TAKE_PROFIT_PCT)
            sl_price = p.entry_price * (1.0 - STOP_LOSS_PCT)

            bars_held = calc_bars_passed(df, p.entry_date, signal_date)

            exit_price = None
            exit_reason = None

            # 保守的に SL 優先
            if low <= sl_price and high >= tp_price:
                exit_price = sl_price
                exit_reason = "sl_tp_same_day"
            elif low <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
            elif high >= tp_price:
                exit_price = tp_price
                exit_reason = "take_profit"
            elif bars_held >= HOLD_DAYS:
                exit_price = close
                exit_reason = "time_exit"

            if exit_price is None:
                next_positions.append(p)
                continue

            proceeds = exit_price * p.qty
            pnl = (exit_price - p.entry_price) * p.qty
            ret = exit_price / p.entry_price - 1.0

            cash += proceeds

            trades.append(
                {
                    "ticker": p.ticker,
                    "entry_date": p.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": signal_date.strftime("%Y-%m-%d"),
                    "entry_price": p.entry_price,
                    "exit_price": exit_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "qty": p.qty,
                    "bars_held": bars_held,
                    "return": ret,
                    "pnl": pnl,
                    "reason": exit_reason,
                }
            )

        positions = next_positions

        # =========================
        # ENTRY
        # =========================
        if len(positions) < MAX_POSITIONS:
            held_tickers = {p.ticker for p in positions}
            candidates = build_candidates(
                signal_date=signal_date,
                held_tickers=held_tickers,
                data_cache=data_cache,
                market_df=market_df,
            )

            slots = MAX_POSITIONS - len(positions)

            for c in candidates[:slots]:
                if c["ticker"] in {p.ticker for p in positions}:
                    continue

                price = float(c["close"])
                usable_cash = cash * RISK_RATIO
                qty = int(usable_cash // price)

                if qty <= 0:
                    continue

                cost = price * qty
                if cost > cash:
                    continue

                cash -= cost
                positions.append(
                    Position(
                        ticker=c["ticker"],
                        entry_date=signal_date,
                        entry_price=price,
                        qty=qty,
                    )
                )

        # =========================
        # EQUITY
        # =========================
        position_value = 0.0

        for p in positions:
            df = data_cache[p.ticker]
            if signal_date in df.index:
                px = float(df.loc[signal_date, "Close"])
            else:
                hist = df.loc[:signal_date]
                px = float(hist["Close"].iloc[-1]) if not hist.empty else p.entry_price

            position_value += px * p.qty

        total_equity = cash + position_value

        equity_rows.append(
            {
                "date": signal_date.strftime("%Y-%m-%d"),
                "cash": cash,
                "position_value": position_value,
                "equity": total_equity,
                "position_count": len(positions),
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    final_equity = (
        float(equity_df["equity"].iloc[-1])
        if not equity_df.empty
        else float(INITIAL_CAPITAL)
    )
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_drawdown = (
        calc_max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0
    )
    trade_count = len(trades_df)

    if trade_count > 0:
        win_rate = float((trades_df["return"] > 0).mean())
        avg_return = float(trades_df["return"].mean())
        total_pnl = float(trades_df["pnl"].sum())
        best_trade = float(trades_df["return"].max())
        worst_trade = float(trades_df["return"].min())
        tp_count = int((trades_df["reason"] == "take_profit").sum())
        sl_count = int((trades_df["reason"] == "stop_loss").sum())
        same_day_count = int((trades_df["reason"] == "sl_tp_same_day").sum())
        time_exit_count = int((trades_df["reason"] == "time_exit").sum())
    else:
        win_rate = 0.0
        avg_return = 0.0
        total_pnl = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        tp_count = 0
        sl_count = 0
        same_day_count = 0
        time_exit_count = 0

    summary_df = pd.DataFrame(
        [
            {
                "period": PERIOD,
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": final_equity,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "total_pnl": total_pnl,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "tp_count": tp_count,
                "sl_count": sl_count,
                "sl_tp_same_day_count": same_day_count,
                "time_exit_count": time_exit_count,
                "ma_days": MA_DAYS,
                "rsi_days": RSI_DAYS,
                "rsi_max": RSI_MAX,
                "pullback_pct": PULLBACK_PCT,
                "tp_pct": TAKE_PROFIT_PCT,
                "sl_pct": STOP_LOSS_PCT,
                "hold_days": HOLD_DAYS,
                "risk_ratio": RISK_RATIO,
                "max_positions": MAX_POSITIONS,
                "min_value20": MIN_VALUE20,
                "use_market_filter": USE_MARKET_FILTER,
                "market_ticker": MARKET_TICKER if USE_MARKET_FILTER else "",
                "market_ma_days": MARKET_MA_DAYS if USE_MARKET_FILTER else "",
                "universe_count": len(data_cache),
                "universe": ",".join(sorted(data_cache.keys())),
            }
        ]
    )

    return summary_df, trades_df, equity_df


# =========================
# SAVE / PRINT
# =========================
def main() -> None:
    ensure_result_dir()

    print("=== BACKTEST STOCK V2 ===")
    print(f"period         : {PERIOD}")
    print(f"initial_capital: {INITIAL_CAPITAL}")
    print(f"risk_ratio     : {RISK_RATIO}")
    print(f"max_positions  : {MAX_POSITIONS}")
    print(f"ma_days        : {MA_DAYS}")
    print(f"rsi_days       : {RSI_DAYS}")
    print(f"rsi_max        : {RSI_MAX}")
    print(f"pullback_pct   : {PULLBACK_PCT}")
    print(f"tp_pct         : {TAKE_PROFIT_PCT}")
    print(f"sl_pct         : {STOP_LOSS_PCT}")
    print(f"hold_days      : {HOLD_DAYS}")
    print(f"min_value20    : {MIN_VALUE20}")
    print(f"use_market_filter: {USE_MARKET_FILTER}")
    print("universe:")
    for t in UNIVERSE:
        print(f"  - {t}")

    summary_df, trades_df, equity_df = run_backtest()

    summary_file = os.path.join(RESULT_DIR, "summary_stock_v2.csv")
    trades_file = os.path.join(RESULT_DIR, "trades_stock_v2.csv")
    equity_file = os.path.join(RESULT_DIR, "equity_stock_v2.csv")

    summary_df.to_csv(summary_file, index=False)
    trades_df.to_csv(trades_file, index=False)
    equity_df.to_csv(equity_file, index=False)

    row = summary_df.iloc[0]

    print("\n=== RESULT ===")
    print(f"final_equity   : {row['final_equity']}")
    print(f"total_return   : {row['total_return']}")
    print(f"max_drawdown   : {row['max_drawdown']}")
    print(f"trade_count    : {int(row['trade_count'])}")
    print(f"win_rate       : {row['win_rate']}")
    print(f"avg_return     : {row['avg_return']}")
    print(f"best_trade     : {row['best_trade']}")
    print(f"worst_trade    : {row['worst_trade']}")
    print(f"tp_count       : {int(row['tp_count'])}")
    print(f"sl_count       : {int(row['sl_count'])}")
    print(f"time_exit_count: {int(row['time_exit_count'])}")
    print(f"same_day_count : {int(row['sl_tp_same_day_count'])}")

    print(f"\nSaved: {summary_file}")
    print(f"Saved: {trades_file}")
    print(f"Saved: {equity_file}")


if __name__ == "__main__":
    main()
