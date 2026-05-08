import pandas as pd
import yfinance as yf

# =====================
# v623 walk forward validation
#
# train : 2022-2023
# test  : 2024-2025
#
# v622固定戦略の頑健性確認
# =====================

TICKERS = [
    "9432.T",
    "9424.T",
]

HOLD_DAYS_MAP = {
    "9432.T": 5,
    "9424.T": 10,
}

TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-01-01"

TEST_START  = "2024-01-01"
TEST_END    = "2026-01-01"

INITIAL_CAPITAL = 20_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03

MIN_SCORE = 0.012
MIN_VALUE20 = 50_000_000

PRICE_MIN = 50
PRICE_MAX = 200

SLIPPAGE = 0.001
FEE_PCT = 0.0005


def calc_indicators(df):
    df["ret3"] = df["Close"].pct_change(3)
    df["value"] = df["Close"] * df["Volume"]
    df["value20"] = df["value"].rolling(20).mean()
    return df


def is_hammer(row):
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])

    body = abs(c - o)
    rng = h - l

    if rng <= 0:
        return False

    lower = min(o, c) - l
    upper = h - max(o, c)
    body_safe = max(body, c * 0.001)

    return (
        lower >= body_safe * 2
        and upper <= body_safe
        and body <= rng * 0.4
    )


def load_data(start, end):
    data = {}

    for ticker in TICKERS:
        print("Downloading:", ticker)

        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=True,
            )
        except Exception as e:
            print("download error:", ticker, e)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 100:
            continue

        df = calc_indicators(df).dropna().copy()

        if len(df) == 0:
            continue

        data[ticker] = df

    return data


def run_backtest(data):
    dates = sorted(set(d for df in data.values() for d in df.index))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []

    for date in dates:

        # EXIT
        new_positions = []

        for pos in positions:

            ticker = pos["ticker"]

            if ticker not in data:
                new_positions.append(pos)
                continue

            df = data[ticker]

            if date not in df.index:
                new_positions.append(pos)
                continue

            row = df.loc[date]

            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            entry_price = float(pos["entry_price"])
            shares = int(pos["shares"])
            cost = float(pos["cost"])
            entry_date = pd.to_datetime(pos["entry_date"])

            hold_days = HOLD_DAYS_MAP[ticker]

            exit_flag = False
            exit_price = None
            reason = None

            if high >= entry_price * (1 + TP_PCT):
                exit_price = entry_price * (1 + TP_PCT)
                reason = "TP"
                exit_flag = True

            elif low <= entry_price * (1 + SL_PCT):
                exit_price = entry_price * (1 + SL_PCT)
                reason = "SL"
                exit_flag = True

            elif (date - entry_date).days >= hold_days:
                exit_price = close
                reason = "TIME"
                exit_flag = True

            if exit_flag:

                exit_price *= (1 - SLIPPAGE)

                gross = exit_price * shares
                fee = gross * FEE_PCT
                proceeds = gross - fee

                pnl = proceeds - cost
                ret = pnl / cost

                cash += proceeds

                trades.append({
                    "ticker": ticker,
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "hold_days": hold_days,
                    "pnl": pnl,
                    "return": ret,
                    "reason": reason,
                })

            else:
                new_positions.append(pos)

        positions = new_positions

        # ENTRY
        if len(positions) < MAX_POSITIONS:

            candidates = []

            for ticker, df in data.items():

                if date not in df.index:
                    continue

                row = df.loc[date]

                close = float(row["Close"])

                if not (PRICE_MIN <= close <= PRICE_MAX):
                    continue

                if (
                    is_hammer(row)
                    and float(row["ret3"]) < 0
                    and float(row["value20"]) >= MIN_VALUE20
                ):

                    score = -float(row["ret3"])

                    if score >= MIN_SCORE:

                        candidates.append({
                            "ticker": ticker,
                            "score": score,
                            "close": close,
                        })

            candidates.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            for cand in candidates:

                entry_price = cand["close"] * (1 + SLIPPAGE)

                gross = entry_price * LOT_SIZE
                fee = gross * FEE_PCT
                total_cost = gross + fee

                if total_cost > INITIAL_CAPITAL:
                    continue

                if cash < total_cost:
                    continue

                cash -= total_cost

                positions.append({
                    "ticker": cand["ticker"],
                    "entry_date": date.strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "shares": LOT_SIZE,
                    "cost": total_cost,
                })

                break

        # EQUITY
        position_value = 0

        for pos in positions:

            ticker = pos["ticker"]

            if ticker in data and date in data[ticker].index:

                close = float(data[ticker].loc[date]["Close"])

                position_value += close * int(pos["shares"])

        equity = cash + position_value

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_curve),
    )


def summarize(name, trades_df, equity_df):

    if len(equity_df) > 0:

        peak = equity_df["equity"].cummax()
        dd = equity_df["equity"] / peak - 1

        final_equity = equity_df["equity"].iloc[-1]
        max_dd = dd.min()

    else:
        final_equity = INITIAL_CAPITAL
        max_dd = 0

    if len(trades_df) == 0:

        return {
            "period": name,
            "trades": 0,
            "win_rate": None,
            "total_pnl": 0,
            "final_equity": final_equity,
            "max_dd": max_dd,
        }

    return {
        "period": name,
        "trades": len(trades_df),
        "win_rate": (trades_df["pnl"] > 0).mean(),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "avg_return": trades_df["return"].mean(),
        "final_equity": final_equity,
        "max_dd": max_dd,
    }


def main():

    print("\n==============================")
    print("TRAIN")
    print("==============================")

    train_data = load_data(
        TRAIN_START,
        TRAIN_END,
    )

    train_trades, train_equity = run_backtest(train_data)

    train_summary = summarize(
        "TRAIN",
        train_trades,
        train_equity,
    )

    print(train_summary)

    print("\n==============================")
    print("TEST")
    print("==============================")

    test_data = load_data(
        TEST_START,
        TEST_END,
    )

    test_trades, test_equity = run_backtest(test_data)

    test_summary = summarize(
        "TEST",
        test_trades,
        test_equity,
    )

    print(test_summary)

    result_df = pd.DataFrame([
        train_summary,
        test_summary,
    ])

    print("\n==============================")
    print("WALK FORWARD RESULT")
    print("==============================")
    print(result_df)

    result_df.to_csv(
        "walkforward_v623.csv",
        index=False,
    )

    train_trades.to_csv(
        "train_trades_v623.csv",
        index=False,
    )

    test_trades.to_csv(
        "test_trades_v623.csv",
        index=False,
    )

    print("\nSaved:")
    print("walkforward_v623.csv")
    print("train_trades_v623.csv")
    print("test_trades_v623.csv")


if __name__ == "__main__":
    main()
