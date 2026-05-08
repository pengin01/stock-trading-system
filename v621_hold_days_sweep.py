import pandas as pd
import yfinance as yf

# =====================
# v621 HOLD_DAYS sweep
# 2万円運用・100株単位
# v620で有望だった銘柄に対して
# HOLD_DAYS を最適化する
# =====================

TICKERS = [
    "9424.T",
    "9432.T",
    "7610.T",
    "4564.T",
]

START = "2022-01-01"
END = "2026-01-01"

INITIAL_CAPITAL = 20_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03

# ここを比較
HOLD_DAYS_LIST = [3, 5, 7, 10, 15]

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


def load_data():
    data = {}

    for ticker in TICKERS:
        print("Downloading:", ticker)

        try:
            df = yf.download(
                ticker,
                start=START,
                end=END,
                auto_adjust=False,
                progress=True,
            )
        except Exception as e:
            print("download error:", ticker, e)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 100:
            print("skip:", ticker)
            continue

        df = calc_indicators(df).dropna().copy()

        if len(df) == 0:
            continue

        data[ticker] = df

    return data


def backtest_single_ticker(ticker, df, hold_days):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    candidate_count = 0

    for date, row in df.iterrows():

        # =====================
        # EXIT
        # =====================
        new_positions = []

        for pos in positions:
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            entry_price = float(pos["entry_price"])
            shares = int(pos["shares"])
            cost = float(pos["cost"])
            entry_date = pd.to_datetime(pos["entry_date"])

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
                    "hold_days": hold_days,
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "cost": cost,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "return": ret,
                    "reason": reason,
                    "score": pos["score"],
                })

            else:
                new_positions.append(pos)

        positions = new_positions

        # =====================
        # ENTRY
        # =====================
        if len(positions) < MAX_POSITIONS:
            close = float(row["Close"])

            if PRICE_MIN <= close <= PRICE_MAX:
                if (
                    is_hammer(row)
                    and float(row["ret3"]) < 0
                    and float(row["value20"]) >= MIN_VALUE20
                ):
                    score = -float(row["ret3"])

                    if score >= MIN_SCORE:
                        candidate_count += 1

                        entry_price = close * (1 + SLIPPAGE)
                        shares = LOT_SIZE

                        gross = entry_price * shares
                        fee = gross * FEE_PCT
                        total_cost = gross + fee

                        if cash >= total_cost:
                            cash -= total_cost

                            positions.append({
                                "ticker": ticker,
                                "entry_date": date.strftime("%Y-%m-%d"),
                                "entry_price": entry_price,
                                "shares": shares,
                                "cost": total_cost,
                                "score": score,
                            })

        # =====================
        # EQUITY
        # =====================
        position_value = 0

        for pos in positions:
            position_value += float(row["Close"]) * int(pos["shares"])

        equity = cash + position_value

        equity_curve.append({
            "ticker": ticker,
            "hold_days": hold_days,
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    return trades_df, equity_df, candidate_count


def calc_summary(ticker, hold_days, trades_df, equity_df, candidate_count):
    if len(equity_df) > 0:
        equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")
        peak = equity_df["equity"].cummax()
        dd = equity_df["equity"] / peak - 1
        final_equity = equity_df["equity"].iloc[-1]
        max_dd = dd.min()
    else:
        final_equity = INITIAL_CAPITAL
        max_dd = 0

    if len(trades_df) == 0:
        return {
            "ticker": ticker,
            "hold_days": hold_days,
            "trades": 0,
            "candidates": candidate_count,
            "win_rate": None,
            "total_pnl": 0,
            "avg_pnl": None,
            "avg_return": None,
            "final_equity": final_equity,
            "max_dd": max_dd,
            "tp_count": 0,
            "sl_count": 0,
            "time_count": 0,
        }

    reason_counts = trades_df["reason"].value_counts()

    return {
        "ticker": ticker,
        "hold_days": hold_days,
        "trades": len(trades_df),
        "candidates": candidate_count,
        "win_rate": (trades_df["pnl"] > 0).mean(),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "avg_return": trades_df["return"].mean(),
        "final_equity": final_equity,
        "max_dd": max_dd,
        "tp_count": int(reason_counts.get("TP", 0)),
        "sl_count": int(reason_counts.get("SL", 0)),
        "time_count": int(reason_counts.get("TIME", 0)),
    }


def main():
    data = load_data()

    summaries = []
    all_trades = []
    all_equity = []

    for ticker, df in data.items():

        for hold_days in HOLD_DAYS_LIST:

            print("\n==============================")
            print("TICKER:", ticker)
            print("HOLD_DAYS:", hold_days)
            print("==============================")

            trades_df, equity_df, candidate_count = backtest_single_ticker(
                ticker,
                df,
                hold_days,
            )

            summary = calc_summary(
                ticker,
                hold_days,
                trades_df,
                equity_df,
                candidate_count,
            )

            summaries.append(summary)

            if len(trades_df) > 0:
                all_trades.append(trades_df)

                print("trades:", len(trades_df))
                print("win_rate:", (trades_df["pnl"] > 0).mean())
                print("total_pnl:", trades_df["pnl"].sum())
                print("max_dd:", summary["max_dd"])

                print("\n=== BY REASON ===")
                print(trades_df.groupby("reason")["pnl"].agg(["count", "mean", "sum"]))
            else:
                print("No trades")

            if len(equity_df) > 0:
                all_equity.append(equity_df)

    summary_df = pd.DataFrame(summaries)

    ranking = summary_df.sort_values(
        ["total_pnl", "win_rate", "max_dd"],
        ascending=[False, False, False],
    )

    print("\n==============================")
    print("HOLD DAYS SUMMARY")
    print("==============================")
    print(ranking.to_string(index=False))

    # ベスト候補
    best = ranking.groupby("ticker").head(1)

    print("\n==============================")
    print("BEST HOLD DAYS")
    print("==============================")
    print(best.to_string(index=False))

    # 保存
    summary_df.to_csv("summary_v621_hold_days.csv", index=False)
    ranking.to_csv("ranking_v621_hold_days.csv", index=False)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            "trades_v621_hold_days.csv",
            index=False,
        )

    if all_equity:
        pd.concat(all_equity, ignore_index=True).to_csv(
            "equity_v621_hold_days.csv",
            index=False,
        )

    print("\nSaved:")
    print("summary_v621_hold_days.csv")
    print("ranking_v621_hold_days.csv")
    print("trades_v621_hold_days.csv")
    print("equity_v621_hold_days.csv")


if __name__ == "__main__":
    main()
