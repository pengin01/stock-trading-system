import pandas as pd
import yfinance as yf

# =====================
# v620 low-cap 20k universe scan
# 2万円運用・100株単位
# 低位株ユニバースから「v616ルールに合う銘柄」を探索する
# =====================

# まずは候補を広めに入れる
# 価格帯・流動性・成績で後から絞る
TICKERS = [
    "9432.T",  # NTT
    "4564.T",
    "8918.T",
    "5856.T",
    "9973.T",
    "2370.T",
    "3664.T",
    "2134.T",
    "2345.T",
    "3315.T",
    "6993.T",
    "7610.T",
    "7647.T",
    "6740.T",
    "8107.T",
    "9424.T",
    "7640.T",
    "8515.T",
    "8897.T",
    "7777.T",
]

START = "2022-01-01"
END = "2026-01-01"

INITIAL_CAPITAL = 20_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03
HOLD_DAYS = 10

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
            print("skip:", ticker, "not enough data")
            continue

        df = calc_indicators(df).dropna().copy()

        if len(df) == 0:
            print("skip:", ticker, "empty after indicators")
            continue

        data[ticker] = df

    return data


def backtest_single_ticker(ticker, df):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    candidate_count = 0
    skipped_count = 0

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

            elif (date - entry_date).days >= HOLD_DAYS:
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

                        if total_cost > INITIAL_CAPITAL or cash < total_cost:
                            skipped_count += 1
                        else:
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
            "date": date.strftime("%Y-%m-%d"),
            "cash": cash,
            "position_value": position_value,
            "equity": equity,
            "position_count": len(positions),
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    return trades_df, equity_df, candidate_count, skipped_count


def calc_summary(ticker, trades_df, equity_df, candidate_count, skipped_count):
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
            "trades": 0,
            "candidates": candidate_count,
            "skipped": skipped_count,
            "win_rate": None,
            "total_pnl": 0,
            "avg_pnl": None,
            "avg_return": None,
            "final_equity": final_equity,
            "max_dd": max_dd,
            "tp_count": 0,
            "sl_count": 0,
            "time_count": 0,
            "tp_rate": None,
            "sl_rate": None,
        }

    reason_counts = trades_df["reason"].value_counts()

    tp_count = int(reason_counts.get("TP", 0))
    sl_count = int(reason_counts.get("SL", 0))
    time_count = int(reason_counts.get("TIME", 0))

    return {
        "ticker": ticker,
        "trades": len(trades_df),
        "candidates": candidate_count,
        "skipped": skipped_count,
        "win_rate": (trades_df["pnl"] > 0).mean(),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "avg_return": trades_df["return"].mean(),
        "final_equity": final_equity,
        "max_dd": max_dd,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "time_count": time_count,
        "tp_rate": tp_count / len(trades_df),
        "sl_rate": sl_count / len(trades_df),
    }


def main():
    data = load_data()

    summaries = []
    all_trades = []
    all_equity = []

    for ticker, df in data.items():
        print("\n==============================")
        print("BACKTEST:", ticker)
        print("==============================")

        trades_df, equity_df, candidate_count, skipped_count = backtest_single_ticker(ticker, df)
        summary = calc_summary(ticker, trades_df, equity_df, candidate_count, skipped_count)

        summaries.append(summary)

        if len(trades_df) > 0:
            all_trades.append(trades_df)

            trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year

            print("trades:", len(trades_df))
            print("candidates:", candidate_count)
            print("win_rate:", (trades_df["pnl"] > 0).mean())
            print("total_pnl:", trades_df["pnl"].sum())
            print("max_dd:", summary["max_dd"])

            print("\n=== YEARLY ===")
            print(trades_df.groupby("year")["pnl"].agg(["count", "mean", "sum"]))

            print("\n=== BY REASON ===")
            print(trades_df.groupby("reason")["pnl"].agg(["count", "mean", "sum"]))
        else:
            print("No trades")
            print("candidates:", candidate_count)

        if len(equity_df) > 0:
            all_equity.append(equity_df)

    summary_df = pd.DataFrame(summaries)

    # 見やすいランキング
    ranking = summary_df.sort_values(
        ["total_pnl", "win_rate", "max_dd"],
        ascending=[False, False, False],
    )

    print("\n==============================")
    print("UNIVERSE SUMMARY")
    print("==============================")
    print(ranking.to_string(index=False))

    # 採用候補
    good = summary_df[
        (summary_df["trades"] >= 2)
        & (summary_df["total_pnl"] > 0)
        & (summary_df["max_dd"] >= -0.08)
    ].sort_values("total_pnl", ascending=False)

    print("\n==============================")
    print("GOOD CANDIDATES")
    print("==============================")
    if len(good) > 0:
        print(good.to_string(index=False))
    else:
        print("No good candidates found")

    # 保存
    summary_df.to_csv("summary_v620_universe_scan.csv", index=False)
    ranking.to_csv("ranking_v620_universe_scan.csv", index=False)
    good.to_csv("good_candidates_v620.csv", index=False)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("trades_v620_universe_scan.csv", index=False)
    else:
        pd.DataFrame().to_csv("trades_v620_universe_scan.csv", index=False)

    if all_equity:
        pd.concat(all_equity, ignore_index=True).to_csv("equity_v620_universe_scan.csv", index=False)
    else:
        pd.DataFrame().to_csv("equity_v620_universe_scan.csv", index=False)

    print("\nSaved:")
    print("summary_v620_universe_scan.csv")
    print("ranking_v620_universe_scan.csv")
    print("good_candidates_v620.csv")
    print("trades_v620_universe_scan.csv")
    print("equity_v620_universe_scan.csv")


if __name__ == "__main__":
    main()
