import pandas as pd
import yfinance as yf

# =====================
# v619 low-cap 20k score sweep
# 2万円運用・100株単位・v616ルール継承
# MIN_SCOREだけ比較するバックテスト
# =====================

TICKERS = [
    "9432.T",
    "4564.T",
    "8918.T",
    "5856.T",
    "9973.T",
    "2370.T",
    "3664.T",
]

START = "2022-01-01"
END = "2026-01-01"

INITIAL_CAPITAL = 20_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03
HOLD_DAYS = 10

SCORE_LIST = [0.012, 0.010, 0.008, 0.006]

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

        df = yf.download(
            ticker,
            start=START,
            end=END,
            auto_adjust=False,
            progress=True,
        )

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


def run_backtest(data, min_score):
    dates = sorted(set(d for df in data.values() for d in df.index))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    candidate_log = []
    skipped = []

    for date in dates:

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
                    "score_min": min_score,
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

                    if score >= min_score:
                        cand = {
                            "score_min": min_score,
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "close": close,
                            "score": score,
                            "ret3": float(row["ret3"]),
                            "value20": float(row["value20"]),
                        }
                        candidate_log.append(cand)
                        candidates.append({
                            "ticker": ticker,
                            "row": row,
                            "score": score,
                        })

            candidates.sort(key=lambda x: x["score"], reverse=True)

            for cand in candidates:
                ticker = cand["ticker"]
                row = cand["row"]

                entry_price = float(row["Close"]) * (1 + SLIPPAGE)
                shares = LOT_SIZE

                gross = entry_price * shares
                fee = gross * FEE_PCT
                total_cost = gross + fee

                if total_cost > INITIAL_CAPITAL:
                    skipped.append({
                        "score_min": min_score,
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "reason": "OVER_INITIAL_CAPITAL",
                        "price": entry_price,
                        "total_cost": total_cost,
                        "cash": cash,
                    })
                    continue

                if cash < total_cost:
                    skipped.append({
                        "score_min": min_score,
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "reason": "NO_CASH",
                        "price": entry_price,
                        "total_cost": total_cost,
                        "cash": cash,
                    })
                    continue

                cash -= total_cost

                positions.append({
                    "ticker": ticker,
                    "entry_date": date.strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "shares": shares,
                    "cost": total_cost,
                    "score": cand["score"],
                })

                break

        position_value = 0

        for pos in positions:
            ticker = pos["ticker"]

            if ticker in data and date in data[ticker].index:
                close = float(data[ticker].loc[date]["Close"])
                position_value += close * int(pos["shares"])

        equity = cash + position_value

        equity_curve.append({
            "score_min": min_score,
            "date": date.strftime("%Y-%m-%d"),
            "cash": cash,
            "position_value": position_value,
            "equity": equity,
            "position_count": len(positions),
        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_curve),
        pd.DataFrame(candidate_log),
        pd.DataFrame(skipped),
    )


def summarize(score, trades_df, equity_df, candidate_df, skipped_df):
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
            "score_min": score,
            "trades": 0,
            "candidates": len(candidate_df),
            "skipped": len(skipped_df),
            "win_rate": None,
            "total_pnl": 0,
            "avg_pnl": None,
            "avg_return": None,
            "final_equity": final_equity,
            "max_dd": max_dd,
        }

    return {
        "score_min": score,
        "trades": len(trades_df),
        "candidates": len(candidate_df),
        "skipped": len(skipped_df),
        "win_rate": (trades_df["pnl"] > 0).mean(),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "avg_return": trades_df["return"].mean(),
        "final_equity": final_equity,
        "max_dd": max_dd,
    }


def main():
    data = load_data()

    all_trades = []
    all_equity = []
    all_candidates = []
    all_skipped = []
    summaries = []

    for score in SCORE_LIST:
        print("\n==============================")
        print("RUN SCORE:", score)
        print("==============================")

        trades_df, equity_df, candidate_df, skipped_df = run_backtest(data, score)

        summaries.append(
            summarize(score, trades_df, equity_df, candidate_df, skipped_df)
        )

        if len(trades_df) > 0:
            all_trades.append(trades_df)

            trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year

            print("\n=== RESULT ===")
            print("score_min:", score)
            print("trades:", len(trades_df))
            print("candidates:", len(candidate_df))
            print("win_rate:", (trades_df["pnl"] > 0).mean())
            print("total_pnl:", trades_df["pnl"].sum())

            print("\n=== YEARLY ===")
            print(trades_df.groupby("year")["pnl"].agg(["count", "mean", "sum"]))

            print("\n=== BY TICKER ===")
            print(trades_df.groupby("ticker")["pnl"].agg(["count", "mean", "sum"]))

            print("\n=== BY REASON ===")
            print(trades_df.groupby("reason")["pnl"].agg(["count", "mean", "sum"]))

        else:
            print("No trades")
            print("candidates:", len(candidate_df))

        if len(equity_df) > 0:
            all_equity.append(equity_df)

        if len(candidate_df) > 0:
            all_candidates.append(candidate_df)

        if len(skipped_df) > 0:
            all_skipped.append(skipped_df)

    summary_df = pd.DataFrame(summaries)

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print(summary_df.sort_values("score_min", ascending=False).to_string(index=False))

    summary_df.to_csv("summary_v619_score_sweep.csv", index=False)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("trades_v619_score_sweep.csv", index=False)
    else:
        pd.DataFrame().to_csv("trades_v619_score_sweep.csv", index=False)

    if all_equity:
        pd.concat(all_equity, ignore_index=True).to_csv("equity_v619_score_sweep.csv", index=False)

    if all_candidates:
        pd.concat(all_candidates, ignore_index=True).to_csv("candidates_v619_score_sweep.csv", index=False)
    else:
        pd.DataFrame().to_csv("candidates_v619_score_sweep.csv", index=False)

    if all_skipped:
        pd.concat(all_skipped, ignore_index=True).to_csv("skipped_v619_score_sweep.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "score_min", "date", "ticker", "reason", "price", "total_cost", "cash"
        ]).to_csv("skipped_v619_score_sweep.csv", index=False)

    print("\nSaved:")
    print("summary_v619_score_sweep.csv")
    print("trades_v619_score_sweep.csv")
    print("equity_v619_score_sweep.csv")
    print("candidates_v619_score_sweep.csv")
    print("skipped_v619_score_sweep.csv")


if __name__ == "__main__":
    main()
