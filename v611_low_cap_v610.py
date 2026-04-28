import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =====================
# v614 low-cap candidate log
# 5万円・100株単位・v610ルール継承
# 候補ログ付き
# =====================

TICKERS = [
    "5020.T",
    "7186.T",
    "8410.T",
    "8308.T",
    "8331.T",
    "8601.T",
]

START = "2022-01-01"
END = "2026-01-01"

INITIAL_CAPITAL = 30_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03
HOLD_DAYS = 10

MIN_SCORE = 0.010
MIN_VALUE20 = 50_000_000

PRICE_MIN = 100
PRICE_MAX = 500

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

    return lower >= body_safe * 2 and upper <= body_safe and body <= rng * 0.4


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
        data[ticker] = df

    return data


def run():
    data = load_data()
    dates = sorted(set(d for df in data.values() for d in df.index))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    skipped = []
    candidate_log = []

    for date in dates:

        # =====================
        # EXIT
        # =====================
        new_positions = []

        for pos in positions:
            ticker = pos["ticker"]
            df = data[ticker]

            if date not in df.index:
                new_positions.append(pos)
                continue

            row = df.loc[date]
            high = float(row["High"])
            low = float(row["Low"])

            exit_flag = False
            exit_price = None
            reason = None

            if high >= pos["entry_price"] * (1 + TP_PCT):
                exit_price = pos["entry_price"] * (1 + TP_PCT)
                reason = "TP"
                exit_flag = True

            elif low <= pos["entry_price"] * (1 + SL_PCT):
                exit_price = pos["entry_price"] * (1 + SL_PCT)
                reason = "SL"
                exit_flag = True

            elif (date - pos["entry_date"]).days >= HOLD_DAYS:
                exit_price = float(row["Close"])
                reason = "TIME"
                exit_flag = True

            if exit_flag:
                exit_price *= 1 - SLIPPAGE

                gross = exit_price * pos["shares"]
                fee = gross * FEE_PCT
                proceeds = gross - fee

                pnl = proceeds - pos["cost"]
                ret = pnl / pos["cost"]

                cash += proceeds

                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "cost": pos["cost"],
                        "proceeds": proceeds,
                        "pnl": pnl,
                        "return": ret,
                        "reason": reason,
                        "score": pos["score"],
                    }
                )
            else:
                new_positions.append(pos)

        positions = new_positions

        # =====================
        # ENTRY
        # =====================
        if len(positions) < MAX_POSITIONS:
            candidates = []

            for ticker, df in data.items():
                if date not in df.index:
                    continue

                row = df.loc[date]
                close = float(row["Close"])

                # 価格帯チェック
                if not (PRICE_MIN <= close <= PRICE_MAX):
                    continue

                # v610シグナルチェック
                if (
                    is_hammer(row)
                    and float(row["ret3"]) < 0
                    and float(row["value20"]) >= MIN_VALUE20
                ):
                    score = -float(row["ret3"])

                    if score >= MIN_SCORE:
                        candidate_log.append(
                            {
                                "date": date,
                                "ticker": ticker,
                                "price": close,
                                "score": score,
                                "ret3": float(row["ret3"]),
                                "value20": float(row["value20"]),
                                "open": float(row["Open"]),
                                "high": float(row["High"]),
                                "low": float(row["Low"]),
                                "close": close,
                            }
                        )

                        candidates.append(
                            {
                                "ticker": ticker,
                                "row": row,
                                "score": score,
                            }
                        )

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
                    skipped.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "OVER_INITIAL_CAPITAL",
                            "price": entry_price,
                            "total_cost": total_cost,
                            "cash": cash,
                        }
                    )
                    continue

                if cash < total_cost:
                    skipped.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "NO_CASH",
                            "price": entry_price,
                            "total_cost": total_cost,
                            "cash": cash,
                        }
                    )
                    continue

                cash -= total_cost

                positions.append(
                    {
                        "ticker": ticker,
                        "entry_date": date,
                        "entry_price": entry_price,
                        "shares": shares,
                        "cost": total_cost,
                        "score": cand["score"],
                    }
                )

                break

        # =====================
        # EQUITY
        # =====================
        equity = cash

        for pos in positions:
            ticker = pos["ticker"]
            df = data[ticker]

            if date in df.index:
                equity += float(df.loc[date]["Close"]) * pos["shares"]

        equity_curve.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "position_count": len(positions),
            }
        )

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_curve),
        pd.DataFrame(skipped),
        pd.DataFrame(candidate_log),
    )


# =====================
# 実行
# =====================
trades_df, equity_df, skipped_df, candidate_df = run()

print("\n=== RESULT v614 low-cap candidate log ===")

if len(trades_df) == 0:
    print("No trades found")
else:
    print("trades:", len(trades_df))
    print("win_rate:", (trades_df["pnl"] > 0).mean())
    print("total_pnl:", trades_df["pnl"].sum())
    print("avg_pnl:", trades_df["pnl"].mean())
    print("avg_return:", trades_df["return"].mean())

    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = equity_df["equity"] / equity_df["peak"] - 1

    print("final_equity:", equity_df["equity"].iloc[-1])
    print("max_drawdown:", equity_df["dd"].min())

    print("\n=== YEARLY ===")
    trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year

    yearly = trades_df.groupby("year").agg(
        trades=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    )

    yearly["cumulative_pnl"] = yearly["total_pnl"].cumsum()
    print(yearly)

    print("\n=== BY TICKER ===")
    print(trades_df.groupby("ticker")["pnl"].agg(["count", "mean", "sum"]))

    print("\n=== BY REASON ===")
    print(trades_df.groupby("reason")["pnl"].agg(["count", "mean", "sum"]))

print("\n=== CANDIDATES ===")
print("candidates:", len(candidate_df))

if len(candidate_df) > 0:
    print(candidate_df["ticker"].value_counts())

print("\n=== SKIPPED ===")
print("skipped:", len(skipped_df))

if len(skipped_df) > 0:
    print(skipped_df["reason"].value_counts())

# =====================
# 保存
# =====================
trades_df.to_csv("trades_v614_low_cap.csv", index=False)
equity_df.to_csv("equity_v614_low_cap.csv", index=False)
skipped_df.to_csv("skipped_v614_low_cap.csv", index=False)
candidate_df.to_csv("candidates_v614_low_cap.csv", index=False)

if len(trades_df) > 0:
    yearly.to_csv("yearly_v614_low_cap.csv")

print("\nSaved:")
print("trades_v614_low_cap.csv")
print("equity_v614_low_cap.csv")
print("skipped_v614_low_cap.csv")
print("candidates_v614_low_cap.csv")
print("yearly_v614_low_cap.csv")

# =====================
# グラフ
# =====================
if len(equity_df) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df["date"], equity_df["equity"])
    plt.title("Equity Curve v614 low-cap")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_v614_low_cap.png")
    plt.show()

print("equity_v614_low_cap.png")
