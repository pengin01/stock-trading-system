import pandas as pd
import yfinance as yf

# =====================
# v622 robust low-cap portfolio
# 頑健性重視版
#
# ・2万円運用
# ・100株単位
# ・銘柄厳選
# ・銘柄別 HOLD_DAYS
# ・過剰最適化を避けるため
#   「極端な最適値」ではなく
#   安定寄り設定を採用
# =====================

# =====================
# 厳選ユニバース
# =====================
TICKERS = [
    "9432.T",  # 安定型
    "9424.T",  # 利益型
]

# =====================
# 銘柄別 HOLD
# 過剰最適化を避けるため
# 少し丸めた値を採用
# =====================
HOLD_DAYS_MAP = {
    "9432.T": 5,
    "9424.T": 10,
}

START = "2022-01-01"
END = "2026-01-01"

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


# =====================
# 指標
# =====================
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


# =====================
# データ取得
# =====================
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


# =====================
# バックテスト
# =====================
def run_backtest(data):
    dates = sorted(set(d for df in data.values() for d in df.index))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    candidate_log = []

    for date in dates:

        # =====================
        # EXIT
        # =====================
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

            # TP
            if high >= entry_price * (1 + TP_PCT):
                exit_price = entry_price * (1 + TP_PCT)
                reason = "TP"
                exit_flag = True

            # SL
            elif low <= entry_price * (1 + SL_PCT):
                exit_price = entry_price * (1 + SL_PCT)
                reason = "SL"
                exit_flag = True

            # TIME
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

                        cand = {
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
                            "score": score,
                            "row": row,
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
                    continue

                if cash < total_cost:
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

        # =====================
        # EQUITY
        # =====================
        position_value = 0

        for pos in positions:
            ticker = pos["ticker"]

            if ticker in data and date in data[ticker].index:
                close = float(data[ticker].loc[date]["Close"])
                position_value += close * int(pos["shares"])

        equity = cash + position_value

        equity_curve.append({
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
    )


# =====================
# 集計
# =====================
def summarize(trades_df, equity_df, candidate_df):

    if len(equity_df) > 0:
        equity_df["equity"] = pd.to_numeric(
            equity_df["equity"],
            errors="coerce",
        )

        peak = equity_df["equity"].cummax()
        dd = equity_df["equity"] / peak - 1

        final_equity = equity_df["equity"].iloc[-1]
        max_dd = dd.min()

    else:
        final_equity = INITIAL_CAPITAL
        max_dd = 0

    if len(trades_df) == 0:
        return {
            "trades": 0,
            "candidates": len(candidate_df),
            "win_rate": None,
            "total_pnl": 0,
            "avg_pnl": None,
            "avg_return": None,
            "final_equity": final_equity,
            "max_dd": max_dd,
        }

    return {
        "trades": len(trades_df),
        "candidates": len(candidate_df),
        "win_rate": (trades_df["pnl"] > 0).mean(),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "avg_return": trades_df["return"].mean(),
        "final_equity": final_equity,
        "max_dd": max_dd,
    }


# =====================
# MAIN
# =====================
def main():

    data = load_data()

    trades_df, equity_df, candidate_df = run_backtest(data)

    summary = summarize(
        trades_df,
        equity_df,
        candidate_df,
    )

    print("\n==============================")
    print("v622 ROBUST LOW-CAP RESULT")
    print("==============================")

    for k, v in summary.items():
        print(f"{k}: {v}")

    if len(trades_df) > 0:

        trades_df["year"] = pd.to_datetime(
            trades_df["exit_date"]
        ).dt.year

        print("\n=== YEARLY ===")
        print(
            trades_df.groupby("year")["pnl"]
            .agg(["count", "mean", "sum"])
        )

        print("\n=== BY TICKER ===")
        print(
            trades_df.groupby("ticker")["pnl"]
            .agg(["count", "mean", "sum"])
        )

        print("\n=== BY REASON ===")
        print(
            trades_df.groupby("reason")["pnl"]
            .agg(["count", "mean", "sum"])
        )

        print("\n=== HOLD DAYS ===")
        print(
            trades_df.groupby("ticker")["hold_days"]
            .first()
        )

    print("\n=== CANDIDATES ===")
    print(
        candidate_df.groupby("ticker")
        .size()
        .sort_values(ascending=False)
    )

    # =====================
    # 保存
    # =====================
    trades_df.to_csv(
        "trades_v622.csv",
        index=False,
    )

    equity_df.to_csv(
        "equity_v622.csv",
        index=False,
    )

    candidate_df.to_csv(
        "candidates_v622.csv",
        index=False,
    )

    pd.DataFrame([summary]).to_csv(
        "summary_v622.csv",
        index=False,
    )

    print("\nSaved:")
    print("trades_v622.csv")
    print("equity_v622.csv")
    print("candidates_v622.csv")
    print("summary_v622.csv")


if __name__ == "__main__":
    main()
