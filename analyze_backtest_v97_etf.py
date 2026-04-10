# analyze_backtest_v97_etf.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_FILE = "backtest_summary_v97_etf.csv"
TRADES_FILE = "backtest_trades_v97_etf.csv"
EQUITY_FILE = "backtest_equity_curve_v97_etf.csv"


def load_required_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)


def calc_drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def main():
    summary = load_required_csv(SUMMARY_FILE)
    trades = load_required_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else pd.DataFrame()
    equity = load_required_csv(EQUITY_FILE)

    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
    equity = equity.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)

    print("=== SUMMARY CSV ===")
    print(summary.to_string(index=False))

    if not trades.empty:
        trades["return"] = pd.to_numeric(trades["return"], errors="coerce")
        trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")

        print("\n=== TRADES SUMMARY ===")
        print("trade_count:", len(trades))
        print("win_rate:", float((trades["return"] > 0).mean()))
        print("avg_return:", float(trades["return"].mean()))
        print("best_trade:", float(trades["return"].max()))
        print("worst_trade:", float(trades["return"].min()))
        print("total_pnl:", float(trades["pnl"].sum()))
    else:
        print("\nNo trades found.")

    equity["drawdown"] = calc_drawdown_series(equity["equity"])
    max_dd = float(equity["drawdown"].min()) if not equity.empty else 0.0

    print("\n=== EQUITY SUMMARY ===")
    print("start_equity:", float(equity["equity"].iloc[0]))
    print("end_equity:", float(equity["equity"].iloc[-1]))
    print("max_drawdown:", max_dd)

    # 月次リターン
    monthly = equity.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly_last = monthly.groupby("month")["equity"].last()
    monthly_ret = monthly_last.pct_change().dropna()

    if not monthly_ret.empty:
        monthly_df = monthly_ret.reset_index()
        monthly_df.columns = ["month", "return"]
        monthly_df.to_csv("backtest_monthly_return_v97_etf.csv", index=False)

        print("\n=== MONTHLY RETURN ===")
        print(monthly_df.to_string(index=False))
        print("\nSaved: backtest_monthly_return_v97_etf.csv")

    # Equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity["date"], equity["equity"])
    plt.title("Backtest Equity Curve v97 ETF")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backtest_equity_curve_v97_etf.png")
    plt.show()

    # Drawdown
    plt.figure(figsize=(12, 4))
    plt.plot(equity["date"], equity["drawdown"])
    plt.title("Backtest Drawdown v97 ETF")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backtest_drawdown_v97_etf.png")
    plt.show()

    # Trade return histogram
    if not trades.empty:
        plt.figure(figsize=(10, 5))
        plt.hist(trades["return"].dropna(), bins=20)
        plt.title("Trade Return Distribution v97 ETF")
        plt.xlabel("Return")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("backtest_trade_return_hist_v97_etf.png")
        plt.show()

    print("\nSaved: backtest_equity_curve_v97_etf.png")
    print("Saved: backtest_drawdown_v97_etf.png")
    if not trades.empty:
        print("Saved: backtest_trade_return_hist_v97_etf.png")


if __name__ == "__main__":
    main()