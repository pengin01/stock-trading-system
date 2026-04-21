# analyze_backtest_202603_202604.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_FILE = "backtest_202603_202604_summary.csv"
TRADES_FILE = "backtest_202603_202604_trades.csv"
EQUITY_FILE = "backtest_202603_202604_equity.csv"
DAILY_LOG_FILE = "backtest_202603_202604_daily_log.csv"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)


def calc_drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def safe_read_optional(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def print_section(title: str):
    print(f"\n=== {title} ===")


def analyze_summary(summary: pd.DataFrame):
    print_section("SUMMARY CSV")
    print(summary.to_string(index=False))


def analyze_trades(trades: pd.DataFrame):
    print_section("TRADES SUMMARY")

    if trades.empty:
        print("No trades found.")
        return

    trades = trades.copy()
    for c in ["entry_price", "exit_price", "qty", "return", "pnl"]:
        if c in trades.columns:
            trades[c] = pd.to_numeric(trades[c], errors="coerce")

    trade_count = len(trades)
    win_rate = float((trades["return"] > 0).mean()) if trade_count > 0 else 0.0
    avg_return = float(trades["return"].mean()) if trade_count > 0 else 0.0
    total_pnl = float(trades["pnl"].sum()) if trade_count > 0 else 0.0
    best_trade = float(trades["return"].max()) if trade_count > 0 else 0.0
    worst_trade = float(trades["return"].min()) if trade_count > 0 else 0.0
    avg_pnl = float(trades["pnl"].mean()) if trade_count > 0 else 0.0

    print("trade_count:", trade_count)
    print("win_rate:", win_rate)
    print("avg_return:", avg_return)
    print("best_trade:", best_trade)
    print("worst_trade:", worst_trade)
    print("avg_pnl:", avg_pnl)
    print("total_pnl:", total_pnl)

    print_section("TRADES DETAIL")
    print(trades.to_string(index=False))

    # 銘柄別集計
    if "ticker" in trades.columns:
        by_ticker = (
            trades.groupby("ticker", dropna=False)
            .agg(
                trade_count=("ticker", "size"),
                win_rate=("return", lambda s: float((s > 0).mean())),
                avg_return=("return", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .reset_index()
            .sort_values("total_pnl", ascending=False)
        )
        print_section("TRADES BY TICKER")
        print(by_ticker.to_string(index=False))
        by_ticker.to_csv("analysis_trades_by_ticker_202603_202604.csv", index=False)
        print("Saved: analysis_trades_by_ticker_202603_202604.csv")


def analyze_equity(equity: pd.DataFrame):
    equity = equity.copy()
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    for c in ["equity", "cash", "position_value", "positions"]:
        if c in equity.columns:
            equity[c] = pd.to_numeric(equity[c], errors="coerce")

    equity = (
        equity.dropna(subset=["date", "equity"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    equity["drawdown"] = calc_drawdown_series(equity["equity"])

    print_section("EQUITY SUMMARY")
    print("start_equity:", float(equity["equity"].iloc[0]))
    print("end_equity:", float(equity["equity"].iloc[-1]))
    print("max_drawdown:", float(equity["drawdown"].min()))
    print("max_equity:", float(equity["equity"].max()))
    print("min_equity:", float(equity["equity"].min()))

    print_section("EQUITY DETAIL")
    print(equity.to_string(index=False))

    # Equity chart
    plt.figure(figsize=(12, 6))
    plt.plot(equity["date"], equity["equity"])
    plt.title("Equity Curve (2026/03 - 2026/04)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("analysis_equity_curve_202603_202604.png")
    plt.show()

    # Drawdown chart
    plt.figure(figsize=(12, 4))
    plt.plot(equity["date"], equity["drawdown"])
    plt.title("Drawdown (2026/03 - 2026/04)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("analysis_drawdown_202603_202604.png")
    plt.show()

    # Cash / Position Value chart
    if "cash" in equity.columns and "position_value" in equity.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(equity["date"], equity["cash"], label="cash")
        plt.plot(equity["date"], equity["position_value"], label="position_value")
        plt.title("Cash / Position Value (2026/03 - 2026/04)")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("analysis_cash_position_202603_202604.png")
        plt.show()

    equity.to_csv("analysis_equity_with_drawdown_202603_202604.csv", index=False)
    print("Saved: analysis_equity_with_drawdown_202603_202604.csv")
    print("Saved: analysis_equity_curve_202603_202604.png")
    print("Saved: analysis_drawdown_202603_202604.png")
    if "cash" in equity.columns and "position_value" in equity.columns:
        print("Saved: analysis_cash_position_202603_202604.png")


def analyze_daily_log(daily: pd.DataFrame):
    if daily.empty:
        print_section("DAILY LOG")
        print("No daily log found.")
        return

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    for c in [
        "entry_count",
        "exit_count",
        "top_candidate_rsi",
        "cash",
        "position_value",
        "equity",
        "positions",
    ]:
        if c in daily.columns:
            daily[c] = pd.to_numeric(daily[c], errors="coerce")

    daily = daily.sort_values("date").reset_index(drop=True)

    print_section("DAILY LOG SUMMARY")
    print("days:", len(daily))
    print(
        "days_with_entry:",
        int((daily["entry_count"] > 0).sum()) if "entry_count" in daily.columns else 0,
    )
    print(
        "days_with_exit:",
        int((daily["exit_count"] > 0).sum()) if "exit_count" in daily.columns else 0,
    )
    if "top_candidate" in daily.columns:
        print(
            "days_with_candidate:", int((daily["top_candidate"].fillna("") != "").sum())
        )

    print_section("DAILY LOG DETAIL")
    print(daily.to_string(index=False))

    # 日別 entry/exit
    if "entry_count" in daily.columns and "exit_count" in daily.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(daily["date"], daily["entry_count"], marker="o", label="entry_count")
        plt.plot(daily["date"], daily["exit_count"], marker="o", label="exit_count")
        plt.title("Daily Entry / Exit Count")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("analysis_daily_entry_exit_202603_202604.png")
        plt.show()
        print("Saved: analysis_daily_entry_exit_202603_202604.png")

    # 候補RSI推移
    if "top_candidate_rsi" in daily.columns:
        x = daily.dropna(subset=["top_candidate_rsi"]).copy()
        if not x.empty:
            plt.figure(figsize=(12, 5))
            plt.plot(x["date"], x["top_candidate_rsi"], marker="o")
            plt.title("Top Candidate RSI")
            plt.xlabel("Date")
            plt.ylabel("RSI")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("analysis_top_candidate_rsi_202603_202604.png")
            plt.show()
            print("Saved: analysis_top_candidate_rsi_202603_202604.png")

    daily.to_csv("analysis_daily_log_202603_202604.csv", index=False)
    print("Saved: analysis_daily_log_202603_202604.csv")


def main():
    summary = load_csv(SUMMARY_FILE)
    equity = load_csv(EQUITY_FILE)
    trades = safe_read_optional(TRADES_FILE)
    daily = safe_read_optional(DAILY_LOG_FILE)

    analyze_summary(summary)
    analyze_trades(trades)
    analyze_equity(equity)
    analyze_daily_log(daily)


if __name__ == "__main__":
    main()
