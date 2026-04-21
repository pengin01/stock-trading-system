import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# FILES
# =========================
RESULT_DIR = "backtest_results_stock_v8"
TRADES_FILE = os.path.join(RESULT_DIR, "trades_v8.csv")
EQUITY_FILE = os.path.join(RESULT_DIR, "equity_v8.csv")

YEARLY_SUMMARY_FILE = os.path.join(RESULT_DIR, "yearly_summary_v8.csv")
YEARLY_RETURN_PNG = os.path.join(RESULT_DIR, "yearly_return_v8.png")
YEARLY_DD_PNG = os.path.join(RESULT_DIR, "yearly_max_dd_v8.png")
YEARLY_TRADE_PNG = os.path.join(RESULT_DIR, "yearly_trade_count_v8.png")


# =========================
# HELPERS
# =========================
def calc_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return series / peak - 1.0


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)


# =========================
# LOAD
# =========================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = safe_read_csv(TRADES_FILE)
    equity = safe_read_csv(EQUITY_FILE)

    if not trades.empty:
        trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
        trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
        trades["return"] = pd.to_numeric(trades["return"], errors="coerce")
        trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
        trades = trades.dropna(subset=["exit_date", "return", "pnl"]).copy()

    if not equity.empty:
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
        equity["cash"] = pd.to_numeric(equity["cash"], errors="coerce")
        equity["position_value"] = pd.to_numeric(
            equity["position_value"], errors="coerce"
        )
        equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
        equity = (
            equity.dropna(subset=["date", "equity"])
            .sort_values("date")
            .reset_index(drop=True)
        )

    return trades, equity


# =========================
# YEARLY ANALYSIS
# =========================
def build_yearly_summary(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()

    eq = equity.copy()
    eq["year"] = eq["date"].dt.year

    rows = []

    for year, eq_year in eq.groupby("year"):
        eq_year = eq_year.sort_values("date").reset_index(drop=True)

        start_equity = float(eq_year["equity"].iloc[0])
        end_equity = float(eq_year["equity"].iloc[-1])
        year_return = end_equity / start_equity - 1.0

        eq_year["drawdown"] = calc_drawdown(eq_year["equity"])
        year_max_dd = float(eq_year["drawdown"].min())

        tr_year = (
            trades[trades["exit_date"].dt.year == year].copy()
            if not trades.empty
            else pd.DataFrame()
        )
        trade_count = len(tr_year)

        if trade_count > 0:
            win_rate = float((tr_year["return"] > 0).mean())
            avg_return = float(tr_year["return"].mean())
            total_pnl = float(tr_year["pnl"].sum())
            best_trade = float(tr_year["return"].max())
            worst_trade = float(tr_year["return"].min())
            tp_count = (
                int((tr_year["reason"] == "take_profit").sum())
                if "reason" in tr_year.columns
                else 0
            )
            sl_count = (
                int((tr_year["reason"] == "stop_loss").sum())
                if "reason" in tr_year.columns
                else 0
            )
            time_exit_count = (
                int((tr_year["reason"] == "time_exit").sum())
                if "reason" in tr_year.columns
                else 0
            )
            same_day_count = (
                int((tr_year["reason"] == "sl_tp_same_day").sum())
                if "reason" in tr_year.columns
                else 0
            )
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

        rows.append(
            {
                "year": int(year),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return": year_return,
                "max_drawdown": year_max_dd,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "total_pnl": total_pnl,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "tp_count": tp_count,
                "sl_count": sl_count,
                "time_exit_count": time_exit_count,
                "sl_tp_same_day_count": same_day_count,
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# =========================
# PLOT
# =========================
def plot_yearly_return(yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.bar(yearly["year"].astype(str), yearly["return"])
    plt.title("Yearly Return (v8)")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(YEARLY_RETURN_PNG, dpi=150)
    plt.show()


def plot_yearly_max_dd(yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.bar(yearly["year"].astype(str), yearly["max_drawdown"])
    plt.title("Yearly Max Drawdown (v8)")
    plt.xlabel("Year")
    plt.ylabel("Max Drawdown")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(YEARLY_DD_PNG, dpi=150)
    plt.show()


def plot_yearly_trade_count(yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.bar(yearly["year"].astype(str), yearly["trade_count"])
    plt.title("Yearly Trade Count (v8)")
    plt.xlabel("Year")
    plt.ylabel("Trades")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(YEARLY_TRADE_PNG, dpi=150)
    plt.show()


# =========================
# MAIN
# =========================
def main() -> None:
    trades, equity = load_data()
    yearly = build_yearly_summary(trades, equity)

    yearly.to_csv(YEARLY_SUMMARY_FILE, index=False)

    print("=== YEARLY SUMMARY ===")
    if yearly.empty:
        print("No yearly data.")
    else:
        print(yearly.to_string(index=False))

    print(f"\nSaved: {YEARLY_SUMMARY_FILE}")

    plot_yearly_return(yearly)
    plot_yearly_max_dd(yearly)
    plot_yearly_trade_count(yearly)

    print(f"Saved: {YEARLY_RETURN_PNG}")
    print(f"Saved: {YEARLY_DD_PNG}")
    print(f"Saved: {YEARLY_TRADE_PNG}")


if __name__ == "__main__":
    main()
