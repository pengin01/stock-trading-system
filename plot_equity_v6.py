import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# FILE PATHS
# =========================
RESULT_DIR = "backtest_results_stock_v9"
EQUITY_FILE = os.path.join(RESULT_DIR, "equity_v9.csv")
OUTPUT_PNG = os.path.join(RESULT_DIR, "equity_curve_v9.png")
OUTPUT_DD_PNG = os.path.join(RESULT_DIR, "drawdown_v9.png")


def main():
    if not os.path.exists(EQUITY_FILE):
        raise FileNotFoundError(f"{EQUITY_FILE} not found")

    df = pd.read_csv(EQUITY_FILE)

    if df.empty:
        raise ValueError("equity_v6.csv is empty")

    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError("equity_v6.csv must contain 'date' and 'equity' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found in equity_v6.csv")

    # Drawdown
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1.0

    # Summary
    start_equity = float(df["equity"].iloc[0])
    end_equity = float(df["equity"].iloc[-1])
    total_return = end_equity / start_equity - 1.0
    max_dd = float(df["drawdown"].min())

    print("=== EQUITY SUMMARY ===")
    print(f"start_equity : {start_equity}")
    print(f"end_equity   : {end_equity}")
    print(f"total_return : {total_return}")
    print(f"max_drawdown : {max_dd}")

    # =========================
    # Equity Curve
    # =========================
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["equity"])
    plt.title("Equity Curve (D)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.show()

    # =========================
    # Drawdown
    # =========================
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["drawdown"])
    plt.title("Drawdown (D)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DD_PNG, dpi=150)
    plt.show()

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_DD_PNG}")


if __name__ == "__main__":
    main()
