import pandas as pd
import matplotlib.pyplot as plt

# =========================
# file
# =========================
CSV_FILE = "equity_v98.csv"

# =========================
# load
# =========================
df = pd.read_csv(CSV_FILE)
df["date"] = pd.to_datetime(df["date"])

# =========================
# drawdown
# =========================
df["peak"] = df["equity"].cummax()
df["drawdown"] = df["equity"] / df["peak"] - 1.0

# =========================
# 1. equity curve
# =========================
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["equity"])
plt.title("Equity Curve v98")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 2. drawdown curve
# =========================
plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["drawdown"])
plt.title("Drawdown Curve v98")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.show()
