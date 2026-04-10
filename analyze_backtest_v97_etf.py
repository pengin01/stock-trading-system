import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 対象ファイル
# =========================
equity_file = r"backtest_results_v97_etf\equity_hd4_pb0007_rsi56_mp1_tp0014_sl0008.csv"
trades_file = r"backtest_results_v97_etf\trades_hd4_pb0007_rsi56_mp1_tp0014_sl0008.csv"

# =========================
# 読み込み
# =========================
eq = pd.read_csv(equity_file)
tr = pd.read_csv(trades_file)

eq["date"] = pd.to_datetime(eq["date"])
tr["entry_date"] = pd.to_datetime(tr["entry_date"])
tr["exit_date"] = pd.to_datetime(tr["exit_date"])

# =========================
# ドローダウン計算
# =========================
eq["peak"] = eq["equity"].cummax()
eq["drawdown"] = eq["equity"] / eq["peak"] - 1.0

# =========================
# 1. 資産推移
# =========================
plt.figure(figsize=(12, 5))
plt.plot(eq["date"], eq["equity"])
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 2. ドローダウン推移
# =========================
plt.figure(figsize=(12, 4))
plt.plot(eq["date"], eq["drawdown"])
plt.title("Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 3. トレード損益分布
# =========================
plt.figure(figsize=(10, 4))
plt.hist(tr["return"], bins=20)
plt.title("Trade Return Distribution")
plt.xlabel("Return")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 4. 累積損益
# =========================
tr = tr.sort_values("exit_date").copy()
tr["cum_pnl"] = tr["pnl"].cumsum()

plt.figure(figsize=(12, 5))
plt.plot(tr["exit_date"], tr["cum_pnl"])
plt.title("Cumulative PnL")
plt.xlabel("Exit Date")
plt.ylabel("Cumulative PnL")
plt.grid(True)
plt.tight_layout()
plt.show()
