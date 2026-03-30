# plot_equity_compare.py

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 比較対象ファイル
# =========================
FILES = [
    ("v97 pb0.004 200000", "equity_curve_v97_cap20000_pb0.004.csv"),
    ("v97 pb0.004 800000", "equity_curve_v97_cap80000_pb0.004.csv"),
    ("v97 pb0.004 1000000", "equity_curve_v97_cap100000_pb0.004.csv")
]

data = {}

# =========================
# 読み込み
# =========================
for label, path in FILES:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    df["peak"] = df["total_equity"].cummax()
    df["drawdown"] = df["total_equity"] / df["peak"] - 1.0

    # 正規化（比較用）
    df["normalized"] = df["total_equity"] / df["total_equity"].iloc[0]

    data[label] = df

# =========================
# Equity比較
# =========================
plt.figure(figsize=(12, 6))

for label, df in data.items():
    plt.plot(df["date"], df["normalized"], label=label)

plt.title("Equity Curve Comparison (Normalized)")
plt.xlabel("Date")
plt.ylabel("Growth Multiple")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Drawdown比較
# =========================
plt.figure(figsize=(12, 4))

for label, df in data.items():
    plt.plot(df["date"], df["drawdown"], label=label)

plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 数値比較
# =========================
print("\n=== SUMMARY ===")

for label, df in data.items():
    final = df["normalized"].iloc[-1]
    max_dd = df["drawdown"].min()

    print(f"{label}")
    print(f"  final_multiple: {final:.2f}")
    print(f"  max_drawdown : {max_dd:.2%}")