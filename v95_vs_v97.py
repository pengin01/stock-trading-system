import pandas as pd
import matplotlib.pyplot as plt

# 比較条件
TARGET_CAPITAL = 20000
TARGET_PULLBACK = 0.004

# --- v95 ---
df95_all = pd.read_csv("v95_equity_curve.csv")
df95_all["date"] = pd.to_datetime(df95_all["date"])

df95 = df95_all[
    (df95_all["initial_capital"] == TARGET_CAPITAL) &
    (df95_all["pullback_pct"] == TARGET_PULLBACK)
].copy()

if df95.empty:
    raise ValueError("v95側に該当データがありません")

df95["plot_equity"] = df95["equity"]

# --- v97 ---
v97_file = f"equity_curve_v97_cap{TARGET_CAPITAL}_pb{TARGET_PULLBACK}.csv"
df97 = pd.read_csv(v97_file)
df97["date"] = pd.to_datetime(df97["date"])
df97["plot_equity"] = df97["total_equity"]

# --- 正規化 ---
df95["normalized"] = df95["plot_equity"] / df95["plot_equity"].iloc[0]
df97["normalized"] = df97["plot_equity"] / df97["plot_equity"].iloc[0]

# --- グラフ ---
plt.figure(figsize=(12, 6))
plt.plot(df95["date"], df95["normalized"], label="v95 full")
plt.plot(df97["date"], df97["normalized"], label="v97 80%")
plt.title(f"v95 vs v97 Normalized Equity Curve (cap={TARGET_CAPITAL}, pb={TARGET_PULLBACK})")
plt.xlabel("Date")
plt.ylabel("Growth Multiple")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()