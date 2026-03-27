# plot_equity_curve.py

import pandas as pd
import matplotlib.pyplot as plt

# 読み込み
df = pd.read_csv("v95_equity_curve.csv")

# 日付変換
df["date"] = pd.to_datetime(df["date"])

# 比較したい条件（ここ変える）
TARGETS = [

    {"initial_capital": 20000, "pullback_pct": 0.002},
    {"initial_capital": 20000, "pullback_pct": 0.003},
    {"initial_capital": 20000, "pullback_pct": 0.004},
    {"initial_capital": 20000, "pullback_pct": 0.007},
        {"initial_capital": 50000, "pullback_pct": 0.002},
    {"initial_capital": 100000, "pullback_pct": 0.007},
]

plt.figure()

for t in TARGETS:
    cond = (
        (df["initial_capital"] == t["initial_capital"]) &
        (df["pullback_pct"] == t["pullback_pct"])
    )

    sub = df[cond].sort_values("date")

    if sub.empty:
        continue

    label = f"{t['initial_capital']} / pb={t['pullback_pct']}"
    plt.plot(sub["date"], sub["equity"], label=label)

plt.legend()
plt.xlabel("Date")
plt.ylabel("Equity")
plt.title("Equity Curve Comparison")

plt.show()