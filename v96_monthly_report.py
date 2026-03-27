# v96_monthly_report.py

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
CSV_FILE = "v95_equity_curve.csv"
TARGET_PULLBACK = 0.007
TARGET_CAPITAL = 20000
#TARGET_PULLBACK = 0.002   # v94で存在する値にする

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_FILE)
df["date"] = pd.to_datetime(df["date"])

# 利用可能な条件を先に確認
print("available initial_capital:", sorted(df["initial_capital"].dropna().unique()))
print("available pullback_pct:", sorted(df["pullback_pct"].dropna().unique()))

# 条件で絞る
df = df[
    (df["initial_capital"] == TARGET_CAPITAL) &
    (df["pullback_pct"] == TARGET_PULLBACK)
].copy()

if df.empty:
    print("\n対象データがありません。")
    print(f"CSV_FILE={CSV_FILE}")
    print(f"initial_capital={TARGET_CAPITAL}")
    print(f"pullback_pct={TARGET_PULLBACK}")
    raise SystemExit(0)

df = df.sort_values("date")

# =========================
# 月次リターン
# =========================
df["month"] = df["date"].dt.to_period("M")

monthly = df.groupby("month").agg(
    start=("equity", "first"),
    end=("equity", "last"),
)

monthly["return"] = monthly["end"] / monthly["start"] - 1

def calc_dd(x):
    peak = x.cummax()
    return (x / peak - 1).min()

monthly["max_dd"] = df.groupby("month")["equity"].apply(calc_dd)

print("\n=== MONTHLY REPORT ===")
print(monthly)

monthly.to_csv("monthly_report.csv")
print("\nSaved: monthly_report.csv")

# =========================
# グラフ
# =========================
if monthly.empty:
    print("monthly is empty, skip plot")
    raise SystemExit(0)

plt.figure()
monthly["return"].plot(kind="bar")
plt.title(f"Monthly Returns | capital={TARGET_CAPITAL}, pb={TARGET_PULLBACK}")
plt.ylabel("Return")
plt.tight_layout()
plt.show()