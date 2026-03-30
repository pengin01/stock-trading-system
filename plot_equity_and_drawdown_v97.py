import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("equity_curve_v97_cap20000_pb0.004.csv")
df["date"] = pd.to_datetime(df["date"])

df["peak"] = df["total_equity"].cummax()
df["drawdown"] = df["total_equity"] / df["peak"] - 1.0

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["total_equity"])
plt.title("v97 Equity Curve (cap=80000, pb=0.004)")
plt.xlabel("Date")
plt.ylabel("Total Equity")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["drawdown"])
plt.title("v97 Drawdown (cap=80000, pb=0.004)")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Max Drawdown:", df["drawdown"].min())