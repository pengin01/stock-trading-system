# compare_equity_curves.py
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


# =========================
# 設定
# =========================
TARGET_FILES = [
    # v97
    "equity_curve_v97_cap20000_pb0.002.csv",
    "equity_curve_v97_cap20000_pb0.004.csv",
    "equity_curve_v97_cap20000_pb0.007.csv",
    # v95（あれば）
    # "equity_curve_v95_cap20000_pb0.004.csv",
]


# =========================
# 読み込み
# =========================
def load_equity(file_path):
    if not os.path.exists(file_path):
        print(f"skip: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 列名吸収（v95/v97差異対策）
    if "total_equity" in df.columns:
        df = df.rename(columns={"total_equity": "equity"})

    if "equity" not in df.columns:
        print(f"invalid file: {file_path}")
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df


# =========================
# ラベル生成
# =========================
def make_label(file_path):
    name = os.path.basename(file_path)

    # v97_cap20000_pb0.004 → v97 pb=0.004
    parts = name.replace(".csv", "").split("_")

    label = ""
    for p in parts:
        if p.startswith("v"):
            label += p + " "
        if p.startswith("pb"):
            label += p

    return label.strip()


# =========================
# プロット（Equity）
# =========================
def plot_equity(dfs):
    plt.figure(figsize=(12, 6))

    for label, df in dfs.items():
        plt.plot(df["date"], df["equity"], label=label)

    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# プロット（Drawdown）
# =========================
def plot_drawdown(dfs):
    plt.figure(figsize=(12, 5))

    for label, df in dfs.items():
        df = df.copy()
        df["peak"] = df["equity"].cummax()
        df["dd"] = df["equity"] / df["peak"] - 1

        plt.plot(df["date"], df["dd"], label=label)

    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# サマリ出力
# =========================
def print_summary(dfs):
    print("\n=== SUMMARY ===")

    rows = []

    for label, df in dfs.items():
        start = df["equity"].iloc[0]
        end = df["equity"].iloc[-1]

        total_return = end / start - 1

        peak = df["equity"].cummax()
        dd = (df["equity"] / peak - 1).min()

        rows.append({"name": label, "total_return": total_return, "max_drawdown": dd})

    out = pd.DataFrame(rows)
    print(out.sort_values("total_return", ascending=False).to_string(index=False))


# =========================
# MAIN
# =========================
def main():
    dfs = {}

    for f in TARGET_FILES:
        df = load_equity(f)
        if df is None:
            continue

        label = make_label(f)
        dfs[label] = df

    if not dfs:
        print("No valid files")
        return

    print_summary(dfs)
    plot_equity(dfs)
    plot_drawdown(dfs)


if __name__ == "__main__":
    main()
