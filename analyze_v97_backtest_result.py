# analyze_v97_backtest_result.py
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = "v97_backtest_result.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "final_capital",
        "total_return",
        "max_drawdown",
        "trades",
        "win_rate",
        "avg_return",
        "passed_candidates",
        "initial_capital",
        "hold_days",
        "pullback_pct",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def print_basic_summary(df: pd.DataFrame):
    print("\n=== RAW RESULT ===")
    print(df.to_string(index=False))

    print("\n=== SORTED BY total_return DESC ===")
    if "total_return" in df.columns:
        print(df.sort_values("total_return", ascending=False).to_string(index=False))

    print("\n=== GROUP SUMMARY BY pullback_pct ===")
    grp = (
        df.groupby("pullback_pct", dropna=False)
        .agg(
            count=("pullback_pct", "size"),
            avg_total_return=("total_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_avg_return=("avg_return", "mean"),
            avg_trades=("trades", "mean"),
            avg_passed_candidates=("passed_candidates", "mean"),
        )
        .reset_index()
        .sort_values("pullback_pct")
    )
    print(grp.to_string(index=False))

    return grp


def make_pivot_tables(df: pd.DataFrame):
    print("\n=== PIVOT: total_return ===")
    p_total = df.pivot(index="initial_capital", columns="pullback_pct", values="total_return")
    print(p_total)

    print("\n=== PIVOT: max_drawdown ===")
    p_dd = df.pivot(index="initial_capital", columns="pullback_pct", values="max_drawdown")
    print(p_dd)

    print("\n=== PIVOT: win_rate ===")
    p_wr = df.pivot(index="initial_capital", columns="pullback_pct", values="win_rate")
    print(p_wr)

    print("\n=== PIVOT: trades ===")
    p_trades = df.pivot(index="initial_capital", columns="pullback_pct", values="trades")
    print(p_trades)

    print("\n=== PIVOT: passed_candidates ===")
    p_candidates = df.pivot(index="initial_capital", columns="pullback_pct", values="passed_candidates")
    print(p_candidates)

    return p_total, p_dd, p_wr, p_trades, p_candidates


def plot_total_return_by_capital(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    for pb, sub in df.groupby("pullback_pct"):
        sub = sub.sort_values("initial_capital")
        plt.plot(
            sub["initial_capital"],
            sub["total_return"],
            marker="o",
            label=f"pullback={pb}"
        )

    plt.title("Total Return by Initial Capital")
    plt.xlabel("Initial Capital")
    plt.ylabel("Total Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drawdown_by_capital(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    for pb, sub in df.groupby("pullback_pct"):
        sub = sub.sort_values("initial_capital")
        plt.plot(
            sub["initial_capital"],
            sub["max_drawdown"],
            marker="o",
            label=f"pullback={pb}"
        )

    plt.title("Max Drawdown by Initial Capital")
    plt.xlabel("Initial Capital")
    plt.ylabel("Max Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_win_rate_by_capital(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    for pb, sub in df.groupby("pullback_pct"):
        sub = sub.sort_values("initial_capital")
        plt.plot(
            sub["initial_capital"],
            sub["win_rate"],
            marker="o",
            label=f"pullback={pb}"
        )

    plt.title("Win Rate by Initial Capital")
    plt.xlabel("Initial Capital")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_avg_return_by_capital(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    for pb, sub in df.groupby("pullback_pct"):
        sub = sub.sort_values("initial_capital")
        plt.plot(
            sub["initial_capital"],
            sub["avg_return"],
            marker="o",
            label=f"pullback={pb}"
        )

    plt.title("Average Return per Trade by Initial Capital")
    plt.xlabel("Initial Capital")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trades_and_candidates(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    for pb, sub in df.groupby("pullback_pct"):
        sub = sub.sort_values("initial_capital")
        plt.plot(
            sub["initial_capital"],
            sub["trades"],
            marker="o",
            label=f"trades pb={pb}"
        )

    plt.title("Trades by Initial Capital")
    plt.xlabel("Initial Capital")
    plt.ylabel("Trades")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if "passed_candidates" in df.columns:
        plt.figure(figsize=(12, 6))

        for pb, sub in df.groupby("pullback_pct"):
            sub = sub.sort_values("initial_capital")
            plt.plot(
                sub["initial_capital"],
                sub["passed_candidates"],
                marker="o",
                label=f"passed pb={pb}"
            )

        plt.title("Passed Candidates by Initial Capital")
        plt.xlabel("Initial Capital")
        plt.ylabel("Passed Candidates")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_bar_group_summary(grp: pd.DataFrame):
    x = grp["pullback_pct"].astype(str)

    plt.figure(figsize=(10, 5))
    plt.bar(x, grp["avg_total_return"])
    plt.title("Average Total Return by Pullback")
    plt.xlabel("Pullback")
    plt.ylabel("Average Total Return")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(x, grp["avg_max_drawdown"])
    plt.title("Average Max Drawdown by Pullback")
    plt.xlabel("Pullback")
    plt.ylabel("Average Max Drawdown")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(x, grp["avg_win_rate"])
    plt.title("Average Win Rate by Pullback")
    plt.xlabel("Pullback")
    plt.ylabel("Average Win Rate")
    plt.tight_layout()
    plt.show()


def print_best_conditions(df: pd.DataFrame):
    print("\n=== BEST BY total_return ===")
    if "total_return" in df.columns:
        best_total = df.sort_values("total_return", ascending=False).iloc[0]
        print(best_total.to_string())

    print("\n=== BEST BY shallowest max_drawdown ===")
    if "max_drawdown" in df.columns:
        best_dd = df.sort_values("max_drawdown", ascending=False).iloc[0]
        print(best_dd.to_string())

    print("\n=== BEST BY win_rate ===")
    if "win_rate" in df.columns:
        best_wr = df.sort_values("win_rate", ascending=False).iloc[0]
        print(best_wr.to_string())


def main():
    df = load_data(FILE_NAME)

    required = ["initial_capital", "pullback_pct", "total_return", "max_drawdown"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grp = print_basic_summary(df)
    make_pivot_tables(df)
    print_best_conditions(df)

    plot_total_return_by_capital(df)
    plot_drawdown_by_capital(df)

    if "win_rate" in df.columns:
        plot_win_rate_by_capital(df)

    if "avg_return" in df.columns:
        plot_avg_return_by_capital(df)

    if "trades" in df.columns:
        plot_trades_and_candidates(df)

    plot_bar_group_summary(grp)


if __name__ == "__main__":
    main()