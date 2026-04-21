# analyze_paper_signal.py
# -*- coding: utf-8 -*-

import os
import pandas as pd

SIGNAL_LOG_FILE = "paper_signal_log.csv"
CANDIDATE_LOG_FILE = "paper_candidates_log.csv"
EQUITY_FILE = "paper_equity_log.csv"


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    signals = read_csv_safe(SIGNAL_LOG_FILE)
    candidates = read_csv_safe(CANDIDATE_LOG_FILE)
    equity = read_csv_safe(EQUITY_FILE)

    print("=== PAPER SIGNAL SUMMARY ===")

    if not signals.empty:
        print("\n[ENTRY / EXIT COUNT]")
        if "action" in signals.columns:
            print(signals["action"].value_counts())

        print("\n[LATEST SIGNALS]")
        print(signals.tail(20).to_string(index=False))
    else:
        print("\nNo signals.")

    if not candidates.empty:
        print("\n[CANDIDATE DAYS]")
        print("days_with_candidates:", candidates["signal_date"].nunique())

        print("\n[LATEST CANDIDATES]")
        print(candidates.tail(20).to_string(index=False))
    else:
        print("\nNo candidates.")

    if not equity.empty:
        print("\n[EQUITY LOG]")
        print(equity.tail(20).to_string(index=False))

        if "equity" in equity.columns:
            eq = pd.to_numeric(equity["equity"], errors="coerce").dropna()
            if not eq.empty:
                peak = eq.cummax()
                dd = eq / peak - 1.0
                print("\n[MAX DRAWDOWN]")
                print(float(dd.min()))
    else:
        print("\nNo equity log.")


if __name__ == "__main__":
    main()