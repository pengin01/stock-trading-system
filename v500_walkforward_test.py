# v500_walkforward_test_fast.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf

# ===== 採用パラメータ =====
INITIAL_CAPITAL = 20000
MA_SHORT = 25
MA_LONG = 75
BREAKOUT = 40
VOL_MULT = 1.5
MA_SLOPE_PCT = 0.02
BREAKOUT_BUFFER = 1.01
HOLD_DAYS = 7
MAX_POSITIONS = 2
RISK_RATIO = 0.5
MIN_VALUE = 100_000_000
YEARS = 5

TICKERS = [
    "7203.T",
    "6758.T",
    "9984.T",
    "8306.T",
    "8035.T",
    "6861.T",
    "6098.T",
    "9432.T",
    "6954.T",
    "4519.T",
    "6501.T",
    "7267.T",
    "6902.T",
    "8031.T",
    "4568.T",
    "4063.T",
    "7751.T",
    "8591.T",
    "9020.T",
    "4502.T",
]


def download_all(tickers):
    """
    全銘柄を一括取得して、銘柄ごとのDataFrameに分解する
    """
    raw = yf.download(
        tickers,
        period=f"{YEARS}y",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    print(type(raw))
    print(raw.shape)
    print(raw.columns)
    print(raw.head())
    result = {}

    # 単一銘柄時の保険
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([tickers[:1], raw.columns])

    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue

        df = raw[t].copy()
        df = df.dropna(subset=["Close", "Volume"])
        if df.empty:
            continue

        df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
        df["MA75"] = df["Close"].rolling(MA_LONG).mean()
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
        df["HH"] = df["Close"].rolling(BREAKOUT).max()

        # エントリー条件を前計算
        df["ENTRY_OK"] = (
            (df["Close"] > df["MA25"])
            & (df["MA25"] > df["MA75"])
            & (df["Close"] > df["HH"].shift(1) * BREAKOUT_BUFFER)
            & (df["Volume"] > df["VOL20"] * VOL_MULT)
            & ((df["MA25"] / df["MA25"].shift(5) - 1) >= MA_SLOPE_PCT)
            & (df["VALUE20"] >= MIN_VALUE)
        )

        df = df.dropna().copy()
        if not df.empty:
            result[t] = df

    return result


def build_runtime_data(data):
    """
    run() 内で pandas をほぼ触らないための前処理
    """
    runtime = {}
    all_dates = set()

    for t, df in data.items():
        idx = df.index.to_numpy()
        close = df["Close"].to_numpy(dtype=np.float64)
        ma25 = df["MA25"].to_numpy(dtype=np.float64)
        entry_ok = df["ENTRY_OK"].to_numpy(dtype=np.bool_)

        date_to_i = {d: i for i, d in enumerate(idx)}
        all_dates.update(idx.tolist())

        runtime[t] = {
            "index": idx,
            "close": close,
            "ma25": ma25,
            "entry_ok": entry_ok,
            "date_to_i": date_to_i,
        }

    all_dates = np.array(sorted(all_dates))
    return runtime, all_dates


data = download_all(TICKERS)
runtime_data, ALL_DATES = build_runtime_data(data)


def run(start, end):
    cash = float(INITIAL_CAPITAL)
    positions = []
    last_equity = float(INITIAL_CAPITAL)

    # 必要区間だけに絞る
    mask = (ALL_DATES >= np.datetime64(start)) & (ALL_DATES <= np.datetime64(end))
    dates = ALL_DATES[mask]

    for d in dates:
        # ===== EXIT =====
        new_positions = []
        for p in positions:
            rt = runtime_data[p["t"]]
            i = rt["date_to_i"].get(d)

            if i is None:
                new_positions.append(p)
                continue

            price = rt["close"][i]
            ma25 = rt["ma25"][i]

            # 元コードはロジックが怪しいので、ここは「現在日までの保有日数」に修正
            hold_days = i - p["entry_i"]

            if price >= ma25 and hold_days < HOLD_DAYS:
                new_positions.append(p)
            else:
                cash += price * p["q"]

        positions = new_positions

        # ===== ENTRY =====
        if len(positions) < MAX_POSITIONS:
            held = {p["t"] for p in positions}

            for t, rt in runtime_data.items():
                if len(positions) >= MAX_POSITIONS:
                    break
                if t in held:
                    continue

                i = rt["date_to_i"].get(d)
                if i is None:
                    continue
                if not rt["entry_ok"][i]:
                    continue

                close = rt["close"][i]
                qty = int((cash * RISK_RATIO) // close)
                if qty <= 0:
                    continue

                cash -= close * qty
                positions.append(
                    {
                        "t": t,
                        "q": qty,
                        "entry_i": i,
                    }
                )
                held.add(t)

        # ===== EQUITY =====
        pv = 0.0
        for p in positions:
            rt = runtime_data[p["t"]]
            i = rt["date_to_i"].get(d)
            if i is not None:
                pv += rt["close"][i] * p["q"]

        last_equity = cash + pv

    return last_equity / INITIAL_CAPITAL - 1.0


print("\n=== WALK FORWARD FAST ===")
print("2021-2023:", run(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")))
print("2024-2026:", run(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31")))
print("2024 only:", run(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")))
print("2025 only:", run(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")))
print("2026 only:", run(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")))
