# v500_paper_signal.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import yfinance as yf

# =========================
# PARAMETERS
# =========================
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
YEARS = 2

TICKERS = [
    "7203.T", "6758.T", "9984.T", "8306.T", "8035.T",
    "6861.T", "6098.T", "9432.T", "6954.T", "4519.T",
    "6501.T", "7267.T", "6902.T", "8031.T", "4568.T",
    "4063.T", "7751.T", "8591.T", "9020.T", "4502.T"
]

SIGNAL_LOG_FILE = "paper_signal_log.csv"
CANDIDATE_LOG_FILE = "paper_candidates_log.csv"
POSITIONS_FILE = "paper_positions.csv"
EQUITY_FILE = "paper_equity_log.csv"


# =========================
# FILE HELPERS
# =========================
def ensure_files() -> None:
    if not os.path.exists(SIGNAL_LOG_FILE):
        pd.DataFrame(columns=[
            "run_date", "signal_date", "action", "ticker",
            "price", "qty", "reason"
        ]).to_csv(SIGNAL_LOG_FILE, index=False)

    if not os.path.exists(CANDIDATE_LOG_FILE):
        pd.DataFrame(columns=[
            "run_date", "signal_date", "rank", "ticker",
            "close", "ma25", "ma75", "hh", "volume", "vol20",
            "value20", "ma_slope_pct"
        ]).to_csv(CANDIDATE_LOG_FILE, index=False)

    if not os.path.exists(POSITIONS_FILE):
        pd.DataFrame(columns=[
            "ticker", "entry_date", "entry_price", "qty"
        ]).to_csv(POSITIONS_FILE, index=False)

    if not os.path.exists(EQUITY_FILE):
        pd.DataFrame(columns=[
            "run_date", "signal_date", "cash", "position_value",
            "equity", "position_count"
        ]).to_csv(EQUITY_FILE, index=False)


def append_csv(path: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=header, index=False)


def load_positions() -> pd.DataFrame:
    if not os.path.exists(POSITIONS_FILE):
        return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "qty"])

    try:
        df = pd.read_csv(POSITIONS_FILE)
    except Exception:
        return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "qty"])

    if df.empty:
        return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "qty"])

    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    if "entry_price" in df.columns:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    return df.dropna(subset=["ticker", "entry_date", "entry_price", "qty"]).copy()


def save_positions(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "qty"]).to_csv(
            POSITIONS_FILE, index=False
        )
        return
    df.to_csv(POSITIONS_FILE, index=False)


# =========================
# DATA
# =========================
def load_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=f"{YEARS}y", progress=False, auto_adjust=False)
    except Exception as e:
        print(f"{ticker}: download error: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if any(x in price_names for x in level0):
            df.columns = df.columns.get_level_values(0)
        elif any(x in price_names for x in level1):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join([str(a), str(b)]) for a, b in df.columns]

    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    for col in ["Close", "Volume"]:
        if col not in df.columns:
            print(f"{ticker}: missing column {col}")
            return pd.DataFrame()

    close = df["Close"]
    volume = df["Volume"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    df["Close"] = pd.to_numeric(close, errors="coerce")
    df["Volume"] = pd.to_numeric(volume, errors="coerce")

    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["HH"] = df["Close"].rolling(BREAKOUT).max()

    return df.dropna()


def load_all_data() -> dict[str, pd.DataFrame]:
    data = {t: load_data(t) for t in TICKERS}
    return {k: v for k, v in data.items() if not v.empty}


def get_signal_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    dates = []
    for df in data.values():
        if df is None or df.empty:
            continue
        dates.append(df.index.max())

    if not dates:
        return None

    return min(dates).normalize()


# =========================
# RULES
# =========================
def bars_passed(df: pd.DataFrame, entry_date: pd.Timestamp, current_date: pd.Timestamp) -> int:
    return int(((df.index > entry_date) & (df.index <= current_date)).sum())


def build_candidates(signal_date: pd.Timestamp, data: dict[str, pd.DataFrame], held: set[str]) -> list[dict]:
    candidates = []

    for ticker, df in data.items():
        if ticker in held:
            continue
        if signal_date not in df.index:
            continue

        i = df.index.get_loc(signal_date)
        if i < BREAKOUT + 5:
            continue

        close = float(df["Close"].iloc[i])
        ma25 = float(df["MA25"].iloc[i])
        ma75 = float(df["MA75"].iloc[i])
        hh = float(df["HH"].iloc[i - 1])
        volume = float(df["Volume"].iloc[i])
        vol20 = float(df["VOL20"].iloc[i])
        value20 = float(df["VALUE20"].iloc[i])

        ma_now = float(df["MA25"].iloc[i])
        ma_past = float(df["MA25"].iloc[i - 5])
        ma_slope_pct = ma_now / ma_past - 1.0 if ma_past != 0 else 0.0

        # ① トレンド
        if not (close > ma25 > ma75):
            continue

        # ② 強いブレイク
        if close <= hh * BREAKOUT_BUFFER:
            continue

        # ③ 出来高ブレイク
        if volume <= vol20 * VOL_MULT:
            continue

        # ④ トレンド加速
        if ma_slope_pct < MA_SLOPE_PCT:
            continue

        # ⑤ 流動性
        if value20 < MIN_VALUE:
            continue

        candidates.append({
            "ticker": ticker,
            "close": close,
            "ma25": ma25,
            "ma75": ma75,
            "hh": hh,
            "volume": volume,
            "vol20": vol20,
            "value20": value20,
            "ma_slope_pct": ma_slope_pct,
        })

    # ブレイクの強さ順にしたいならここを変更可
    candidates.sort(key=lambda x: x["ma_slope_pct"], reverse=True)
    return candidates


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_files()

    run_date = pd.Timestamp.now().normalize()
    print("RUN:", pd.Timestamp.now())

    data = load_all_data()
    print("Universe size:", len(data))

    if not data:
        raise RuntimeError("No market data loaded")

    signal_date = get_signal_date(data)
    if signal_date is None:
        raise RuntimeError("No signal date found")

    print("Today:", run_date)
    print("Signal date:", signal_date)

    positions = load_positions()
    cash = float(INITIAL_CAPITAL)
    actions = []

    # 現在cashは固定の紙トレード簡易版
    if not positions.empty:
        invested = float((positions["entry_price"] * positions["qty"]).sum())
        cash = max(0.0, INITIAL_CAPITAL - invested)

    # EXIT判定
    remain_rows = []
    for _, p in positions.iterrows():
        ticker = str(p["ticker"])
        entry_date = pd.Timestamp(p["entry_date"]).normalize()

        df = data.get(ticker, pd.DataFrame())
        if df.empty or signal_date not in df.index:
            remain_rows.append(p.to_dict())
            continue

        price = float(df.loc[signal_date, "Close"])
        ma25 = float(df.loc[signal_date, "MA25"])
        hold = bars_passed(df, entry_date, signal_date)

        if price >= ma25 and hold < HOLD_DAYS:
            remain_rows.append(p.to_dict())
            continue

        actions.append({
            "run_date": run_date.strftime("%Y-%m-%d"),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "action": "EXIT",
            "ticker": ticker,
            "price": price,
            "qty": int(p["qty"]),
            "reason": "ma25_break" if price < ma25 else "time_exit",
        })

    positions = pd.DataFrame(remain_rows, columns=["ticker", "entry_date", "entry_price", "qty"])

    # ENTRY候補
    held = set(positions["ticker"].tolist()) if not positions.empty else set()
    candidates = build_candidates(signal_date, data, held)

    cand_rows = []
    for rank, c in enumerate(candidates, start=1):
        cand_rows.append({
            "run_date": run_date.strftime("%Y-%m-%d"),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "rank": rank,
            "ticker": c["ticker"],
            "close": c["close"],
            "ma25": c["ma25"],
            "ma75": c["ma75"],
            "hh": c["hh"],
            "volume": c["volume"],
            "vol20": c["vol20"],
            "value20": c["value20"],
            "ma_slope_pct": c["ma_slope_pct"],
        })

    # 仮想ENTRY
    slots = MAX_POSITIONS - (len(positions) if not positions.empty else 0)
    if slots > 0 and candidates:
        for c in candidates[:slots]:
            ticker = c["ticker"]
            price = c["close"]
            qty = int((cash * RISK_RATIO) // price)

            if qty <= 0:
                continue

            cost = price * qty
            if cost > cash:
                continue

            cash -= cost

            new_row = {
                "ticker": ticker,
                "entry_date": signal_date.strftime("%Y-%m-%d"),
                "entry_price": price,
                "qty": qty,
            }
            positions = pd.concat([positions, pd.DataFrame([new_row])], ignore_index=True)

            actions.append({
                "run_date": run_date.strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "action": "ENTRY",
                "ticker": ticker,
                "price": price,
                "qty": qty,
                "reason": "breakout_signal",
            })

    # equity計算
    position_value = 0.0
    if not positions.empty:
        for _, p in positions.iterrows():
            ticker = str(p["ticker"])
            qty = float(p["qty"])
            df = data.get(ticker, pd.DataFrame())

            if df.empty or signal_date not in df.index:
                px = float(p["entry_price"])
            else:
                px = float(df.loc[signal_date, "Close"])

            position_value += px * qty

    equity = cash + position_value

    # 保存
    append_csv(SIGNAL_LOG_FILE, pd.DataFrame(actions))
    append_csv(CANDIDATE_LOG_FILE, pd.DataFrame(cand_rows))
    save_positions(positions)

    append_csv(EQUITY_FILE, pd.DataFrame([{
        "run_date": run_date.strftime("%Y-%m-%d"),
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "cash": cash,
        "position_value": position_value,
        "equity": equity,
        "position_count": 0 if positions.empty else len(positions),
    }]))

    # 表示
    print("== ENTRY / EXIT ==")
    if actions:
        print(pd.DataFrame(actions).to_string(index=False))
    else:
        print("(none)")

    print("\n== TOP CANDIDATES ==")
    if cand_rows:
        print(pd.DataFrame(cand_rows).head(10).to_string(index=False))
    else:
        print("(none)")

    print("\n== POSITIONS ==")
    if not positions.empty:
        print(positions.to_string(index=False))
    else:
        print("(empty)")

    print("\n== EQUITY ==")
    print(equity)


if __name__ == "__main__":
    main()