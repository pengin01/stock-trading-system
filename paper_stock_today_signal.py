# -*- coding: utf-8 -*-
# pip install pandas numpy yfinance ta

from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import ta


# =========================
# PARAMETERS
# =========================
@dataclass
class Params:
    capital_yen: int = 80000

    ma_days: int = 25
    rsi_days: int = 14

    rsi_max: float = 65.0
    pullback_pct: float = 0.005

    min_avg_value20: float = 300_000_000

    hold_days: int = 4
    years: int = 1

    pos_file: str = "positions.csv"
    entry_file: str = "today_entry.csv"
    exit_file: str = "today_exit.csv"
    log_file: str = "daily_result_log.csv"


P = Params()

STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]


# =========================
# DATA
# =========================
def download(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{P.years}y",
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns: {missing}")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["MA"] = close.rolling(P.ma_days).mean()
    out["RSI"] = ta.momentum.RSIIndicator(close, window=P.rsi_days).rsi()
    out["VALUE20"] = (close * out["Volume"]).rolling(20).mean()

    return out


# =========================
# SIGNAL
# =========================
def entry_signal(df: pd.DataFrame, i: int) -> bool:
    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]
    ma = df["MA"].iloc[i]
    rsi = df["RSI"].iloc[i]
    v = df["VALUE20"].iloc[i]

    vals = np.array([c, prev, ma, rsi, v], dtype=float)
    if not np.isfinite(vals).all():
        return False

    if c < ma:
        return False

    if c > prev * (1 - P.pullback_pct):
        return False

    if rsi > P.rsi_max:
        return False

    if v < P.min_avg_value20:
        return False

    return True


# =========================
# TODAY ENTRY
# =========================
def today_entry() -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    candidates = []

    for ticker in STOCK_UNIVERSE:
        try:
            df = download(ticker)
            if df.empty:
                continue

            df = add_features(df)

            hist = df[df.index.normalize() < today]
            if len(hist) < P.ma_days + 2:
                continue

            i = len(hist) - 1

            if not entry_signal(hist, i):
                continue

            close_price = float(hist["Close"].iloc[i])
            shares = int(P.capital_yen // close_price) if close_price > 0 else 0

            candidates.append({
                "run_date": today.date(),
                "ticker": ticker,
                "signal_date": hist.index[i].date(),
                "close": round(close_price, 4),
                "ma": round(float(hist["MA"].iloc[i]), 4),
                "rsi": round(float(hist["RSI"].iloc[i]), 4),
                "value20": round(float(hist["VALUE20"].iloc[i]), 0),
                "shares": shares,
                "score": round(float(hist["RSI"].iloc[i]), 4),
            })

        except Exception as e:
            print(f"[WARN] entry check failed: {ticker}: {e}")

    if not candidates:
        return pd.DataFrame(columns=[
            "run_date", "ticker", "signal_date", "close", "ma", "rsi", "value20", "shares", "score"
        ])

    out = pd.DataFrame(candidates).sort_values(["score", "ticker"]).reset_index(drop=True)

    # 1銘柄採用
    return out.head(1)


# =========================
# TODAY EXIT
# =========================
def today_exit(pos_df: pd.DataFrame) -> pd.DataFrame:
    if pos_df.empty:
        return pd.DataFrame(columns=[
            "run_date", "ticker", "entry_date", "last_date", "hold_days_calendar",
            "entry_price", "close", "ma", "shares", "reason"
        ])

    today = pd.Timestamp.now().normalize()
    exits = []

    for _, row in pos_df.iterrows():
        try:
            ticker = row["ticker"]
            entry_date = pd.to_datetime(row["entry_date"]).tz_localize(None)
            entry_price = float(row.get("entry_price", np.nan))
            shares = int(row.get("shares", 0))

            df = download(ticker)
            if df.empty:
                continue

            df = add_features(df)

            hist = df[df.index.normalize() < today]
            if hist.empty:
                continue

            last_i = len(hist) - 1
            last_date = hist.index[last_i]
            last_close = float(hist["Close"].iloc[last_i])
            last_ma = float(hist["MA"].iloc[last_i])

            hold_days_calendar = (last_date.normalize() - entry_date.normalize()).days

            if not np.isfinite([last_close, last_ma]).all():
                continue

            if last_close < last_ma:
                exits.append({
                    "run_date": today.date(),
                    "ticker": ticker,
                    "entry_date": entry_date.date(),
                    "last_date": last_date.date(),
                    "hold_days_calendar": hold_days_calendar,
                    "entry_price": round(entry_price, 4) if np.isfinite(entry_price) else np.nan,
                    "close": round(last_close, 4),
                    "ma": round(last_ma, 4),
                    "shares": shares,
                    "reason": "MA_EXIT",
                })
                continue

            if hold_days_calendar >= P.hold_days:
                exits.append({
                    "run_date": today.date(),
                    "ticker": ticker,
                    "entry_date": entry_date.date(),
                    "last_date": last_date.date(),
                    "hold_days_calendar": hold_days_calendar,
                    "entry_price": round(entry_price, 4) if np.isfinite(entry_price) else np.nan,
                    "close": round(last_close, 4),
                    "ma": round(last_ma, 4),
                    "shares": shares,
                    "reason": "TIME",
                })
                continue

        except Exception as e:
            print(f"[WARN] exit check failed: {row.get('ticker', 'UNKNOWN')}: {e}")

    if not exits:
        return pd.DataFrame(columns=[
            "run_date", "ticker", "entry_date", "last_date", "hold_days_calendar",
            "entry_price", "close", "ma", "shares", "reason"
        ])

    return pd.DataFrame(exits).reset_index(drop=True)


# =========================
# POSITIONS
# =========================
def empty_positions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ticker", "entry_date", "entry_price", "shares", "source"
    ])


def load_positions() -> pd.DataFrame:
    pos_path = Path(P.pos_file)
    if not pos_path.exists():
        return empty_positions_df()

    try:
        pos = pd.read_csv(pos_path, parse_dates=["entry_date"])
    except Exception as e:
        print(f"[WARN] could not read {P.pos_file}: {e}")
        return empty_positions_df()

    if pos.empty:
        return empty_positions_df()

    required = ["ticker", "entry_date", "entry_price", "shares", "source"]
    for col in required:
        if col not in pos.columns:
            pos[col] = np.nan

    return pos[required].copy()


def save_positions(pos_df: pd.DataFrame) -> None:
    pos_df = pos_df.copy()
    if not pos_df.empty:
        pos_df["entry_date"] = pd.to_datetime(pos_df["entry_date"]).dt.strftime("%Y-%m-%d")
    pos_df.to_csv(P.pos_file, index=False, encoding="utf-8-sig")


def apply_position_updates(pos_df: pd.DataFrame, entry_df: pd.DataFrame, exit_df: pd.DataFrame) -> pd.DataFrame:
    updated = pos_df.copy()

    # EXITを先に反映
    if not exit_df.empty and not updated.empty:
        exit_keys = set(
            zip(
                exit_df["ticker"].astype(str),
                pd.to_datetime(exit_df["entry_date"]).dt.strftime("%Y-%m-%d")
            )
        )

        updated["_entry_date_key"] = pd.to_datetime(updated["entry_date"]).dt.strftime("%Y-%m-%d")
        updated = updated[
            ~updated.apply(lambda r: (str(r["ticker"]), r["_entry_date_key"]) in exit_keys, axis=1)
        ].drop(columns=["_entry_date_key"])

    # ENTRYを反映（同一ticker保有中なら重複追加しない）
    if not entry_df.empty:
        if updated.empty:
            current_tickers = set()
        else:
            current_tickers = set(updated["ticker"].astype(str).tolist())

        new_rows = []
        for _, row in entry_df.iterrows():
            ticker = str(row["ticker"])
            if ticker in current_tickers:
                continue

            new_rows.append({
                "ticker": ticker,
                "entry_date": pd.to_datetime(row["signal_date"]).strftime("%Y-%m-%d"),
                "entry_price": float(row["close"]),
                "shares": int(row["shares"]),
                "source": "SIGNAL",
            })

        if new_rows:
            updated = pd.concat([updated, pd.DataFrame(new_rows)], ignore_index=True)

    if updated.empty:
        return empty_positions_df()

    updated = updated.sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    return updated


# =========================
# CSV OUTPUT
# =========================
def save_daily_csv(entry_df: pd.DataFrame, exit_df: pd.DataFrame) -> None:
    entry_df.to_csv(P.entry_file, index=False, encoding="utf-8-sig")
    exit_df.to_csv(P.exit_file, index=False, encoding="utf-8-sig")


def append_daily_log(entry_df: pd.DataFrame, exit_df: pd.DataFrame, before_pos: pd.DataFrame, after_pos: pd.DataFrame) -> None:
    now = pd.Timestamp.now()
    today = now.normalize()

    log_row = {
        "run_timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "run_date": today.date(),
        "universe_count": len(STOCK_UNIVERSE),
        "entry_count": len(entry_df),
        "exit_count": len(exit_df),
        "positions_before": len(before_pos),
        "positions_after": len(after_pos),
        "entry_tickers": ",".join(entry_df["ticker"].astype(str).tolist()) if not entry_df.empty else "",
        "exit_tickers": ",".join(exit_df["ticker"].astype(str).tolist()) if not exit_df.empty else "",
        "params_capital_yen": P.capital_yen,
        "params_ma_days": P.ma_days,
        "params_rsi_days": P.rsi_days,
        "params_rsi_max": P.rsi_max,
        "params_pullback_pct": P.pullback_pct,
        "params_min_avg_value20": P.min_avg_value20,
        "params_hold_days": P.hold_days,
        "params_years": P.years,
    }

    log_df = pd.DataFrame([log_row])
    log_path = Path(P.log_file)

    if log_path.exists():
        old = pd.read_csv(log_path)
        new_df = pd.concat([old, log_df], ignore_index=True)
        new_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    else:
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")


# =========================
# DISPLAY
# =========================
def print_section(title: str) -> None:
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)


def print_params() -> None:
    print_section("PARAMETERS")
    for k, v in asdict(P).items():
        print(f"{k}: {v}")


# =========================
# MAIN
# =========================
def main() -> None:
    now = pd.Timestamp.now()
    print("RUN:", now)

    print_params()

    before_pos = load_positions()

    print_section("POSITIONS BEFORE")
    if before_pos.empty:
        print("(no positions)")
    else:
        print(before_pos.to_string(index=False))

    print_section("TODAY ENTRY")
    entry_df = today_entry()
    if entry_df.empty:
        print("(no entry)")
    else:
        print(entry_df.to_string(index=False))

    print_section("TODAY EXIT")
    exit_df = today_exit(before_pos)
    if exit_df.empty:
        print("(no exit)")
    else:
        print(exit_df.to_string(index=False))

    save_daily_csv(entry_df, exit_df)

    after_pos = apply_position_updates(before_pos, entry_df, exit_df)
    save_positions(after_pos)

    append_daily_log(entry_df, exit_df, before_pos, after_pos)

    print_section("POSITIONS AFTER")
    if after_pos.empty:
        print("(no positions)")
    else:
        print(after_pos.to_string(index=False))

    print_section("CSV OUTPUT")
    print(f"saved: {P.entry_file}")
    print(f"saved: {P.exit_file}")
    print(f"saved: {P.pos_file}")
    print(f"appended: {P.log_file}")


if __name__ == "__main__":
    main()