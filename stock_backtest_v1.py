import itertools
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# GLOBAL SETTINGS
# =========================================================
INITIAL_CAPITAL = 300000
LOT_SIZE = 100
MAX_POSITIONS = 3
YEARS = 10

UNIVERSE = [
    "7203.T",
    "6758.T",
    "8306.T",
    "8035.T",
    "9984.T",
    "9432.T",
    "4063.T",
    "6861.T",
    "6501.T",
    "6098.T",
]

START_DATE = None  # 例: "2018-01-01"
END_DATE = None  # 例: "2026-01-01"

OUTPUT_ROOT_PREFIX = "backtest_results_stock_v1"


# =========================================================
# PARAMETER GRID
# 複数指定対応
# =========================================================
PARAM_GRID = {
    "ma_days": [25],
    "ma_slope_days": [5],
    "rsi_days": [14],
    "rsi_max": [48, 50, 52],
    "pullback_pct": [0.03],
    "pullback_upper": [0.02],
    "min_avg_value20": [500_000_000],
    "tp_pct": [0.05, 0.07, 0.09],
    "sl_pct": [0.04],
    "hold_days": [3, 4, 5],
    "buy_fee_pct": [0.0005],
    "sell_fee_pct": [0.0005],
    "slippage_pct": [0.0005],
}


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_cost: float
    signal_date: pd.Timestamp
    hold_days_elapsed: int = 0


# =========================================================
# UTILS
# =========================================================
def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_fee(amount: float, fee_pct: float) -> float:
    return float(amount) * float(fee_pct)


def safe_float(x) -> float:
    if pd.isna(x):
        return np.nan
    return float(x)


def build_param_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[k] for k in keys))
    combos = [dict(zip(keys, values)) for values in values_product]
    return combos


def create_output_dir(prefix: str) -> Path:
    now_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{prefix}_{now_str}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def print_global_settings():
    print("=== STOCK BACKTEST V1 MULTI ===")
    print(f"initial_capital: {INITIAL_CAPITAL}")
    print(f"lot_size: {LOT_SIZE}")
    print(f"max_positions: {MAX_POSITIONS}")
    print(f"years: {YEARS}")
    print(f"universe: {UNIVERSE}")
    print(f"start_date: {START_DATE}")
    print(f"end_date: {END_DATE}")
    print()


def print_param_grid():
    print("=== PARAM GRID ===")
    for k, v in PARAM_GRID.items():
        print(f"{k}: {v}")
    print()


def print_run_header(run_id: int, total_runs: int, params: Dict):
    print(f"=== RUN {run_id}/{total_runs} ===")
    for k, v in params.items():
        print(f"{k}: {v}")
    print()


# =========================================================
# DATA DOWNLOAD
# MA / RSI 日数違いを考慮して、最大必要期間で一括取得
# =========================================================
def download_price_data(
    tickers: List[str],
    years: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, pd.DataFrame]:
    if start_date is None:
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    else:
        start = pd.Timestamp(start_date)

    end = (
        pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
    )

    raw = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    data_map: Dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        t = tickers[0]
        df = raw.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        data_map[t] = df
    else:
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            data_map[t] = raw[t].copy()

    cleaned: Dict[str, pd.DataFrame] = {}
    for t, df in data_map.items():
        if df.empty:
            continue

        df = df.rename(columns=str.title)
        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] {t}: missing columns {missing}, skipped")
            continue

        df = df[required].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if len(df) < 100:
            print(f"[WARN] {t}: too short, skipped")
            continue

        cleaned[t] = df

    return cleaned


# =========================================================
# FEATURE BUILD
# 各パラメータセットごとに特徴量とシグナルを生成
# =========================================================
def build_features_for_params(
    base_map: Dict[str, pd.DataFrame], params: Dict
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}

    ma_days = params["ma_days"]
    ma_slope_days = params["ma_slope_days"]
    rsi_days = params["rsi_days"]
    rsi_max = params["rsi_max"]
    pullback_pct = params["pullback_pct"]
    pullback_upper = params["pullback_upper"]
    min_avg_value20 = params["min_avg_value20"]

    for ticker, base_df in base_map.items():
        df = base_df.copy()

        df["Value"] = df["Close"] * df["Volume"]
        df["Value20"] = df["Value"].rolling(20).mean()

        df["MA"] = df["Close"].rolling(ma_days).mean()
        df["RSI"] = compute_rsi(df["Close"], rsi_days)

        # トレンド判定: MAが過去より上
        df["MA_Slope"] = df["MA"] > df["MA"].shift(ma_slope_days)
        df["Cond_Trend"] = df["MA_Slope"]

        df["Cond_RSI"] = df["RSI"] <= rsi_max
        df["Cond_Liquidity"] = df["Value20"] >= min_avg_value20

        # MA近辺の押し目
        df["Ma_Distance"] = (df["Close"] - df["MA"]) / df["MA"]
        df["Cond_Pullback"] = (df["Ma_Distance"] >= -pullback_pct) & (
            df["Ma_Distance"] <= pullback_upper
        )

        df["Signal"] = (
            df["Cond_Trend"]
            & df["Cond_RSI"]
            & df["Cond_Liquidity"]
            & df["Cond_Pullback"]
        )

        df = df.dropna(subset=["MA", "RSI", "Value20"]).copy()
        result[ticker] = df

    return result


# =========================================================
# BACKTEST CORE
# =========================================================
def backtest_portfolio(
    data_map: Dict[str, pd.DataFrame],
    params: Dict,
    run_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tp_pct = params["tp_pct"]
    sl_pct = params["sl_pct"]
    hold_days = params["hold_days"]
    buy_fee_pct = params["buy_fee_pct"]
    sell_fee_pct = params["sell_fee_pct"]
    slippage_pct = params["slippage_pct"]

    all_dates = sorted(set().union(*[df.index.tolist() for df in data_map.values()]))

    positions: List[Position] = []
    pending_entries: List[dict] = []

    trades: List[dict] = []
    equity_rows: List[dict] = []

    cash = float(INITIAL_CAPITAL)

    for current_date in all_dates:
        # -------------------------------------------------
        # 1) EXIT
        # -------------------------------------------------
        next_positions: List[Position] = []

        for pos in positions:
            df = data_map[pos.ticker]
            if current_date not in df.index:
                next_positions.append(pos)
                continue

            row = df.loc[current_date]
            high = safe_float(row["High"])
            low = safe_float(row["Low"])
            close = safe_float(row["Close"])

            tp_price = pos.entry_price * (1 + tp_pct)
            sl_price = pos.entry_price * (1 - sl_pct)

            exit_flag = False
            exit_reason = None
            exit_price = None

            pos.hold_days_elapsed += 1

            # 同日両方ヒット時は保守的にSL優先
            if low <= sl_price:
                exit_flag = True
                exit_reason = "SL"
                exit_price = sl_price * (1 - slippage_pct)
            elif high >= tp_price:
                exit_flag = True
                exit_reason = "TP"
                exit_price = tp_price * (1 - slippage_pct)
            elif pos.hold_days_elapsed >= hold_days:
                exit_flag = True
                exit_reason = "TIME"
                exit_price = close * (1 - slippage_pct)

            if exit_flag:
                gross_exit_amount = exit_price * pos.shares
                sell_fee = calc_fee(gross_exit_amount, sell_fee_pct)
                net_exit_amount = gross_exit_amount - sell_fee
                cash += net_exit_amount

                pnl = net_exit_amount - pos.entry_cost
                ret = pnl / pos.entry_cost if pos.entry_cost > 0 else np.nan

                trade_row = {
                    "run_id": run_id,
                    "ticker": pos.ticker,
                    "signal_date": pos.signal_date,
                    "entry_date": pos.entry_date,
                    "exit_date": current_date,
                    "entry_price": round(pos.entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "shares": pos.shares,
                    "entry_cost": round(pos.entry_cost, 2),
                    "exit_amount_net": round(net_exit_amount, 2),
                    "pnl": round(pnl, 2),
                    "return": round(ret, 6),
                    "hold_days": pos.hold_days_elapsed,
                    "exit_reason": exit_reason,
                }
                trade_row.update(params)
                trades.append(trade_row)
            else:
                next_positions.append(pos)

        positions = next_positions

        # -------------------------------------------------
        # 2) ENTRY
        # -------------------------------------------------
        todays_entries = [x for x in pending_entries if x["entry_date"] == current_date]
        pending_entries = [
            x for x in pending_entries if x["entry_date"] != current_date
        ]

        open_slots = max(0, MAX_POSITIONS - len(positions))

        # 優先順位: RSI低い順 → MAから深く押している順
        todays_entries = sorted(
            todays_entries, key=lambda x: (x["rsi"], x["ma_distance"])
        )

        for order in todays_entries[:open_slots]:
            ticker = order["ticker"]
            df = data_map[ticker]
            if current_date not in df.index:
                continue

            entry_open = safe_float(df.loc[current_date, "Open"])
            if np.isnan(entry_open) or entry_open <= 0:
                continue

            entry_price = entry_open * (1 + slippage_pct)
            shares = LOT_SIZE

            gross_buy_amount = entry_price * shares
            buy_fee = calc_fee(gross_buy_amount, buy_fee_pct)
            total_buy_cost = gross_buy_amount + buy_fee

            if cash >= total_buy_cost:
                cash -= total_buy_cost
                positions.append(
                    Position(
                        ticker=ticker,
                        entry_date=current_date,
                        entry_price=entry_price,
                        shares=shares,
                        entry_cost=total_buy_cost,
                        signal_date=order["signal_date"],
                    )
                )

        # -------------------------------------------------
        # 3) SIGNAL -> NEXT ENTRY
        # -------------------------------------------------
        held_tickers = {p.ticker for p in positions}
        pending_tickers = {p["ticker"] for p in pending_entries}

        for ticker, df in data_map.items():
            if current_date not in df.index:
                continue
            if ticker in held_tickers or ticker in pending_tickers:
                continue

            row = df.loc[current_date]
            signal = bool(row["Signal"])

            if not signal:
                continue

            future_dates = df.index[df.index > current_date]
            if len(future_dates) == 0:
                continue

            next_date = future_dates[0]

            pending_entries.append(
                {
                    "ticker": ticker,
                    "signal_date": current_date,
                    "entry_date": next_date,
                    "rsi": safe_float(row["RSI"]),
                    "ma_distance": safe_float(row["Ma_Distance"]),
                }
            )

        # -------------------------------------------------
        # 4) EQUITY
        # -------------------------------------------------
        market_value = 0.0
        for pos in positions:
            df = data_map[pos.ticker]
            if current_date in df.index:
                px = safe_float(df.loc[current_date, "Close"])
                if not np.isnan(px):
                    market_value += px * pos.shares

        total_equity = cash + market_value

        equity_row = {
            "run_id": run_id,
            "date": current_date,
            "cash": round(cash, 2),
            "market_value": round(market_value, 2),
            "total_equity": round(total_equity, 2),
            "positions": len(positions),
            "pending_entries": len(pending_entries),
        }
        equity_row.update(params)
        equity_rows.append(equity_row)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if equity_df.empty:
        summary_row = {
            "run_id": run_id,
            "initial_capital": INITIAL_CAPITAL,
            "final_equity": INITIAL_CAPITAL,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "total_pnl": 0.0,
            "years": YEARS,
            "lot_size": LOT_SIZE,
            "max_positions": MAX_POSITIONS,
        }
        summary_row.update(params)
        return pd.DataFrame([summary_row]), trades_df, equity_df

    equity_df["cummax"] = equity_df["total_equity"].cummax()
    equity_df["drawdown"] = equity_df["total_equity"] / equity_df["cummax"] - 1.0

    final_equity = float(equity_df["total_equity"].iloc[-1])
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_drawdown = float(equity_df["drawdown"].min())

    if not trades_df.empty:
        trade_count = int(len(trades_df))
        win_rate = float((trades_df["pnl"] > 0).mean())
        avg_return = float(trades_df["return"].mean())
        total_pnl = float(trades_df["pnl"].sum())
        best_trade = float(trades_df["return"].max())
        worst_trade = float(trades_df["return"].min())
    else:
        trade_count = 0
        win_rate = np.nan
        avg_return = np.nan
        total_pnl = 0.0
        best_trade = np.nan
        worst_trade = np.nan

    summary_row = {
        "run_id": run_id,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trade_count": trade_count,
        "win_rate": round(win_rate, 6) if not pd.isna(win_rate) else np.nan,
        "avg_return": round(avg_return, 6) if not pd.isna(avg_return) else np.nan,
        "best_trade": round(best_trade, 6) if not pd.isna(best_trade) else np.nan,
        "worst_trade": round(worst_trade, 6) if not pd.isna(worst_trade) else np.nan,
        "total_pnl": round(total_pnl, 2),
        "years": YEARS,
        "lot_size": LOT_SIZE,
        "max_positions": MAX_POSITIONS,
    }
    summary_row.update(params)

    equity_df = equity_df.drop(columns=["cummax"], errors="ignore")

    return pd.DataFrame([summary_row]), trades_df, equity_df


# =========================================================
# SIGNAL DIAGNOSTICS
# 各ランごとに条件通過数を見える化
# =========================================================
def make_signal_diagnostics(
    data_map: Dict[str, pd.DataFrame], run_id: int, params: Dict
) -> pd.DataFrame:
    rows = []
    for ticker, df in data_map.items():
        row = {
            "run_id": run_id,
            "ticker": ticker,
            "rows": len(df),
            "trend_count": int(df["Cond_Trend"].sum()),
            "rsi_count": int(df["Cond_RSI"].sum()),
            "liquidity_count": int(df["Cond_Liquidity"].sum()),
            "pullback_count": int(df["Cond_Pullback"].sum()),
            "signal_count": int(df["Signal"].sum()),
        }
        row.update(params)
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    print_global_settings()
    print_param_grid()

    param_combinations = build_param_combinations(PARAM_GRID)
    total_runs = len(param_combinations)

    print(f"total_parameter_combinations: {total_runs}")
    print()

    output_dir = create_output_dir(OUTPUT_ROOT_PREFIX)
    print(f"output_dir: {output_dir}")
    print()

    print("Downloading price data...")
    base_map = download_price_data(
        tickers=UNIVERSE,
        years=YEARS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if not base_map:
        print("No valid data downloaded.")
        return

    print(f"downloaded_tickers: {list(base_map.keys())}")
    print()

    all_summary = []
    all_trades = []
    all_equity = []
    all_diagnostics = []

    for idx, params in enumerate(param_combinations, start=1):
        run_id = idx
        print_run_header(run_id, total_runs, params)

        feature_map = build_features_for_params(base_map, params)

        diagnostics_df = make_signal_diagnostics(feature_map, run_id, params)
        all_diagnostics.append(diagnostics_df)

        # コンソールにも簡易表示
        total_signals = int(diagnostics_df["signal_count"].sum())
        print("signal diagnostics:")
        for _, r in diagnostics_df.iterrows():
            print(
                f"  {r['ticker']}: "
                f"trend={r['trend_count']} "
                f"rsi={r['rsi_count']} "
                f"liq={r['liquidity_count']} "
                f"pullback={r['pullback_count']} "
                f"signal={r['signal_count']}"
            )
        print(f"total_signals: {total_signals}")
        print()

        summary_df, trades_df, equity_df = backtest_portfolio(
            data_map=feature_map,
            params=params,
            run_id=run_id,
        )

        all_summary.append(summary_df)
        all_trades.append(trades_df)
        all_equity.append(equity_df)

        print("run summary:")
        print(summary_df.to_string(index=False))
        print()

    summary_all = (
        pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    )
    trades_all = (
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    )
    equity_all = (
        pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    )
    diagnostics_all = (
        pd.concat(all_diagnostics, ignore_index=True)
        if all_diagnostics
        else pd.DataFrame()
    )

    # 並び替え
    if not summary_all.empty:
        summary_all = summary_all.sort_values(
            by=["total_return", "max_drawdown", "trade_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    # 出力
    summary_path = output_dir / "summary_all.csv"
    trades_path = output_dir / "trades_all.csv"
    equity_path = output_dir / "equity_all.csv"
    diagnostics_path = output_dir / "signal_diagnostics_all.csv"

    summary_all.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades_all.to_csv(trades_path, index=False, encoding="utf-8-sig")
    equity_all.to_csv(equity_path, index=False, encoding="utf-8-sig")
    diagnostics_all.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    print("=== FINAL RESULT ===")
    if not summary_all.empty:
        print(summary_all.to_string(index=False))
    else:
        print("No summary rows.")
    print()

    print(f"Saved: {summary_path}")
    print(f"Saved: {trades_path}")
    print(f"Saved: {equity_path}")
    print(f"Saved: {diagnostics_path}")

    if not summary_all.empty:
        print()
        print("=== TOP 10 BY TOTAL_RETURN ===")
        top_cols = [
            "run_id",
            "total_return",
            "max_drawdown",
            "trade_count",
            "win_rate",
            "avg_return",
            "ma_days",
            "ma_slope_days",
            "rsi_days",
            "rsi_max",
            "pullback_pct",
            "pullback_upper",
            "min_avg_value20",
            "tp_pct",
            "sl_pct",
            "hold_days",
        ]
        existing_cols = [c for c in top_cols if c in summary_all.columns]
        print(summary_all[existing_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
