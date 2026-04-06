# Stock Trading System

RSI + 移動平均 + 押し目戦略による日本株の自動売買システムです。  
日経225銘柄を対象に、毎日シグナルを生成し、Discord通知・GitHub Actionsによる自動運用をサポートします。

---

## 目次

- [戦略概要](#戦略概要)
- [システム構成](#システム構成)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
  - [日次シグナル（本番運用）](#日次シグナル本番運用)
  - [ペーパートレード](#ペーパートレード)
  - [バックテスト](#バックテスト)
  - [分析・可視化](#分析可視化)
- [GitHub Actions 自動運用](#github-actions-自動運用)
- [パラメータ詳細](#パラメータ詳細)
- [データファイル](#データファイル)
- [戦略バージョン履歴](#戦略バージョン履歴)
- [Discord 通知の見方](#discord-通知の見方)
- [バックテスト結果](#バックテスト結果)

---

## 戦略概要

### エントリー条件

以下すべてを満たす場合にエントリーシグナルを生成します：

| 条件 | 説明 |
|------|------|
| 株価 > MA(25) | 上昇トレンド中であること |
| 前日比で押し目（pullback） | 株価が前日終値から一定割合下落していること |
| RSI(14) < 65 | 過熱状態でないこと |
| 20日平均売買代金 >= 3億円 | 流動性が十分であること |

### イグジット条件

- **4日後に強制決済**（time exit）

### 資金管理（v97）

- 1トレードで資金の **70%** を使用（`RISK_RATIO = 0.7`）
- 残り30%はキャッシュとして保持
- 同時保有：**1銘柄のみ**
- 複利で運用

---

## システム構成

```
stock-trading-system/
├── v97_system.py                  # 本番運用（日次シグナル生成 + ポジション管理）
├── daily_signal.py                # 日次シグナル（v97ベース、別実装）
├── paper_stock_today_signal.py    # ペーパートレード（シグナル + 診断）
├── paper_stock_backtest.py        # ペーパートレード用バックテスト
├── v97_backtest_nikkei225.py      # v97 日経225バックテスト（パラメータグリッド）
├── v97_risk_control_with_equity.py # v97 リスク管理バックテスト（エクイティ出力付き）
├── v95_with_equity_output.py      # v95 バックテスト（エクイティカーブ出力）
├── v92_unified_system.py          # v92 統合システム（エントリー/イグジット統一ロジック）
├── v92_experiment.py              # v92 パラメータ実験
├── v92_experiment_small_capital.py # v92 少額資金実験
├── v85_backtest.py                # v85 バックテスト（手数料・スリッページ込み）
├── analyze_v97_backtest_result.py # v97 バックテスト結果の分析・可視化
├── compare_equity_curves.py       # エクイティカーブ比較
├── v95_vs_v97.py                  # v95 vs v97 比較グラフ
├── v96_monthly_report.py          # 月次レポート生成
├── plot_equity_and_drawdown_v97.py # v97 エクイティ & ドローダウン描画
├── plot_equity_curve.py           # エクイティカーブ描画
├── nikkei225.csv                  # 日経225銘柄リスト（222銘柄）
├── stok.md                        # v85 運用ルール
├── requirements.txt               # Python依存パッケージ
├── .github/workflows/
│   └── daily-signal.yml           # GitHub Actions（毎朝自動実行 + Discord通知）
└── .gitignore
```

---

## セットアップ

### 必要環境

- Python 3.11+

### インストール

```bash
git clone https://github.com/pengin01/stock-trading-system.git
cd stock-trading-system

pip install -r requirements.txt
```

### 依存パッケージ

| パッケージ | 用途 |
|-----------|------|
| `pandas` | データ処理 |
| `yfinance` | Yahoo Finance から株価データ取得 |
| `numpy` | 数値計算 |
| `ta` | テクニカル指標（RSI等）の計算 |
| `matplotlib` | グラフ描画（分析・可視化スクリプトで使用） |
| `requests` | Discord Webhook 通知（GitHub Actionsで使用） |

---

## 使い方

### 日次シグナル（本番運用）

メインの運用スクリプトです。日経225全銘柄をスクリーニングし、エントリー/イグジットシグナルを生成します。

```bash
python v97_system.py
```

**処理フロー：**

1. `nikkei225.csv` から銘柄ユニバースを読み込み
2. Yahoo Finance から直近1年分の株価データを取得
3. 保有中のポジションのイグジット判定（4日経過で決済）
4. 新規エントリー候補をスクリーニング（RSIが低い順にランキング）
5. エントリー/イグジット/ポジション/エクイティを CSV に保存

**出力ファイル：**

| ファイル | 内容 |
|---------|------|
| `today_entry.csv` | 当日のエントリーシグナル（ticker, signal_date, entry_price, qty, rsi, score） |
| `today_exit.csv` | 当日のイグジットシグナル（ticker, reason） |
| `positions.csv` | 現在の保有ポジション（ticker, entry_date, entry_price, qty, exit_date） |
| `equity.csv` | エクイティ推移（date, equity, cash, position_value） |
| `cashflow.csv` | 入出金履歴（date, amount, note） |
| `candidate_rank.csv` | エントリー候補ランキング（date, rank, ticker, close, prev_close, ma, rsi, value20, pullback_ratio, score） |

### ペーパートレード

少数銘柄（10銘柄）を対象にしたペーパートレード用スクリプトです。  
エントリー診断（各条件の合否）も出力します。

```bash
# 今日のシグナル確認 + ポジション更新
python paper_stock_today_signal.py

# バックテスト
python paper_stock_backtest.py
```

**ペーパートレード対象銘柄：**

| ティッカー | 銘柄名 |
|-----------|--------|
| 9432.T | NTT |
| 6758.T | ソニー |
| 9984.T | ソフトバンクグループ |
| 7203.T | トヨタ |
| 8306.T | 三菱UFJ |
| 8035.T | 東京エレクトロン |
| 6501.T | 日立 |
| 6861.T | キーエンス |
| 4063.T | 信越化学 |
| 7267.T | ホンダ |

**ペーパートレード出力ファイル：**

| ファイル | 内容 |
|---------|------|
| `today_entry.csv` | エントリーシグナル |
| `today_exit.csv` | イグジットシグナル |
| `entry_diagnostics.csv` | 各銘柄のエントリー条件診断（条件ごとの合否） |
| `daily_result_log.csv` | 日次結果ログ（損益、エクイティ等） |
| `paper_positions.csv` | 保有ポジション |

### バックテスト

#### v97 日経225バックテスト（パラメータグリッド）

初期資金とpullback値の組み合わせを総当たりでバックテストします。

```bash
python v97_backtest_nikkei225.py
```

**パラメータグリッド：**

| パラメータ | テスト値 |
|-----------|---------|
| `initial_capital` | 20,000 / 30,000 / 50,000 / 80,000 / 100,000 |
| `hold_days` | 4 |
| `pullback_pct` | 0.002 / 0.004 / 0.007 |

**出力：** `v97_backtest_result.csv`、各条件のエクイティカーブCSV

#### v97 リスク管理バックテスト（3銘柄）

3銘柄（9432.T, 6758.T, 9984.T）を対象に、10年分のデータでバックテストします。

```bash
python v97_risk_control_with_equity.py
```

**出力：** `v97_result.csv`、各条件のエクイティカーブCSV

#### v95 バックテスト

v97のベースとなったバージョンです。資金を100%使用する点がv97と異なります。

```bash
python v95_with_equity_output.py
```

**出力：** `v95_result.csv`、`v95_equity_curve.csv`

#### v92 パラメータ実験

```bash
# 標準実験（hold_days x pullback_pct のグリッドサーチ）
python v92_experiment.py

# 少額資金での実験（initial_capital x pullback_pct）
python v92_experiment_small_capital.py
```

#### v85 バックテスト

松井証券の手数料体系とスリッページ（片道0.1%）を考慮したバックテストです。

```bash
python v85_backtest.py
```

### 分析・可視化

#### バックテスト結果の分析

```bash
python analyze_v97_backtest_result.py
```

`v97_backtest_result.csv` を読み込み、以下を出力します：
- ピボットテーブル（リターン、ドローダウン、勝率、取引回数）
- グラフ（初期資金別のリターン・ドローダウン・勝率）
- 最適条件の特定

#### エクイティカーブの比較

```bash
python compare_equity_curves.py
```

複数条件のエクイティカーブを重ねて描画し、ドローダウンも比較します。

#### v95 vs v97 比較

```bash
python v95_vs_v97.py
```

同一条件（capital=20000, pullback=0.004）で v95（資金100%）と v97（資金70%）の正規化エクイティカーブを比較します。

#### エクイティ & ドローダウン描画

```bash
python plot_equity_and_drawdown_v97.py
```

#### 月次レポート

```bash
python v96_monthly_report.py
```

月ごとのリターンとドローダウンをテーブル+棒グラフで出力します。

---

## GitHub Actions 自動運用

`.github/workflows/daily-signal.yml` により、毎朝自動でシグナルが生成されます。

| 項目 | 内容 |
|------|------|
| **スケジュール** | 毎週月〜金 JST 07:17（UTC 22:17） |
| **実行内容** | `v97_system.py` を実行 |
| **通知** | Discord Webhookで結果を送信 |
| **手動実行** | `workflow_dispatch` で手動トリガーも可能 |
| **成果物** | CSVファイル一式を Artifacts としてアップロード |

### 必要なSecrets

| Secret名 | 説明 |
|----------|------|
| `DISCORD_WEBHOOK_URL` | Discord Webhook URL（リポジトリのSettings > Secrets and variablesで設定） |

---

## パラメータ詳細

### v97（現行メインバージョン）

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `INITIAL_CAPITAL` | 20,000 | 初期資金（円） |
| `HOLD_DAYS` | 4 | 保有日数（固定） |
| `PULLBACK` | 0.004 | 押し目の閾値（前日比0.4%下落） |
| `RISK_RATIO` | 0.7 | 資金使用率（70%） |
| `MAX_POSITIONS` | 1 | 最大同時保有銘柄数 |
| `MA_DAYS` | 25 | 移動平均の期間（日） |
| `RSI_DAYS` | 14 | RSIの計算期間（日） |
| `RSI_MAX` | 65 | RSI上限（これ以上は過熱と判断し除外） |
| `MIN_VALUE` | 300,000,000 | 20日平均売買代金の下限（3億円） |

### v92

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `initial_capital` | 80,000 | 初期資金（円） |
| `risk_per_trade` | 0.02 | 1トレードあたりのリスク（2%ルール） |
| `max_positions` | 3 | 最大同時保有銘柄数 |
| `lot_size` | 100 | 売買単位（100株） |
| `pullback_pct` | 0.005 | 押し目の閾値 |
| `hold_days` | 4 | 保有日数 |

v92では、イグジット条件に **MA割れ**（`MA_EXIT`）が追加されています。  
株価がMAを下回った場合、保有日数に関係なく決済します。

### v85

松井証券の手数料体系（50万以下: 0円、100万以下: 1,100円、100万超: 2,200円）と  
片道0.1%のスリッページを考慮したバックテストを行います。

---

## データファイル

### nikkei225.csv

日経225構成銘柄のティッカーシンボル一覧です（222銘柄）。  
`v97_system.py`、`daily_signal.py`、`v97_backtest_nikkei225.py` で銘柄ユニバースとして使用されます。

### cashflow.csv

入出金を記録するファイルです。初回実行時に自動生成されます。

| カラム | 説明 |
|--------|------|
| `date` | 日付 |
| `amount` | 金額（正: 入金、負: 出金） |
| `note` | メモ |

### stok.md

v85戦略の運用ルールドキュメントです。  
ターゲット銘柄、資金配分、日次運用フロー、禁止事項などが記載されています。

---

## 戦略バージョン履歴

| バージョン | 主な特徴 |
|-----------|---------|
| **v85** | 基本戦略。手数料・スリッページ考慮。翌日寄りエントリー。6銘柄固定 |
| **v92** | 統合システム化。2%リスクルール。MA割れイグジット追加。最大3ポジション |
| **v95** | パラメータグリッド対応。エクイティカーブ出力。資金100%使用 |
| **v96** | 月次レポート機能追加 |
| **v97** | **現行バージョン**。資金70%運用。日経225全銘柄対応。Discord通知。GitHub Actions自動運用 |

---

## Discord 通知の見方

GitHub Actionsから毎朝送信される通知の構造：

```
Daily Signal v97

RUN INFO
- run_date / signal_date

ENTRY          ... エントリーシグナル（あれば成行/指値で購入）
EXIT           ... イグジットシグナル（該当銘柄を全株売却）
POSITIONS      ... 現在の保有銘柄
CANDIDATES     ... エントリー候補上位3銘柄
EQUITY         ... 現在の資産額と前日比
WEEK           ... 週間パフォーマンス
MONTH          ... 月間パフォーマンス
TOTAL          ... 累計パフォーマンス
CASHFLOW       ... 累計入出金
PNL            ... 損益（対入出金）
DRAWDOWN ALERT ... ドローダウン10%超で警告
NEW HIGH       ... 過去最高益更新時
```

### シグナルに従った操作

| シグナル | 操作 |
|---------|------|
| ENTRY | 指定銘柄を成行 or 指値で購入。数量（qty）をそのまま使用。当日中に約定させる |
| EXIT | 該当銘柄を全株売却。理由は基本 `time_exit`（4日経過） |

---

## バックテスト結果

### v97 推奨設定: pullback = 0.004

| 指標 | 結果 |
|------|------|
| 総リターン | 約 +290% 〜 +310% |
| 最大ドローダウン | 約 -22% |
| 勝率 | 約 54% |
| 平均リターン | 約 +0.4% / トレード |
| 取引回数 | 約 570回 |

### pullback値の比較

| pullback | 特徴 | 評価 |
|----------|------|------|
| 0.002 | 頻度多、リターン控えめ | 安定 |
| **0.004** | **バランスが良い** | **最適（推奨）** |
| 0.007 | 頻度少、リターン高いがDD大 | 攻撃的 |
