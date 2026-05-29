# Excursion Bands Research System Methodology

## Executive Summary

This project is a systematic research framework for testing intraday NQ breakout strategies using a custom feature family called **excursion bands**. The system combines session-aware data processing, volatility-normalized feature engineering, rule-based breakout strategies, machine-learning meta-labeling, and walk-forward optimization.

The core research hypothesis is that intraday price movement is not random around the session open. On directional days, price often first establishes an **adverse excursion** against the eventual trend before expanding into a **favorable excursion** in the trend direction. Excursion bands attempt to estimate these statistically recurring adverse and favorable price zones from prior sessions, then use them as contextual filters for breakout strategies.

For a long trend-following setup, the working hypothesis is:

Price moving above the positive adverse-excursion zone indicates that the market has invalidated the expected pullback area and may be entering an expansion phase. If price is still below the favorable-excursion objective zone, there may be enough remaining directional range to justify a long breakout trade.

For a short trend-following setup, the analogous hypothesis is:

Price moving below the negative adverse-excursion zone indicates downside expansion beyond the expected adverse region, while remaining above the negative favorable-excursion objective zone leaves enough downside continuation potential.

This produces the currently implemented band filters:

```text
Long filter:  Band_AE_Pos_Upper < Close < Band_FE_Pos_Lower
Short filter: Band_FE_Neg_Upper < Close < Band_AE_Neg_Lower
```

The report intentionally does not include performance results. Results should be attached after full walk-forward validation is complete.

## Project Objectives

The project is designed around four practical research objectives:

1. Build a reusable backtesting framework for intraday breakout research.
2. Derive a volatility-normalized excursion-band feature set from historical session behavior.
3. Test whether excursion bands improve trend-following breakout selection.
4. Evaluate whether ML meta-labeling can add or reject trades without introducing lookahead bias.

The system is intentionally modular. Strategies, feature construction, execution assumptions, sizing, ML filtering, and walk-forward optimization are separated so that each research component can be tested independently.

## System Architecture

The codebase is organized around a feature pipeline and a backtesting pipeline.

### Feature Pipeline

The feature pipeline lives primarily under:

```text
src/excursion_bands/features/
src/excursion_bands/pipeline/
configs/features/
configs/data/
configs/sessions/
```

Its main responsibilities are:

1. Load raw intraday OHLCV data.
2. Convert raw bars into processed session-aware data.
3. Aggregate intraday data into configured session buckets.
4. Estimate volatility using Yang-Zhang-style features.
5. Construct excursion-band levels from historical session behavior.

The intended processing order is:

```text
load_raw_data
load_processed_data
process_raw_data
load_aggregated_data
calculate volatility
calculate excursion bands
```

The session pipeline assumes four intraday buckets and drops days that do not contain all required buckets. This is important because the excursion-band calculations depend on consistent daily session structure.

### Backtesting Pipeline

The backtesting pipeline lives primarily under:

```text
src/excursion_bands/backtesting/
configs/strategies/
notebooks/strategy_orb_backtesting.py
notebooks/strategy_donchian_backtesting.py
```

Its main components are:

```text
models.py          Trade, signal, position, and result models
engine.py          Single-position event-driven bar engine
costs.py           Slippage and commission handling
sizing.py          Fixed and risk-based position sizing
metrics.py         Return, drawdown, Sharpe, and trade metrics
reports.py         CSV, YAML, Markdown, and chart outputs
benchmarks.py      Benchmark comparison utilities
wfo.py             Walk-forward optimization and stitched OOS equity
ml.py              ML candidate labeling and meta-label filtering
strategies/orb.py  Opening range breakout logic
strategies/donchian.py Donchian breakout logic
```

The backtest engine is custom rather than delegated to an external backtesting library. This gives direct control over next-bar execution, same-bar stop/take-profit priority, break-even logic, commission conversion, slippage, trade source attribution, ML probabilities, and WFO stitching.

## Data Model And Session Structure

The system uses intraday OHLCV bars with at least these fields:

```text
DateTime
Open
High
Low
Close
Volume
Session
```

The aggregated feature data uses session-bucket columns such as:

```text
O_pre_target_1
H_pre_target_1
L_pre_target_1
C_pre_target_2
H_target_1
L_target_1
H_target_2
L_target_2
C_target_2
O_ref
```

`O_ref` is the reference price used to anchor excursion calculations. Conceptually, it is the session reference open from which adverse and favorable excursions are measured.

The feature pipeline uses both pre-target and target session buckets. This lets the system estimate how far price historically moves away from a reference open across a structured intraday window.

## Volatility Estimation

The system uses a Yang-Zhang-style volatility estimator. This is useful because it incorporates overnight movement, open-to-close movement, and intraday range information.

For each session, the implementation computes:

```text
log_overnight = log(Open_t / PrevClose_{t-1})
log_oc        = log(Close_t / Open_t)
```

It also computes a Rogers-Satchell range component:

```text
RS_t = log(High_t / Close_t) * log(High_t / Open_t)
     + log(Low_t / Close_t)  * log(Low_t / Open_t)
```

For historical volatility over a rolling window `n`, the estimator is:

```text
k = 0.34 / (1.34 + (n + 1) / (n - 1))

Sigma_historical = sqrt(
    rolling_var(log_overnight, n)
  + k * rolling_var(log_oc, n)
  + (1 - k) * rolling_mean(RS, n)
)
```

The default volatility lookback is:

```yaml
lookback_window: 10
```

`Sigma_historical` is central to excursion-band construction because adverse and favorable excursions are normalized by volatility before their rolling expectations are estimated.

Important implementation note: `Sigma_historical` is generally calculated using current-day realized information in the feature pipeline. For any prediction-time feature, it must be shifted or otherwise restricted so today's future information is not used. The excursion-band construction handles its expected excursion centers with lagged rolling calculations, and the project treats lookahead control as a first-class research constraint.

## Excursion Bands: Conceptual Foundation

Excursion bands are based on an empirical observation about candle and session formation.

On a bullish directional day, price often does not move in a straight line from open to high. It may first probe lower, absorb liquidity, or establish a local low. This downside move is adverse to the eventual bullish direction. After that adverse excursion is complete, price may rotate upward and distribute range toward the session high. The upward movement is favorable to the bullish direction.

On a bearish directional day, the structure is mirrored. Price may first probe higher, establishing an adverse upside excursion, before rotating lower and distributing range toward the session low.

This gives two path-dependent concepts:

```text
Adverse excursion:  distance from reference open against the final directional move
Favorable excursion: distance from reference open in the final directional move
```

The purpose of the bands is not to forecast the exact high or low. The purpose is to estimate statistically relevant zones where price behavior transitions from normal adverse movement into directional expansion.

## Direction Classification

Each historical session is classified as bullish, bearish, or neutral using a volatility-normalized body size.

The normalized body magnitude is:

```text
z_body_t = abs(log(C_target_2,t / O_ref,t)) / Sigma_historical,t
```

This measures the size of the session body relative to the current volatility estimate. A large body in low volatility is more meaningful than the same absolute move in high volatility.

The system also computes a volatility-regime score:

```text
z_sigma_t = Sigma_historical,t / rolling_mean(Sigma_historical, n)_{t-1}
```

The adaptive body threshold is:

```text
tau_t = clip(tau_0 * z_sigma_t^(-0.5), tau_min, tau_max)
```

Default values are:

```yaml
lookback_window: 10
threshold:
  tau_0: 0.4
  tau_min: 0.26
  tau_max: 1.75
  k: 0.1
```

Direction is assigned as:

```text
if z_body_t > tau_t and C_target_2,t > O_ref,t:
    direction_t = bullish

if z_body_t > tau_t and C_target_2,t < O_ref,t:
    direction_t = bearish

otherwise:
    direction_t = neutral
```

The adaptive threshold is intended to avoid over-classifying noise as direction. In elevated volatility regimes, the threshold is adjusted so that classification remains regime-aware rather than fixed in raw price terms.

## Day Boundary Extraction

For each aggregated session row, the system computes the full-session low and high across the configured buckets:

```text
L_day,t = min(
    L_pre_target_1,t,
    L_pre_target_2,t,
    L_target_1,t,
    L_target_2,t
)

H_day,t = max(
    H_pre_target_1,t,
    H_pre_target_2,t,
    H_target_1,t,
    H_target_2,t
)
```

These are the realized intraday extremes used to measure how far the session traveled away from `O_ref`.

## Adverse And Favorable Excursion Measurement

For bullish sessions:

```text
epsilon_AE,t = O_ref,t - L_day,t
epsilon_FE,t = H_day,t - O_ref,t
```

For bearish sessions:

```text
epsilon_AE,t = H_day,t - O_ref,t
epsilon_FE,t = O_ref,t - L_day,t
```

For neutral sessions, both adverse and favorable excursion are assigned the larger side of the realized range:

```text
epsilon_AE,t = max(H_day,t - O_ref,t, O_ref,t - L_day,t)
epsilon_FE,t = max(H_day,t - O_ref,t, O_ref,t - L_day,t)
```

The neutral treatment is conservative. Instead of forcing a directional interpretation on a non-directional day, the system records a broad reference excursion magnitude.

## Volatility Normalization

Raw excursion distances are not directly comparable across volatility regimes. A 40-point move in NQ has different meaning in a low-volatility regime than in a high-volatility regime.

The system normalizes adverse and favorable excursion distances by historical volatility:

```text
epsilon_AE_norm,t = epsilon_AE,t / Sigma_historical,t
epsilon_FE_norm,t = epsilon_FE,t / Sigma_historical,t
```

The rolling expected normalized excursions are then calculated using only prior sessions:

```text
mu_AE,t = rolling_mean(shift(epsilon_AE_norm, 1), n)
mu_FE,t = rolling_mean(shift(epsilon_FE_norm, 1), n)
```

The `shift(1)` is critical. It prevents the current session's realized excursion from influencing the band levels for that same session.

## Scaling Back Into Price Space

After estimating expected excursion centers in normalized space, the system scales them back into price units:

```text
mu_AE_scaled,t = mu_AE,t * Sigma_historical,t-1
mu_FE_scaled,t = mu_FE,t * Sigma_historical,t-1
```

The implementation uses lagged `Sigma_historical` for the scaling step. This preserves the principle that today's band levels should be derived from information available before the current session.

## Band Center Construction

Excursion-band centers are constructed symmetrically around `O_ref`.

Positive-side adverse and favorable centers:

```text
Band_AE_Pos_Center,t = O_ref,t + mu_AE_scaled,t
Band_FE_Pos_Center,t = O_ref,t + mu_FE_scaled,t
```

Negative-side adverse and favorable centers:

```text
Band_AE_Neg_Center,t = O_ref,t - mu_AE_scaled,t
Band_FE_Neg_Center,t = O_ref,t - mu_FE_scaled,t
```

This symmetry allows the same excursion statistics to define both upside and downside contextual levels.

The terms mean:

```text
AE_Pos: upside adverse excursion zone for a bearish-style path, reused as an upside transition level
FE_Pos: upside favorable excursion zone for a bullish-style path
AE_Neg: downside adverse excursion zone for a bullish-style path, reused as a downside transition level
FE_Neg: downside favorable excursion zone for a bearish-style path
```

For the current long trend-following filter, the key upside corridor is between `AE_Pos` and `FE_Pos`. For the current short trend-following filter, the key downside corridor is between `AE_Neg` and `FE_Neg`.

## Band Width Construction

The system creates a half-width around each band center:

```text
delta_t = k * Sigma_historical,t-1 * O_ref,t
```

Default:

```yaml
k: 0.1
```

Each center becomes a zone with an upper and lower boundary:

```text
Band_X_Upper,t = Band_X_Center,t + delta_t
Band_X_Lower,t = Band_X_Center,t - delta_t
```

The purpose of `delta_t` is to avoid treating a single price level as exact. Markets rarely respect precise points. A band is more realistic as a zone around a statistically estimated excursion center.

## Trend-Following Filter Hypothesis

The current band filter is designed for breakout continuation systems, not mean-reversion systems.

For long trades, the rule is:

```text
Band_AE_Pos_Upper < Close < Band_FE_Pos_Lower
```

This expresses three ideas:

1. Price has moved beyond the positive adverse-excursion zone.
2. Price may therefore be transitioning from auction/noise into upside expansion.
3. Price has not yet reached the lower boundary of the positive favorable-excursion zone, so the trade still has projected room before the expected favorable excursion area.

The professional interpretation is that `AE_Pos` acts as an expansion confirmation threshold, while `FE_Pos` acts as a forward range objective or exhaustion-proximity threshold. A long signal below `AE_Pos_Upper` may be too early because price has not yet escaped the historically normal adverse/noise region. A long signal above `FE_Pos_Lower` may be too late because price has already entered the statistically expected favorable destination zone.

For short trades, the rule is:

```text
Band_FE_Neg_Upper < Close < Band_AE_Neg_Lower
```

This is the mirrored downside corridor. Price must be below the negative adverse threshold but still above the negative favorable objective zone.

The practical interpretation is:

```text
Long:  trade the expansion corridor after upside AE disrespect but before FE exhaustion.
Short: trade the expansion corridor after downside AE disrespect but before FE exhaustion.
```

## Why This Filter Is Applied To Breakout Strategies

ORB and Donchian systems are both trend-following breakout strategies. Their main weakness is that many breakouts occur in the wrong part of the intraday range distribution.

A raw breakout can fail because:

1. It triggers inside a normal adverse/noise zone.
2. It triggers after too much favorable movement has already occurred.
3. It occurs in a volatility regime where the breakout distance is not meaningful.
4. It occurs against the session's path structure.

Excursion bands attempt to address these weaknesses by adding a session-path context layer. The breakout is only accepted when price is in a corridor that is hypothesized to represent directional expansion with remaining expected range.

This makes the band filter different from a generic trend filter. It is not simply asking whether price is above a moving average. It asks whether price is located between historically derived adverse and favorable excursion zones for the current session.

## Opening Range Breakout Strategy

The ORB strategy defines an opening range over a configured time window. The current default research setup uses a short opening range such as `09:30` to `09:35`.

The strategy computes:

```text
OR_High = max(High during opening range)
OR_Low  = min(Low during opening range)
OR_Mid  = (OR_High + OR_Low) / 2
```

Long breakout condition:

```text
Close > OR_High + optional_ATR_buffer
```

Short breakout condition:

```text
Close < OR_Low - optional_ATR_buffer
```

Optional filters include VWAP, ATR buffer, side mode, and excursion bands.

The ORB strategy is constrained to one trade per session. This reduces overtrading and makes the trade sample easier to interpret as a session-level breakout model.

## Donchian Breakout Strategy

The Donchian strategy uses a rolling high/low channel. The channel is shifted by one bar to avoid lookahead bias.

```text
Donchian_High_t = rolling_max(High, lookback_bars)_{t-1}
Donchian_Low_t  = rolling_min(Low, lookback_bars)_{t-1}
Donchian_Mid_t  = (Donchian_High_t + Donchian_Low_t) / 2
```

Long breakout condition:

```text
Close_t > Donchian_High_t + optional_ATR_buffer
```

Short breakout condition:

```text
Close_t < Donchian_Low_t - optional_ATR_buffer
```

The current Donchian research focus is long-only with the excursion-band filter and optional additive ML. Like ORB, it is constrained to one trade per session.

## Execution Methodology

Signals are generated on a completed bar. Entries are executed on the next bar open.

This prevents same-bar close-to-close execution assumptions and better reflects a tradable workflow:

```text
signal at bar t close
entry at bar t+1 open
```

Stops and take profits are evaluated on subsequent bar high/low data. If a stop and take profit are both touched in the same bar, the system uses conservative priority:

```text
same-bar stop/take-profit priority = stop first
```

Positions can also be closed by a forced session exit time.

## Risk, Stops, And Profit Targets

The strategies support ATR-based stops and boundary-based stops.

For ORB:

```text
boundary stop for long  = OR_Low
boundary stop for short = OR_High
```

For Donchian:

```text
boundary stop for long  = Donchian_Low
boundary stop for short = Donchian_High
```

ATR stop mode uses:

```text
ATR_Stop = RMA(TrueRange, length) * multiplier
```

For a long trade:

```text
stop_loss = bar.Low - ATR_Stop
```

For a short trade:

```text
stop_loss = bar.High + ATR_Stop
```

Take profit is expressed as a risk multiple:

```text
risk = abs(entry_reference - stop_loss)

long_take_profit  = entry_reference + rr * risk
short_take_profit = entry_reference - rr * risk
```

Break-even movement is supported but not always enabled. When enabled, the stop can move to entry plus or minus an offset after price reaches a configured R multiple.

## Cost And Sizing Model

The system models NQ-like price movement but uses CFD-style sizing for research granularity.

Important assumptions:

```text
point_value: 20.0
cfd_point_value: 1.0
tick_size: 0.25
slippage_ticks_per_side: 3.0
commission_per_contract_side: 4.5
```

The commission field is interpreted as NQ mini commission per side and converted into CFD unit cost internally.

The standard comparable research baseline is:

```yaml
sizing:
  mode: risk_current_equity
  risk_pct: 0.01
```

This means model comparisons should be made at the same risk percentage. Higher risk can produce much larger compounded returns but does not necessarily imply a better signal model.

## ML Meta-Labeling Methodology

The ML layer is not designed as an independent alpha generator. It is a meta-labeling layer placed on top of strategy candidates.

The workflow is:

```text
strategy candidate generation
candidate outcome simulation
feature extraction
label assignment
model training
probability prediction
trade filter or additive inclusion
```

The label is:

```text
label = 1 if net_r > 0 else 0
```

`net_r` is used instead of raw PnL because it is more stable across risk sizing and equity compounding regimes.

The model is an XGBoost classifier. Features include:

```text
side indicators
entry timing
opening range or channel width
breakout distance
VWAP relationship
ATR/session volatility
bar range and body structure
recent returns and volatility
day/month calendar features
distance to active AE and FE bands
price location inside the active band corridor
interactions between breakout, VWAP, ATR, and band position
```

The ML layer supports two modes:

```text
filter:   only accept strategy candidates whose ML probability exceeds threshold
additive: keep base band-filtered trades and add high-confidence raw candidates
```

The additive mode is important because it separates two hypotheses:

1. Excursion bands provide a robust structural base filter.
2. ML may recover additional valid breakouts that occur outside the strict band corridor.

In additive mode, the current preferred candidate scope is raw strategy candidates:

```yaml
ml_execution_mode: additive
ml_candidate_scope: raw
```

This allows the model to learn from the broader breakout opportunity set while preserving the band-filtered base strategy.

## Lookahead Control In ML

The ML implementation uses prior candidates only. In expanding mode, the model trains on previous candidates before predicting the current candidate.

The warmup period allows the model to learn before the evaluated backtest period begins:

```yaml
ml:
  warmup_start_date: 2014-01-01
```

Warmup candidates can be used for training, but trades are only allowed from `backtest.start_date` onward.

In WFO mode, each fold trains only on the fold's training window and predicts the fold's out-of-sample test window.

## Walk-Forward Optimization Methodology

Walk-forward optimization is used to reduce the risk of selecting parameters that only work in a full-sample backtest.

The WFO process is:

```text
split history into sequential train/test folds
optimize parameter grid on train fold
select best train parameters by objective
apply selected parameters to next OOS test fold
stitch OOS fold equity curves by variant
report OOS metrics and fold diagnostics
```

The WFO implementation supports both calendar-based and session-based folds. Calendar WFO is anchored to the configured backtest start date.

The key principle is that every test fold represents data not used to select the fold's parameters.

For ML-enabled WFO, the model must also be trained only on data available before the OOS fold. The implementation handles this by building train and test candidate datasets separately per fold.

## Current Research Hypotheses

The project is currently organized around these hypotheses:

1. Excursion bands identify statistically meaningful intraday path zones derived from prior session behavior.
2. Breakouts that occur after adverse-zone disrespect but before favorable-zone exhaustion have better continuation characteristics than raw breakouts.
3. The band filter should reduce low-quality trades that trigger inside noise or after range exhaustion.
4. Donchian and ORB strategies are suitable base models because their entries are simple and interpretable, making the effect of the band filter easier to isolate.
5. Additive ML can improve coverage by identifying high-confidence raw breakout candidates missed by strict band rules.
6. Walk-forward validation is required because both the band parameters and ML threshold can overfit if judged only on full-sample results.
7. Drawdown should be controlled through robust parameter selection, stop/target design, ML thresholding, and WFO objectives rather than by increasing or decreasing sizing during model comparison.

## Validation Plan

The intended validation sequence is:

1. Compare raw breakout vs band-filtered breakout.
2. Compare band-filtered breakout vs band-plus-additive-ML breakout.
3. Compare normal expanding backtest behavior against WFO OOS behavior.
4. Inspect fold-level performance to identify unstable regimes.
5. Inspect ML diagnostics to verify that additive trades are genuinely being selected OOS.
6. Tune WFO objective functions to penalize drawdown and unstable trade selection.
7. Preserve consistent risk sizing across comparisons.

The most important next evidence should come from full Donchian WFO outputs:

```text
wfo_oos_metrics.csv
ml_wfo_diagnostics.csv
wfo_test_folds.csv
```

These artifacts will allow analysis of whether the excursion-band filter and additive ML layer are improving true out-of-sample performance or simply increasing exposure.

## Interpretation Guidelines

When reading future results, the key question is not only whether total return increases. The stronger question is whether the method improves the quality of the return stream.

Important diagnostics include:

```text
OOS total return
OOS max drawdown
Sharpe ratio
profit factor
trade count
win rate
average R
worst fold behavior
source attribution: band vs ml_additive
ML allowed/rejected candidate counts
probability distribution of accepted trades
feature importance stability
```

An acceptable research result should show that the band filter and ML layer improve risk-adjusted behavior, not merely that higher trade count or higher risk generated more profit.

## Summary

The project is a structured attempt to turn an observed intraday market behavior into a testable quantitative feature.

The central idea is that directional sessions often contain both an adverse excursion and a favorable excursion relative to a reference open. By estimating these distances in volatility-normalized space and converting them into tradable price zones, the system creates a contextual filter for breakout strategies.

The current long filter accepts breakouts only after price has moved beyond the positive adverse-excursion zone and before it reaches the positive favorable-excursion zone. This encodes the hypothesis that the best continuation trades occur in the expansion corridor, not inside early-session noise and not after the expected favorable move is already mature.

The broader framework then tests this idea through controlled execution assumptions, risk-based sizing, ML meta-labeling, and walk-forward optimization.
