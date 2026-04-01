from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    max_position_size: float = 0.20
    fee_rate: float = 0.004
    slippage_rate: float = 0.001
    entry_threshold: float = 0.003
    exit_threshold: float = 0.000
    min_direction_prob: float = 0.55
    exit_direction_prob: float = 0.50
    min_signal_strength: float = 0.00015
    min_hold_days: int = 3
    cooldown_days: int = 2
    confirmation_period: int = 2
    volatility_threshold: Optional[float] = 0.035
    sideways_entry_multiplier: float = 1.15
    bearish_entry_multiplier: float = 1.35
    sideways_position_scale: float = 0.50
    bearish_min_direction_prob: float = 0.65
    bearish_min_signal_strength: float = 0.00100
    high_volatility_blocks_entry: bool = True
    max_drawdown_threshold: Optional[float] = 0.10
    drawdown_scaling: float = 0.50
    pause_trading_on_drawdown: bool = False
    stop_loss_pct: Optional[float] = 0.05
    daily_loss_cap_pct: Optional[float] = 0.03


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    holding_days: int
    position_weight: float
    gross_return_pct: float
    net_pnl: float
    signal_strength: float
    regime: str
    volatility: Optional[float]
    entry_reason: str
    exit_reason: str


@dataclass
class BacktestResult:
    summary: Dict[str, float]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _clip_weight(weight: float, max_position_size: float) -> float:
    return float(np.clip(weight, 0.0, float(max_position_size)))


def _signal_strength(pred_ret: float, dir_prob: float) -> float:
    return float(pred_ret) * (float(dir_prob) - 0.5)


def _confidence_scalar(dir_prob: float) -> float:
    return float(np.clip((float(dir_prob) - 0.5) * 2.0, 0.0, 1.0))


def _position_size(
    dir_prob: float,
    volatility_value: float,
    cfg: BacktestConfig,
    size_multiplier: float = 1.0,
) -> float:
    confidence = _confidence_scalar(dir_prob)
    normalized_volatility = 0.0
    if np.isfinite(volatility_value):
        normalized_volatility = max(float(volatility_value), 0.0)

    raw_size = float(cfg.max_position_size) * confidence / max(1.0 + normalized_volatility, 1e-9)
    raw_size *= max(float(size_multiplier), 0.0)
    return _clip_weight(raw_size, cfg.max_position_size)


def _infer_volatility(row: pd.Series) -> float:
    for col in ("atr_pct_14", "vol_20d", "volatility_20", "atr_pct_7"):
        value = _safe_float(row.get(col), np.nan)
        if np.isfinite(value) and value >= 0.0:
            return float(value)
    regime_vol_pct = _safe_float(row.get("regime_volatility_pct"), np.nan)
    if np.isfinite(regime_vol_pct) and regime_vol_pct >= 0.0:
        return float(regime_vol_pct) / 100.0
    return np.nan


def _regime_entry_multiplier(regime: Any, cfg: BacktestConfig) -> float:
    label = str(regime or "").lower()
    if "bear" in label:
        return float(cfg.bearish_entry_multiplier)
    if "sideways" in label or "neutral" in label or "accumulation" in label:
        return float(cfg.sideways_entry_multiplier)
    return 1.0


def _is_high_volatility_regime(regime: Any) -> bool:
    label = str(regime or "").lower()
    high_vol_tokens = ("high_vol", "high-vol", "high volatility", "volatile", "panic")
    return any(token in label for token in high_vol_tokens)


def _regime_behavior(regime: Any, volatility_value: float, cfg: BacktestConfig) -> Dict[str, Any]:
    label = str(regime or "").lower()
    behavior = {
        "label": label,
        "behavior": "trending",
        "entry_blocked": False,
        "size_multiplier": 1.0,
        "min_direction_prob": float(cfg.min_direction_prob),
        "min_signal_strength": float(cfg.min_signal_strength),
    }

    if _regime_blocks_entry(regime):
        behavior["behavior"] = "blocked"
        behavior["entry_blocked"] = True
        return behavior

    volatility_breach = (
        bool(cfg.high_volatility_blocks_entry)
        and cfg.volatility_threshold is not None
        and np.isfinite(volatility_value)
        and float(volatility_value) > float(cfg.volatility_threshold)
    )
    if _is_high_volatility_regime(regime) or volatility_breach:
        behavior["behavior"] = "high_volatility"
        behavior["entry_blocked"] = True
        return behavior

    if any(token in label for token in ("sideways", "neutral", "accumulation", "range")):
        behavior["behavior"] = "sideways"
        behavior["size_multiplier"] = float(np.clip(cfg.sideways_position_scale, 0.0, 1.0))
        return behavior

    if any(token in label for token in ("bear", "distribution", "downtrend")):
        behavior["behavior"] = "bearish"
        behavior["min_direction_prob"] = max(float(cfg.min_direction_prob), float(cfg.bearish_min_direction_prob))
        behavior["min_signal_strength"] = max(float(cfg.min_signal_strength), float(cfg.bearish_min_signal_strength))
        return behavior

    return behavior


def _drawdown_scale(current_drawdown: float, cfg: BacktestConfig) -> Dict[str, Any]:
    if cfg.max_drawdown_threshold is None:
        return {"active": False, "pause": False, "scale": 1.0}

    threshold = -abs(float(cfg.max_drawdown_threshold))
    if float(current_drawdown) > threshold:
        return {"active": False, "pause": False, "scale": 1.0}

    scale = float(np.clip(cfg.drawdown_scaling, 0.0, 1.0))
    pause = bool(cfg.pause_trading_on_drawdown) or scale <= 0.0
    return {"active": True, "pause": pause, "scale": 0.0 if pause else scale}


def _regime_blocks_entry(regime: Any) -> bool:
    label = str(regime or "").lower()
    blocked_tokens = ("manipulation", "pump", "dump", "parabolic", "overextended")
    return any(token in label for token in blocked_tokens)


def _close_trade(
    trades: List[Dict[str, Any]],
    current_trade: Optional[Dict[str, Any]],
    exit_date: str,
    exit_price: float,
    net_pnl: float,
    exit_reason: str,
) -> Optional[Dict[str, Any]]:
    if current_trade is None:
        return None

    avg_weight = float(current_trade["weight_sum"]) / max(int(current_trade["holding_days"]), 1)
    trade = TradeRecord(
        entry_date=str(current_trade["entry_date"]),
        exit_date=str(exit_date),
        entry_price=round(float(current_trade["entry_price"]), 4),
        exit_price=round(float(exit_price), 4),
        holding_days=int(current_trade["holding_days"]),
        position_weight=round(avg_weight, 6),
        gross_return_pct=round((float(exit_price) / (float(current_trade["entry_price"]) + 1e-9) - 1.0) * 100.0, 4),
        net_pnl=round(float(net_pnl), 4),
        signal_strength=round(_safe_float(current_trade.get("signal_strength"), 0.0), 6),
        regime=str(current_trade.get("regime", "")),
        volatility=(
            None
            if not np.isfinite(_safe_float(current_trade.get("volatility"), np.nan))
            else round(_safe_float(current_trade.get("volatility"), 0.0), 6)
        ),
        entry_reason=str(current_trade.get("entry_reason", "entry")),
        exit_reason=str(exit_reason),
    )
    trades.append(asdict(trade))
    return None


def generate_signals(predictions: pd.DataFrame, config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    cfg = config or BacktestConfig()
    out = predictions.copy().sort_values("trade_date").reset_index(drop=True)
    if out.empty:
        return out

    state_rows: List[Dict[str, Any]] = []
    current_weight = 0.0
    holding_days = 0
    cooldown_remaining = 0
    confirmation_streak = 0
    confirmation_period = max(int(cfg.confirmation_period), 1)
    total_cost_rate = float(cfg.fee_rate) + float(cfg.slippage_rate)

    for _, row in out.iterrows():
        pred_ret = _safe_float(row.get("predicted_return"), 0.0)
        dir_prob = float(np.clip(_safe_float(row.get("direction_prob"), 0.5), 0.0, 1.0))
        strength = _signal_strength(pred_ret, dir_prob)
        confidence = _confidence_scalar(dir_prob)
        volatility_value = _infer_volatility(row)
        regime_behavior = _regime_behavior(row.get("regime"), volatility_value, cfg)
        adjusted_entry_threshold = float(cfg.entry_threshold)
        adjusted_prob_threshold = float(regime_behavior["min_direction_prob"])
        adjusted_strength_threshold = float(regime_behavior["min_signal_strength"])
        size = _position_size(
            dir_prob=dir_prob,
            volatility_value=volatility_value,
            cfg=cfg,
            size_multiplier=float(regime_behavior["size_multiplier"]),
        )
        in_cooldown = current_weight <= 0.0 and cooldown_remaining > 0

        entry_triggered = False
        exit_triggered = False
        exit_blocked_by_hold = False
        confirmation_ready = False
        decision_reason = ""

        if current_weight > 0.0:
            exit_reason = ""
            if pred_ret < float(cfg.exit_threshold):
                exit_reason = "exit_predicted_return"
            elif dir_prob < float(cfg.exit_direction_prob):
                exit_reason = "exit_direction_prob"
            elif strength <= 0.0:
                exit_reason = "exit_signal_strength"

            if exit_reason and holding_days >= int(cfg.min_hold_days):
                current_weight = 0.0
                holding_days = 0
                cooldown_remaining = max(int(cfg.cooldown_days), 0)
                exit_triggered = True
                decision_reason = exit_reason
            else:
                if exit_reason:
                    exit_blocked_by_hold = True
                    decision_reason = "min_hold"
                else:
                    current_weight = size if size > 0.0 else current_weight
                    decision_reason = "hold"
                holding_days += 1
            confirmation_streak = 0
        else:
            meets_entry_threshold = pred_ret > adjusted_entry_threshold
            meets_cost_threshold = pred_ret > total_cost_rate
            meets_prob_threshold = dir_prob >= adjusted_prob_threshold
            meets_strength_threshold = strength >= adjusted_strength_threshold
            base_entry_ready = (
                not in_cooldown
                and not bool(regime_behavior["entry_blocked"])
                and meets_entry_threshold
                and meets_cost_threshold
                and meets_prob_threshold
                and meets_strength_threshold
                and size > 0.0
            )

            if base_entry_ready:
                confirmation_streak += 1
            else:
                confirmation_streak = 0
            confirmation_ready = confirmation_streak >= confirmation_period

            if in_cooldown:
                decision_reason = "cooldown"
            elif _regime_blocks_entry(row.get("regime")):
                confirmation_streak = 0
                decision_reason = "regime"
            elif bool(regime_behavior["entry_blocked"]):
                confirmation_streak = 0
                decision_reason = str(regime_behavior["behavior"])
            elif not meets_cost_threshold:
                confirmation_streak = 0
                decision_reason = "cost"
            elif not meets_strength_threshold:
                confirmation_streak = 0
                decision_reason = "strength"
            elif not (meets_entry_threshold and meets_prob_threshold):
                confirmation_streak = 0
                decision_reason = "threshold"
            elif size <= 0.0:
                confirmation_streak = 0
                decision_reason = "size"
            elif not confirmation_ready:
                decision_reason = "confirmation"
            else:
                current_weight = size
                holding_days = 1
                entry_triggered = True
                decision_reason = "confirmed_entry"

            if not entry_triggered:
                current_weight = 0.0
                holding_days = 0
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

        state_rows.append(
            {
                "signal": int(current_weight > 0.0),
                "position_weight": round(current_weight, 6),
                "signal_strength": round(strength, 6),
                "size_scalar": round(confidence, 6),
                "confidence_scalar": round(confidence, 6),
                "volatility_value": None if not np.isfinite(volatility_value) else round(float(volatility_value), 6),
                "entry_threshold_used": round(adjusted_entry_threshold, 6),
                "direction_prob_threshold_used": round(adjusted_prob_threshold, 6),
                "signal_strength_threshold_used": round(adjusted_strength_threshold, 6),
                "cost_threshold_used": round(total_cost_rate, 6),
                "confirmation_count": int(confirmation_streak),
                "confirmation_period": int(confirmation_period),
                "confirmation_ready": bool(confirmation_ready),
                "regime_behavior": str(regime_behavior["behavior"]),
                "regime_size_multiplier": round(float(regime_behavior["size_multiplier"]), 6),
                "holding_days": int(holding_days),
                "cooldown_remaining": int(cooldown_remaining),
                "in_cooldown": bool(in_cooldown),
                "entry_triggered": bool(entry_triggered),
                "exit_triggered": bool(exit_triggered),
                "exit_blocked_by_hold": bool(exit_blocked_by_hold),
                "decision_reason": decision_reason,
            }
        )

    return pd.concat([out, pd.DataFrame(state_rows, index=out.index)], axis=1)


def run_backtest(
    signals: pd.DataFrame,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    df = signals.copy().sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        return BacktestResult(summary={}, equity_curve=[], trades=[], signals=[])

    equity = float(cfg.initial_capital)
    prev_end_weight = 0.0
    forced_cooldown_remaining = 0
    running_peak = float(cfg.initial_capital)
    equity_curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    portfolio_returns: List[float] = []
    current_trade: Optional[Dict[str, Any]] = None

    total_cost_rate = float(cfg.fee_rate) + float(cfg.slippage_rate)

    for _, row in df.iterrows():
        equity_before = equity
        current_drawdown = equity_before / max(running_peak, 1e-9) - 1.0
        drawdown_control = _drawdown_scale(current_drawdown, cfg)
        signal_weight = _clip_weight(_safe_float(row.get("position_weight"), 0.0), cfg.max_position_size)
        target_weight = _clip_weight(signal_weight * float(drawdown_control["scale"]), cfg.max_position_size)
        execution_cooldown_before = forced_cooldown_remaining
        cooldown_blocked = prev_end_weight <= 0.0 and forced_cooldown_remaining > 0 and target_weight > 0.0
        drawdown_pause_active = bool(drawdown_control["pause"]) and signal_weight > 0.0
        actual_weight = 0.0 if cooldown_blocked or drawdown_pause_active else target_weight

        if current_trade is not None and prev_end_weight > 0.0 and actual_weight <= 0.0:
            exit_reason = str(row.get("decision_reason") or "signal_exit")
            if drawdown_pause_active:
                exit_reason = "drawdown_pause"
            current_trade = _close_trade(
                trades=trades,
                current_trade=current_trade,
                exit_date=str(row.get("signal_date", row.get("trade_date", ""))),
                exit_price=_safe_float(row.get("signal_close"), _safe_float(row.get("target_close"), 0.0)),
                net_pnl=equity_before - float(current_trade["entry_equity"]),
                exit_reason=exit_reason,
            )

        if current_trade is None and prev_end_weight <= 0.0 and actual_weight > 0.0:
            entry_reason = str(row.get("decision_reason") or "entry")
            if signal_weight > actual_weight:
                entry_reason = f"{entry_reason}+drawdown_scaled"
            current_trade = {
                "entry_date": str(row.get("signal_date", row.get("trade_date", ""))),
                "entry_price": _safe_float(row.get("signal_close"), _safe_float(row.get("target_close"), 0.0)),
                "holding_days": 0,
                "entry_equity": equity_before,
                "weight_sum": 0.0,
                "signal_strength": _safe_float(row.get("signal_strength"), 0.0),
                "regime": str(row.get("regime", "")),
                "volatility": row.get("volatility_value"),
                "entry_reason": entry_reason,
            }

        actual_ret_raw = _safe_float(row.get("actual_return"), 0.0)
        actual_ret_used = actual_ret_raw
        stop_loss_triggered = False
        if actual_weight > 0.0 and cfg.stop_loss_pct is not None and actual_ret_raw <= -float(cfg.stop_loss_pct):
            actual_ret_used = -float(cfg.stop_loss_pct)
            stop_loss_triggered = True

        turnover = abs(actual_weight - prev_end_weight)
        trading_cost = turnover * total_cost_rate
        daily_portfolio_return = actual_weight * actual_ret_used - trading_cost
        stop_exit_cost = 0.0
        end_weight = actual_weight

        if current_trade is not None and actual_weight > 0.0:
            current_trade["holding_days"] += 1
            current_trade["weight_sum"] += actual_weight

        if stop_loss_triggered:
            stop_exit_cost = actual_weight * total_cost_rate
            daily_portfolio_return -= stop_exit_cost
            end_weight = 0.0

        if cfg.daily_loss_cap_pct is not None:
            daily_portfolio_return = max(daily_portfolio_return, -float(cfg.daily_loss_cap_pct))

        equity *= (1.0 + daily_portfolio_return)
        running_peak = max(running_peak, equity)
        portfolio_returns.append(daily_portfolio_return)

        if stop_loss_triggered and current_trade is not None:
            stop_exit_price = _safe_float(row.get("signal_close"), _safe_float(row.get("target_close"), 0.0)) * (1.0 - float(cfg.stop_loss_pct))
            current_trade = _close_trade(
                trades=trades,
                current_trade=current_trade,
                exit_date=str(row.get("trade_date", "")),
                exit_price=stop_exit_price,
                net_pnl=equity - float(current_trade["entry_equity"]),
                exit_reason="stop_loss",
            )
            forced_cooldown_remaining = max(int(cfg.cooldown_days), forced_cooldown_remaining)

        equity_curve.append(
            {
                "date": str(row["trade_date"]),
                "equity": round(equity, 4),
                "signal": int(actual_weight > 0.0),
                "predicted_return": round(_safe_float(row.get("predicted_return"), 0.0), 6),
                "actual_return": round(actual_ret_raw, 6),
                "actual_return_used": round(actual_ret_used, 6),
                "portfolio_return": round(daily_portfolio_return, 6),
                "signal_position_weight": round(signal_weight, 6),
                "target_position_weight": round(target_weight, 6),
                "actual_position_weight": round(actual_weight, 6),
                "ending_position_weight": round(end_weight, 6),
                "trading_cost": round(trading_cost + stop_exit_cost, 6),
                "stop_loss_triggered": bool(stop_loss_triggered),
                "execution_cooldown_remaining": int(execution_cooldown_before),
                "cooldown_blocked_entry": bool(cooldown_blocked),
                "signal_strength": round(_safe_float(row.get("signal_strength"), 0.0), 6),
                "regime": str(row.get("regime", "")),
                "volatility": round(_safe_float(row.get("volatility_value"), 0.0), 6),
                "current_drawdown_pct": round(float(current_drawdown) * 100.0, 4),
                "drawdown_risk_active": bool(drawdown_control["active"]),
                "drawdown_scale_applied": round(float(drawdown_control["scale"]), 6),
                "drawdown_pause_active": bool(drawdown_pause_active),
            }
        )
        prev_end_weight = end_weight
        if prev_end_weight <= 0.0 and forced_cooldown_remaining > 0 and not stop_loss_triggered:
            forced_cooldown_remaining -= 1

    if current_trade is not None:
        last_row = df.iloc[-1]
        current_trade = _close_trade(
            trades=trades,
            current_trade=current_trade,
            exit_date=str(last_row.get("trade_date", "")),
            exit_price=_safe_float(last_row.get("target_close"), _safe_float(last_row.get("signal_close"), 0.0)),
            net_pnl=equity - float(current_trade["entry_equity"]),
            exit_reason="mark_to_market",
        )

    eq = np.asarray([p["equity"] for p in equity_curve], dtype=float)
    running_peak = np.maximum.accumulate(eq)
    drawdowns = eq / (running_peak + 1e-9) - 1.0
    wins = [t for t in trades if float(t["net_pnl"]) > 0]

    mean_ret = float(np.mean(portfolio_returns)) if portfolio_returns else 0.0
    std_ret = float(np.std(portfolio_returns)) if portfolio_returns else 0.0
    sharpe = 0.0 if std_ret <= 1e-12 else float(np.sqrt(252.0) * mean_ret / std_ret)
    strategy_return = (float(eq[-1]) / float(cfg.initial_capital) - 1.0) if len(eq) else 0.0

    initial_price = _safe_float(df.iloc[0].get("signal_close"), _safe_float(df.iloc[0].get("target_close"), 0.0))
    final_price = _safe_float(df.iloc[-1].get("target_close"), _safe_float(df.iloc[-1].get("signal_close"), initial_price))
    buy_hold_return = (final_price / initial_price - 1.0) if initial_price > 0.0 and final_price > 0.0 else 0.0
    alpha = strategy_return - buy_hold_return

    summary = {
        "initial_capital": round(float(cfg.initial_capital), 2),
        "ending_capital": round(float(eq[-1]) if len(eq) else float(cfg.initial_capital), 2),
        "total_return_pct": round(strategy_return * 100.0, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(float(drawdowns.min()) * 100.0 if len(drawdowns) else 0.0, 4),
        "win_rate_pct": round((len(wins) / len(trades)) * 100.0, 2) if trades else 0.0,
        "num_trades": float(len(trades)),
        "exposure_pct": round(float(np.mean([row["actual_position_weight"] for row in equity_curve])) * 100.0, 2) if equity_curve else 0.0,
        "avg_position_size_pct": round(float(np.mean([row["actual_position_weight"] for row in equity_curve if row["actual_position_weight"] > 0.0])) * 100.0, 2) if any(row["actual_position_weight"] > 0.0 for row in equity_curve) else 0.0,
        "buy_hold_return_pct": round(float(buy_hold_return) * 100.0, 4),
        "alpha_pct": round(float(alpha) * 100.0, 4),
    }

    return BacktestResult(
        summary=summary,
        equity_curve=equity_curve,
        trades=trades,
        signals=df.to_dict(orient="records"),
    )
