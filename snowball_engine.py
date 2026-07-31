"""Monte Carlo pricing and risk engine for the European Snowball model.

The implementation is intentionally aligned with the supplied
Snowball_Pricing(2).ipynb, especially the Cell 3 / Cell 4 logic:
- risk-neutral GBM drift r - q;
- monthly knock-out observations after a lock-in period;
- linearly decreasing knock-out barrier;
- European knock-in, checked only at maturity;
- deterministic r, q and volatility;
- common-random-number central finite-difference Greeks.

The engine adds vectorized/chunked curve calculations and reusable analytics for
an external-facing Streamlit dashboard without changing the core payoff logic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import json
from typing import Iterable
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnowballParams:
    S0: float = 100.0
    KI_barrier: float = 65.0
    KO_start: float = 100.0
    SD_rate: float = 0.5
    lock_months: int = 3
    tenure_months: int = 24
    coupon: float = 0.10
    r: float = 0.025
    q: float = 0.03
    vol: float = 0.22
    n_sims: int = 30_000
    face_value: float = 100.0

    @property
    def maturity_years(self) -> float:
        return self.tenure_months / 12.0

    def validate(self) -> None:
        if self.S0 <= 0:
            raise ValueError("S0 must be positive.")
        if self.KI_barrier <= 0 or self.KO_start <= 0:
            raise ValueError("Barrier levels must be positive.")
        if self.SD_rate < 0:
            raise ValueError("Monthly step-down cannot be negative.")
        if self.lock_months < 0:
            raise ValueError("Lock-in months cannot be negative.")
        if self.tenure_months <= 0:
            raise ValueError("Tenure must be positive.")
        if self.lock_months >= self.tenure_months:
            raise ValueError("Lock-in period must be shorter than total tenure.")
        if self.coupon < -1.0:
            raise ValueError("Coupon is implausibly low.")
        if self.vol < 0:
            raise ValueError("Volatility cannot be negative.")
        if self.n_sims < 100:
            raise ValueError("n_sims must be at least 100.")
        if self.face_value <= 0:
            raise ValueError("Face value must be positive.")

    def ko_barrier(self, month: int) -> float:
        """Monthly step-down KO barrier, matching the notebook."""
        return self.KO_start - self.SD_rate * (month - self.lock_months - 1)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "S0": self.S0,
            "KI_barrier": self.KI_barrier,
            "KO_start": self.KO_start,
            "SD_rate": self.SD_rate,
            "lock_months": self.lock_months,
            "tenure_months": self.tenure_months,
            "coupon": self.coupon,
            "r": self.r,
            "q": self.q,
            "vol": self.vol,
            "n_sims": self.n_sims,
            "face_value": self.face_value,
        }


def generate_normals(params: SnowballParams, seed: int) -> np.ndarray:
    """Generate normals using the legacy RandomState sequence used by the notebook.

    RandomState(seed).normal(...) matches the sequence produced by
    np.random.seed(seed); np.random.normal(...), preserving reproducibility with
    the original notebook convention.
    """
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 1.0, size=(params.n_sims, params.tenure_months))


def future_schedule(
    t_curr_years: float, params: SnowballParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return future observation months, times from valuation, and increments."""
    if t_curr_years < 0:
        raise ValueError("Current time cannot be negative.")

    m_curr = int(np.floor(t_curr_years * 12.0 + 1e-12))
    obs_months = np.arange(m_curr + 1, params.tenure_months + 1, dtype=int)
    if obs_months.size == 0:
        return obs_months, np.array([], dtype=float), np.array([], dtype=float)

    dt_steps = (obs_months - t_curr_years * 12.0) / 12.0
    dt_steps = np.maximum(dt_steps, 0.0)
    dts = np.diff(np.insert(dt_steps, 0, 0.0))
    return obs_months, dt_steps, dts


def _growth_factors(
    t_curr_years: float,
    Z_mat: np.ndarray,
    params: SnowballParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate multiplicative GBM growth from valuation time to future observations."""
    obs_months, dt_steps, dts = future_schedule(t_curr_years, params)
    if obs_months.size == 0:
        return obs_months, dt_steps, np.empty((params.n_sims, 0))

    if Z_mat.ndim != 2 or Z_mat.shape[0] != params.n_sims:
        raise ValueError(
            f"Z_mat must have shape ({params.n_sims}, at least {len(obs_months)})."
        )
    if Z_mat.shape[1] < len(obs_months):
        raise ValueError("Z_mat does not have enough observation columns.")

    Z = Z_mat[:, : len(obs_months)]
    log_returns = (
        (params.r - params.q - 0.5 * params.vol**2) * dts
        + params.vol * np.sqrt(dts) * Z
    )
    growth = np.exp(np.cumsum(log_returns, axis=1))
    return obs_months, dt_steps, growth


def price_many_spots(
    spots: Iterable[float] | np.ndarray,
    t_curr_years: float,
    Z_mat: np.ndarray,
    params: SnowballParams,
    coupon_rate: float | None = None,
    chunk_size: int = 8,
) -> np.ndarray:
    """Price multiple spot levels using the notebook's payoff logic."""
    params.validate()
    coupon = params.coupon if coupon_rate is None else float(coupon_rate)
    spots_arr = np.asarray(
        list(spots) if not isinstance(spots, np.ndarray) else spots,
        dtype=float,
    )
    original_shape = spots_arr.shape
    spots_flat = spots_arr.reshape(-1)

    if np.any(spots_flat <= 0):
        raise ValueError("All spot levels must be positive.")

    # At/after maturity, use the same terminal logic as the notebook.
    if t_curr_years >= params.maturity_years:
        maturity_payoff = np.where(
            spots_flat >= params.KI_barrier,
            params.face_value * (1.0 + coupon * params.maturity_years),
            params.face_value * spots_flat / params.S0,
        )
        return maturity_payoff.reshape(original_shape)

    obs_months, dt_steps, growth = _growth_factors(t_curr_years, Z_mat, params)
    output = np.empty_like(spots_flat)
    chunk_size = max(1, int(chunk_size))

    for start in range(0, len(spots_flat), chunk_size):
        stop = min(start + chunk_size, len(spots_flat))
        spot_chunk = spots_flat[start:stop]

        # shape: (spot_chunk, path, observation)
        paths = spot_chunk[:, None, None] * growth[None, :, :]
        batch = len(spot_chunk)
        payoffs = np.zeros((batch, params.n_sims), dtype=float)
        alive = np.ones((batch, params.n_sims), dtype=bool)

        # Monthly KO observations after the lock-in period.
        for idx, month in enumerate(obs_months):
            if month > params.lock_months:
                barrier = params.ko_barrier(int(month))
                is_ko = (paths[:, :, idx] >= barrier) & alive
                if np.any(is_ko):
                    ko_payoff = (
                        params.face_value
                        * (1.0 + coupon * month / 12.0)
                        * np.exp(-params.r * dt_steps[idx])
                    )
                    payoffs[is_ko] = ko_payoff
                    alive[is_ko] = False

        # Maturity payoff for paths not knocked out.
        terminal_spot = paths[:, :, -1]
        maturity_payoff = np.where(
            terminal_spot >= params.KI_barrier,
            params.face_value * (1.0 + coupon * params.maturity_years),
            params.face_value * terminal_spot / params.S0,
        ) * np.exp(-params.r * dt_steps[-1])
        payoffs[alive] = maturity_payoff[alive]

        output[start:stop] = np.mean(payoffs, axis=1)

    return output.reshape(original_shape)


def price_snowball(
    S_curr: float,
    t_curr_years: float,
    Z_mat: np.ndarray,
    params: SnowballParams,
    coupon_rate: float | None = None,
) -> float:
    return float(
        price_many_spots(
            np.array([S_curr]),
            t_curr_years,
            Z_mat,
            params,
            coupon_rate=coupon_rate,
        )[0]
    )


def greeks_curve(
    spots: Iterable[float] | np.ndarray,
    t_curr_years: float,
    Z_mat: np.ndarray,
    params: SnowballParams,
    dS: float = 0.5,
) -> pd.DataFrame:
    """Calculate price, Delta and Gamma with central finite differences."""
    if dS <= 0:
        raise ValueError("dS must be positive.")

    spots_arr = np.asarray(
        list(spots) if not isinstance(spots, np.ndarray) else spots,
        dtype=float,
    )
    if np.any(spots_arr - dS <= 0):
        raise ValueError("The minimum spot must be greater than dS.")

    price_mid = price_many_spots(spots_arr, t_curr_years, Z_mat, params)
    price_up = price_many_spots(spots_arr + dS, t_curr_years, Z_mat, params)
    price_down = price_many_spots(spots_arr - dS, t_curr_years, Z_mat, params)

    delta = (price_up - price_down) / (2.0 * dS)
    gamma = (price_up - 2.0 * price_mid + price_down) / (dS**2)

    return pd.DataFrame(
        {
            "Spot": spots_arr,
            "Price": price_mid,
            "Delta": delta,
            "Gamma": gamma,
            "Time_Years": t_curr_years,
            "Time_Months": t_curr_years * 12.0,
        }
    )


def greeks_at_spot(
    S_curr: float,
    t_curr_years: float,
    Z_mat: np.ndarray,
    params: SnowballParams,
    dS: float = 0.5,
) -> dict[str, float]:
    row = greeks_curve([S_curr], t_curr_years, Z_mat, params, dS=dS).iloc[0]
    return {
        "price": float(row["Price"]),
        "delta": float(row["Delta"]),
        "gamma": float(row["Gamma"]),
    }


def target_coupon_for_par(
    params: SnowballParams,
    Z_mat: np.ndarray,
    target_price: float | None = None,
) -> float:
    """Solve the coupon that makes Day-0 PV equal the target issue price.

    In the notebook payoff, KO decisions do not depend on coupon and each payoff
    is affine in coupon. Therefore two common-random-number prices (coupon 0 and
    coupon 100%) give an exact linear solution without root-finding noise.
    """
    target = params.face_value if target_price is None else float(target_price)
    pv_zero = price_snowball(
        params.S0, 0.0, Z_mat, params, coupon_rate=0.0
    )
    pv_one = price_snowball(
        params.S0, 0.0, Z_mat, params, coupon_rate=1.0
    )
    slope = pv_one - pv_zero
    if abs(slope) < 1e-12:
        raise RuntimeError("Coupon sensitivity is too small to determine a target coupon.")
    return (target - pv_zero) / slope


def barrier_schedule(params: SnowballParams) -> pd.DataFrame:
    months = np.arange(params.lock_months + 1, params.tenure_months + 1)
    return pd.DataFrame(
        {
            "Observation_Month": months,
            "KO_Barrier": [params.ko_barrier(int(m)) for m in months],
            "KO_Percent_of_S0": [params.ko_barrier(int(m)) / params.S0 for m in months],
        }
    )


def observation_delta_table(
    S_curr: float,
    params: SnowballParams,
    Z_mat: np.ndarray,
    dS: float = 0.5,
    day_fraction: float = 1.0 / 365.0,
) -> pd.DataFrame:
    """Compare Greeks one day before and after each monthly observation."""
    rows: list[dict[str, float]] = []
    for month in range(params.lock_months + 1, params.tenure_months + 1):
        t_obs = month / 12.0
        t_before = max(0.0, t_obs - day_fraction)
        before = greeks_at_spot(S_curr, t_before, Z_mat, params, dS=dS)

        if month < params.tenure_months:
            t_after = t_obs + day_fraction
            after = greeks_at_spot(S_curr, t_after, Z_mat, params, dS=dS)
            delta_after = after["delta"]
            gamma_after = after["gamma"]
            price_after = after["price"]
            delta_jump = delta_after - before["delta"]
        else:
            delta_after = np.nan
            gamma_after = np.nan
            price_after = np.nan
            delta_jump = np.nan

        rows.append(
            {
                "Observation_Month": month,
                "KO_Barrier": params.ko_barrier(month),
                "Spot": S_curr,
                "Price_Before": before["price"],
                "Price_After": price_after,
                "Delta_Before": before["delta"],
                "Delta_After": delta_after,
                "Delta_Jump": delta_jump,
                "Gamma_Before": before["gamma"],
                "Gamma_After": gamma_after,
            }
        )
    return pd.DataFrame(rows)


def outcome_analysis(
    params: SnowballParams,
    Z_mat: np.ndarray,
    coupon_rate: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Estimate first-KO probabilities and terminal outcome probabilities at inception."""
    coupon = params.coupon if coupon_rate is None else float(coupon_rate)
    obs_months, dt_steps, growth = _growth_factors(0.0, Z_mat, params)
    paths = params.S0 * growth

    alive = np.ones(params.n_sims, dtype=bool)
    first_ko_month = np.zeros(params.n_sims, dtype=int)
    discounted_payoff = np.zeros(params.n_sims, dtype=float)

    for idx, month in enumerate(obs_months):
        if month > params.lock_months:
            is_ko = (paths[:, idx] >= params.ko_barrier(int(month))) & alive
            if np.any(is_ko):
                first_ko_month[is_ko] = int(month)
                discounted_payoff[is_ko] = (
                    params.face_value
                    * (1.0 + coupon * month / 12.0)
                    * np.exp(-params.r * dt_steps[idx])
                )
                alive[is_ko] = False

    terminal_spot = paths[:, -1]
    maturity_safe = alive & (terminal_spot >= params.KI_barrier)
    maturity_loss = alive & (terminal_spot < params.KI_barrier)

    discounted_payoff[maturity_safe] = (
        params.face_value
        * (1.0 + coupon * params.maturity_years)
        * np.exp(-params.r * dt_steps[-1])
    )
    discounted_payoff[maturity_loss] = (
        params.face_value
        * terminal_spot[maturity_loss]
        / params.S0
        * np.exp(-params.r * dt_steps[-1])
    )

    schedule_rows = []
    for month in range(params.lock_months + 1, params.tenure_months + 1):
        schedule_rows.append(
            {
                "Observation_Month": month,
                "KO_Barrier": params.ko_barrier(month),
                "First_KO_Probability": float(np.mean(first_ko_month == month)),
            }
        )
    ko_df = pd.DataFrame(schedule_rows)

    categories = pd.DataFrame(
        {
            "Outcome": ["Knock-out", "Maturity safe", "Maturity loss"],
            "Probability": [
                float(np.mean(first_ko_month > 0)),
                float(np.mean(maturity_safe)),
                float(np.mean(maturity_loss)),
            ],
        }
    )

    ko_months = first_ko_month[first_ko_month > 0]
    summary = {
        "pv": float(np.mean(discounted_payoff)),
        "ko_probability": float(np.mean(first_ko_month > 0)),
        "maturity_safe_probability": float(np.mean(maturity_safe)),
        "maturity_loss_probability": float(np.mean(maturity_loss)),
        "expected_ko_month_conditional": (
            float(np.mean(ko_months)) if len(ko_months) else np.nan
        ),
        "payoff_std": float(np.std(discounted_payoff, ddof=1)),
        "payoff_p05": float(np.quantile(discounted_payoff, 0.05)),
        "payoff_p50": float(np.quantile(discounted_payoff, 0.50)),
        "payoff_p95": float(np.quantile(discounted_payoff, 0.95)),
    }
    return ko_df, categories, summary


def target_coupon_sensitivity(
    params: SnowballParams,
    seed: int,
    q_values: Iterable[float] | None = None,
    vol_values: Iterable[float] | None = None,
    target_price: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate par-coupon sensitivity to dividend yield and volatility."""
    q_rows: list[dict[str, float]] = []
    vol_rows: list[dict[str, float]] = []

    if q_values is not None:
        for q_value in q_values:
            scenario = replace(params, q=float(q_value))
            Z = generate_normals(scenario, seed)
            q_rows.append(
                {
                    "Dividend_Yield": float(q_value),
                    "Target_Coupon": target_coupon_for_par(scenario, Z, target_price),
                }
            )

    if vol_values is not None:
        for vol_value in vol_values:
            scenario = replace(params, vol=float(vol_value))
            Z = generate_normals(scenario, seed)
            vol_rows.append(
                {
                    "Volatility": float(vol_value),
                    "Target_Coupon": target_coupon_for_par(scenario, Z, target_price),
                }
            )

    return pd.DataFrame(q_rows), pd.DataFrame(vol_rows)


def seed_stability(
    params: SnowballParams,
    seeds: Iterable[int],
    target_price: float,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        Z = generate_normals(params, int(seed))
        rows.append(
            {
                "Seed": int(seed),
                "Target_Coupon": target_coupon_for_par(params, Z, target_price),
            }
        )
    return pd.DataFrame(rows)


def make_analysis_zip(
    params: SnowballParams,
    analysis_tables: dict[str, pd.DataFrame],
    metadata: dict[str, object],
) -> bytes:
    """Package the current analysis into a portable ZIP download."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysis_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        zf.writestr("parameters.json", json.dumps(params.as_dict(), ensure_ascii=False, indent=2))
        for name, df in analysis_tables.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            zf.writestr(f"{safe_name}.csv", df.to_csv(index=False))
    return buffer.getvalue()
