"""
FINANCIAL KUNDLI v2.0 — Institutional-Grade Financial Planning Engine
=====================================================================
Single-file Streamlit application. Run:  streamlit run financial_kundli_v2.py
Self-test the computation engine:        python financial_kundli_v2.py --selftest

Major corrections vs v1 (see AUDIT section in the accompanying report):
  1.  Income-tax engine rebuilt: correct FY 2026-27 slabs (new & old regime),
      Section 87A rebates for BOTH regimes, marginal relief on rebate and on
      surcharge, correct surcharge tiers (10/15/25/37%, capped at 25% in the
      new regime), cess applied last. Config-driven so future FYs are a
      10-line table edit, not a code rewrite.
  2.  Retirement corpus now uses an INFLATION-ADJUSTED (real-return) annuity.
      v1 kept post-retirement expenses flat for 25+ years, understating the
      required corpus by roughly 40-60% at 6% inflation. This was the single
      largest error in v1.
  3.  Monte Carlo is now a full LIFECYCLE simulation: correlated equity/debt
      returns, monthly rebalancing, an accumulation phase and a decumulation
      phase with inflation-growing withdrawals. "Success" = the corpus
      survives to life expectancy — not merely "corpus >= a number on day 1
      of retirement". Vectorised with numpy (1,000 paths run in <1s).
  4.  Goal planning uses a HORIZON-BASED glide path (debt for <3y goals,
      blended 3-7y, risk-profiled equity for 7y+) instead of assuming 12%
      equity returns on a 2-year car goal.
  5.  Emergency-fund double counting fixed: v1 added loan EMIs to expenses
      that already contained "Housing (Rent / EMI)". Expenses are now
      strictly ex-EMI; EMIs flow only from the Debts register.
  6.  Savings rate now uses the actual tax engine (better regime) instead of
      a flat 15% guess, and nets out EMIs. One definition of "surplus" is
      used everywhere.
  7.  Debt module: exact months-to-payoff via amortisation math, detection of
      EMIs that don't cover interest, avalanche schedule.
  8.  Insurance: needs-based Human-Life-Value cover (income replacement +
      liabilities + unfunded critical goals - liquid assets) instead of a
      bare 10x-income rule.
  9.  Deterministic projections use geometric monthly rates ((1+r)^(1/12)-1),
      consistent with the Monte Carlo drift.
 10.  Session save/load restores widget state; input validation throughout.

DISCLAIMER / COMPLIANCE NOTE FOR THE OPERATOR:
  In India, charging fees for personalised investment advice generally
  requires registration as an Investment Adviser with SEBI under the
  SEBI (Investment Advisers) Regulations, 2013. Distributing this tool
  commercially as "advice" without RIA registration carries regulatory risk.
  Position it as an educational/planning calculator, or operate under an RIA.
"""

from __future__ import annotations

import io
import csv
import json
import math
import sys
import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

INF = float("inf")

# ============================================================================
# 1. MARKET & PLANNING ASSUMPTIONS (single source of truth, documented)
# ============================================================================

@dataclass(frozen=True)
class Assumptions:
    """Capital-market and planning assumptions (nominal, annualised, INR).

    These are long-run planning assumptions, not forecasts. Review annually.
    """
    equity_return:  float = 0.12
    debt_return:    float = 0.07
    gold_return:    float = 0.08
    equity_vol:     float = 0.18
    debt_vol:       float = 0.045
    gold_vol:       float = 0.14
    equity_debt_corr: float = 0.10   # long-run India equity/bond correlation

    general_inflation:   float = 0.06
    housing_inflation:   float = 0.08
    education_inflation: float = 0.10
    wedding_inflation:   float = 0.07

    post_retirement_return: float = 0.075   # ~30/70 conservative mix
    post_retirement_equity: float = 0.30

    emergency_fund_months:   int   = 6
    high_interest_threshold: float = 0.12
    home_loan_rate:          float = 0.085
    home_loan_tenure_years:  int   = 20
    down_payment_pct:        float = 0.20
    max_emi_to_income:       float = 0.40

    min_health_cover: float = 500_000

    allocation_targets: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "conservative": {"equity": 30, "debt": 55, "gold": 15},
        "moderate":     {"equity": 55, "debt": 35, "gold": 10},
        "aggressive":   {"equity": 75, "debt": 20, "gold": 5},
    })

    goal_inflation: Dict[str, float] = field(default_factory=lambda: {
        "emergency_fund": 0.06, "retirement": 0.06, "home_purchase": 0.08,
        "education": 0.10, "vehicle": 0.06, "vacation": 0.06,
        "wedding": 0.07, "other": 0.06,
    })

ASSUMPTIONS = Assumptions()


# ============================================================================
# 2. CORE FINANCIAL MATHEMATICS (annuity / SIP helpers, geometric rates)
# ============================================================================

def monthly_rate(annual: float) -> float:
    """Geometric monthly rate equivalent to an annual compounded rate."""
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def real_rate(nominal: float, inflation: float) -> float:
    """Fisher real rate: (1+n)/(1+i) - 1."""
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


def fv_lumpsum(pv: float, annual_rate: float, years: float) -> float:
    return pv * (1.0 + annual_rate) ** years


def fv_sip(monthly_amt: float, annual_rate: float, years: float) -> float:
    """Future value of an ordinary monthly SIP (end-of-month contributions)."""
    i, n = monthly_rate(annual_rate), round(years * 12)
    if n <= 0:
        return 0.0
    if abs(i) < 1e-12:
        return monthly_amt * n
    return monthly_amt * (((1 + i) ** n - 1) / i)


def sip_for_target(target_fv: float, annual_rate: float, years: float) -> float:
    """Monthly SIP required to reach target_fv in `years`."""
    i, n = monthly_rate(annual_rate), round(years * 12)
    if n <= 0:
        return target_fv
    if abs(i) < 1e-12:
        return target_fv / n
    return target_fv / (((1 + i) ** n - 1) / i)


def inflation_adjusted_corpus(monthly_expense_at_start: float,
                              years: float,
                              nominal_return: float,
                              inflation: float) -> float:
    """Corpus needed at start of a drawdown period so that a withdrawal which
    starts at `monthly_expense_at_start` and GROWS with inflation every month
    is sustainable for `years`, with the balance earning `nominal_return`.

    This is a growing-annuity PV, equivalently a level annuity at the REAL
    monthly rate. (v1 ignored inflation during retirement — the corpus it
    produced funded a fixed nominal expense, which loses ~half its purchasing
    power over a 25-year retirement at 6% inflation.)
    """
    i = (1.0 + nominal_return) ** (1 / 12) / (1.0 + inflation) ** (1 / 12) - 1.0
    n = round(years * 12)
    if n <= 0:
        return 0.0
    if abs(i) < 1e-12:
        return monthly_expense_at_start * n
    return monthly_expense_at_start * (1 - (1 + i) ** (-n)) / i


def loan_emi(principal: float, annual_rate: float, months: int) -> float:
    i = annual_rate / 12.0
    if months <= 0:
        return principal
    if abs(i) < 1e-12:
        return principal / months
    f = (1 + i) ** months
    return principal * i * f / (f - 1)


def months_to_payoff(principal: float, annual_rate: float, emi: float) -> Optional[int]:
    """Exact amortisation payoff time; None if EMI doesn't cover interest."""
    i = annual_rate / 12.0
    if principal <= 0:
        return 0
    if emi <= 0:
        return None
    if abs(i) < 1e-12:
        return math.ceil(principal / emi)
    if emi <= principal * i:
        return None  # negative amortisation — EMI never repays the loan
    return math.ceil(-math.log(1 - principal * i / emi) / math.log(1 + i))


def total_interest_on_loan(principal: float, annual_rate: float, emi: float) -> float:
    n = months_to_payoff(principal, annual_rate, emi)
    if n is None:
        return float("nan")
    return max(0.0, emi * n - principal)


# ============================================================================
# 3. INCOME TAX ENGINE — India, config-driven per financial year
# ============================================================================
# Verified against Income Tax Dept guidance for AY 2027-28 (FY 2026-27):
# Budget 2026 retained the Budget-2025 structure — new regime basic exemption
# Rs 4L, 87A rebate up to Rs 60,000 for taxable income <= Rs 12L (with
# marginal relief), std deduction Rs 75,000; old regime unchanged (2.5L
# exemption, Rs 12,500 rebate <= 5L, Rs 50,000 std deduction). Surcharge
# 10/15/25/37% with the 37% tier NOT applicable in the new regime (cap 25%);
# marginal relief applies at every surcharge threshold; 4% cess on tax+surcharge.

TAX_CONFIG: Dict[str, Dict] = {
    "FY 2026-27": {
        "old": {
            "std_deduction": 50_000,
            "slabs": [(250_000, 0.00), (250_000, 0.05), (500_000, 0.20), (INF, 0.30)],
            "rebate_income_limit": 500_000, "rebate_max": 12_500, "rebate_marginal_relief": False,
            "surcharge": [(50_00_000, 0.10), (1_00_00_000, 0.15), (2_00_00_000, 0.25), (5_00_00_000, 0.37)],
            "deduction_caps": {"sec_80c": 150_000, "sec_80d": 100_000,
                               "nps_80ccd1b": 50_000, "home_loan_int": 200_000,
                               "hra_exemption": None, "other_deductions": None},
        },
        "new": {
            "std_deduction": 75_000,
            "slabs": [(400_000, 0.00), (400_000, 0.05), (400_000, 0.10),
                      (400_000, 0.15), (400_000, 0.20), (400_000, 0.25), (INF, 0.30)],
            "rebate_income_limit": 1_200_000, "rebate_max": 60_000, "rebate_marginal_relief": True,
            "surcharge": [(50_00_000, 0.10), (1_00_00_000, 0.15), (2_00_00_000, 0.25)],
            "deduction_caps": {},  # no Chapter VI-A deductions modelled
        },
        "cess": 0.04,
    },
}
DEFAULT_TAX_YEAR = "FY 2026-27"


def _slab_tax(taxable: float, slabs: List[Tuple[float, float]]) -> float:
    tax, remaining = 0.0, max(0.0, taxable)
    for width, rate in slabs:
        chunk = min(remaining, width)
        tax += chunk * rate
        remaining -= chunk
        if remaining <= 0:
            break
    return tax


def _apply_rebate(tax: float, taxable: float, cfg: Dict) -> float:
    limit, cap = cfg["rebate_income_limit"], cfg["rebate_max"]
    if taxable <= limit:
        return max(0.0, tax - cap)
    if cfg.get("rebate_marginal_relief"):
        # Payable tax can't exceed income earned above the rebate threshold
        # (tax at the threshold itself is nil after rebate).
        return min(tax, taxable - limit)
    return tax


def _surcharge_rate(taxable: float, tiers: List[Tuple[float, float]]) -> float:
    rate = 0.0
    for threshold, r in tiers:
        if taxable > threshold:
            rate = r
    return rate


def _pre_cess_liability(taxable: float, cfg: Dict) -> float:
    """Tax + surcharge (post-rebate, with marginal relief on surcharge)."""
    base = _apply_rebate(_slab_tax(taxable, cfg["slabs"]), taxable, cfg)
    rate = _surcharge_rate(taxable, cfg["surcharge"])
    total = base * (1 + rate)
    # Marginal relief at each surcharge threshold crossed:
    for threshold, _r in cfg["surcharge"]:
        if taxable > threshold:
            at_thr = _apply_rebate(_slab_tax(threshold, cfg["slabs"]), threshold, cfg)
            at_thr *= (1 + _surcharge_rate(threshold, cfg["surcharge"]))
            total = min(total, at_thr + (taxable - threshold))
    return max(0.0, total)


def _regime_tax(gross_income: float, deductions: Dict, regime_cfg: Dict, cess: float) -> Dict:
    caps = regime_cfg["deduction_caps"]
    allowed = 0.0
    used = {}
    for key, cap in caps.items():
        amt = max(0.0, float(deductions.get(key, 0) or 0))
        take = amt if cap is None else min(amt, cap)
        used[key] = take
        allowed += take
    taxable = max(0.0, gross_income - regime_cfg["std_deduction"] - allowed)
    pre_cess = _pre_cess_liability(taxable, regime_cfg)
    return {"taxable": taxable, "tax": pre_cess * (1 + cess), "deductions_used": used}


def calculate_tax(annual_income: float, deductions: Optional[Dict] = None,
                  fy: str = DEFAULT_TAX_YEAR) -> Dict:
    """Compare old vs new regime for a resident individual (<60), salaried.

    Scope notes: slab income only (no special-rate capital gains), resident
    <60 exemption limits, 87A rebate assumed available on the full liability.
    """
    deductions = deductions or {}
    cfg = TAX_CONFIG[fy]
    old = _regime_tax(annual_income, deductions, cfg["old"], cfg["cess"])
    new = _regime_tax(annual_income, {}, cfg["new"], cfg["cess"])

    better = "New Regime" if new["tax"] <= old["tax"] else "Old Regime"
    caps = cfg["old"]["deduction_caps"]
    gap = lambda k: max(0.0, (caps.get(k) or 0) - old["deductions_used"].get(k, 0))

    return {
        "fy": fy,
        "old_regime_tax": round(old["tax"]),
        "new_regime_tax": round(new["tax"]),
        "old_taxable": round(old["taxable"]),
        "new_taxable": round(new["taxable"]),
        "better_regime": better,
        "savings_by_switching": round(abs(old["tax"] - new["tax"])),
        "effective_rate_old": (old["tax"] / annual_income * 100) if annual_income > 0 else 0.0,
        "effective_rate_new": (new["tax"] / annual_income * 100) if annual_income > 0 else 0.0,
        "80c_gap": round(gap("sec_80c")),
        "80d_gap": round(gap("sec_80d")),
        "nps_gap": round(gap("nps_80ccd1b")),
        "total_deduction_gap": round(gap("sec_80c") + gap("sec_80d") + gap("nps_80ccd1b")),
        "annual_tax_best": round(min(old["tax"], new["tax"])),
        "note": ("Deduction gaps (80C/80D/NPS) only reduce tax under the OLD regime. "
                 "If the new regime is better for you even with maxed deductions, "
                 "additional 80C investing gives no tax benefit."),
    }


# ============================================================================
# 4. RISK PROFILING
# ============================================================================

class LifeStage(Enum):
    EARLY_CAREER    = "early_career"
    MID_CAREER      = "mid_career"
    PRE_RETIREMENT  = "pre_retirement"
    NEAR_RETIREMENT = "near_retirement"
    RETIRED         = "retired"


def get_life_stage(age: int) -> LifeStage:
    if age < 30:  return LifeStage.EARLY_CAREER
    if age < 45:  return LifeStage.MID_CAREER
    if age < 55:  return LifeStage.PRE_RETIREMENT
    if age < 60:  return LifeStage.NEAR_RETIREMENT
    return LifeStage.RETIRED


RISK_QUESTIONS = [
    {"id": "q1", "text": "1. How would you describe your primary investment objective?",
     "options": [("Preserve capital — I cannot afford to lose money", 1),
                 ("Steady income with some growth", 2),
                 ("Balanced growth with moderate risk", 3),
                 ("Aggressive growth — I can handle significant ups and downs", 4)]},
    {"id": "q2", "text": "2. If your portfolio dropped 25% in a quarter, what would you do?",
     "options": [("Sell everything immediately", 1),
                 ("Shift most to safer assets", 2),
                 ("Hold and wait for recovery", 3),
                 ("Buy more — it's a great opportunity", 4)]},
    {"id": "q3", "text": "3. What is your investment horizon?",
     "options": [("Less than 2 years", 1), ("2–5 years", 2),
                 ("5–10 years", 3), ("More than 10 years", 4)]},
    {"id": "q4", "text": "4. How stable is your income?",
     "options": [("Very unstable — freelance / variable", 1),
                 ("Somewhat variable — commission / bonus heavy", 2),
                 ("Mostly stable with occasional variability", 3),
                 ("Very stable — government / large corporate", 4)]},
    {"id": "q5", "text": "5. What share of monthly income can you invest without impacting lifestyle?",
     "options": [("Less than 10%", 1), ("10–20%", 2), ("20–35%", 3), ("More than 35%", 4)]},
]


def score_risk_profile(answers: Dict) -> Tuple[str, int, str]:
    total = sum(answers.values())
    max_score = len(RISK_QUESTIONS) * 4
    if total <= max_score * 0.40:
        return ("conservative", total,
                "You prefer capital preservation over returns and are uncomfortable with losses.")
    if total <= max_score * 0.65:
        return ("moderate", total,
                "You seek balance — steady growth with manageable volatility.")
    return ("aggressive", total,
            "You aim for maximum long-term returns and can tolerate significant short-term swings.")


def capacity_adjusted_risk(risk_level: str, age: int, years_to_retirement: int) -> str:
    """Risk WILLINGNESS (questionnaire) capped by risk CAPACITY (horizon).
    An aggressive 58-year-old retiring at 60 does not have aggressive capacity."""
    order = ["conservative", "moderate", "aggressive"]
    cap = "aggressive"
    if years_to_retirement < 5 or age >= 55:
        cap = "conservative"
    elif years_to_retirement < 10:
        cap = "moderate"
    return order[min(order.index(risk_level), order.index(cap))]


# ============================================================================
# 5. GOAL RETURN GLIDE PATH
# ============================================================================

def goal_asset_mix(years: float, risk_level: str, a: Assumptions = ASSUMPTIONS) -> Dict:
    """Horizon-appropriate return/vol for a goal. v1 assumed 12% equity on
    every goal including 2-year ones — indefensible for short horizons."""
    eq_target = a.allocation_targets.get(risk_level, a.allocation_targets["moderate"])["equity"] / 100
    if years < 3:
        w_eq = 0.0
    elif years < 7:
        w_eq = min(0.50, eq_target)
    else:
        w_eq = eq_target
    ret = w_eq * a.equity_return + (1 - w_eq) * a.debt_return
    vol = portfolio_vol(w_eq, a)
    label = "Debt" if w_eq == 0 else (f"{w_eq*100:.0f}% Equity / {100-w_eq*100:.0f}% Debt")
    return {"equity_weight": w_eq, "return": ret, "vol": vol, "label": label}


def portfolio_vol(w_eq: float, a: Assumptions = ASSUMPTIONS) -> float:
    """Two-asset volatility WITH correlation (v1 linearly averaged vols,
    which assumes correlation = 1 and misstates portfolio risk)."""
    w_d = 1 - w_eq
    var = (w_eq ** 2 * a.equity_vol ** 2 + w_d ** 2 * a.debt_vol ** 2
           + 2 * w_eq * w_d * a.equity_debt_corr * a.equity_vol * a.debt_vol)
    return math.sqrt(max(0.0, var))


# ============================================================================
# 6. MONTE CARLO — vectorised lifecycle engine
# ============================================================================

class MonteCarloEngine:
    """Correlated equity/debt monthly returns; monthly rebalancing to the
    target mix; accumulation then decumulation with inflation-growing
    withdrawals. Success = corpus never depletes before life expectancy."""

    def __init__(self, n_simulations: int = 1000, seed: int = 42,
                 a: Assumptions = ASSUMPTIONS):
        self.n = n_simulations
        self.a = a
        self.rng = np.random.default_rng(seed)

    def _correlated_returns(self, months: int) -> Tuple[np.ndarray, np.ndarray]:
        a = self.a
        mu = np.array([monthly_rate(a.equity_return), monthly_rate(a.debt_return)])
        sig = np.array([a.equity_vol, a.debt_vol]) / math.sqrt(12)
        corr = np.array([[1.0, a.equity_debt_corr], [a.equity_debt_corr, 1.0]])
        cov = np.outer(sig, sig) * corr
        z = self.rng.multivariate_normal(mu, cov, size=(self.n, months))
        return z[:, :, 0], z[:, :, 1]   # (n_sims, months) each

    def lifecycle(self,
                  current_corpus: float,
                  monthly_contrib: float,
                  years_to_retire: int,
                  years_in_retire: int,
                  monthly_expense_today: float,
                  equity_pct_accum: float,
                  inflation: Optional[float] = None,
                  contrib_growth: float = 0.0) -> Dict:
        a = self.a
        inflation = a.general_inflation if inflation is None else inflation
        m_acc, m_ret = years_to_retire * 12, years_in_retire * 12
        months = m_acc + m_ret
        re, rd = self._correlated_returns(months)
        infl_m = monthly_rate(inflation)
        cg_m = monthly_rate(contrib_growth)

        w_acc, w_ret = equity_pct_accum, a.post_retirement_equity
        val = np.full(self.n, float(current_corpus))
        alive = np.ones(self.n, dtype=bool)
        yearly = np.empty((self.n, years_to_retire + years_in_retire + 1))
        yearly[:, 0] = val
        corpus_at_retirement = None
        contrib = monthly_contrib
        withdrawal = monthly_expense_today * (1 + infl_m) ** m_acc  # first month of retirement

        for m in range(months):
            in_accum = m < m_acc
            w = w_acc if in_accum else w_ret
            r = w * re[:, m] + (1 - w) * rd[:, m]
            val = val * (1 + r)
            if in_accum:
                val += contrib
                contrib *= (1 + cg_m)
            else:
                wd = withdrawal * (1 + infl_m) ** (m - m_acc)
                val = val - wd
                depleted = val < 0
                alive &= ~depleted
                val = np.maximum(val, 0.0)
            if m == m_acc - 1:
                corpus_at_retirement = val.copy()
            if (m + 1) % 12 == 0:
                yearly[:, (m + 1) // 12] = val

        if corpus_at_retirement is None:                 # years_to_retire == 0
            corpus_at_retirement = np.full(self.n, float(current_corpus))

        fv = corpus_at_retirement
        pct = lambda arr, q: float(np.percentile(arr, q))
        return {
            "success_probability": float(alive.mean() * 100),
            "corpus_at_retirement": {
                "mean": float(fv.mean()), "median": pct(fv, 50),
                "p5": pct(fv, 5), "p25": pct(fv, 25), "p50": pct(fv, 50),
                "p75": pct(fv, 75), "p95": pct(fv, 95), "std": float(fv.std()),
            },
            "yearly_percentiles": {
                "years": list(range(yearly.shape[1])),
                "p5":  np.percentile(yearly, 5, axis=0).tolist(),
                "p50": np.percentile(yearly, 50, axis=0).tolist(),
                "p95": np.percentile(yearly, 95, axis=0).tolist(),
            },
            "sample_paths": yearly[:60].tolist(),
            "retirement_year_index": years_to_retire,
            "first_month_withdrawal": float(withdrawal),
        }

    def goal_simulation(self, current_saved: float, monthly_sip: float,
                        years: int, target_fv: float, equity_weight: float) -> Dict:
        months = max(1, years * 12)
        re, rd = self._correlated_returns(months)
        r = equity_weight * re + (1 - equity_weight) * rd
        val = np.full(self.n, float(current_saved))
        for m in range(months):
            val = val * (1 + r[:, m]) + monthly_sip
        pct = lambda q: float(np.percentile(val, q))
        return {
            "success_probability": float((val >= target_fv).mean() * 100),
            "statistics": {"mean": float(val.mean()), "median": pct(50),
                           "p5": pct(5), "p50": pct(50), "p95": pct(95)},
        }


# ============================================================================
# 7. EMERGENCY FUND (expenses are strictly ex-EMI; EMIs from debt register)
# ============================================================================

class EmergencyFundAnalyser:
    def analyse(self, monthly_expenses_ex_emi: float, cash_bank: float,
                debts: List[Dict], months_target: Optional[int] = None,
                income_stability_score: int = 3) -> Dict:
        a = ASSUMPTIONS
        # Unstable income -> larger buffer (9 months); very stable -> 6.
        target = months_target or (9 if income_stability_score <= 2 else a.emergency_fund_months)
        total_emi = sum(d.get("emi", 0) for d in debts)
        oblig = monthly_expenses_ex_emi + total_emi
        required = oblig * target
        pct = (cash_bank / required * 100) if required > 0 else 100.0
        months_cov = (cash_bank / oblig) if oblig > 0 else 99.0
        if pct >= 100:   status, color = "Adequate", "#1D8A4E"
        elif pct >= 50:  status, color = "Partial", "#C99700"
        else:            status, color = "Insufficient", "#C94F4F"
        return {"required_fund": round(required), "current_fund": cash_bank,
                "shortfall": round(max(0.0, required - cash_bank)),
                "months_coverage": round(months_cov, 1), "target_months": target,
                "adequacy_percentage": min(150.0, pct), "status": status,
                "color": color, "monthly_obligations": round(oblig)}


# ============================================================================
# 8. RETIREMENT PLANNER (deterministic, inflation-adjusted)
# ============================================================================

class RetirementPlanner:
    def __init__(self, a: Assumptions = ASSUMPTIONS):
        self.a = a

    def plan(self, age: int, retirement_age: int, life_expectancy: int,
             monthly_expense_today: float, current_corpus: float,
             monthly_contrib: float, risk_level: str = "moderate",
             expense_replacement: float = 1.0) -> Dict:
        a = self.a
        ytr = max(1, retirement_age - age)
        yir = max(1, life_expectancy - retirement_age)
        eff_risk = capacity_adjusted_risk(risk_level, age, ytr)
        eq = a.allocation_targets[risk_level]["equity"] / 100  # accumulation mix from willingness
        blend = eq * a.equity_return + (1 - eq) * a.debt_return

        retirement_expense = monthly_expense_today * expense_replacement
        future_exp = retirement_expense * (1 + a.general_inflation) ** ytr
        corpus_needed = inflation_adjusted_corpus(
            future_exp, yir, a.post_retirement_return, a.general_inflation)

        projected = fv_lumpsum(current_corpus, blend, ytr) + fv_sip(monthly_contrib, blend, ytr)
        readiness = min(100.0, projected / corpus_needed * 100) if corpus_needed > 0 else 100.0
        gap = max(0.0, corpus_needed - projected)
        extra = sip_for_target(gap, blend, ytr) if gap > 0 else 0.0

        if readiness >= 80:  status, color = "On Track", "#1D8A4E"
        elif readiness >= 50: status, color = "Needs Work", "#C99700"
        else:                 status, color = "Underfunded", "#C94F4F"

        return {"years_to_retirement": ytr, "years_in_retirement": yir,
                "corpus_needed": round(corpus_needed),
                "projected_corpus": round(projected),
                "readiness_percentage": round(readiness, 1), "gap": round(gap),
                "additional_monthly_saving_needed": round(extra),
                "status": status, "color": color,
                "future_monthly_expense": round(future_exp),
                "accumulation_equity_pct": round(eq * 100),
                "capacity_adjusted_risk": eff_risk,
                "assumed_pre_retirement_return": blend,
                "methodology": ("Corpus = PV of a withdrawal stream growing with "
                                f"{a.general_inflation*100:.0f}% inflation through retirement, "
                                f"discounted at a {a.post_retirement_return*100:.1f}% nominal "
                                "post-retirement return (i.e. a real-return annuity).")}


# ============================================================================
# 9. DEBT ANALYSER (exact amortisation)
# ============================================================================

class DebtAnalyser:
    def analyse(self, debts: List[Dict], monthly_income: float) -> Dict:
        if not debts:
            return {"total_debt": 0, "monthly_emi": 0, "debt_to_income_ratio": 0.0,
                    "high_interest_debt_count": 0, "total_interest_payable": 0,
                    "status": "Debt-Free", "color": "#1D8A4E",
                    "priority_payoff_order": [], "warnings": []}
        total_debt = sum(d["outstanding_amount"] for d in debts)
        total_emi = sum(d.get("emi", 0) for d in debts)
        dti = (total_emi / monthly_income * 100) if monthly_income > 0 else 0.0
        warnings, total_int = [], 0.0
        enriched = []
        for d in debts:
            n = months_to_payoff(d["outstanding_amount"], d.get("interest_rate", 0), d.get("emi", 0))
            if n is None:
                warnings.append(f"'{d['name']}': EMI ₹{d.get('emi',0):,.0f} does not even cover "
                                f"monthly interest — this loan will never be repaid at this EMI.")
                interest, n_disp = None, None
            else:
                interest = total_interest_on_loan(d["outstanding_amount"], d.get("interest_rate", 0), d.get("emi", 0))
                total_int += interest
                n_disp = n
            enriched.append({"name": d["name"], "rate": d.get("interest_rate", 0),
                             "outstanding": d["outstanding_amount"], "emi": d.get("emi", 0),
                             "months_to_payoff": n_disp, "interest_payable": None if interest is None else round(interest)})
        hi = sum(1 for d in debts if d.get("interest_rate", 0) >= ASSUMPTIONS.high_interest_threshold)
        if dti <= 20:   status, color = "Healthy", "#1D8A4E"
        elif dti <= 40: status, color = "Manageable", "#C99700"
        else:           status, color = "Stressed", "#C94F4F"
        return {"total_debt": total_debt, "monthly_emi": total_emi,
                "debt_to_income_ratio": round(dti, 1),
                "high_interest_debt_count": hi,
                "total_interest_payable": round(total_int),
                "status": status, "color": color,
                "priority_payoff_order": sorted(enriched, key=lambda x: x["rate"], reverse=True),
                "warnings": warnings}


# ============================================================================
# 10. GOAL PLANNER (horizon glide path, consistent asset growth)
# ============================================================================

class GoalPlanner:
    def __init__(self, a: Assumptions = ASSUMPTIONS):
        self.a = a

    def plan_goals(self, goals: List[Dict], monthly_surplus: float,
                   risk_level: str) -> Dict:
        if not goals:
            return {"total_goals": 0, "total_target_amount_pv": 0,
                    "total_target_amount_fv": 0, "total_monthly_investment_needed": 0,
                    "goals_details": []}
        details, total_pv, total_fv, total_sip = [], 0.0, 0.0, 0.0
        for g in goals:
            infl = self.a.goal_inflation.get(g["type"], self.a.general_inflation)
            pv, years = g["target_amount"], g["timeframe_years"]
            saved = g.get("current_saved", 0)
            mix = goal_asset_mix(years, risk_level, self.a)
            fv = pv * (1 + infl) ** years
            fv_saved = fv_lumpsum(saved, mix["return"], years)
            fv_gap = max(0.0, fv - fv_saved)
            sip = sip_for_target(fv_gap, mix["return"], years) if fv_gap > 0 else 0.0
            progress = min(100.0, saved / pv * 100) if pv > 0 else 100.0
            if fv_gap <= 0:                       gstatus = "✅ Funded"
            elif sip <= monthly_surplus * 0.5:    gstatus = "🟡 On Track"
            else:                                 gstatus = "🔴 Needs Attention"
            total_pv, total_fv, total_sip = total_pv + pv, total_fv + fv, total_sip + sip
            details.append({"name": g["name"], "type": g["type"],
                            "target_amount_pv": round(pv), "target_amount_fv": round(fv),
                            "current_saved": saved, "monthly_saving_needed": round(sip),
                            "timeframe_years": years, "priority": g.get("priority", "medium"),
                            "progress_percentage": round(progress, 1),
                            "inflation_used": round(infl * 100, 1),
                            "asset_mix": mix["label"],
                            "assumed_return": round(mix["return"] * 100, 1),
                            "equity_weight": mix["equity_weight"],
                            "completion_year": datetime.now().year + years,
                            "status": gstatus})
        return {"total_goals": len(goals), "total_target_amount_pv": round(total_pv),
                "total_target_amount_fv": round(total_fv),
                "total_monthly_investment_needed": round(total_sip),
                "goals_details": details}


# ============================================================================
# 11. REBALANCING
# ============================================================================

class RebalancingEngine:
    def analyse(self, investments: List[Dict], risk_level: str) -> Dict:
        targets = ASSUMPTIONS.allocation_targets.get(risk_level, ASSUMPTIONS.allocation_targets["moderate"])
        total = sum(i["current_value"] for i in investments)
        if total == 0:
            return {"current_allocation": {"equity": 0, "debt": 0, "gold": 0},
                    "target_allocation": targets, "needs_rebalancing": False,
                    "actions": [], "total_portfolio": 0}
        current = {"equity": 0.0, "debt": 0.0, "gold": 0.0}
        for inv in investments:
            t = inv.get("type", "equity")
            if t in current:
                current[t] += inv["current_value"] / total * 100
        actions, needs = [], False
        for asset in ("equity", "debt", "gold"):
            drift = current[asset] - targets[asset]
            if abs(drift) > 5:
                needs = True
                actions.append({"asset": asset.title(),
                                "action": "SELL" if drift > 0 else "BUY",
                                "amount": round(abs(drift / 100) * total),
                                "current_pct": round(current[asset], 1),
                                "target_pct": targets[asset],
                                "drift_pct": round(drift, 1)})
        return {"current_allocation": {k: round(v, 1) for k, v in current.items()},
                "target_allocation": targets, "needs_rebalancing": needs,
                "actions": actions, "total_portfolio": total,
                "note": "Consider capital-gains tax and exit loads before selling; "
                        "prefer directing NEW SIPs to underweight assets where possible."}


# ============================================================================
# 12. INSURANCE — needs-based Human Life Value
# ============================================================================

class InsuranceAnalyser:
    def analyse(self, annual_income: float, total_debt: float,
                critical_goal_pv: float, liquid_assets: float,
                life_cover: float, health_cover: float, dependents: int) -> Dict:
        a = ASSUMPTIONS
        income_replacement = annual_income * 10 if dependents > 0 else annual_income * 3
        needed_life = max(0.0, income_replacement + total_debt + critical_goal_pv - liquid_assets)
        life_gap = max(0.0, needed_life - life_cover)
        rec_health = max(a.min_health_cover, 500_000 + 300_000 * max(0, dependents))
        health_gap = max(0.0, rec_health - health_cover)
        life_pct = min(150.0, (life_cover / needed_life * 100)) if needed_life > 0 else 150.0
        return {"life_insurance": life_cover, "health_insurance": health_cover,
                "recommended_life_cover": round(needed_life),
                "recommended_health_cover": round(rec_health),
                "life_cover_gap": round(life_gap), "health_cover_gap": round(health_gap),
                "life_adequacy_pct": round(life_pct, 1),
                "method": ("Needs-based: 10× income replacement (3× if no dependents) "
                           "+ outstanding debt + unfunded critical goals − liquid assets.")}


# ============================================================================
# 13. SCENARIO PLANNER
# ============================================================================

class ScenarioPlanner:
    def __init__(self, a: Assumptions = ASSUMPTIONS):
        self.a = a

    def job_loss(self, monthly_income, monthly_oblig, emergency_fund, months) -> Dict:
        shortfall = max(0.0, monthly_oblig * months - emergency_fund)
        covers = (emergency_fund / monthly_oblig) if monthly_oblig > 0 else 99
        verdict = ("Survivable" if covers >= months
                   else "Stressful" if covers >= months * 0.5 else "Critical")
        return {"scenario": f"Job Loss for {months} Months",
                "income_lost": round(monthly_income * months),
                "emergency_covers": round(covers, 1),
                "fund_shortfall": round(shortfall), "verdict": verdict}

    def market_crash(self, portfolio_value, crash_pct, target_corpus,
                     years_to_target, equity_return, recovery_years=3) -> Dict:
        after = portfolio_value * (1 - crash_pct / 100)
        recovery = fv_lumpsum(after, equity_return, recovery_years)
        gap = max(0.0, target_corpus - fv_lumpsum(after, equity_return, max(recovery_years, years_to_target)))
        return {"scenario": f"{crash_pct:.0f}% Market Crash",
                "portfolio_after": round(after),
                "projected_recovery": round(recovery),
                "gap_to_target": round(gap),
                "verdict": "Resilient" if gap == 0 else "Needs top-up"}

    def early_retirement(self, age, monthly_expense_today, current_corpus,
                         monthly_contrib, risk_level, target_age=55, life_exp=85) -> Dict:
        a = self.a
        ytr, yir = max(1, target_age - age), max(1, life_exp - target_age)
        eq = a.allocation_targets.get(risk_level, a.allocation_targets["moderate"])["equity"] / 100
        blend = eq * a.equity_return + (1 - eq) * a.debt_return
        future_exp = monthly_expense_today * (1 + a.general_inflation) ** ytr
        needed = inflation_adjusted_corpus(future_exp, yir, a.post_retirement_return, a.general_inflation)
        projected = fv_lumpsum(current_corpus, blend, ytr) + fv_sip(monthly_contrib, blend, ytr)
        gap = max(0.0, needed - projected)
        extra = sip_for_target(gap, blend, ytr) if gap > 0 else 0.0
        return {"scenario": f"Early Retirement at {target_age}",
                "corpus_needed": round(needed), "projected_corpus": round(projected),
                "gap": round(gap), "feasible": gap == 0,
                "extra_monthly_needed": round(extra)}

    def home_purchase(self, monthly_income, existing_emi, current_savings,
                      property_value_today, years=5) -> Dict:
        a = self.a
        prop_at_purchase = property_value_today * (1 + a.housing_inflation) ** years
        down = prop_at_purchase * a.down_payment_pct
        loan = prop_at_purchase - down
        emi = loan_emi(loan, a.home_loan_rate, a.home_loan_tenure_years * 12)
        savings_at_purchase = fv_lumpsum(current_savings, a.debt_return, years)
        need = max(0.0, down - savings_at_purchase)
        sip = sip_for_target(need, a.debt_return, years) if need > 0 else 0.0
        emi_ratio = ((emi + existing_emi) / monthly_income * 100) if monthly_income > 0 else 100.0
        return {"property_value_today": round(property_value_today),
                "property_value_at_purchase": round(prop_at_purchase),
                "down_payment": round(down), "loan_amount": round(loan),
                "estimated_emi": round(emi),
                "emi_to_income_pct": round(emi_ratio, 1),
                "savings_gap_at_purchase": round(need),
                "monthly_sip_for_downpayment": round(sip),
                "affordable": emi_ratio <= a.max_emi_to_income * 100,
                "note": (f"Property inflated at {a.housing_inflation*100:.0f}%/yr to purchase date; "
                         f"loan at {a.home_loan_rate*100:.1f}% for {a.home_loan_tenure_years}y; "
                         "affordability includes existing EMIs.")}


# ============================================================================
# 14. FINANCIAL HEALTH SCORE
# ============================================================================

class FinancialHealthScorer:
    def score(self, emergency: Dict, savings_rate: float, retirement: Dict,
              debt: Dict, insurance: Dict, assets: Dict) -> Dict:
        components = {}
        components["emergency_fund"] = {
            "score": min(100.0, emergency["adequacy_percentage"]), "weight": 20,
            "label": emergency["status"]}
        components["savings_rate"] = {
            "score": min(100.0, max(0.0, savings_rate * 5)), "weight": 20,
            "label": f"{savings_rate:.1f}% (target ≥ 20%)"}
        components["retirement"] = {
            "score": retirement["readiness_percentage"], "weight": 20,
            "label": retirement["status"]}
        components["debt"] = {
            "score": max(0.0, 100 - debt["debt_to_income_ratio"] * 2), "weight": 15,
            "label": debt["status"]}
        components["insurance"] = {
            "score": min(100.0, insurance["life_adequacy_pct"]) * 0.6
                     + min(100.0, (insurance["health_insurance"] /
                                   max(1, insurance["recommended_health_cover"]) * 100)) * 0.4,
            "weight": 15,
            "label": f"Life adequacy {insurance['life_adequacy_pct']:.0f}%"}
        n_classes = sum(1 for i in assets.get("investments", []) if i["current_value"] > 0)
        components["investment_diversification"] = {
            "score": min(100.0, n_classes / 3 * 100), "weight": 10,
            "label": f"{n_classes} asset classes"}

        total = sum(v["score"] * v["weight"] / 100 for v in components.values())
        if total >= 80:   cat, desc, col = "Excellent", "Your finances are in great shape. Focus on optimisation.", "#1D8A4E"
        elif total >= 65: cat, desc, col = "Good", "Solid foundation with specific areas for improvement.", "#3E9E6B"
        elif total >= 50: cat, desc, col = "Fair", "Functional but with notable gaps that need attention.", "#C99700"
        elif total >= 35: cat, desc, col = "Needs Work", "Several critical areas require immediate action.", "#C97B3D"
        else:             cat, desc, col = "Critical", "Urgent financial attention required across multiple areas.", "#C94F4F"
        for v in components.values():
            v["score"] = round(v["score"], 1)
        return {"total_score": round(total, 1), "category": cat,
                "description": desc, "color": col, "components": components}


# ============================================================================
# 15. RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    def generate(self, emergency, retirement, debt, tax, goals, insurance,
                 rebalance, savings_rate, monthly_surplus) -> List[Dict]:
        recs = []
        if emergency["shortfall"] > 0:
            recs.append({"title": "Build Emergency Fund", "priority": "critical",
                "description": (f"Your emergency fund covers only {emergency['months_coverage']} months "
                                f"of obligations. Target: {emergency['target_months']} months "
                                f"(₹{emergency['required_fund']:,.0f})."),
                "actions": [f"Park ₹{emergency['shortfall']:,.0f} more in liquid funds or a sweep-in FD.",
                            f"Automate ₹{max(1, round(emergency['shortfall'] / 6)):,.0f}/month until the target is met.",
                            "Keep this money separate from goal investments — it is insurance, not returns."],
                "timeline": "3–6 months", "impact": "High — protects against income disruption"})

        for w in debt.get("warnings", []):
            recs.append({"title": "Fix Negative-Amortisation Loan", "priority": "critical",
                "description": w,
                "actions": ["Increase the EMI above the monthly interest immediately, or",
                            "Refinance/restructure the loan at a lower rate."],
                "timeline": "This month", "impact": "Critical — balance is growing, not shrinking"})

        hi = [d for d in debt.get("priority_payoff_order", [])
              if d["rate"] >= ASSUMPTIONS.high_interest_threshold]
        if hi:
            names = ", ".join(d["name"] for d in hi[:3])
            recs.append({"title": "Pay Down High-Interest Debt",
                "priority": "critical" if len(hi) > 1 else "high",
                "description": f"High-interest loans ({names}) are eroding wealth. "
                               "Use the avalanche method: highest rate first.",
                "actions": ["Direct all surplus cash to the highest-rate loan first.",
                            "Consider a balance transfer for credit-card debt.",
                            "Prepaying a loan at 14% is a GUARANTEED 14% post-tax return — "
                            "better than any market expectation."],
                "timeline": "12–36 months", "impact": "High — guaranteed return equal to the rate saved"})

        if retirement.get("additional_monthly_saving_needed", 0) > 0:
            extra = retirement["additional_monthly_saving_needed"]
            recs.append({"title": "Boost Retirement Savings", "priority": "high",
                "description": f"You need ₹{extra:,.0f}/month extra to fund an inflation-adjusted retirement.",
                "actions": [f"Increase your SIP by ₹{extra:,.0f}/month (allocation per your risk profile).",
                            "Step up SIPs by 10% annually as income grows.",
                            "Max NPS 80CCD(1B) if you file under the old regime."],
                "timeline": "Start immediately",
                "impact": "Critical — each year of delay compounds the required SIP"})

        if tax["better_regime"] == "Old Regime" and tax.get("total_deduction_gap", 0) > 10_000:
            recs.append({"title": "Optimise Old-Regime Deductions", "priority": "high",
                "description": (f"The old regime is better for you. You still have "
                                f"₹{tax['total_deduction_gap']:,.0f} of unused deduction headroom."),
                "actions": [f"80C gap ₹{tax['80c_gap']:,.0f}: ELSS / PPF / EPF top-up.",
                            f"80D gap ₹{tax['80d_gap']:,.0f}: health-insurance premium.",
                            f"NPS 80CCD(1B) gap ₹{tax['nps_gap']:,.0f}."],
                "timeline": "Before March 31",
                "impact": f"Old-regime tax: ₹{tax['old_regime_tax']:,.0f}/yr"})
        elif tax["savings_by_switching"] > 5_000:
            recs.append({"title": f"File Under the {tax['better_regime']}", "priority": "high",
                "description": (f"Switching saves ₹{tax['savings_by_switching']:,.0f}/year. "
                                + ("Note: 80C/80D investments give NO tax benefit under the new regime — "
                                   "invest for goals, not for Section 80C." if tax["better_regime"] == "New Regime" else "")),
                "actions": [f"Declare the {tax['better_regime']} to your employer for TDS.",
                            "Re-check every year — the better regime changes with income and deductions."],
                "timeline": "Next payroll declaration",
                "impact": f"Save ₹{tax['savings_by_switching']:,.0f}/year"})

        if insurance["life_cover_gap"] > 0:
            recs.append({"title": "Close the Life-Insurance Gap", "priority": "high",
                "description": (f"Needs-based required cover: ₹{insurance['recommended_life_cover']:,.0f}. "
                                f"Gap: ₹{insurance['life_cover_gap']:,.0f}."),
                "actions": ["Buy a pure TERM plan for the gap (cheapest per crore of cover).",
                            "Avoid ULIPs/endowment plans — keep insurance and investment separate."],
                "timeline": "1 month", "impact": "Protects dependents from income loss"})
        if insurance["health_cover_gap"] > 0:
            recs.append({"title": "Increase Health Cover", "priority": "high",
                "description": f"Recommended: ₹{insurance['recommended_health_cover']:,.0f}. "
                               f"Gap: ₹{insurance['health_cover_gap']:,.0f}.",
                "actions": ["Add a super top-up policy — far cheaper than raising the base cover."],
                "timeline": "1–2 months", "impact": "One hospitalisation can erase years of savings"})

        if rebalance.get("needs_rebalancing"):
            recs.append({"title": "Rebalance Portfolio", "priority": "medium",
                "description": "Allocation has drifted more than 5% from your target.",
                "actions": [f"{a['action']} {a['asset']} by ₹{a['amount']:,.0f}"
                            for a in rebalance.get("actions", [])] +
                           ["Prefer redirecting new SIPs over selling (avoids capital-gains tax)."],
                "timeline": "This month", "impact": "Maintains your intended risk level"})

        at_risk = [g for g in goals.get("goals_details", []) if g["status"] == "🔴 Needs Attention"]
        if at_risk:
            recs.append({"title": "Goals Needing Attention: " + ", ".join(g["name"] for g in at_risk[:3]),
                "priority": "medium",
                "description": f"Combined shortfall SIP: ₹{sum(g['monthly_saving_needed'] for g in at_risk):,.0f}/month.",
                "actions": [f"'{g['name']}': ₹{g['monthly_saving_needed']:,.0f}/mo in {g['asset_mix']}"
                            for g in at_risk[:3]] +
                           ["If the surplus can't cover all goals, extend timelines or trim targets — "
                            "don't reach for extra risk on short-horizon goals."],
                "timeline": "Start this month", "impact": "Avoid shortfalls at critical milestones"})

        if savings_rate < 20:
            recs.append({"title": "Raise Savings Rate to 20%+", "priority": "medium",
                "description": f"Current post-tax savings rate: {savings_rate:.1f}%.",
                "actions": ["Automate investments on salary day (pay yourself first).",
                            "Audit subscriptions and discretionary spending.",
                            "Apply the 50-30-20 rule as a floor, not a ceiling."],
                "timeline": "Ongoing", "impact": "Compounding multiplies every extra 1% saved"})

        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recs.sort(key=lambda x: order.get(x["priority"], 4))
        return recs


# ============================================================================
# 16. MASTER ANALYSIS RUNNER
# ============================================================================

def run_full_analysis(personal_info: Dict, income_data: Dict, assets_data: Dict,
                      debts: List[Dict], insurance: Dict, goals: List[Dict],
                      tax_inputs: Dict, risk_answers: Dict,
                      n_simulations: int = 1000) -> Dict:
    a = ASSUMPTIONS
    risk_level = personal_info.get("risk_level", "moderate")

    monthly_income = sum(income_data["income"].values())
    monthly_expenses = income_data["monthly_expenses"]          # strictly ex-EMI
    annual_income = monthly_income * 12
    total_emi = sum(d.get("emi", 0) for d in debts)

    # ── Tax first: the savings rate depends on it (v1 guessed 15% flat).
    tax = calculate_tax(annual_income, tax_inputs or {})
    monthly_tax = tax["annual_tax_best"] / 12
    post_tax_income = monthly_income - monthly_tax
    monthly_surplus = max(0.0, post_tax_income - monthly_expenses - total_emi)
    savings_rate = (monthly_surplus / post_tax_income * 100) if post_tax_income > 0 else 0.0

    income_stability = risk_answers.get("q4", 3)
    emergency = EmergencyFundAnalyser().analyse(
        monthly_expenses, assets_data["cash_and_bank"], debts,
        income_stability_score=income_stability)

    # Retirement uses financial assets (equity/debt/gold + EPF/PPF/NPS).
    # Real estate and cash are deliberately excluded (illiquid / emergency).
    current_ret_corpus = (assets_data.get("retirement_corpus", 0)
                          + sum(i["current_value"] for i in assets_data.get("investments", [])))
    monthly_inv = assets_data.get("monthly_investment", 0)

    retirement = RetirementPlanner(a).plan(
        age=personal_info["age"], retirement_age=personal_info["retirement_age"],
        life_expectancy=personal_info["life_expectancy"],
        monthly_expense_today=monthly_expenses,
        current_corpus=current_ret_corpus, monthly_contrib=monthly_inv,
        risk_level=risk_level)

    debt = DebtAnalyser().analyse(debts, monthly_income)
    goals_analysis = GoalPlanner(a).plan_goals(goals, monthly_surplus, risk_level)
    rebalancing = RebalancingEngine().analyse(assets_data.get("investments", []), risk_level)

    critical_goal_pv = sum(g["target_amount"] - g.get("current_saved", 0)
                           for g in goals if g.get("priority") == "critical")
    liquid = assets_data.get("total_investments", 0) + assets_data.get("cash_and_bank", 0)
    ins = InsuranceAnalyser().analyse(
        annual_income, debt["total_debt"], max(0.0, critical_goal_pv), liquid,
        insurance.get("life_insurance", 0), insurance.get("health_insurance", 0),
        personal_info.get("dependents", 0))

    monthly_oblig = monthly_expenses + total_emi
    sp = ScenarioPlanner(a)
    eq_pct = a.allocation_targets[risk_level]["equity"] / 100
    blend = eq_pct * a.equity_return + (1 - eq_pct) * a.debt_return
    total_portfolio = assets_data.get("total_investments", 0) + assets_data.get("retirement_corpus", 0)
    scenarios = {
        "job_loss_3m": sp.job_loss(monthly_income, monthly_oblig, assets_data["cash_and_bank"], 3),
        "job_loss_6m": sp.job_loss(monthly_income, monthly_oblig, assets_data["cash_and_bank"], 6),
        "crash_20": sp.market_crash(total_portfolio, 20, retirement["corpus_needed"],
                                    retirement["years_to_retirement"], blend),
        "crash_40": sp.market_crash(total_portfolio, 40, retirement["corpus_needed"],
                                    retirement["years_to_retirement"], blend),
        "early_ret_55": sp.early_retirement(personal_info["age"], monthly_expenses,
                                            current_ret_corpus, monthly_inv, risk_level,
                                            55, personal_info["life_expectancy"]),
    }

    mc = MonteCarloEngine(n_simulations=n_simulations)
    ret_mc = mc.lifecycle(current_corpus=current_ret_corpus, monthly_contrib=monthly_inv,
                          years_to_retire=retirement["years_to_retirement"],
                          years_in_retire=retirement["years_in_retirement"],
                          monthly_expense_today=monthly_expenses,
                          equity_pct_accum=eq_pct)
    ret_mc["required_corpus"] = retirement["corpus_needed"]

    goal_mc = []
    for gd in goals_analysis["goals_details"]:
        r = mc.goal_simulation(gd["current_saved"], gd["monthly_saving_needed"],
                               gd["timeframe_years"], gd["target_amount_fv"],
                               gd["equity_weight"])
        goal_mc.append({"goal_name": gd["name"], "target_amount": gd["target_amount_fv"],
                        "success_probability": round(r["success_probability"], 1),
                        "note": "Assumes the required SIP is actually invested."})

    health = FinancialHealthScorer().score(emergency, savings_rate, retirement, debt, ins, assets_data)
    recs = RecommendationEngine().generate(emergency, retirement, debt, tax,
                                           goals_analysis, ins, rebalancing,
                                           savings_rate, monthly_surplus)

    total_assets = (assets_data["cash_and_bank"] + assets_data.get("total_investments", 0)
                    + assets_data.get("retirement_corpus", 0)
                    + assets_data.get("real_estate_value", 0) + assets_data.get("other_assets", 0))
    return {"health_score": health, "emergency": emergency, "retirement": retirement,
            "debt": debt, "goals": goals_analysis, "tax": tax,
            "rebalancing": rebalancing, "scenarios": scenarios,
            "monte_carlo": {"retirement": ret_mc, "goals": goal_mc},
            "recommendations": recs, "insurance_analysis": ins,
            "risk_level": risk_level, "life_stage": personal_info.get("life_stage", ""),
            "summary": {"net_worth": round(total_assets - debt["total_debt"]),
                        "annual_income": round(annual_income),
                        "monthly_tax": round(monthly_tax),
                        "monthly_savings": round(monthly_surplus),
                        "savings_rate": round(savings_rate, 1),
                        "total_assets": round(total_assets),
                        "total_liabilities": round(debt["total_debt"])}}


# ============================================================================
# 17. SELF-TESTS (python financial_kundli_v2.py --selftest)
# ============================================================================

def run_self_tests() -> None:
    ok = lambda cond, msg: print(("PASS  " if cond else "FAIL  ") + msg) or (cond or sys.exit(1))

    # --- Tax: FY 2026-27 new regime, salaried 12.75L -> 0 (87A rebate)
    t = calculate_tax(1_275_000)
    ok(t["new_regime_tax"] == 0, f"New regime 12.75L gross -> 0 tax (got {t['new_regime_tax']})")
    # 15L salary new regime -> 97,500 + 4% cess = 101,400 (taxable 14.25L)
    t = calculate_tax(1_500_000)
    slab = _slab_tax(1_425_000, TAX_CONFIG["FY 2026-27"]["new"]["slabs"])
    ok(abs(slab - 93_750) < 1, f"Slab tax on 14.25L taxable = 93,750 (got {slab:,.0f})")
    ok(t["new_regime_tax"] == round(93_750 * 1.04), f"New regime 15L gross -> 97,500 incl cess, matches ITD example (got {t['new_regime_tax']:,})")
    # Marginal relief just above rebate threshold: taxable 12.1L -> pay <= 10,000+... = 10,000? tax capped at income above 12L
    t = calculate_tax(1_285_000)  # taxable 12.10L
    pre = t["new_regime_tax"] / 1.04
    ok(abs(pre - 10_000) < 2, f"87A marginal relief: taxable 12.10L -> 10,000 pre-cess (got {pre:,.0f})")
    # Old regime rebate: taxable <= 5L -> 0
    t = calculate_tax(540_000, {"sec_80c": 0})  # 5.40L - 50k std = 4.90L taxable
    ok(t["old_regime_tax"] == 0, f"Old regime taxable 4.9L -> 0 via 87A (got {t['old_regime_tax']})")
    # Surcharge: 60L taxable-ish old regime -> 10% surcharge, with marginal relief sanity
    t_hi = calculate_tax(6_050_000)
    t_lo = calculate_tax(5_049_999)
    ok(t_hi["old_regime_tax"] > t_lo["old_regime_tax"], "Tax is monotonic across the 50L surcharge threshold")
    t_just = calculate_tax(5_060_000)  # marginal relief: extra tax <= extra income
    extra_tax = t_just["old_regime_tax"] - t_lo["old_regime_tax"]
    ok(extra_tax <= 10_001 * 1.05, f"Surcharge marginal relief works (extra tax {extra_tax:,.0f} on 10k extra income)")

    # --- Annuity math
    emi = loan_emi(1_000_000, 0.085, 240)
    ok(abs(emi - 8678) < 5, f"EMI 10L @8.5% 20y = ~8,678 (got {emi:,.0f})")
    n = months_to_payoff(1_000_000, 0.085, emi)
    ok(n == 240, f"months_to_payoff inverts EMI (got {n})")
    ok(months_to_payoff(1_000_000, 0.24, 5_000) is None, "Negative amortisation detected")

    # --- Inflation-adjusted corpus is materially larger than v1's flat annuity
    flat = 100_000 * (1 - (1 + 0.075/12) ** (-300)) / (0.075/12)     # v1 formula
    real = inflation_adjusted_corpus(100_000, 25, 0.075, 0.06)
    ok(real > flat * 1.4, f"Real-return corpus {real/1e7:.2f}Cr vs v1 flat {flat/1e7:.2f}Cr (+{(real/flat-1)*100:.0f}%)")

    # --- SIP inversion
    fv = fv_sip(10_000, 0.12, 10)
    sip = sip_for_target(fv, 0.12, 10)
    ok(abs(sip - 10_000) < 0.01, "fv_sip and sip_for_target are inverses")

    # --- Portfolio vol with correlation < linear average
    ok(portfolio_vol(0.55) < 0.55 * ASSUMPTIONS.equity_vol + 0.45 * ASSUMPTIONS.debt_vol,
       "Correlated vol below linear blend")

    # --- Monte Carlo lifecycle sanity
    mc = MonteCarloEngine(400, seed=7)
    rich = mc.lifecycle(5_00_00_000, 100_000, 20, 25, 50_000, 0.55)
    poor = mc.lifecycle(0, 1_000, 20, 25, 100_000, 0.55)
    ok(rich["success_probability"] > 95, f"Well-funded plan succeeds (got {rich['success_probability']:.0f}%)")
    ok(poor["success_probability"] < 5, f"Unfunded plan fails (got {poor['success_probability']:.0f}%)")
    ok(len(rich["yearly_percentiles"]["p50"]) == 46, "Yearly path covers full lifecycle")

    # --- Goal glide path
    ok(goal_asset_mix(2, "aggressive")["equity_weight"] == 0.0, "2y goal -> 100% debt even for aggressive")
    ok(goal_asset_mix(10, "aggressive")["equity_weight"] == 0.75, "10y aggressive goal -> 75% equity")

    # --- Emergency fund: EMIs added exactly once
    e = EmergencyFundAnalyser().analyse(50_000, 300_000, [{"emi": 20_000}])
    ok(e["required_fund"] == 420_000, f"6 × (50k expenses + 20k EMI) = 4.2L (got {e['required_fund']:,})")

    # --- Full pipeline smoke test
    res = run_full_analysis(
        personal_info={"name": "Test", "age": 32, "retirement_age": 60,
                       "life_expectancy": 85, "dependents": 1,
                       "risk_level": "moderate", "life_stage": "mid_career"},
        income_data={"income": {"salary": 150_000}, "monthly_expenses": 70_000, "savings_rate": 0},
        assets_data={"cash_and_bank": 300_000, "retirement_corpus": 800_000,
                     "real_estate_value": 0, "other_assets": 0,
                     "monthly_investment": 25_000, "total_investments": 900_000,
                     "investments": [{"name": "Equity", "type": "equity", "current_value": 500_000},
                                     {"name": "Debt", "type": "debt", "current_value": 300_000},
                                     {"name": "Gold", "type": "gold", "current_value": 100_000}]},
        debts=[{"name": "Car Loan", "type": "car_loan", "outstanding_amount": 400_000,
                "interest_rate": 0.095, "emi": 12_000, "tenure_months": 48, "tenure_months_remaining": 40}],
        insurance={"life_insurance": 5_000_000, "health_insurance": 1_000_000},
        goals=[{"name": "House", "type": "home_purchase", "target_amount": 3_000_000,
                "timeframe_years": 6, "priority": "high", "current_saved": 400_000},
               {"name": "Car", "type": "vehicle", "target_amount": 800_000,
                "timeframe_years": 2, "priority": "medium", "current_saved": 100_000}],
        tax_inputs={"sec_80c": 150_000, "sec_80d": 25_000},
        risk_answers={"q1": 3, "q2": 3, "q3": 4, "q4": 4, "q5": 3},
        n_simulations=300)
    ok(0 <= res["health_score"]["total_score"] <= 100, "Health score in range")
    ok(res["summary"]["monthly_tax"] > 0, f"Real tax used in savings rate (₹{res['summary']['monthly_tax']:,}/mo)")
    ok(res["retirement"]["corpus_needed"] > 0, "Retirement corpus computed")
    ok(res["goals"]["goals_details"][1]["equity_weight"] == 0.0, "2y car goal planned in debt")
    ok(json.dumps(res, default=str) is not None, "Results JSON-serialisable")
    print("\nAll self-tests passed. Sample outputs:")
    print(f"  Retirement corpus needed : ₹{res['retirement']['corpus_needed']:,}")
    print(f"  Projected corpus         : ₹{res['retirement']['projected_corpus']:,}")
    print(f"  MC lifecycle success     : {res['monte_carlo']['retirement']['success_probability']:.1f}%")
    print(f"  Savings rate (post-tax)  : {res['summary']['savings_rate']}%")
    print(f"  Better regime            : {res['tax']['better_regime']} "
          f"(old ₹{res['tax']['old_regime_tax']:,} vs new ₹{res['tax']['new_regime_tax']:,})")
    print(f"  Health score             : {res['health_score']['total_score']} ({res['health_score']['category']})")


# ############################################################################
#                             STREAMLIT UI LAYER
#  (lazy imports so the engine above is importable/testable without streamlit)
# ############################################################################

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False

# ── Design tokens: "financial almanac" — ink indigo + gold on warm ivory ──
INK    = "#16263D"   # deep indigo ink (display type, headers)
ROYAL  = "#0057B8"   # brand royal blue (interactive, series 1)
GOLD   = "#C29A3B"   # burnished gold (accents, targets, medians)
GREEN  = "#1D8A4E"   # positive
RED    = "#C94F4F"   # negative
BLUE   = "#0072CE"   # series 2
ORANGE = "#B7791F"   # caution / annotations
IVORY  = "#FBF9F3"   # paper
HAIR   = "#E7E1D2"   # hairline rules on ivory
MUTED  = "#6B7688"   # secondary text
PIE_COLORS = ["#16263D", "#0057B8", "#C29A3B", "#1D8A4E", "#7A8CB0",
              "#C97B3D", "#8A5A83", "#4E7F8C"]

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* ── Base: warm ivory paper, ink typography ─────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #FBF9F3;
    color: #3E4C63;
}
.block-container { padding: 1.2rem 2rem 2.5rem; max-width: 1180px; }
h1, h2, h3 { font-family: 'Fraunces', serif; color: #16263D; }

/* ── Hero ───────────────────────────────────────────────────────────── */
.hero-wrap { text-align:center; padding: 0.6rem 0 0.2rem; }
.hero-emblem { margin-bottom: 0.4rem; }
.hero-title {
    font-family: 'Fraunces', serif; font-weight: 600;
    font-size: clamp(2rem, 4.5vw, 3.2rem);
    color: #16263D; letter-spacing: 0.14em; margin: 0;
}
.hero-title .gold { color: #C29A3B; }
.hero-sub {
    color: #6B7688; font-size: 0.82rem; text-transform: uppercase;
    letter-spacing: 0.32em; margin-top: 0.45rem;
}
.hero-rule {
    width: 220px; height: 1px; margin: 1.1rem auto 0.2rem;
    background: linear-gradient(90deg, transparent, #C29A3B 25%, #C29A3B 75%, transparent);
}

/* ── Section titles: gold diamond + serif + hairline ────────────────── */
.section-title {
    font-family: 'Fraunces', serif; font-weight: 600;
    font-size: 1.45rem; color: #16263D;
    margin: 1.3rem 0 0.9rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #E7E1D2; position: relative;
}
.section-title::after {
    content:""; position:absolute; left:0; bottom:-1px;
    width:64px; height:2px; background:#C29A3B;
}

/* ── Cards ──────────────────────────────────────────────────────────── */
.card {
    background: #FFFFFF; border: 1px solid #E7E1D2; border-top: 2px solid #C29A3B;
    border-radius: 10px; box-shadow: 0 1px 0 rgba(22,38,61,0.04), 0 6px 18px rgba(22,38,61,0.05);
    padding: 0.9rem 1.1rem; transition: transform .15s ease, box-shadow .15s ease;
}
.card:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(22,38,61,0.09); }
.card-gold {
    background: linear-gradient(160deg, #FFFFFF 0%, #F6F1E4 100%);
    border: 1px solid #D8C89B; border-radius: 12px; padding: 1.4rem 1.6rem;
    box-shadow: 0 8px 24px rgba(22,38,61,0.06);
}
.metric-label {
    font-size: 0.68rem; color: #8A93A5; text-transform: uppercase;
    letter-spacing: 0.14em; margin-bottom: 0.25rem; font-weight: 600;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    font-size: 1.42rem; color: #16263D; line-height: 1.15;
    font-variant-numeric: tabular-nums;
}
.metric-delta-pos { color: #1D8A4E; font-size: 0.78rem; margin-top: 0.2rem; }
.metric-delta-neg { color: #C94F4F; font-size: 0.78rem; margin-top: 0.2rem; }

/* ── Scenario cards ─────────────────────────────────────────────────── */
.scenario-card {
    background: #FFFFFF; border: 1px solid #E7E1D2; border-left: 3px solid #C29A3B;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}

/* ── Alert boxes ────────────────────────────────────────────────────── */
.info-box    { background:#F0F4FA; border-left:3px solid #0057B8; border-radius:0 8px 8px 0;
  padding:0.7rem 1rem; font-size:0.87rem; color:#2C4A75; margin:0.6rem 0; }
.success-box { background:#EDF6EF; border-left:3px solid #1D8A4E; border-radius:0 8px 8px 0;
  padding:0.7rem 1rem; font-size:0.87rem; color:#1B6B41; margin:0.6rem 0; }
.warning-box { background:#FAF3DC; border-left:3px solid #C99700; border-radius:0 8px 8px 0;
  padding:0.7rem 1rem; font-size:0.87rem; color:#7A5D00; margin:0.6rem 0; }
.error-box   { background:#FBEAE5; border-left:3px solid #C94F4F; border-radius:0 8px 8px 0;
  padding:0.7rem 1rem; font-size:0.87rem; color:#9C3A3A; margin:0.6rem 0; }

/* ── Pills ──────────────────────────────────────────────────────────── */
.pill { display:inline-block; padding:0.22rem 0.7rem; border-radius:3px; font-size:0.68rem;
  font-weight:700; letter-spacing:0.12em; text-transform:uppercase; }
.pill-blue   { background:#F0F4FA; color:#0057B8; border:1px solid #C4D4EA; }
.pill-green  { background:#EDF6EF; color:#1B6B41; border:1px solid #BCDCC8; }
.pill-orange { background:#FAF3DC; color:#7A5D00; border:1px solid #E4D194; }
.pill-red    { background:#FBEAE5; color:#9C3A3A; border:1px solid #EBC2BA; }

/* ── Sidebar: deep indigo panel ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16263D 0%, #0F1B2E 100%);
    border-right: 1px solid #C29A3B;
}
section[data-testid="stSidebar"] * { color: #D9DEE8; }
section[data-testid="stSidebar"] hr { border-color: rgba(194,154,59,0.25); }
.sidebar-title {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.45rem;
    color: #F4EFE2 !important; letter-spacing: 0.06em; margin-bottom: 0.1rem;
}
.sidebar-title .gold { color: #C29A3B !important; }
.sb-caption { font-size: 0.7rem; color: #8FA0BC !important; letter-spacing: 0.12em; text-transform: uppercase; }
section[data-testid="stSidebar"] .step-item { display:flex; align-items:center; gap:0.55rem;
  padding:0.28rem 0.2rem; font-size:0.86rem; border-radius:6px; }
section[data-testid="stSidebar"] .step-diamond { font-size:0.62rem; line-height:1; }
section[data-testid="stSidebar"] .step-done, section[data-testid="stSidebar"] .step-done * { color:#8FA0BC; }
section[data-testid="stSidebar"] .step-done .step-diamond { color:#C29A3B !important; }
section[data-testid="stSidebar"] .step-current, section[data-testid="stSidebar"] .step-current * { color:#F4EFE2; }
section[data-testid="stSidebar"] .step-current { font-weight:700; background:rgba(194,154,59,0.14);
  border-left:2px solid #C29A3B; padding-left:0.5rem; }
section[data-testid="stSidebar"] .step-current .step-diamond { color:#C29A3B !important; }
section[data-testid="stSidebar"] .step-todo, section[data-testid="stSidebar"] .step-todo * { color:#5E6E88; }
section[data-testid="stSidebar"] .step-todo .step-diamond { color:#3A4B66 !important; }
section[data-testid="stSidebar"] .sb-score { text-align:center; padding:0.6rem 0 0.2rem; }
section[data-testid="stSidebar"] .sb-score .num { font-family:'Fraunces',serif; font-size:2rem; font-weight:700; }
section[data-testid="stSidebar"] .sb-score .cat { font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid rgba(194,154,59,0.5);
    color: #E8DFC8 !important; border-radius: 6px; font-size: 0.82rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #C29A3B; background: rgba(194,154,59,0.12);
}
section[data-testid="stSidebar"] .stProgress > div > div { background: #C29A3B; }

/* ── Buttons (main area) ────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #16263D, #0057B8);
    color: #FFFFFF; font-weight: 700; border: none; border-radius: 8px;
    box-shadow: inset 0 -2px 0 rgba(194,154,59,0.9);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); }
.stButton > button[kind="secondary"] {
    background: #FFFFFF; border: 1px solid #D9D2BF; color: #3E4C63; border-radius: 8px;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid #E7E1D2; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px 8px 0 0; padding: 0.45rem 0.9rem;
    font-size: 0.82rem; color: #6B7688; letter-spacing: 0.02em;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #16263D !important; font-weight: 700;
    border: 1px solid #E7E1D2 !important; border-bottom: 2px solid #C29A3B !important;
}

/* ── Inputs / misc ──────────────────────────────────────────────────── */
.stProgress > div > div { background: #C29A3B; }
div[data-testid="stExpander"] { border: 1px solid #E7E1D2; border-radius: 10px; background: #FFFFFF; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
</style>
"""

def fmt(n) -> str:
    try:
        return f"₹{float(n):,.0f}"
    except (TypeError, ValueError):
        return "—"

def card_metric(label, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta:
        cls = "metric-delta-pos" if delta_pos else "metric-delta-neg"
        delta_html = f'<div class="{cls}">{"▲" if delta_pos else "▼"} {delta}</div>'
    return (f'<div class="card" style="margin-bottom:0.55rem"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{delta_html}</div>')

KUNDLI_EMBLEM = """
<svg width="56" height="56" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#C29A3B" stroke-width="2.4">
    <rect x="18" y="18" width="64" height="64" transform="rotate(45 50 50)"/>
    <rect x="32" y="32" width="36" height="36" transform="rotate(45 50 50)"/>
    <path d="M50 5 L50 27 M50 73 L50 95 M5 50 L27 50 M73 50 L95 50" stroke-width="1.6"/>
  </g>
  <circle cx="50" cy="50" r="4.5" fill="#C29A3B"/>
</svg>"""

def light_plotly(fig, height=350):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(color="#3E4C63", family="DM Sans", size=12),
                      title_font=dict(family="Fraunces, serif", size=16, color=INK),
                      xaxis=dict(gridcolor=HAIR, linecolor=HAIR, zerolinecolor=HAIR),
                      yaxis=dict(gridcolor=HAIR, linecolor=HAIR, zerolinecolor=HAIR),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      colorway=PIE_COLORS,
                      margin=dict(l=10, r=10, t=44, b=10))
    return fig


# ============================================================================
# INPUT SECTIONS
# ============================================================================

def section_risk_profiling():
    st.markdown('<div class="section-title">🧭 Risk Profile Questionnaire</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Answer honestly — this drives your recommended allocation and all simulations. '
                'Your final allocation is also capped by your risk <strong>capacity</strong> (time to retirement).</div>',
                unsafe_allow_html=True)
    answers = {}
    for q in RISK_QUESTIONS:
        st.markdown(f"**{q['text']}**")
        labels = [o[0] for o in q["options"]]
        stored = st.session_state.get(f"rq_{q['id']}")
        idx = labels.index(stored) if stored in labels else 0
        choice = st.radio(q["id"], labels, index=idx, key=f"rq_{q['id']}", label_visibility="collapsed")
        answers[q["id"]] = dict(q["options"])[choice]
        st.markdown("")
    risk_level, total_score, description = score_risk_profile(answers)
    col = {"conservative": BLUE, "moderate": GOLD, "aggressive": GREEN}[risk_level]
    st.markdown(f"""
    <div class="card-gold" style="text-align:center; margin-top:1rem">
        <div style="font-size:0.8rem; color:#8A93A5; letter-spacing:1px; text-transform:uppercase">Your Risk Profile</div>
        <div style="font-family:'Fraunces',serif; font-size:2.5rem; color:{col}; margin:0.25rem 0">{risk_level.title()}</div>
        <div style="color:#3E4C63; font-size:0.9rem">{description}</div>
        <div style="color:#8A93A5; font-size:0.8rem; margin-top:0.5rem">Score: {total_score}/{len(RISK_QUESTIONS)*4}</div>
    </div>""", unsafe_allow_html=True)
    return risk_level, total_score, answers


def section_personal_info():
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full Name", value=st.session_state.get("pi_name", ""), key="pi_name_w")
        age = st.number_input("Current Age", 18, 100, int(st.session_state.get("pi_age", 32)), 1, key="pi_age_w")
    with c2:
        ret_age = st.number_input("Planned Retirement Age", 40, 75, int(st.session_state.get("pi_ret_age", 60)), 1, key="pi_ret_age_w")
        dependents = st.number_input("Number of Dependents", 0, 10, int(st.session_state.get("pi_dep", 1)), 1, key="pi_dep_w")
    with c3:
        opts = ["Tier 1 (Metro)", "Tier 2", "Tier 3"]
        saved = st.session_state.get("pi_city", opts[0])
        city_tier = st.selectbox("City Tier", opts, index=opts.index(saved) if saved in opts else 0, key="pi_city_w")
        life_exp = st.number_input("Life Expectancy (for planning)", 70, 105, int(st.session_state.get("pi_le", 90)), 1,
                                   key="pi_le_w", help="Plan long: outliving your money is the bigger risk. 90+ recommended.")
    errors = []
    if ret_age <= age:
        errors.append("Retirement age must be greater than current age.")
    if life_exp <= ret_age:
        errors.append("Life expectancy must exceed retirement age.")
    for e in errors:
        st.markdown(f'<div class="error-box">⚠️ {e}</div>', unsafe_allow_html=True)

    stage = get_life_stage(age)
    labels = {LifeStage.EARLY_CAREER: ("🌱", "Early Career"), LifeStage.MID_CAREER: ("📈", "Mid Career"),
              LifeStage.PRE_RETIREMENT: ("⏳", "Pre-Retirement"), LifeStage.NEAR_RETIREMENT: ("🔑", "Near Retirement"),
              LifeStage.RETIRED: ("🏖️", "Retired")}
    icon, slabel = labels[stage]
    st.markdown(f'<div class="success-box">{icon} Life Stage: <strong>{slabel}</strong></div>', unsafe_allow_html=True)
    st.session_state.update(pi_name=name, pi_age=age, pi_ret_age=ret_age, pi_dep=dependents,
                            pi_city=city_tier, pi_le=life_exp)
    return {"name": name, "age": age, "retirement_age": ret_age, "dependents": dependents,
            "city_tier": city_tier, "life_expectancy": life_exp, "life_stage": stage.value,
            "valid": not errors}


def section_income_expenses():
    st.markdown('<div class="section-title">💰 Income & Expenses</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><strong>Do NOT include loan EMIs below.</strong> EMIs are captured in the '
                'Debts section and counted exactly once everywhere (v1 double-counted them in the emergency fund).</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Income Sources (Monthly, Gross)**")
        salary = st.number_input("Salary / Regular Income (₹)", 0, value=int(st.session_state.get("inc_sal", 100_000)), step=5_000, key="inc_sal_w")
        business = st.number_input("Business Income (₹)", 0, value=int(st.session_state.get("inc_biz", 0)), step=5_000, key="inc_biz_w")
        rental = st.number_input("Rental Income (₹)", 0, value=int(st.session_state.get("inc_ren", 0)), step=2_000, key="inc_ren_w")
        other_inc = st.number_input("Other Income (₹)", 0, value=int(st.session_state.get("inc_oth", 0)), step=2_000, key="inc_oth_w")
    with c2:
        st.markdown("**Monthly Expenses (excluding all EMIs)**")
        housing = st.number_input("Housing — rent & maintenance only (₹)", 0, value=int(st.session_state.get("exp_hse", 25_000)), step=1_000, key="exp_hse_w")
        groceries = st.number_input("Groceries & Food (₹)", 0, value=int(st.session_state.get("exp_gro", 15_000)), step=1_000, key="exp_gro_w")
        transport = st.number_input("Transport (₹)", 0, value=int(st.session_state.get("exp_trn", 6_000)), step=1_000, key="exp_trn_w")
        utilities = st.number_input("Utilities (₹)", 0, value=int(st.session_state.get("exp_utl", 4_000)), step=500, key="exp_utl_w")
        education = st.number_input("Education (₹)", 0, value=int(st.session_state.get("exp_edu", 8_000)), step=1_000, key="exp_edu_w")
        healthcare = st.number_input("Healthcare (₹)", 0, value=int(st.session_state.get("exp_hlt", 3_000)), step=500, key="exp_hlt_w")
        entertain = st.number_input("Entertainment (₹)", 0, value=int(st.session_state.get("exp_ent", 5_000)), step=500, key="exp_ent_w")
        other_exp = st.number_input("Other Expenses (₹)", 0, value=int(st.session_state.get("exp_oth", 8_000)), step=1_000, key="exp_oth_w")

    total_income = salary + business + rental + other_inc
    total_expenses = housing + groceries + transport + utilities + education + healthcare + entertain + other_exp

    # Real tax estimate from the tax engine (best regime, entered deductions if any)
    tax_prev = calculate_tax(total_income * 12, st.session_state.get("tax_inputs") or {})
    monthly_tax = tax_prev["annual_tax_best"] / 12
    total_emi = sum(d.get("emi", 0) for d in st.session_state.get("debts", []))
    post_tax = total_income - monthly_tax
    surplus = post_tax - total_expenses - total_emi
    savings_rate = (surplus / post_tax * 100) if post_tax > 0 else 0.0

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(card_metric("Monthly Income (Gross)", fmt(total_income),
                                     f"− {fmt(monthly_tax)} tax ({tax_prev['better_regime']})", delta_pos=False), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("Expenses + EMIs", fmt(total_expenses + total_emi),
                                     f"incl. {fmt(total_emi)} EMIs" if total_emi else None, delta_pos=False), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("Post-Tax Savings Rate", f"{savings_rate:.1f}%", "Target ≥ 20%",
                                     delta_pos=savings_rate >= 20), unsafe_allow_html=True)
    if surplus < 0:
        st.markdown('<div class="error-box">⚠️ Outgo exceeds post-tax income. Review the budget before planning goals.</div>', unsafe_allow_html=True)
    elif savings_rate < 10:
        st.markdown('<div class="warning-box">Savings rate below 10%. Aim for 20%+ for long-term wealth building.</div>', unsafe_allow_html=True)

    exp_data = {"Housing": housing, "Groceries": groceries, "Transport": transport, "Utilities": utilities,
                "Education": education, "Healthcare": healthcare, "Entertainment": entertain, "Other": other_exp}
    nz = {k: v for k, v in exp_data.items() if v > 0}
    if nz:
        fig = px.pie(names=list(nz.keys()), values=list(nz.values()), hole=0.55,
                     color_discrete_sequence=PIE_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig = light_plotly(fig, 280)
        fig.update_layout(title="Expense Breakdown (ex-EMI)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="expense_pie")

    st.session_state.update(inc_sal=salary, inc_biz=business, inc_ren=rental, inc_oth=other_inc,
                            exp_hse=housing, exp_gro=groceries, exp_trn=transport, exp_utl=utilities,
                            exp_edu=education, exp_hlt=healthcare, exp_ent=entertain, exp_oth=other_exp)
    return {"income": {"salary": salary, "business": business, "rental": rental, "other": other_inc},
            "monthly_expenses": total_expenses, "savings_rate": savings_rate}


def section_assets():
    st.markdown('<div class="section-title">🏦 Assets & Investments</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cash = st.number_input("Cash & Bank Balance (₹)", 0, value=int(st.session_state.get("a_cash", 200_000)), step=10_000, key="a_cash_w")
        re_v = st.number_input("Real Estate Value (₹)", 0, value=int(st.session_state.get("a_re", 2_000_000)), step=100_000, key="a_re_w",
                               help="Counted in net worth only — excluded from the retirement corpus (illiquid).")
    with c2:
        equity = st.number_input("Equity (Stocks/MFs) (₹)", 0, value=int(st.session_state.get("a_eq", 500_000)), step=25_000, key="a_eq_w")
        debt_v = st.number_input("Debt (FDs/Bonds) (₹)", 0, value=int(st.session_state.get("a_db", 300_000)), step=25_000, key="a_db_w")
        gold = st.number_input("Gold (₹)", 0, value=int(st.session_state.get("a_gd", 100_000)), step=10_000, key="a_gd_w")
    with c3:
        ret_c = st.number_input("EPF/PPF/NPS Corpus (₹)", 0, value=int(st.session_state.get("a_ret", 800_000)), step=50_000, key="a_ret_w")
        oth = st.number_input("Other Assets (₹)", 0, value=int(st.session_state.get("a_oth", 50_000)), step=10_000, key="a_oth_w")
    st.markdown("**Monthly SIPs / Contributions (incl. EPF)**")
    c1, c2, c3 = st.columns(3)
    with c1: m_eq = st.number_input("Equity SIP (₹)", 0, value=int(st.session_state.get("a_meq", 15_000)), step=1_000, key="a_meq_w")
    with c2: m_db = st.number_input("Debt SIP / EPF (₹)", 0, value=int(st.session_state.get("a_mdb", 5_000)), step=1_000, key="a_mdb_w")
    with c3: m_ot = st.number_input("Other SIP (₹)", 0, value=int(st.session_state.get("a_mot", 0)), step=1_000, key="a_mot_w")

    total_inv = equity + debt_v + gold
    total_mo = m_eq + m_db + m_ot
    total_all = cash + total_inv + ret_c + re_v + oth
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_metric("Investable Assets", fmt(total_inv), f"{fmt(total_mo)}/month SIP"), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("Total Assets (Gross)", fmt(total_all)), unsafe_allow_html=True)
    if total_inv > 0:
        fig = go.Figure(go.Bar(x=["Equity", "Debt", "Gold"],
                               y=[equity / total_inv * 100, debt_v / total_inv * 100, gold / total_inv * 100],
                               marker_color=[GREEN, BLUE, GOLD],
                               text=[f"{v/total_inv*100:.1f}%" for v in (equity, debt_v, gold)],
                               textposition="outside"))
        fig = light_plotly(fig, 220)
        fig.update_layout(title="Current Investment Allocation", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="asset_alloc")

    st.session_state.update(a_cash=cash, a_re=re_v, a_eq=equity, a_db=debt_v, a_gd=gold,
                            a_ret=ret_c, a_oth=oth, a_meq=m_eq, a_mdb=m_db, a_mot=m_ot)
    return {"cash_and_bank": cash,
            "investments": [{"name": "Equity", "type": "equity", "current_value": equity},
                            {"name": "Debt", "type": "debt", "current_value": debt_v},
                            {"name": "Gold", "type": "gold", "current_value": gold}],
            "retirement_corpus": ret_c, "real_estate_value": re_v, "other_assets": oth,
            "monthly_investment": total_mo, "total_investments": total_inv}


def section_debts():
    st.markdown('<div class="section-title">💳 Debts & Liabilities</div>', unsafe_allow_html=True)
    if "debts" not in st.session_state:
        st.session_state.debts = []
    with st.expander("➕ Add a Loan/Debt", expanded=not bool(st.session_state.debts)):
        c1, c2, c3 = st.columns(3)
        with c1:
            dname = st.text_input("Loan Name", key="d_name")
            dtype = st.selectbox("Type", ["Home Loan", "Car Loan", "Personal Loan", "Credit Card",
                                          "Education Loan", "Other"], key="d_type")
        with c2:
            d_out = st.number_input("Outstanding Amount (₹)", 0, value=0, step=10_000, key="d_out")
            d_rate = st.number_input("Interest Rate (% p.a.)", 0.0, 50.0, 10.0, 0.25, key="d_rate")
        with c3:
            d_emi = st.number_input("Monthly EMI (₹)", 0, value=0, step=500, key="d_emi")
        if st.button("Add Debt", type="primary", use_container_width=True):
            if dname and d_out > 0 and d_emi > 0:
                n = months_to_payoff(d_out, d_rate / 100, d_emi)
                st.session_state.debts.append({"name": dname, "type": dtype.lower().replace(" ", "_"),
                                               "outstanding_amount": d_out, "interest_rate": d_rate / 100,
                                               "emi": d_emi})
                if n is None:
                    st.warning(f"'{dname}': this EMI does not cover monthly interest — flagged in the analysis.")
                st.rerun()
            else:
                st.markdown('<div class="error-box">Fill name, outstanding amount and EMI.</div>', unsafe_allow_html=True)
    if st.session_state.debts:
        st.markdown("**Your Loans** (payoff time computed exactly from the amortisation schedule)")
        for i, d in enumerate(st.session_state.debts):
            n = months_to_payoff(d["outstanding_amount"], d["interest_rate"], d["emi"])
            hi = d["interest_rate"] >= ASSUMPTIONS.high_interest_threshold
            c1, c2, c3, c4, c5, c6 = st.columns([2, 1.4, 1, 1.2, 1.2, 0.5])
            with c1: st.write(f"**{d['name']}**"); st.caption(d["type"].replace("_", " ").title())
            with c2: st.write(fmt(d["outstanding_amount"])); st.caption("Outstanding")
            with c3: st.write(("🔴 " if hi else "") + f"{d['interest_rate']*100:.1f}%")
            with c4: st.write(fmt(d["emi"])); st.caption("EMI")
            with c5:
                st.write("∞ never" if n is None else f"{n} mo"); st.caption("To payoff")
            with c6:
                if st.button("✕", key=f"del_d_{i}"):
                    st.session_state.debts.pop(i); st.rerun()
        c1, c2 = st.columns(2)
        with c1: st.markdown(card_metric("Total Outstanding", fmt(sum(d["outstanding_amount"] for d in st.session_state.debts))), unsafe_allow_html=True)
        with c2: st.markdown(card_metric("Total Monthly EMI", fmt(sum(d["emi"] for d in st.session_state.debts))), unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">✅ No debts added — debt-free is great! You can proceed.</div>', unsafe_allow_html=True)
    return st.session_state.debts


def section_insurance():
    st.markdown('<div class="section-title">🛡️ Insurance Coverage</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        life = st.number_input("Life Insurance Sum Assured (₹)", 0, value=int(st.session_state.get("ins_life", 5_000_000)), step=500_000, key="ins_life_w")
        lp = st.number_input("Life Annual Premium (₹)", 0, value=int(st.session_state.get("ins_lp", 25_000)), step=1_000, key="ins_lp_w")
    with c2:
        hlth = st.number_input("Health Insurance Coverage (₹)", 0, value=int(st.session_state.get("ins_hlth", 1_000_000)), step=100_000, key="ins_hlth_w")
        hp = st.number_input("Health Annual Premium (₹)", 0, value=int(st.session_state.get("ins_hp", 15_000)), step=1_000, key="ins_hp_w")
    st.markdown('<div class="info-box">Adequacy is assessed needs-based in the Analysis step: income replacement '
                '+ outstanding debt + unfunded critical goals − liquid assets.</div>', unsafe_allow_html=True)
    st.session_state.update(ins_life=life, ins_lp=lp, ins_hlth=hlth, ins_hp=hp)
    return {"life_insurance": life, "life_premium": lp, "health_insurance": hlth, "health_premium": hp}


def section_goals():
    st.markdown('<div class="section-title">🎯 Financial Goals</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Enter targets in <strong>today\'s value</strong>. Each goal is inflated at the right '
                'rate AND invested per its horizon: &lt;3y → debt, 3–7y → blended, 7y+ → your risk-profile equity mix.</div>',
                unsafe_allow_html=True)
    INFL = {"emergency_fund": "6%", "retirement": "6%", "home_purchase": "8% (housing)",
            "education": "10% (education)", "vehicle": "6%", "vacation": "6%",
            "wedding": "7% (wedding)", "other": "6%"}
    if "goals" not in st.session_state:
        st.session_state.goals = []
    with st.expander("➕ Add a Financial Goal", expanded=not bool(st.session_state.goals)):
        c1, c2 = st.columns(2)
        with c1:
            gname = st.text_input("Goal Name", key="g_name")
            gtype = st.selectbox("Goal Type", list(INFL.keys()), key="g_type",
                                 format_func=lambda x: x.replace("_", " ").title())
            gtarget = st.number_input("Target Amount (₹) — Today's Value", 0, value=0, step=10_000, key="g_target")
        with c2:
            gyears = st.number_input("Timeframe (Years)", 1, 50, 5, key="g_years")
            gpri = st.selectbox("Priority", ["critical", "high", "medium", "low"], index=2, key="g_pri", format_func=str.title)
            gsaved = st.number_input("Already Saved (₹)", 0, value=0, step=5_000, key="g_saved")
        st.caption(f"Inflation applied: {INFL[gtype]} · Asset mix: "
                   f"{goal_asset_mix(gyears, st.session_state.get('risk_level', 'moderate'))['label']}")
        if st.button("Add Goal", type="primary", use_container_width=True):
            if gname and gtarget > 0:
                st.session_state.goals.append({"name": gname, "type": gtype, "target_amount": gtarget,
                                               "timeframe_years": gyears, "priority": gpri, "current_saved": gsaved})
                st.rerun()
            else:
                st.markdown('<div class="error-box">Enter goal name and target amount.</div>', unsafe_allow_html=True)
    if st.session_state.goals:
        for i, g in enumerate(st.session_state.goals):
            pct = g["current_saved"] / g["target_amount"] * 100 if g["target_amount"] > 0 else 0
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1, 1, 0.5])
            with c1: st.write(f"**{g['name']}**"); st.caption(g["type"].replace("_", " ").title())
            with c2: st.write(fmt(g["target_amount"])); st.caption("Today's value")
            with c3: st.write(f"{g['timeframe_years']}y"); st.caption(g["priority"].title())
            with c4: st.progress(min(1.0, pct / 100)); st.caption(f"{pct:.0f}%")
            with c5:
                if st.button("✕", key=f"del_g_{i}"):
                    st.session_state.goals.pop(i); st.rerun()
    else:
        st.markdown('<div class="info-box">No goals yet. Add at least one to enable goal analysis.</div>', unsafe_allow_html=True)
    return st.session_state.goals


def section_tax():
    st.markdown(f'<div class="section-title">📋 Tax Planning — {DEFAULT_TAX_YEAR}</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Old vs New regime comparison with Section 87A rebates, surcharge tiers and '
                'marginal relief. <strong>Deductions below only apply under the old regime</strong> — the new regime '
                'ignores them (except the ₹75,000 standard deduction, applied automatically).</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        s80c = st.number_input("Section 80C Investments (₹)", 0, 150_000, int(st.session_state.get("tx_80c", 100_000)), 5_000, key="tx_80c_w")
        s80d = st.number_input("Section 80D Health Premiums (₹)", 0, 100_000, int(st.session_state.get("tx_80d", 15_000)), 1_000, key="tx_80d_w")
        nps = st.number_input("NPS 80CCD(1B) extra (₹)", 0, 50_000, int(st.session_state.get("tx_nps", 0)), 5_000, key="tx_nps_w")
    with c2:
        hl = st.number_input("Home Loan Interest — Sec 24(b) (₹)", 0, 200_000, int(st.session_state.get("tx_hl", 0)), 5_000, key="tx_hl_w")
        hra = st.number_input("HRA Exemption claimed (₹)", 0, 2_000_000, int(st.session_state.get("tx_hra", 0)), 5_000, key="tx_hra_w")
        oth = st.number_input("Other Deductions (₹)", 0, 500_000, int(st.session_state.get("tx_oth", 0)), 5_000, key="tx_oth_w")
    deductions = {"sec_80c": s80c, "sec_80d": s80d, "nps_80ccd1b": nps,
                  "home_loan_int": hl, "hra_exemption": hra, "other_deductions": oth}
    ai = st.session_state.get("income_data") or {}
    minc = sum(ai.get("income", {}).values()) if ai else 0
    if minc > 0:
        p = calculate_tax(minc * 12, deductions)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(card_metric("Old Regime (Annual)", fmt(p["old_regime_tax"]), f"Eff. {p['effective_rate_old']:.1f}%"), unsafe_allow_html=True)
        with c2: st.markdown(card_metric("New Regime (Annual)", fmt(p["new_regime_tax"]), f"Eff. {p['effective_rate_new']:.1f}%"), unsafe_allow_html=True)
        with c3: st.markdown(card_metric("Better Choice", p["better_regime"], f"Save {fmt(p['savings_by_switching'])}"), unsafe_allow_html=True)
        if p["better_regime"] == "New Regime":
            st.markdown('<div class="warning-box">The new regime is better for you: further 80C/80D investing gives '
                        '<strong>no tax benefit</strong>. Invest for goals, not for tax sections.</div>', unsafe_allow_html=True)
    st.session_state.update(tx_80c=s80c, tx_80d=s80d, tx_nps=nps, tx_hl=hl, tx_hra=hra, tx_oth=oth)
    return deductions


# ============================================================================
# RESULT DISPLAYS
# ============================================================================

def display_health_score(h: Dict):
    st.markdown('<div class="section-title">❤️ Financial Health Score</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=h["total_score"],
            number={"suffix": "/100", "font": {"size": 36, "color": h["color"]}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#8A93A5"},
                   "bar": {"color": h["color"], "thickness": 0.3}, "bgcolor": "#FFF", "borderwidth": 0,
                   "steps": [{"range": [0, 50], "color": "#F6E4DE"}, {"range": [50, 65], "color": "#F6EFD8"},
                             {"range": [65, 100], "color": "#E6F0E4"}],
                   "threshold": {"line": {"color": h["color"], "width": 3}, "thickness": 0.8, "value": h["total_score"]}}))
        fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3E4C63", family="DM Sans"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="health_gauge")
        st.markdown(f'<div style="text-align:center; font-family:\'Fraunces\',serif; font-size:1.4rem; '
                    f'color:{h["color"]}">{h["category"]}</div>'
                    f'<div style="text-align:center; color:#8A93A5; font-size:0.85rem; margin-top:0.3rem">{h["description"]}</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown("**Component Breakdown**")
        for k, v in h["components"].items():
            label = k.replace("_", " ").title()
            bar = GREEN if v["score"] >= 70 else (ORANGE if v["score"] >= 45 else RED)
            st.markdown(f"""
            <div style="margin-bottom:0.5rem">
              <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#3E4C63; margin-bottom:2px">
                <span>{label} <span style="color:#8A93A5">(wt {v['weight']}%)</span></span>
                <span style="color:{bar}; font-weight:600">{v['score']:.0f}</span></div>
              <div style="background:#E7E1D2; border-radius:4px; height:6px">
                <div style="background:{bar}; width:{min(100, v['score'])}%; height:6px; border-radius:4px"></div></div>
              <div style="font-size:0.72rem; color:#8A93A5; margin-top:2px">{v['label']}</div></div>""",
                        unsafe_allow_html=True)


def display_tax(tax: Dict):
    st.markdown(f'<div class="section-title">📋 Tax Analysis — {tax["fy"]}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(card_metric("Old Regime Tax", fmt(tax["old_regime_tax"]), f"Eff. {tax['effective_rate_old']:.1f}% · taxable {fmt(tax['old_taxable'])}"), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("New Regime Tax", fmt(tax["new_regime_tax"]), f"Eff. {tax['effective_rate_new']:.1f}% · taxable {fmt(tax['new_taxable'])}"), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("Better Choice", tax["better_regime"], f"Save {fmt(tax['savings_by_switching'])}"), unsafe_allow_html=True)
    if tax["better_regime"] == "Old Regime" and tax["total_deduction_gap"] > 0:
        st.markdown(f"""<div class="warning-box">⚡ <strong>Untapped old-regime deductions</strong> — ₹{tax['total_deduction_gap']:,.0f} unused:
        80C ₹{tax['80c_gap']:,.0f} · 80D ₹{tax['80d_gap']:,.0f} · NPS ₹{tax['nps_gap']:,.0f}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">{tax["note"]}</div>', unsafe_allow_html=True)


def display_monte_carlo(mc: Dict, ret: Dict):
    st.markdown('<div class="section-title">🎲 Monte Carlo — Lifecycle Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">1,000 correlated equity/debt paths through BOTH accumulation and retirement, '
                'with withdrawals growing at inflation. <strong>Success = the corpus never runs out before life '
                'expectancy</strong> — a stricter, more honest test than "hit a number at retirement".</div>',
                unsafe_allow_html=True)
    r = mc["retirement"]
    prob = r["success_probability"]
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(card_metric("Plan Success Probability", f"{prob:.1f}%",
                                     "Corpus survives to life expectancy", delta_pos=prob >= 75), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("Median Corpus at Retirement", fmt(r["corpus_at_retirement"]["p50"])), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("90% Range at Retirement",
                                     f"{fmt(r['corpus_at_retirement']['p5'])} – {fmt(r['corpus_at_retirement']['p95'])}"), unsafe_allow_html=True)
    yp = r["yearly_percentiles"]
    fig = go.Figure()
    for path in r["sample_paths"]:
        fig.add_trace(go.Scatter(x=yp["years"], y=path, mode="lines",
                                 line=dict(color="rgba(22,38,61,0.055)", width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yp["years"], y=yp["p5"], mode="lines", line=dict(color=RED, width=2, dash="dash"), name="5th pct"))
    fig.add_trace(go.Scatter(x=yp["years"], y=yp["p50"], mode="lines", line=dict(color=GOLD, width=3), name="Median"))
    fig.add_trace(go.Scatter(x=yp["years"], y=yp["p95"], mode="lines", line=dict(color=GREEN, width=2, dash="dash"), name="95th pct"))
    fig.add_vline(x=r["retirement_year_index"], line_dash="dot", line_color=ORANGE,
                  annotation_text="Retirement", annotation_font_color=ORANGE)
    fig = light_plotly(fig, 400)
    fig.update_layout(title="Portfolio Value — Accumulation & Drawdown (1,000 paths)",
                      xaxis_title="Years from Now", yaxis_title="Portfolio Value (₹)")
    st.plotly_chart(fig, use_container_width=True, key="mc_paths")

    if mc.get("goals"):
        st.markdown("**Goal Achievement Probabilities** (assuming the recommended SIP is invested)")
        for gs in mc["goals"]:
            p = gs["success_probability"]
            col = GREEN if p >= 70 else (ORANGE if p >= 40 else RED)
            c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1.5])
            with c1: st.write(f"**{gs['goal_name']}**")
            with c2: st.write(fmt(gs["target_amount"]))
            with c3: st.markdown(f'<span style="color:{col}; font-weight:700">{p:.1f}%</span>', unsafe_allow_html=True)
            with c4: st.write("✅ Likely" if p >= 70 else ("⚠️ Uncertain" if p >= 40 else "❌ Unlikely"))


def display_rebalancing(rebal: Dict, risk_level: str):
    st.markdown('<div class="section-title">⚖️ Portfolio Rebalancing</div>', unsafe_allow_html=True)
    cur, tgt = rebal["current_allocation"], rebal["target_allocation"]
    c1, c2 = st.columns(2)
    with c1:
        cats = ["Equity", "Debt", "Gold"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=cats, y=[cur["equity"], cur["debt"], cur["gold"]], marker_color=BLUE))
        fig.add_trace(go.Bar(name="Target", x=cats, y=[tgt["equity"], tgt["debt"], tgt["gold"]], marker_color=GOLD))
        fig = light_plotly(fig, 280)
        fig.update_layout(title=f"Current vs Target ({risk_level.title()} Profile)", barmode="group", yaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, key="rebal_chart")
    with c2:
        if rebal["needs_rebalancing"]:
            st.markdown('<div class="warning-box">⚖️ Allocation has drifted — rebalancing recommended.</div>', unsafe_allow_html=True)
            for a in rebal["actions"]:
                col = GREEN if a["action"] == "BUY" else RED
                st.markdown(f"""<div class="card" style="border-left:3px solid {col}; margin-bottom:0.5rem">
                    <span style="color:{col}; font-weight:700">{a['action']}</span> <strong>{a['asset']}</strong> — {fmt(a['amount'])}<br>
                    <span style="color:#8A93A5; font-size:0.8rem">{a['current_pct']}% → {a['target_pct']}% (drift {a['drift_pct']}%)</span></div>""",
                            unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">{rebal["note"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ Portfolio is within the 5% drift band. No action needed.</div>', unsafe_allow_html=True)


def display_scenarios(scenarios: Dict):
    st.markdown('<div class="section-title">🔮 Scenario Planning</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Job Loss", "Market Crash", "Early Retirement", "Home Purchase"])
    with tabs[0]:
        for key in ("job_loss_3m", "job_loss_6m"):
            s = scenarios[key]
            cls = {"Survivable": "green", "Stressful": "orange"}.get(s["verdict"], "red")
            st.markdown(f"""<div class="scenario-card"><div style="font-weight:600; color:#16263D">{s['scenario']}</div>
                <div style="color:#3E4C63; font-size:0.85rem; margin:0.3rem 0">Income lost: {fmt(s['income_lost'])} |
                Emergency fund covers {s['emergency_covers']} months of obligations | Shortfall: {fmt(s['fund_shortfall'])}</div>
                <span class="pill pill-{cls}">{s['verdict']}</span></div>""", unsafe_allow_html=True)
    with tabs[1]:
        for key in ("crash_20", "crash_40"):
            s = scenarios[key]
            ok = s["gap_to_target"] == 0
            st.markdown(f"""<div class="scenario-card"><div style="font-weight:600; color:#16263D">{s['scenario']}</div>
                <div style="color:#3E4C63; font-size:0.85rem; margin:0.3rem 0">Portfolio drops to {fmt(s['portfolio_after'])} |
                3-year recovery projection: {fmt(s['projected_recovery'])} | Gap to retirement target: {fmt(s['gap_to_target'])}</div>
                <span class="pill pill-{'green' if ok else 'orange'}">{s['verdict']}</span></div>""", unsafe_allow_html=True)
    with tabs[2]:
        s = scenarios["early_ret_55"]
        st.markdown(f"""<div class="scenario-card"><div style="font-weight:600; color:#16263D">{s['scenario']}</div>
            <div style="color:#3E4C63; font-size:0.85rem; margin:0.4rem 0">Corpus needed (inflation-adjusted):
            {fmt(s['corpus_needed'])} | Projected: {fmt(s['projected_corpus'])} | Gap: {fmt(s['gap'])}</div>
            <span class="pill pill-{'green' if s['feasible'] else 'red'}">{'Feasible' if s['feasible'] else 'Needs more savings'}</span></div>""",
                    unsafe_allow_html=True)
        if not s["feasible"]:
            st.markdown(f'<div class="warning-box">To retire at 55, increase SIP by <strong>{fmt(s["extra_monthly_needed"])}/month</strong>.</div>',
                        unsafe_allow_html=True)
    with tabs[3]:
        hp = next((g for g in st.session_state.get("goals", []) if g.get("type") == "home_purchase"), None)
        prop = hp["target_amount"] if hp else 8_000_000
        yrs = hp["timeframe_years"] if hp else 5
        ai = st.session_state.get("income_data") or {}
        minc = sum(ai.get("income", {}).values()) if ai else 100_000
        emi_now = sum(d.get("emi", 0) for d in st.session_state.get("debts", []))
        assets_d = st.session_state.get("assets_data") or {}
        saved = (hp.get("current_saved", 0) if hp else assets_d.get("cash_and_bank", 0))
        s = ScenarioPlanner().home_purchase(minc, emi_now, saved, prop, yrs)
        st.markdown(f"""<div class="scenario-card">
            <div style="font-weight:600; color:#16263D">Property {fmt(s['property_value_today'])} today →
            {fmt(s['property_value_at_purchase'])} in {yrs}y (housing inflation)</div>
            <div style="color:#3E4C63; font-size:0.85rem; margin:0.4rem 0">Down payment: {fmt(s['down_payment'])} |
            EMI: {fmt(s['estimated_emi'])}/mo | (EMI + existing EMIs)/income: {s['emi_to_income_pct']:.1f}%</div>
            {f"<div style='color:#B03030;font-size:0.85rem'>SIP needed for down payment: {fmt(s['monthly_sip_for_downpayment'])}/mo</div>" if s['savings_gap_at_purchase'] > 0 else ""}
            <span class="pill pill-{'green' if s['affordable'] else 'red'}">{'Affordable' if s['affordable'] else 'Stretch (EMI > 40% of income)'}</span></div>""",
                    unsafe_allow_html=True)
        st.caption(s["note"])


def display_recommendations(recs: List[Dict]):
    st.markdown('<div class="section-title">🎯 Priority Action Plan</div>', unsafe_allow_html=True)
    cfg = {"critical": ("🚨", "pill-red"), "high": ("⚠️", "pill-orange"),
           "medium": ("💡", "pill-blue"), "low": ("✅", "pill-green")}
    for i, r in enumerate(recs):
        icon, pill_cls = cfg.get(r["priority"], ("📌", "pill-blue"))
        with st.expander(f"{icon} {r['title']}", expanded=(i < 3)):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(r["description"])
                st.markdown("**Actions:**")
                for a in r["actions"]:
                    st.markdown(f"- {a}")
            with c2:
                st.markdown(f'<span class="pill {pill_cls}">{r["priority"].upper()}</span>', unsafe_allow_html=True)
                st.write(f"**Timeline:** {r['timeline']}")
                st.write(f"**Impact:** {r['impact']}")


def display_full_results(ar: Dict):
    st.markdown("---")
    s = ar["summary"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card_metric("Net Worth", fmt(s["net_worth"]), f"{fmt(s['monthly_savings'])}/mo surplus"), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("Annual Income", fmt(s["annual_income"]), f"−{fmt(s['monthly_tax'])}/mo tax", delta_pos=False), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("Savings Rate (post-tax)", f"{s['savings_rate']:.1f}%", "Target ≥ 20%", delta_pos=s["savings_rate"] >= 20), unsafe_allow_html=True)
    with c4: st.markdown(card_metric("Profile", ar["risk_level"].title(), ar.get("life_stage", "").replace("_", " ").title()), unsafe_allow_html=True)
    st.markdown("---")

    tabs = st.tabs(["🏥 Health", "📋 Tax", "🎲 Monte Carlo", "⚖️ Rebalance", "🔮 Scenarios",
                    "🚨 Emergency", "🏖 Retirement", "💳 Debt", "🎯 Goals", "🛡 Insurance", "💡 Actions"])
    with tabs[0]: display_health_score(ar["health_score"])
    with tabs[1]: display_tax(ar["tax"])
    with tabs[2]: display_monte_carlo(ar["monte_carlo"], ar["retirement"])
    with tabs[3]: display_rebalancing(ar["rebalancing"], ar["risk_level"])
    with tabs[4]: display_scenarios(ar["scenarios"])

    with tabs[5]:
        e = ar["emergency"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=e["adequacy_percentage"], number={"suffix": "%"},
                gauge={"axis": {"range": [0, 150]}, "bar": {"color": e["color"]},
                       "steps": [{"range": [0, 50], "color": "#F6E4DE"}, {"range": [50, 100], "color": "#F6EFD8"},
                                 {"range": [100, 150], "color": "#E6F0E4"}],
                       "threshold": {"line": {"color": GOLD, "width": 3}, "value": 100}}))
            fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3E4C63", family="DM Sans"), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="ef_gauge")
        with c2:
            st.markdown(card_metric("Current Fund", fmt(e["current_fund"])), unsafe_allow_html=True)
            st.markdown(card_metric("Required Fund", fmt(e["required_fund"]),
                                    f"{e['target_months']} months × {fmt(e['monthly_obligations'])} obligations"), unsafe_allow_html=True)
            st.markdown(card_metric("Coverage", f"{e['months_coverage']} months"), unsafe_allow_html=True)
            if e["shortfall"] > 0:
                st.markdown(card_metric("Shortfall", fmt(e["shortfall"]), delta_pos=False), unsafe_allow_html=True)
        st.markdown('<div class="info-box">Obligations = living expenses (ex-EMI) + loan EMIs, counted once. '
                    'Target rises to 9 months for unstable income.</div>', unsafe_allow_html=True)

    with tabs[6]:
        ret = ar["retirement"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=ret["readiness_percentage"], number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": ret["color"]},
                       "steps": [{"range": [0, 40], "color": "#F6E4DE"}, {"range": [40, 70], "color": "#F6EFD8"},
                                 {"range": [70, 100], "color": "#E6F0E4"}],
                       "threshold": {"line": {"color": GOLD, "width": 3}, "value": 70}}))
            fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3E4C63", family="DM Sans"), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="ret_gauge")
        with c2:
            st.markdown(card_metric("Years to Retire", f"{ret['years_to_retirement']}"), unsafe_allow_html=True)
            st.markdown(card_metric("Corpus Needed (inflation-adjusted)", fmt(ret["corpus_needed"])), unsafe_allow_html=True)
            st.markdown(card_metric("Projected Corpus", fmt(ret["projected_corpus"]),
                                    f"@ {ret['assumed_pre_retirement_return']*100:.1f}%/yr blended"), unsafe_allow_html=True)
            if ret["additional_monthly_saving_needed"] > 0:
                st.markdown(card_metric("Extra Monthly Needed", fmt(ret["additional_monthly_saving_needed"]), delta_pos=False), unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">📌 {ret["methodology"]}</div>', unsafe_allow_html=True)
        if ret["capacity_adjusted_risk"] != ar["risk_level"]:
            st.markdown(f'<div class="warning-box">Your questionnaire says <strong>{ar["risk_level"]}</strong>, but with '
                        f'{ret["years_to_retirement"]} years to retirement your risk <strong>capacity</strong> is '
                        f'{ret["capacity_adjusted_risk"]}. Consider de-risking as retirement approaches.</div>', unsafe_allow_html=True)

    with tabs[7]:
        d = ar["debt"]
        for w in d.get("warnings", []):
            st.markdown(f'<div class="error-box">🚨 {w}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=d["debt_to_income_ratio"], number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": d["color"]},
                       "steps": [{"range": [0, 20], "color": "#E6F0E4"}, {"range": [20, 40], "color": "#F6EFD8"},
                                 {"range": [40, 100], "color": "#F6E4DE"}],
                       "threshold": {"line": {"color": GOLD, "width": 3}, "value": 40}}))
            fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3E4C63", family="DM Sans"), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="dti_gauge")
        with c2:
            st.markdown(card_metric("Total Debt", fmt(d["total_debt"])), unsafe_allow_html=True)
            st.markdown(card_metric("Monthly EMI", fmt(d["monthly_emi"])), unsafe_allow_html=True)
            st.markdown(card_metric("High-Interest Loans", str(d["high_interest_debt_count"])), unsafe_allow_html=True)
            st.markdown(card_metric("Total Interest Payable", fmt(d["total_interest_payable"]), "exact amortisation"), unsafe_allow_html=True)
        if d["priority_payoff_order"]:
            st.markdown("**Avalanche Payoff Order** (highest rate first)")
            for p in d["priority_payoff_order"]:
                mo = "never at this EMI" if p["months_to_payoff"] is None else f"{p['months_to_payoff']} months"
                st.markdown(f"- **{p['name']}** — {p['rate']*100:.1f}% · {fmt(p['outstanding'])} outstanding · payoff in {mo}")

    with tabs[8]:
        g = ar["goals"]
        if g["total_goals"] == 0:
            st.info("No goals set.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(card_metric("Total Goals", str(g["total_goals"])), unsafe_allow_html=True)
            with c2: st.markdown(card_metric("Target (Today)", fmt(g["total_target_amount_pv"])), unsafe_allow_html=True)
            with c3: st.markdown(card_metric("Target (Future)", fmt(g["total_target_amount_fv"])), unsafe_allow_html=True)
            st.markdown(card_metric("Combined Monthly SIP Required", fmt(g["total_monthly_investment_needed"])), unsafe_allow_html=True)
            for goal in g["goals_details"]:
                with st.expander(f"{goal['name']} — {goal['status']}", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(card_metric("Today's Value", fmt(goal["target_amount_pv"])), unsafe_allow_html=True)
                        st.markdown(card_metric("Future Value", fmt(goal["target_amount_fv"]),
                                                f"@ {goal['inflation_used']}% inflation"), unsafe_allow_html=True)
                    with c2:
                        st.markdown(card_metric("Timeframe", f"{goal['timeframe_years']} yrs"), unsafe_allow_html=True)
                        st.markdown(card_metric("Asset Mix", goal["asset_mix"], f"~{goal['assumed_return']}% expected"), unsafe_allow_html=True)
                    with c3:
                        st.markdown(card_metric("Saved So Far", fmt(goal["current_saved"])), unsafe_allow_html=True)
                        st.markdown(card_metric("Monthly SIP Needed", fmt(goal["monthly_saving_needed"])), unsafe_allow_html=True)
                    st.progress(min(1.0, goal["progress_percentage"] / 100))
                    st.caption(f"Progress: {goal['progress_percentage']:.1f}% | Target year: {goal['completion_year']}")

    with tabs[9]:
        ins = ar["insurance_analysis"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card_metric("Current Life Cover", fmt(ins["life_insurance"])), unsafe_allow_html=True)
            st.markdown(card_metric("Needs-Based Requirement", fmt(ins["recommended_life_cover"])), unsafe_allow_html=True)
            if ins["life_cover_gap"] > 0:
                st.markdown(card_metric("Life Cover Gap", fmt(ins["life_cover_gap"]), delta_pos=False), unsafe_allow_html=True)
        with c2:
            st.markdown(card_metric("Current Health Cover", fmt(ins["health_insurance"])), unsafe_allow_html=True)
            st.markdown(card_metric("Recommended Health Cover", fmt(ins["recommended_health_cover"])), unsafe_allow_html=True)
            if ins["health_cover_gap"] > 0:
                st.markdown(card_metric("Health Cover Gap", fmt(ins["health_cover_gap"]), delta_pos=False), unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{ins["method"]}</div>', unsafe_allow_html=True)

    with tabs[10]:
        display_recommendations(ar["recommendations"])


# ============================================================================
# PDF REPORT
# ============================================================================

def _pdf_text(value) -> str:
    """PDF-safe text for ReportLab built-in fonts (no ₹ glyph in Helvetica)."""
    return (str(value).replace("₹", "Rs.").replace("—", "-").replace("–", "-")
            .replace("✅", "").replace("🟡", "").replace("🔴", "").strip())


def generate_pdf(personal_info: Dict, ar: Dict) -> bytes:
    """Branded 'financial almanac' report: ink-indigo header band with gold rule
    and kundli emblem, drawn score gauge + component bars, disciplined tables."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, KeepTogether)
    from reportlab.graphics.shapes import Drawing, Wedge, Circle, String, Rect

    C_INK   = colors.HexColor("#16263D")
    C_GOLD  = colors.HexColor("#C29A3B")
    C_IVORY = colors.HexColor("#FBF9F3")
    C_HAIR  = colors.HexColor("#E7E1D2")
    C_MUT   = colors.HexColor("#6B7688")
    C_TRACK = colors.HexColor("#EDE8DB")
    C_BODY  = colors.HexColor("#3E4C63")
    C_CREAM = colors.HexColor("#F4EFE2")
    STATUS  = {"green": colors.HexColor("#1D8A4E"), "amber": colors.HexColor("#C99700"),
               "red": colors.HexColor("#C94F4F")}

    W, H = letter
    client = _pdf_text(personal_info.get("name", "Client"))

    # ── page furniture: header band + emblem + footer ──────────────────────
    def page_decor(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_INK)
        canvas.rect(0, H - 58, W, 58, fill=1, stroke=0)
        canvas.setFillColor(C_GOLD)
        canvas.rect(0, H - 60, W, 2, fill=1, stroke=0)
        cx, cy, r = 68, H - 29, 11
        canvas.setStrokeColor(C_GOLD)
        canvas.setLineWidth(1.1)
        for rr in (r, r * 0.5):
            p = canvas.beginPath()
            p.moveTo(cx, cy + rr); p.lineTo(cx + rr, cy)
            p.lineTo(cx, cy - rr); p.lineTo(cx - rr, cy); p.close()
            canvas.drawPath(p, stroke=1, fill=0)
        canvas.setFillColor(C_GOLD)
        canvas.setFont("Times-Bold", 15)
        canvas.drawString(88, H - 34, "FINANCIAL KUNDLI")
        canvas.setFillColor(colors.HexColor("#8FA0BC"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(W - 60, H - 33, "F I N A N C I A L   P L A N N I N G   R E P O R T")
        canvas.setStrokeColor(C_HAIR)
        canvas.setLineWidth(0.6)
        canvas.line(60, 46, W - 60, 46)
        canvas.setFillColor(C_MUT)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(60, 34, f"Confidential - prepared for {client}")
        canvas.drawRightString(W - 60, 34, f"Page {doc.page}")
        canvas.restoreState()

    # ── drawn widgets ───────────────────────────────────────────────────────
    def score_gauge(score, hexcolor, size=124):
        d = Drawing(size, size)
        c = size / 2.0
        rad = c - 3
        d.add(Circle(c, c, rad, fillColor=C_TRACK, strokeColor=None))
        sweep = max(1.0, 360.0 * min(100.0, score) / 100.0)
        d.add(Wedge(c, c, rad, 90 - sweep, 90, fillColor=colors.HexColor(hexcolor), strokeColor=None))
        d.add(Circle(c, c, rad * 0.66, fillColor=colors.white, strokeColor=None))
        d.add(String(c, c - 4, f"{score:.0f}", fontName="Times-Bold", fontSize=28,
                     fillColor=C_INK, textAnchor="middle"))
        d.add(String(c, c - 17, "of 100", fontName="Helvetica", fontSize=7.5,
                     fillColor=C_MUT, textAnchor="middle"))
        return d

    def bar_color(score):
        return STATUS["green"] if score >= 70 else (STATUS["amber"] if score >= 45 else STATUS["red"])

    def component_bars(components, width=300):
        rows = list(components.items())
        rh = 27
        h = rh * len(rows)
        d = Drawing(width, h)
        for i, (k, v) in enumerate(rows):
            score = float(v["score"])
            col = bar_color(score)
            y = h - (i + 1) * rh + 7
            d.add(String(0, y + 9, k.replace("_", " ").title() + f"   wt {v['weight']}%",
                         fontName="Helvetica", fontSize=7.6, fillColor=C_INK))
            d.add(String(width, y + 9, f"{score:.0f}", fontName="Helvetica-Bold",
                         fontSize=8.4, fillColor=col, textAnchor="end"))
            d.add(Rect(0, y, width, 5, rx=2.5, ry=2.5, fillColor=C_TRACK, strokeColor=None))
            d.add(Rect(0, y, max(3.0, width * min(100.0, score) / 100.0), 5,
                       rx=2.5, ry=2.5, fillColor=col, strokeColor=None))
        return d

    # ── text styles ─────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    P = lambda text, style: Paragraph(_pdf_text(text), style)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=13,
                          textColor=C_BODY, spaceAfter=4)
    bold = ParagraphStyle("BOLD", parent=body, fontName="Helvetica-Bold", textColor=C_INK)
    note = ParagraphStyle("N", parent=styles["Normal"], fontSize=7.6, leading=10.5,
                          textColor=C_MUT, spaceAfter=3, alignment=TA_JUSTIFY)
    meta = ParagraphStyle("M", parent=styles["Normal"], fontSize=9, textColor=C_MUT)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Times-Bold",
                        fontSize=14, textColor=C_INK, spaceBefore=16, spaceAfter=1)
    kpi_label = ParagraphStyle("KL", parent=styles["Normal"], fontSize=6.6,
                               textColor=C_MUT, spaceAfter=2)
    kpi_value = ParagraphStyle("KV", parent=styles["Normal"], fontName="Times-Bold",
                               fontSize=14, textColor=C_INK, leading=16)
    cat_style = ParagraphStyle("CAT", parent=styles["Normal"], fontName="Times-Bold",
                               fontSize=17, leading=21, spaceAfter=5)

    def section(title):
        return [P(title, h2),
                HRFlowable(width=46, thickness=2, color=C_GOLD, hAlign="LEFT",
                           spaceBefore=1, spaceAfter=8)]

    def data_table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_CREAM),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7.8),
            ("LINEBELOW", (0, 0), (-1, 0), 1.4, C_GOLD),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_IVORY]),
            ("GRID", (0, 1), (-1, -1), 0.4, C_HAIR),
            ("FONTSIZE", (0, 1), (-1, -1), 8.2),
            ("TEXTCOLOR", (0, 1), (-1, -1), C_BODY),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (0, -1), 8),
        ]))
        return t

    PRIORITY_HEX = {"critical": "#C94F4F", "high": "#C99700",
                    "medium": "#0057B8", "low": "#1D8A4E"}

    # ── assemble ────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=60, leftMargin=60,
                            topMargin=84, bottomMargin=64,
                            title="Financial Kundli - Financial Planning Report",
                            author="Financial Kundli")
    s = ar["summary"]
    h = ar["health_score"]
    ret = ar["retirement"]
    story = []

    story.append(P(f"<b>Client:</b> {client} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Age:</b> "
                   f"{personal_info.get('age', 'N/A')} &nbsp;&nbsp;|&nbsp;&nbsp; "
                   f"<b>Tax Year:</b> {ar['tax']['fy']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                   f"<b>Date:</b> {datetime.now():%d %B %Y}", meta))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=C_HAIR, spaceAfter=2))

    # Health score panel: gauge + category on the left, component bars right
    story.extend(section("Financial Health"))
    cat_style.textColor = colors.HexColor(h["color"])
    left_cell = [score_gauge(h["total_score"], h["color"])]
    right_cell = [P(h["category"], cat_style), P(h["description"], body),
                  Spacer(1, 6), component_bars(h["components"], width=290)]
    panel = Table([[left_cell, right_cell]], colWidths=[150, 342])
    panel.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, C_HAIR),
        ("LINEABOVE", (0, 0), (-1, 0), 2, C_GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(panel)
    story.append(Spacer(1, 10))

    # KPI band
    mc_p = ar["monte_carlo"]["retirement"]["success_probability"]
    kpis = [("NET WORTH", f"Rs.{s['net_worth']:,.0f}"),
            ("SAVINGS RATE (POST-TAX)", f"{s['savings_rate']:.1f}%"),
            ("LIFECYCLE SUCCESS (MC)", f"{mc_p:.1f}%"),
            ("BETTER TAX REGIME", ar["tax"]["better_regime"])]
    kpi_cells = [[P(lbl, kpi_label), P(val, kpi_value)] for lbl, val in kpis]
    kpi = Table([kpi_cells], colWidths=[123] * 4)
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, C_HAIR),
        ("LINEABOVE", (0, 0), (-1, 0), 2, C_GOLD),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, C_HAIR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(kpi)

    # Key metrics table
    story.extend(section("Key Metrics"))
    story.append(data_table([
        ["Area", "Value", "Status"],
        ["Emergency Fund", f"{ar['emergency']['adequacy_percentage']:.1f}%", ar["emergency"]["status"]],
        ["Retirement Readiness", f"{ret['readiness_percentage']:.1f}%", ret["status"]],
        ["Lifecycle Success Probability", f"{mc_p:.1f}%", "Monte Carlo, 1,000 paths"],
        ["Debt-to-Income", f"{ar['debt']['debt_to_income_ratio']:.1f}%", ar["debt"]["status"]],
        ["Savings Rate (post-tax)", f"{s['savings_rate']:.1f}%",
         "Good" if s["savings_rate"] >= 20 else "Below 20% target"],
        ["Annual Tax (best regime)", f"Rs.{ar['tax']['annual_tax_best']:,.0f}",
         f"Saves Rs.{ar['tax']['savings_by_switching']:,.0f}/yr vs other"],
        ["Life Cover Gap", f"Rs.{ar['insurance_analysis']['life_cover_gap']:,.0f}", "Needs-based method"],
    ], [190, 132, 170]))

    # Retirement
    story.extend(section("Retirement Planning"))
    story.append(data_table([
        ["Horizon", "Corpus Needed", "Projected Corpus", "Extra SIP Needed"],
        [f"{ret['years_to_retirement']}y to retire / {ret['years_in_retirement']}y in retirement",
         f"Rs.{ret['corpus_needed']:,.0f}", f"Rs.{ret['projected_corpus']:,.0f}",
         f"Rs.{ret['additional_monthly_saving_needed']:,.0f}/mo"
         if ret["additional_monthly_saving_needed"] > 0 else "None - on track"],
    ], [170, 112, 112, 98]))
    story.append(Spacer(1, 4))
    story.append(P(ret["methodology"], note))

    # Goals
    if ar["goals"]["total_goals"] > 0:
        story.extend(section("Financial Goals"))
        mc_map = {g["goal_name"]: g["success_probability"] for g in ar["monte_carlo"]["goals"]}
        rows = [["Goal", "Today", "Future Value", "Infl.", "Asset Mix", "SIP / mo", "Success"]]
        for g in ar["goals"]["goals_details"]:
            rows.append([_pdf_text(g["name"]), f"Rs.{g['target_amount_pv']:,.0f}",
                         f"Rs.{g['target_amount_fv']:,.0f}", f"{g['inflation_used']}%",
                         _pdf_text(g["asset_mix"]), f"Rs.{g['monthly_saving_needed']:,.0f}",
                         f"{mc_map.get(g['name'], 0):.0f}%"])
        story.append(data_table(rows, [86, 62, 72, 34, 92, 64, 42]))

    # Recommendations
    story.extend(section("Priority Action Plan"))
    recs = ar.get("recommendations", [])
    if recs:
        for i, r in enumerate(recs[:6]):
            chip = PRIORITY_HEX.get(r["priority"], "#0057B8")
            head = (f'<font color="{chip}"><b>{r["priority"].upper()}</b></font>'
                    f'&nbsp;&nbsp;<font color="#16263D"><b>{i + 1}. {_pdf_text(r["title"])}</b></font>')
            block = [Paragraph(head, body), P(r["description"], body)]
            if r.get("actions"):
                for a_item in r["actions"][:3]:
                    block.append(P("-  " + a_item, note))
            block.append(Spacer(1, 6))
            story.append(KeepTogether(block))
    else:
        story.append(P("No priority recommendations generated.", body))

    # Assumptions + disclaimer
    story.append(Spacer(1, 8))
    story.extend(section("Key Assumptions"))
    a = ASSUMPTIONS
    story.append(P(
        f"Equity {a.equity_return*100:.0f}% return / {a.equity_vol*100:.0f}% volatility; "
        f"Debt {a.debt_return*100:.0f}% / {a.debt_vol*100:.1f}%; correlation {a.equity_debt_corr:.2f}. "
        f"Inflation: general {a.general_inflation*100:.0f}%, housing {a.housing_inflation*100:.0f}%, "
        f"education {a.education_inflation*100:.0f}%. Post-retirement return "
        f"{a.post_retirement_return*100:.1f}% at {a.post_retirement_equity*100:.0f}% equity. "
        f"These are long-run planning assumptions, not forecasts.", note))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.6, color=C_HAIR, spaceAfter=5))
    story.append(P("Disclaimer", ParagraphStyle("DT", parent=bold, fontSize=8.5,
                                                textColor=STATUS["red"])))
    story.append(P(
        "This report is generated by an automated financial planning tool and is for informational and "
        "educational purposes only. It does not constitute investment advice under the SEBI (Investment "
        "Advisers) Regulations, 2013, and the operator of this tool is not acting as your investment adviser "
        "unless separately registered and engaged. All projections are estimates based on stated assumptions "
        "and may differ materially from actual outcomes. Tax computations follow the rules for a resident "
        "individual below 60 with slab income only; verify with a qualified tax professional. Please consult "
        "a SEBI-registered Investment Adviser and a chartered accountant before acting on this report.", note))

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    buf.seek(0)
    return buf.getvalue()


# ============================================================================
# SESSION PERSISTENCE (restores widget state — v1 loaded data the widgets ignored)
# ============================================================================

# Widget-backing keys that hold user inputs, grouped for save/load.
_PERSIST_KEYS = [
    "pi_name", "pi_age", "pi_ret_age", "pi_dep", "pi_city", "pi_le",
    "inc_sal", "inc_biz", "inc_ren", "inc_oth",
    "exp_hse", "exp_gro", "exp_trn", "exp_utl", "exp_edu", "exp_hlt", "exp_ent", "exp_oth",
    "a_cash", "a_re", "a_eq", "a_db", "a_gd", "a_ret", "a_oth", "a_meq", "a_mdb", "a_mot",
    "ins_life", "ins_lp", "ins_hlth", "ins_hp",
    "tx_80c", "tx_80d", "tx_nps", "tx_hl", "tx_hra", "tx_oth",
]
_RISK_KEYS = [f"rq_{q['id']}" for q in RISK_QUESTIONS]


def save_session_to_json() -> str:
    data = {
        "app_version": "2.0",
        "inputs": {k: st.session_state.get(k) for k in _PERSIST_KEYS},
        "risk_choices": {k: st.session_state.get(k) for k in _RISK_KEYS},
        "debts": st.session_state.get("debts", []),
        "goals": st.session_state.get("goals", []),
        "derived": {
            "personal_info": st.session_state.get("personal_info"),
            "income_data": st.session_state.get("income_data"),
            "assets_data": st.session_state.get("assets_data"),
            "insurance_data": st.session_state.get("insurance_data"),
            "tax_inputs": st.session_state.get("tax_inputs"),
            "risk_answers": st.session_state.get("risk_answers"),
            "risk_level": st.session_state.get("risk_level"),
        },
        "saved_at": datetime.now().isoformat(),
    }
    return json.dumps(data, default=str, indent=2)


def load_session_from_json(raw: str) -> str:
    d = json.loads(raw)
    if not isinstance(d, dict):
        raise ValueError("Not a valid session file.")
    for k, v in (d.get("inputs") or {}).items():
        if k in _PERSIST_KEYS and v is not None:
            st.session_state[k] = v
    for k, v in (d.get("risk_choices") or {}).items():
        if k in _RISK_KEYS and v is not None:
            st.session_state[k] = v
    st.session_state.debts = d.get("debts", []) or []
    st.session_state.goals = d.get("goals", []) or []
    for k, v in (d.get("derived") or {}).items():
        if v is not None:
            st.session_state[k] = v
    st.session_state.analysis_done = False
    st.session_state.analysis_results = None
    return d.get("saved_at", "unknown")


# ============================================================================
# MAIN APP
# ============================================================================

SECTIONS = ["🧭 Risk Profile", "👤 Personal Info", "💰 Income & Expenses", "🏦 Assets",
            "💳 Debts", "🛡️ Insurance", "🎯 Goals", "📋 Tax Planning", "📊 Analysis"]


def init_state():
    defaults = {"section": 0, "analysis_done": False, "analysis_results": None,
                "personal_info": None, "income_data": None, "assets_data": None,
                "debts": [], "insurance_data": None, "goals": [], "tax_inputs": None,
                "risk_level": "moderate", "risk_answers": {},
                "show_save": False, "show_load": False}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(page_title="Financial Kundli — Financial Planning", page_icon="💎",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    init_state()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Financial <span class="gold">Kundli</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sb-caption">Planning Engine v2.0 · {DEFAULT_TAX_YEAR}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.progress((st.session_state.section + 1) / len(SECTIONS))
        st.caption(f"Step {st.session_state.section + 1} of {len(SECTIONS)}")
        st.markdown("---")
        steps_html = ""
        for i, s in enumerate(SECTIONS):
            plain = s.split(" ", 1)[1] if " " in s else s   # drop the emoji, keep the label
            if i == st.session_state.section:
                steps_html += (f'<div class="step-item step-current">'
                               f'<span class="step-diamond">◆</span>{plain}</div>')
            elif i < st.session_state.section:
                steps_html += (f'<div class="step-item step-done">'
                               f'<span class="step-diamond">◆</span>{plain}</div>')
            else:
                steps_html += (f'<div class="step-item step-todo">'
                               f'<span class="step-diamond">◇</span>{plain}</div>')
        st.markdown(steps_html, unsafe_allow_html=True)

        if st.session_state.analysis_done and st.session_state.analysis_results:
            st.markdown("---")
            h = st.session_state.analysis_results["health_score"]
            st.markdown(f'<div class="sb-score">'
                        f'<div class="num" style="color:{h["color"]}">{h["total_score"]}<span '
                        f'style="font-size:0.9rem; color:#8FA0BC">/100</span></div>'
                        f'<div class="cat" style="color:{h["color"]}">{h["category"]}</div></div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", use_container_width=True):
                st.session_state.show_save = True
                st.rerun()
        with c2:
            if st.button("📂 Load", use_container_width=True):
                st.session_state.show_load = True
                st.rerun()
        if st.button("🔄 Reset", use_container_width=True, type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ---- SAVE / LOAD SCREENS ----
    if st.session_state.show_save:
        st.markdown('<div class="section-title">💾 Save Session</div>', unsafe_allow_html=True)
        json_str = save_session_to_json()
        name = str(st.session_state.get("pi_name") or "Financial_Kundli").replace(" ", "_")
        st.download_button("⬇ Download Session File", data=json_str,
                           file_name=f"{name}_session_{datetime.now():%Y%m%d_%H%M%S}.json",
                           mime="application/json", type="primary")
        if st.button("← Back"):
            st.session_state.show_save = False
            st.rerun()
        return

    if st.session_state.show_load:
        st.markdown('<div class="section-title">📂 Load Session</div>', unsafe_allow_html=True)
        f = st.file_uploader("Upload .json session file", type=["json"])
        if f:
            try:
                saved_at = load_session_from_json(f.read().decode())
                st.success(f"Session loaded (saved {saved_at}). Inputs restored across all steps.")
                st.session_state.show_load = False
                st.rerun()
            except Exception as e:
                st.error(f"Could not load this file: {e}")
        if st.button("← Back"):
            st.session_state.show_load = False
            st.rerun()
        return

    # ---- HERO ----
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-emblem">{KUNDLI_EMBLEM}</div>
        <div class="hero-title">FINANCIAL <span class="gold">KUNDLI</span></div>
        <div class="hero-sub">Conviction in every strategy</div>
        <div class="hero-rule"></div>
    </div>""", unsafe_allow_html=True)

    # ---- NAV ----
    c1, _, c3 = st.columns([1, 6, 1])
    with c1:
        if st.session_state.section > 0:
            if st.button("← Back", type="secondary", use_container_width=True):
                st.session_state.section -= 1
                st.rerun()
    with c3:
        if st.session_state.section < len(SECTIONS) - 1:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.section += 1
                st.rerun()
    st.markdown("---")

    sname = SECTIONS[st.session_state.section]

    if sname == "🧭 Risk Profile":
        risk_level, _score, answers = section_risk_profiling()
        st.session_state.risk_level = risk_level
        st.session_state.risk_answers = answers

    elif sname == "👤 Personal Info":
        pi = section_personal_info()
        if pi["valid"]:
            pi["risk_level"] = st.session_state.risk_level
            st.session_state.personal_info = pi
        else:
            st.session_state.personal_info = None

    elif sname == "💰 Income & Expenses":
        if not st.session_state.personal_info:
            st.markdown('<div class="error-box">Complete Personal Info first.</div>', unsafe_allow_html=True)
        else:
            st.session_state.income_data = section_income_expenses()

    elif sname == "🏦 Assets":
        if not st.session_state.income_data:
            st.markdown('<div class="error-box">Complete Income & Expenses first.</div>', unsafe_allow_html=True)
        else:
            st.session_state.assets_data = section_assets()

    elif sname == "💳 Debts":
        st.session_state.debts = section_debts()

    elif sname == "🛡️ Insurance":
        st.session_state.insurance_data = section_insurance()

    elif sname == "🎯 Goals":
        st.session_state.goals = section_goals()

    elif sname == "📋 Tax Planning":
        st.session_state.tax_inputs = section_tax()

    elif sname == "📊 Analysis":
        required = {"Personal Info": st.session_state.personal_info,
                    "Income/Expenses": st.session_state.income_data,
                    "Assets": st.session_state.assets_data,
                    "Insurance": st.session_state.insurance_data,
                    "Tax Inputs": st.session_state.tax_inputs}
        missing = [k for k, v in required.items() if not v]
        if missing:
            st.markdown(f'<div class="error-box">⚠️ Complete these sections first: '
                        f'<strong>{", ".join(missing)}</strong></div>', unsafe_allow_html=True)
        elif not st.session_state.analysis_done:
            st.markdown('<div class="card-gold" style="text-align:center; padding:2rem">', unsafe_allow_html=True)
            st.markdown("### Ready to analyse your complete financial picture?")
            st.markdown("Runs a 1,000-path lifecycle Monte Carlo plus full tax, rebalancing & scenario analysis.")
            if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("Running 1,000 lifecycle simulations and all analyses..."):
                    st.session_state.analysis_results = run_full_analysis(
                        st.session_state.personal_info, st.session_state.income_data,
                        st.session_state.assets_data, st.session_state.debts,
                        st.session_state.insurance_data, st.session_state.goals,
                        st.session_state.tax_inputs, st.session_state.risk_answers)
                    st.session_state.analysis_done = True
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.button("🔄 Re-run Analysis", type="secondary"):
                st.session_state.analysis_done = False
                st.rerun()
            display_full_results(st.session_state.analysis_results)

            st.markdown("---")
            st.markdown("### 📥 Export Reports")
            c1, c2 = st.columns(2)
            with c1:
                try:
                    pdf_bytes = generate_pdf(st.session_state.personal_info,
                                             st.session_state.analysis_results)
                    name = str(st.session_state.personal_info.get("name") or "Report").replace(" ", "_")
                    st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                       file_name=f"{name}_FinancialKundli_{datetime.now():%Y%m%d}.pdf",
                                       mime="application/pdf", type="primary", use_container_width=True)
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
            with c2:
                ar = st.session_state.analysis_results
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["Financial Kundli Report", datetime.now().strftime("%Y-%m-%d"), ar["tax"]["fy"]])
                w.writerow(["Metric", "Value", "Status"])
                w.writerow(["Health Score", ar["health_score"]["total_score"], ar["health_score"]["category"]])
                w.writerow(["Emergency Fund", f"{ar['emergency']['adequacy_percentage']:.1f}%", ar["emergency"]["status"]])
                w.writerow(["Retirement Readiness", f"{ar['retirement']['readiness_percentage']:.1f}%", ar["retirement"]["status"]])
                w.writerow(["Lifecycle Success Probability", f"{ar['monte_carlo']['retirement']['success_probability']:.1f}%", "Monte Carlo"])
                w.writerow(["Debt-to-Income", f"{ar['debt']['debt_to_income_ratio']:.1f}%", ar["debt"]["status"]])
                w.writerow(["Better Tax Regime", ar["tax"]["better_regime"], f"Save {ar['tax']['savings_by_switching']:,.0f}"])
                w.writerow([])
                w.writerow(["Goal", "Target Today", "Target Future", "Inflation%", "Asset Mix", "Monthly SIP"])
                for g in ar["goals"]["goals_details"]:
                    w.writerow([g["name"], g["target_amount_pv"], g["target_amount_fv"],
                                g["inflation_used"], g["asset_mix"], g["monthly_saving_needed"]])
                name = str(st.session_state.personal_info.get("name") or "FinancialKundli").replace(" ", "_")
                st.download_button("📊 Download CSV", data=buf.getvalue(),
                                   file_name=f"{name}_data.csv", mime="text/csv",
                                   type="secondary", use_container_width=True)

    st.markdown("---")
    st.caption("Educational planning tool — not investment advice. Projections are assumption-based estimates. "
               "Consult a SEBI-registered Investment Adviser before acting.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        run_self_tests()
    elif _UI_AVAILABLE:
        main()
    else:
        print("Streamlit is not installed. Install requirements and run:\n"
              "  pip install streamlit plotly reportlab numpy\n"
              "  streamlit run financial_kundli_v2.py\n"
              "Or verify the computation engine with:\n"
              "  python financial_kundli_v2.py --selftest")
