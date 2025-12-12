"""
FINANCIAL PLANNING ENGINE - INTERACTIVE DASHBOARD
Streamlit-based web application with beautiful UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import pickle
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from math import pow, sqrt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# For PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import base64

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Financial Planning Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #34495e;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #7f8c8d;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 5px solid #3498db;
    }
    .positive {
        color: #27ae60;
        font-weight: 600;
    }
    .negative {
        color: #e74c3c;
        font-weight: 600;
    }
    .warning {
        color: #f39c12;
        font-weight: 600;
    }
    .stButton > button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .download-btn {
        background-color: #2ecc71 !important;
    }
    .download-btn:hover {
        background-color: #27ae60 !important;
    }
    .save-btn {
        background-color: #9b59b6 !important;
    }
    .save-btn:hover {
        background-color: #8e44ad !important;
    }
    .load-btn {
        background-color: #e67e22 !important;
    }
    .load-btn:hover {
        background-color: #d35400 !important;
    }
    .health-score-excellent {
        color: #27ae60;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .health-score-good {
        color: #2ecc71;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .health-score-fair {
        color: #f39c12;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .health-score-poor {
        color: #e74c3c;
        font-weight: 800;
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONSTANTS & CONFIGURATION
# ============================================================================

class FinancialConstants:
    """Financial planning constants"""
    EMERGENCY_FUND_SINGLE = 6
    EMERGENCY_FUND_WITH_DEPENDENTS = 9
    LIFE_EXPECTANCY = 85
    INFLATION_RATE = 0.06
    POST_RETURN_RATE = 0.07
    HIGH_INTEREST_THRESHOLD = 0.08
    LIFE_INSURANCE_MULTIPLIER = 10
    HEALTH_INSURANCE_MIN = 500000
    AGE_BASED_ALLOCATION = {
        "conservative": {"equity": 0.4, "debt": 0.5, "cash": 0.10},
        "moderate": {"equity": 0.6, "debt": 0.35, "cash": 0.05},
        "aggressive": {"equity": 0.8, "debt": 0.15, "cash": 0.05}
    }
    # Investment return assumptions (mean, std deviation)
    EQUITY_RETURN = (0.12, 0.18)  # 12% mean, 18% std
    DEBT_RETURN = (0.07, 0.05)    # 7% mean, 5% std
    GOLD_RETURN = (0.08, 0.10)    # 8% mean, 10% std
    # Monte Carlo parameters
    MONTE_CARLO_SIMULATIONS = 1000
    MONTE_CARLO_YEARS = 30

# ============================================================================
# 3. DATA MODELS
# ============================================================================

class RiskAppetite(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class LifeStage(str, Enum):
    EARLY_CAREER = "early_career"
    MID_CAREER = "mid_career"
    PRE_RETIREMENT = "pre_retirement"
    RETIRED = "retired"

class GoalPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class GoalType(str, Enum):
    EMERGENCY_FUND = "emergency_fund"
    RETIREMENT = "retirement"
    HOME_PURCHASE = "home_purchase"
    EDUCATION = "education"
    VEHICLE = "vehicle"
    VACATION = "vacation"
    WEDDING = "wedding"
    OTHER = "other"

# ============================================================================
# 4. CALCULATORS
# ============================================================================

class FinancialCalculator:
    """Core financial calculations"""
    
    @staticmethod
    def future_value(present_value: float, rate: float, years: int) -> float:
        """Calculate future value with compound interest"""
        return present_value * pow((1 + rate), years)
    
    @staticmethod
    def future_value_monthly(monthly_investment: float, rate: float, years: int) -> float:
        """Future value of monthly investments"""
        monthly_rate = rate / 12
        months = years * 12
        
        if monthly_rate == 0:
            return monthly_investment * months
        
        fv = monthly_investment * (pow((1 + monthly_rate), months) - 1) / monthly_rate
        return fv
    
    @staticmethod
    def inflation_adjusted_value(amount: float, inflation_rate: float, years: int) -> float:
        """Calculate inflation-adjusted future value"""
        return amount * pow((1 + inflation_rate), years)
    
    @staticmethod
    def retirement_corpus_needed(
        annual_expenses: float,
        inflation_rate: float,
        retirement_age: int,
        life_expectancy: int,
        current_age: int
    ) -> Tuple[float, float]:
        """Calculate retirement corpus needed"""
        years_to_retirement = retirement_age - current_age
        retirement_years = life_expectancy - retirement_age
        
        # Future annual expenses at retirement
        future_annual_expenses = FinancialCalculator.inflation_adjusted_value(
            annual_expenses, inflation_rate, years_to_retirement
        )
        
        # Simple corpus calculation with buffer
        corpus_needed = future_annual_expenses * retirement_years
        corpus_needed *= 1.2  # 20% buffer
        
        return future_annual_expenses, corpus_needed
    
    @staticmethod
    def monthly_saving_for_goal(
        target_amount: float,
        years: int,
        expected_return: float,
        current_saved: float = 0
    ) -> float:
        """Calculate monthly saving needed for a goal"""
        if years == 0:
            return 0
        
        months = years * 12
        monthly_rate = expected_return / 12
        
        # Future value of current savings
        fv_current = FinancialCalculator.future_value(
            current_saved, expected_return, years
        )
        
        amount_needed = target_amount - fv_current
        if amount_needed <= 0:
            return 0
        
        if monthly_rate == 0:
            return amount_needed / months
        
        monthly_saving = amount_needed * monthly_rate / (pow((1 + monthly_rate), months) - 1)
        return monthly_saving

# ============================================================================
# 5. ANALYZERS
# ============================================================================

class EmergencyFundAnalyzer:
    def __init__(self, constants: FinancialConstants):
        self.constants = constants
    
    def analyze(self, monthly_expenses: float, dependents: int, cash_and_bank: float) -> Dict[str, Any]:
        """Analyze emergency fund status"""
        
        # Calculate required emergency fund
        if dependents > 0:
            months_needed = self.constants.EMERGENCY_FUND_WITH_DEPENDENTS
        else:
            months_needed = self.constants.EMERGENCY_FUND_SINGLE
        
        required_emergency_fund = monthly_expenses * months_needed
        current_emergency_fund = cash_and_bank
        
        # Determine adequacy
        adequacy_percentage = (current_emergency_fund / required_emergency_fund * 100) if required_emergency_fund > 0 else 0
        shortfall = max(0, required_emergency_fund - current_emergency_fund)
        
        # Calculate months coverage
        months_coverage = current_emergency_fund / monthly_expenses if monthly_expenses > 0 else 0
        
        # Status classification
        if adequacy_percentage >= 100:
            status = "Adequate"
            priority = "low"
            color = "green"
        elif adequacy_percentage >= 50:
            status = "Partially Adequate"
            priority = "medium"
            color = "orange"
        else:
            status = "Inadequate"
            priority = "high"
            color = "red"
        
        return {
            "required_fund": round(required_emergency_fund, 2),
            "current_fund": round(current_emergency_fund, 2),
            "adequacy_percentage": round(adequacy_percentage, 2),
            "shortfall": round(shortfall, 2),
            "months_coverage": round(months_coverage, 1),
            "status": status,
            "priority": priority,
            "color": color,
            "recommended_months": months_needed
        }

class RetirementAnalyzer:
    def __init__(self, constants: FinancialConstants):
        self.constants = constants
        self.calculator = FinancialCalculator()
    
    def analyze(
        self,
        age: int,
        retirement_age: int,
        monthly_expenses: float,
        current_corpus: float,
        monthly_investment: float
    ) -> Dict[str, Any]:
        """Analyze retirement planning"""
        
        annual_expenses = monthly_expenses * 12
        
        # Calculate retirement needs
        future_annual_expenses, corpus_needed = self.calculator.retirement_corpus_needed(
            annual_expenses=annual_expenses,
            inflation_rate=self.constants.INFLATION_RATE,
            retirement_age=retirement_age,
            life_expectancy=self.constants.LIFE_EXPECTANCY,
            current_age=age
        )
        
        # Project future value
        years_to_retirement = retirement_age - age
        expected_return = 0.10
        
        # Future value of current corpus
        fv_current_corpus = self.calculator.future_value(
            current_corpus, expected_return, years_to_retirement
        )
        
        # Future value of monthly investments
        fv_monthly_investments = self.calculator.future_value_monthly(
            monthly_investment, expected_return, years_to_retirement
        )
        
        total_projected_corpus = fv_current_corpus + fv_monthly_investments
        
        # Calculate shortfall/surplus
        shortfall = max(0, corpus_needed - total_projected_corpus)
        surplus = max(0, total_projected_corpus - corpus_needed)
        
        # Calculate additional monthly saving needed
        if shortfall > 0:
            additional_monthly_saving = self.calculator.monthly_saving_for_goal(
                shortfall, years_to_retirement, expected_return
            )
        else:
            additional_monthly_saving = 0
        
        # Retirement readiness score
        readiness_score = min(100, (total_projected_corpus / corpus_needed) * 100) if corpus_needed > 0 else 100
        
        # Determine status
        if readiness_score >= 100:
            status = "On Track"
            priority = "low"
            color = "green"
        elif readiness_score >= 70:
            status = "Moderately On Track"
            priority = "medium"
            color = "orange"
        elif readiness_score >= 40:
            status = "Needs Attention"
            priority = "high"
            color = "red"
        else:
            status = "Critical"
            priority = "critical"
            color = "darkred"
        
        return {
            "years_to_retirement": years_to_retirement,
            "retirement_age": retirement_age,
            "current_annual_expenses": round(annual_expenses, 2),
            "future_annual_expenses_at_retirement": round(future_annual_expenses, 2),
            "corpus_needed": round(corpus_needed, 2),
            "current_corpus": round(current_corpus, 2),
            "projected_corpus": round(total_projected_corpus, 2),
            "readiness_percentage": round(readiness_score, 2),
            "shortfall": round(shortfall, 2),
            "surplus": round(surplus, 2),
            "additional_monthly_saving_needed": round(additional_monthly_saving, 2),
            "current_monthly_investment": round(monthly_investment, 2),
            "status": status,
            "priority": priority,
            "color": color
        }

class DebtAnalyzer:
    def __init__(self, constants: FinancialConstants):
        self.constants = constants
    
    def analyze(
        self,
        debts: List[Dict],
        client_annual_income: float
    ) -> Dict[str, Any]:
        """Analyze debt burden"""
        
        if not debts:
            return {
                "total_debt": 0,
                "monthly_emi": 0,
                "debt_to_income_ratio": 0,
                "high_interest_debt_count": 0,
                "total_interest_payable": 0,
                "status": "No Debt",
                "priority": "low",
                "color": "green"
            }
        
        # Calculate totals
        total_debt = sum(debt.get('outstanding_amount', 0) for debt in debts)
        total_monthly_emi = sum(debt.get('emi', 0) for debt in debts)
        monthly_income = client_annual_income / 12
        
        # Debt-to-income ratio
        debt_to_income = (total_monthly_emi / monthly_income) * 100 if monthly_income > 0 else 0
        
        # Identify high-interest debts
        high_interest_debts = [
            debt for debt in debts 
            if debt.get('interest_rate', 0) >= self.constants.HIGH_INTEREST_THRESHOLD
        ]
        
        # Calculate interest burden (simplified)
        total_interest_payable = sum(
            (debt.get('emi', 0) * debt.get('tenure_months', 0)) - debt.get('principal', 0)
            for debt in debts
        )
        
        # Calculate potential savings from prepayment
        high_interest_savings = 0
        for debt in high_interest_debts:
            prepayment = debt.get('outstanding_amount', 0) * 0.20
            interest_saved = prepayment * debt.get('interest_rate', 0) * (debt.get('tenure_months', 0) / 12)
            high_interest_savings += interest_saved
        
        # Status classification
        if debt_to_income <= 20:
            status = "Healthy"
            priority = "low"
            color = "green"
        elif debt_to_income <= 40:
            status = "Moderate"
            priority = "medium"
            color = "orange"
        elif debt_to_income <= 60:
            status = "Concerning"
            priority = "high"
            color = "red"
        else:
            status = "Critical"
            priority = "critical"
            color = "darkred"
        
        return {
            "total_debt": round(total_debt, 2),
            "monthly_emi": round(total_monthly_emi, 2),
            "debt_to_income_ratio": round(debt_to_income, 2),
            "high_interest_debt_count": len(high_interest_debts),
            "total_interest_payable": round(total_interest_payable, 2),
            "high_interest_savings_possible": round(high_interest_savings, 2),
            "status": status,
            "priority": priority,
            "color": color,
            "debts_details": debts
        }

class GoalAnalyzer:
    def __init__(self, constants: FinancialConstants):
        self.constants = constants
        self.calculator = FinancialCalculator()
    
    def analyze(
        self,
        goals: List[Dict],
        age: int,
        current_investment_rate: float = 0.10
    ) -> Dict[str, Any]:
        """Analyze financial goals"""
        
        if not goals:
            return {
                "total_goals": 0,
                "total_target_amount_pv": 0,
                "total_target_amount_fv": 0,
                "monthly_investment_needed": 0,
                "goals_details": [],
                "priority_summary": {}
            }
        
        goals_analysis = []
        total_monthly_investment_needed = 0
        total_target_amount_pv = 0
        total_target_amount_fv = 0
        priority_summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for goal in goals:
            goal_name = goal.get("name", "Unnamed Goal")
            target_amount = goal.get("target_amount", 0)
            timeframe_years = goal.get("timeframe_years", 0)
            priority = goal.get("priority", "medium")
            current_saved = goal.get("current_saved", 0)
            
            # Target amount is in TODAY'S VALUE (PV)
            # Calculate Future Value considering inflation
            inflated_target_amount = self.calculator.inflation_adjusted_value(
                target_amount, self.constants.INFLATION_RATE, timeframe_years
            )
            
            total_target_amount_pv += target_amount
            total_target_amount_fv += inflated_target_amount
            
            # Calculate monthly saving needed
            monthly_saving_needed = self.calculator.monthly_saving_for_goal(
                target_amount=inflated_target_amount,
                years=timeframe_years,
                expected_return=current_investment_rate,
                current_saved=current_saved
            )
            
            total_monthly_investment_needed += monthly_saving_needed
            
            # Track priority
            if priority in priority_summary:
                priority_summary[priority] += 1
            
            # Calculate progress
            progress_percentage = (current_saved / target_amount * 100) if target_amount > 0 else 0
            
            # Determine status
            if progress_percentage >= 100:
                status = "Completed"
                color = "green"
            elif progress_percentage >= 70:
                status = "On Track"
                color = "green"
            elif progress_percentage >= 40:
                status = "Needs Attention"
                color = "orange"
            else:
                status = "Behind Schedule"
                color = "red"
            
            goals_analysis.append({
                "name": goal_name,
                "type": goal.get("type", "other"),
                "target_amount_pv": round(target_amount, 2),  # Present value
                "target_amount_fv": round(inflated_target_amount, 2),  # Future value (inflation-adjusted)
                "timeframe_years": timeframe_years,
                "priority": priority,
                "current_saved": round(current_saved, 2),
                "monthly_saving_needed": round(monthly_saving_needed, 2),
                "progress_percentage": round(progress_percentage, 2),
                "status": status,
                "color": color,
                "completion_year": age + timeframe_years
            })
        
        # Sort goals by priority (critical first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        goals_analysis.sort(key=lambda x: (priority_order.get(x["priority"], 4), x["timeframe_years"]))
        
        return {
            "total_goals": len(goals),
            "total_target_amount_pv": round(total_target_amount_pv, 2),
            "total_target_amount_fv": round(total_target_amount_fv, 2),
            "total_monthly_investment_needed": round(total_monthly_investment_needed, 2),
            "goals_details": goals_analysis,
            "priority_summary": priority_summary
        }

# ============================================================================
# NEW: FINANCIAL HEALTH SCORE CALCULATOR
# ============================================================================

class FinancialHealthScore:
    """Calculate overall financial health score (0-100)"""
    
    def __init__(self, constants: FinancialConstants):
        self.constants = constants
    
    def calculate_score(
        self,
        emergency_analysis: Dict,
        retirement_analysis: Dict,
        debt_analysis: Dict,
        goal_analysis: Dict,
        personal_info: Dict,
        monthly_savings_rate: float,
        insurance_coverage: Dict
    ) -> Dict[str, Any]:
        """Calculate comprehensive financial health score"""
        
        # Component weights (sum = 100)
        weights = {
            "emergency_fund": 20,
            "debt_management": 20,
            "savings_rate": 15,
            "retirement_readiness": 25,
            "goal_progress": 10,
            "insurance_adequacy": 10
        }
        
        # 1. Emergency Fund Score (0-100)
        emergency_score = min(100, emergency_analysis["adequacy_percentage"])
        
        # 2. Debt Management Score (0-100)
        if debt_analysis["debt_to_income_ratio"] == 0:
            debt_score = 100
        else:
            # Inverted scale: lower DTI = higher score
            debt_score = max(0, 100 - (debt_analysis["debt_to_income_ratio"] * 1.5))
            debt_score = min(100, debt_score)
        
        # 3. Savings Rate Score (0-100)
        # Target: 20% savings rate gets 100 points
        savings_score = min(100, monthly_savings_rate * 5)  # 20% * 5 = 100
        
        # 4. Retirement Readiness Score (0-100)
        retirement_score = min(100, retirement_analysis["readiness_percentage"])
        
        # 5. Goal Progress Score (0-100)
        if goal_analysis["total_goals"] > 0:
            avg_progress = sum(g["progress_percentage"] for g in goal_analysis["goals_details"]) / goal_analysis["total_goals"]
            goal_score = min(100, avg_progress)
        else:
            goal_score = 50  # Neutral score if no goals set
        
        # 6. Insurance Adequacy Score (0-100)
        insurance_score = self._calculate_insurance_score(personal_info, insurance_coverage)
        
        # Calculate weighted total score
        total_score = (
            (emergency_score * weights["emergency_fund"]) +
            (debt_score * weights["debt_management"]) +
            (savings_score * weights["savings_rate"]) +
            (retirement_score * weights["retirement_readiness"]) +
            (goal_score * weights["goal_progress"]) +
            (insurance_score * weights["insurance_adequacy"])
        ) / 100
        
        total_score = round(total_score, 1)
        
        # Determine health category
        if total_score >= 80:
            category = "Excellent"
            color = "#27ae60"
            description = "Your financial health is excellent! Keep up the good work."
        elif total_score >= 65:
            category = "Good"
            color = "#2ecc71"
            description = "Your financial health is good, with room for improvement in some areas."
        elif total_score >= 50:
            category = "Fair"
            color = "#f39c12"
            description = "Your financial health needs attention in several areas."
        else:
            category = "Poor"
            color = "#e74c3c"
            description = "Your financial health requires immediate attention and improvement."
        
        return {
            "total_score": total_score,
            "category": category,
            "color": color,
            "description": description,
            "components": {
                "emergency_fund": {
                    "score": round(emergency_score, 1),
                    "weight": weights["emergency_fund"],
                    "description": f"Emergency fund: {emergency_analysis['adequacy_percentage']:.1f}% adequate"
                },
                "debt_management": {
                    "score": round(debt_score, 1),
                    "weight": weights["debt_management"],
                    "description": f"Debt-to-income: {debt_analysis['debt_to_income_ratio']:.1f}%"
                },
                "savings_rate": {
                    "score": round(savings_score, 1),
                    "weight": weights["savings_rate"],
                    "description": f"Savings rate: {monthly_savings_rate:.1f}%"
                },
                "retirement_readiness": {
                    "score": round(retirement_score, 1),
                    "weight": weights["retirement_readiness"],
                    "description": f"Retirement: {retirement_analysis['readiness_percentage']:.1f}% ready"
                },
                "goal_progress": {
                    "score": round(goal_score, 1),
                    "weight": weights["goal_progress"],
                    "description": f"Goals: {goal_analysis['total_goals']} goals set"
                },
                "insurance_adequacy": {
                    "score": round(insurance_score, 1),
                    "weight": weights["insurance_adequacy"],
                    "description": f"Insurance: {insurance_score:.1f}/100"
                }
            }
        }
    
    def _calculate_insurance_score(self, personal_info: Dict, insurance: Dict) -> float:
        """Calculate insurance adequacy score"""
        age = personal_info.get("age", 35)
        dependents = personal_info.get("dependents", 0)
        
        # Life insurance adequacy (40 points max)
        annual_income_needed = 100000 * 12  # Example annual income
        recommended_life_insurance = annual_income_needed * 10
        life_coverage = insurance.get("life_insurance", 0)
        
        if life_coverage >= recommended_life_insurance:
            life_score = 40
        elif life_coverage >= recommended_life_insurance * 0.5:
            life_score = 30
        elif life_coverage >= recommended_life_insurance * 0.25:
            life_score = 20
        else:
            life_score = 10
        
        # Health insurance adequacy (40 points max)
        recommended_health_insurance = 1000000
        health_coverage = insurance.get("health_insurance", 0)
        
        if health_coverage >= recommended_health_insurance * 2:
            health_score = 40
        elif health_coverage >= recommended_health_insurance:
            health_score = 30
        elif health_coverage >= recommended_health_insurance * 0.5:
            health_score = 20
        else:
            health_score = 10
        
        # Other insurance (20 points max)
        other_insurance = insurance.get("vehicle_insurance", 0) + insurance.get("other_insurance", 0)
        if other_insurance > 0:
            other_score = 20
        else:
            other_score = 0
        
        total_insurance_score = life_score + health_score + other_score
        return min(100, total_insurance_score)

# ============================================================================
# NEW: MONTE CARLO SIMULATOR
# ============================================================================

class MonteCarloSimulator:
    """Run Monte Carlo simulations for financial projections"""
    
    def __init__(self, constants: FinancialConstants):
        self.constants = constants
    
    def simulate_portfolio_growth(
        self,
        initial_investment: float,
        monthly_contribution: float,
        years: int,
        equity_allocation: float = 0.6,
        debt_allocation: float = 0.35,
        gold_allocation: float = 0.05
    ) -> Dict[str, Any]:
        """Simulate portfolio growth using Monte Carlo"""
        
        n_simulations = self.constants.MONTE_CARLO_SIMULATIONS
        n_years = years
        
        # Annual returns for each asset class
        equity_mean, equity_std = self.constants.EQUITY_RETURN
        debt_mean, debt_std = self.constants.DEBT_RETURN
        gold_mean, gold_std = self.constants.GOLD_RETURN
        
        # Correlation matrix (simplified)
        # In reality, correlations are time-varying and complex
        
        # Generate random returns for each simulation
        final_values = []
        yearly_paths = []
        
        for _ in range(n_simulations):
            # Initialize portfolio value
            portfolio_value = initial_investment
            
            # Store yearly values for this simulation
            yearly_values = [portfolio_value]
            
            for year in range(n_years):
                # Generate random returns for each asset class
                equity_return = np.random.normal(equity_mean, equity_std)
                debt_return = np.random.normal(debt_mean, debt_std)
                gold_return = np.random.normal(gold_mean, gold_std)
                
                # Calculate weighted portfolio return
                portfolio_return = (
                    equity_allocation * equity_return +
                    debt_allocation * debt_return +
                    gold_allocation * gold_return
                )
                
                # Apply return to existing portfolio
                portfolio_value *= (1 + portfolio_return)
                
                # Add monthly contributions (compounded monthly)
                for month in range(12):
                    portfolio_value += monthly_contribution
                    # Apply monthly return (annual return / 12)
                    portfolio_value *= (1 + portfolio_return / 12)
                
                yearly_values.append(portfolio_value)
            
            final_values.append(portfolio_value)
            yearly_paths.append(yearly_values)
        
        # Calculate statistics
        final_values_array = np.array(final_values)
        
        # Calculate percentiles
        percentiles = {
            "p5": np.percentile(final_values_array, 5),
            "p25": np.percentile(final_values_array, 25),
            "p50": np.percentile(final_values_array, 50),  # Median
            "p75": np.percentile(final_values_array, 75),
            "p95": np.percentile(final_values_array, 95)
        }
        
        # Calculate probability of reaching target (if provided)
        success_probability = None
        
        return {
            "final_values": final_values_array,
            "yearly_paths": yearly_paths,
            "statistics": {
                "mean": float(np.mean(final_values_array)),
                "median": float(np.median(final_values_array)),
                "std": float(np.std(final_values_array)),
                "min": float(np.min(final_values_array)),
                "max": float(np.max(final_values_array)),
                "percentiles": percentiles
            },
            "success_probability": success_probability
        }
    
    def simulate_retirement(
        self,
        current_age: int,
        retirement_age: int,
        current_corpus: float,
        monthly_investment: float,
        target_corpus: float,
        equity_allocation: float = 0.6,
        debt_allocation: float = 0.35,
        gold_allocation: float = 0.05
    ) -> Dict[str, Any]:
        """Simulate retirement corpus growth"""
        
        years_to_retirement = retirement_age - current_age
        
        if years_to_retirement <= 0:
            return {
                "success_probability": 100,
                "median_corpus": current_corpus,
                "required_corpus": target_corpus,
                "percentiles": {
                    "p5": current_corpus,
                    "p25": current_corpus,
                    "p50": current_corpus,
                    "p75": current_corpus,
                    "p95": current_corpus
                }
            }
        
        simulation = self.simulate_portfolio_growth(
            initial_investment=current_corpus,
            monthly_contribution=monthly_investment,
            years=years_to_retirement,
            equity_allocation=equity_allocation,
            debt_allocation=debt_allocation,
            gold_allocation=gold_allocation
        )
        
        # Calculate success probability
        final_values = simulation["final_values"]
        success_count = np.sum(final_values >= target_corpus)
        success_probability = (success_count / len(final_values)) * 100
        
        simulation["success_probability"] = success_probability
        simulation["required_corpus"] = target_corpus
        
        return simulation
    
    def simulate_goal(
        self,
        goal_name: str,
        current_saved: float,
        monthly_saving: float,
        target_amount: float,
        years: int,
        equity_allocation: float = 0.6,
        debt_allocation: float = 0.35,
        gold_allocation: float = 0.05
    ) -> Dict[str, Any]:
        """Simulate goal achievement probability"""
        
        if years <= 0:
            return {
                "goal_name": goal_name,
                "success_probability": 100 if current_saved >= target_amount else 0,
                "median_value": current_saved,
                "target_amount": target_amount,
                "percentiles": {
                    "p5": current_saved,
                    "p25": current_saved,
                    "p50": current_saved,
                    "p75": current_saved,
                    "p95": current_saved
                }
            }
        
        simulation = self.simulate_portfolio_growth(
            initial_investment=current_saved,
            monthly_contribution=monthly_saving,
            years=years,
            equity_allocation=equity_allocation,
            debt_allocation=debt_allocation,
            gold_allocation=gold_allocation
        )
        
        # Calculate success probability
        final_values = simulation["final_values"]
        success_count = np.sum(final_values >= target_amount)
        success_probability = (success_count / len(final_values)) * 100
        
        return {
            "goal_name": goal_name,
            "success_probability": success_probability,
            "median_value": float(np.median(final_values)),
            "target_amount": target_amount,
            "percentiles": simulation["statistics"]["percentiles"],
            "simulation_data": simulation
        }

# ============================================================================
# 6. DASHBOARD COMPONENTS
# ============================================================================

def display_welcome():
    """Display welcome section"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">💰 Financial Planning Engine</h1>', unsafe_allow_html=True)
        st.markdown("### Your Personal Financial Advisor")
        st.markdown("Get a comprehensive analysis of your finances with personalized recommendations")

def create_personal_info_section():
    """Create personal information input section"""
    st.markdown('<h2 class="section-header">👤 Personal Information</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("Full Name", value="John Doe")
        age = st.number_input("Current Age", min_value=18, max_value=100, value=35, step=1)
    
    with col2:
        retirement_age = st.number_input("Planned Retirement Age", min_value=40, max_value=75, value=60, step=1)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
    
    with col3:
        city_tier = st.selectbox("City Tier", ["Tier 1 (Metro)", "Tier 2", "Tier 3"], index=0)
        risk_appetite = st.selectbox(
            "Risk Appetite",
            ["Conservative (Low Risk)", "Moderate (Balanced)", "Aggressive (High Risk)"],
            index=1
        )
    
    # Map risk appetite
    risk_map = {
        "Conservative (Low Risk)": "conservative",
        "Moderate (Balanced)": "moderate", 
        "Aggressive (High Risk)": "aggressive"
    }
    
    return {
        "name": name,
        "age": age,
        "retirement_age": retirement_age,
        "dependents": dependents,
        "city_tier": city_tier,
        "risk_appetite": risk_map[risk_appetite]
    }

def create_income_expenses_section():
    """Create income and expenses input section"""
    st.markdown('<h2 class="section-header">💰 Income & Expenses</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">Income Sources</h3>', unsafe_allow_html=True)
        
        salary = st.number_input("Monthly Salary (₹)", min_value=0, value=100000, step=10000)
        business = st.number_input("Monthly Business Income (₹)", min_value=0, value=0, step=10000)
        rental = st.number_input("Monthly Rental Income (₹)", min_value=0, value=0, step=5000)
        other_income = st.number_input("Other Monthly Income (₹)", min_value=0, value=0, step=5000)
    
    with col2:
        st.markdown('<h3 class="sub-header">Monthly Expenses</h3>', unsafe_allow_html=True)
        
        housing = st.number_input("Housing (Rent/EMI) (₹)", min_value=0, value=25000, step=1000)
        groceries = st.number_input("Groceries & Food (₹)", min_value=0, value=15000, step=1000)
        transportation = st.number_input("Transportation (₹)", min_value=0, value=8000, step=1000)
        utilities = st.number_input("Utilities (₹)", min_value=0, value=5000, step=1000)
        education = st.number_input("Education (₹)", min_value=0, value=10000, step=1000)
        healthcare = st.number_input("Healthcare (₹)", min_value=0, value=3000, step=1000)
        entertainment = st.number_input("Entertainment (₹)", min_value=0, value=5000, step=1000)
        other_expenses = st.number_input("Other Expenses (₹)", min_value=0, value=7000, step=1000)
    
    # Calculate totals
    total_income = salary + business + rental + other_income
    total_expenses = housing + groceries + transportation + utilities + education + healthcare + entertainment + other_expenses
    
    # Display summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Monthly Income", f"₹{total_income:,.0f}")
    with col2:
        st.metric("Total Monthly Expenses", f"₹{total_expenses:,.0f}")
    
    # Savings rate
    savings_rate = 0
    if total_income > 0:
        savings_rate = ((total_income - total_expenses) / total_income) * 100
        st.progress(min(100, max(0, savings_rate)) / 100)
        st.caption(f"Savings Rate: {savings_rate:.1f}%")
    
    return {
        "income": {
            "salary": salary,
            "business": business,
            "rental": rental,
            "other": other_income
        },
        "monthly_expenses": total_expenses,
        "savings_rate": savings_rate,
        "expense_breakdown": {
            "housing": housing,
            "groceries": groceries,
            "transportation": transportation,
            "utilities": utilities,
            "education": education,
            "healthcare": healthcare,
            "entertainment": entertainment,
            "other": other_expenses
        }
    }

def create_assets_section():
    """Create assets input section"""
    st.markdown('<h2 class="section-header">🏦 Assets & Investments</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h3 class="sub-header">Liquid Assets</h3>', unsafe_allow_html=True)
        cash_bank = st.number_input("Cash & Bank Balance (₹)", min_value=0, value=200000, step=10000)
    
    with col2:
        st.markdown('<h3 class="sub-header">Investments</h3>', unsafe_allow_html=True)
        equity = st.number_input("Equity (Stocks/MFs) (₹)", min_value=0, value=500000, step=50000)
        debt = st.number_input("Debt (FDs/Bonds) (₹)", min_value=0, value=300000, step=50000)
        gold = st.number_input("Gold (₹)", min_value=0, value=100000, step=10000)
    
    with col3:
        st.markdown('<h3 class="sub-header">Other Assets</h3>', unsafe_allow_html=True)
        retirement = st.number_input("Retirement Corpus (EPF/PPF/NPS) (₹)", min_value=0, value=800000, step=50000)
        real_estate = st.number_input("Real Estate Value (₹)", min_value=0, value=2000000, step=100000)
        other_assets = st.number_input("Other Assets (₹)", min_value=0, value=100000, step=10000)
    
    # Monthly investments
    st.markdown('<h3 class="sub-header">Monthly Investments</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monthly_equity = st.number_input("Monthly Equity SIP (₹)", min_value=0, value=20000, step=1000)
    with col2:
        monthly_debt = st.number_input("Monthly Debt SIP (₹)", min_value=0, value=10000, step=1000)
    with col3:
        monthly_other = st.number_input("Other Monthly Investments (₹)", min_value=0, value=5000, step=1000)
    
    total_investments = equity + debt + gold
    total_assets = cash_bank + total_investments + retirement + real_estate + other_assets
    total_monthly_investment = monthly_equity + monthly_debt + monthly_other
    
    # Display summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Investments", f"₹{total_investments:,.0f}")
    with col2:
        st.metric("Total Assets", f"₹{total_assets:,.0f}")
    
    # Investment allocation
    if total_investments > 0:
        equity_pct = (equity / total_investments) * 100
        debt_pct = (debt / total_investments) * 100
        gold_pct = (gold / total_investments) * 100
        
        st.markdown("#### Current Investment Allocation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Equity", f"{equity_pct:.1f}%")
        with col2:
            st.metric("Debt", f"{debt_pct:.1f}%")
        with col3:
            st.metric("Gold", f"{gold_pct:.1f}%")
    
    return {
        "cash_and_bank": cash_bank,
        "investments": [
            {"name": "Equity", "type": "equity", "current_value": equity, "monthly_contribution": monthly_equity},
            {"name": "Debt", "type": "debt", "current_value": debt, "monthly_contribution": monthly_debt},
            {"name": "Gold", "type": "gold", "current_value": gold, "monthly_contribution": 0}
        ],
        "retirement_corpus": retirement,
        "real_estate_value": real_estate,
        "other_assets": other_assets,
        "monthly_investment": total_monthly_investment,
        "total_investments": total_investments
    }

def create_debts_section():
    """Create debts input section"""
    st.markdown('<h2 class="section-header">💳 Debts & Liabilities</h2>', unsafe_allow_html=True)
    
    # Initialize session state for debts if not exists
    if 'debts' not in st.session_state:
        st.session_state.debts = []
    
    # Debt input form
    with st.expander("Add a Loan/Debt", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            debt_name = st.text_input("Loan Name", key="debt_name")
            debt_type = st.selectbox(
                "Loan Type",
                ["Home Loan", "Car Loan", "Personal Loan", "Credit Card", "Education Loan", "Other"],
                key="debt_type"
            )
        
        with col2:
            principal = st.number_input("Principal Amount (₹)", min_value=0, value=0, step=10000, key="principal")
            outstanding = st.number_input("Outstanding Amount (₹)", min_value=0, value=0, step=10000, key="outstanding")
        
        with col3:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5, key="interest_rate")
            emi = st.number_input("Monthly EMI (₹)", min_value=0, value=0, step=1000, key="emi")
        
        with col4:
            tenure_years = st.number_input("Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="tenure_years")
            tenure_months = tenure_years * 12
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add This Debt", use_container_width=True):
                if debt_name and outstanding > 0:
                    new_debt = {
                        "name": debt_name,
                        "type": debt_type.lower().replace(" ", "_"),
                        "principal": principal if principal > 0 else outstanding,
                        "interest_rate": interest_rate / 100,
                        "emi": emi,
                        "tenure_months": tenure_months,
                        "outstanding_amount": outstanding
                    }
                    st.session_state.debts.append(new_debt)
                    st.success(f"Added {debt_name}")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Clear Form", use_container_width=True):
                st.rerun()
    
    # Display current debts
    if st.session_state.debts:
        st.markdown("### Current Debts")
        
        debt_data = []
        for i, debt in enumerate(st.session_state.debts):
            debt_data.append({
                "Loan": debt["name"],
                "Type": debt["type"].replace("_", " ").title(),
                "Outstanding": f"₹{debt['outstanding_amount']:,.0f}",
                "Interest Rate": f"{debt['interest_rate']*100:.1f}%",
                "Monthly EMI": f"₹{debt['emi']:,.0f}",
                "Actions": i
            })
        
        df = pd.DataFrame(debt_data)
        
        # Display table with delete buttons
        for i, row in df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1.5, 1, 1, 1])
            with col1:
                st.write(row["Loan"])
            with col2:
                st.write(row["Type"])
            with col3:
                st.write(row["Outstanding"])
            with col4:
                st.write(row["Interest Rate"])
            with col5:
                st.write(row["Monthly EMI"])
            with col6:
                if st.button("❌", key=f"delete_{i}"):
                    st.session_state.debts.pop(i)
                    st.rerun()
        
        # Calculate totals
        total_debt = sum(d["outstanding_amount"] for d in st.session_state.debts)
        total_emi = sum(d["emi"] for d in st.session_state.debts)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Debt Outstanding", f"₹{total_debt:,.0f}")
        with col2:
            st.metric("Total Monthly EMI", f"₹{total_emi:,.0f}")
    else:
        st.info("No debts added yet. Click 'Add a Loan/Debt' to add your loans.")
    
    return st.session_state.debts

def create_insurance_section():
    """Create insurance input section"""
    st.markdown('<h2 class="section-header">🛡️ Insurance Coverage</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">Life Insurance</h3>', unsafe_allow_html=True)
        life_insurance = st.number_input("Life Insurance Sum Assured (₹)", min_value=0, value=5000000, step=100000)
        life_premium = st.number_input("Annual Premium (₹)", min_value=0, value=25000, step=1000)
    
    with col2:
        st.markdown('<h3 class="sub-header">Health Insurance</h3>', unsafe_allow_html=True)
        health_insurance = st.number_input("Health Insurance Coverage (₹)", min_value=0, value=1000000, step=100000)
        health_premium = st.number_input("Annual Premium (₹)", min_value=0, value=15000, step=1000)
    
    # Other insurance
    st.markdown('<h3 class="sub-header">Other Insurance</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_insurance = st.number_input("Vehicle Insurance (₹)", min_value=0, value=0, step=1000)
    with col2:
        other_insurance = st.number_input("Other Insurance (₹)", min_value=0, value=0, step=1000)
    
    return {
        "life_insurance": life_insurance,
        "life_premium": life_premium,
        "health_insurance": health_insurance,
        "health_premium": health_premium,
        "vehicle_insurance": vehicle_insurance,
        "other_insurance": other_insurance
    }

def create_goals_section():
    """Create financial goals input section"""
    st.markdown('<h2 class="section-header">🎯 Financial Goals</h2>', unsafe_allow_html=True)
    
    # Add clarification about PV/FV
    st.info("💡 **Note**: Enter the target amount in today's value (present value). The calculator will automatically adjust for inflation to show you the future value needed.")
    
    # Initialize session state for goals if not exists
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    
    # Goal input form
    with st.expander("Add a Financial Goal", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            goal_name = st.text_input("Goal Name", key="goal_name")
            goal_type = st.selectbox(
                "Goal Type",
                ["Emergency Fund", "Retirement", "Home Purchase", "Education", "Vehicle", "Vacation", "Wedding", "Other"],
                key="goal_type"
            )
            target_amount = st.number_input("Target Amount (₹) - Today's Value", min_value=0, value=0, step=10000, key="target_amount")
        
        with col2:
            timeframe = st.number_input("Timeframe (Years)", min_value=1, max_value=50, value=5, step=1, key="timeframe")
            priority = st.selectbox(
                "Priority",
                ["Critical", "High", "Medium", "Low"],
                key="goal_priority"
            )
            current_saved = st.number_input("Already Saved (₹)", min_value=0, value=0, step=10000, key="current_saved")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add This Goal", use_container_width=True, key="add_goal"):
                if goal_name and target_amount > 0:
                    new_goal = {
                        "name": goal_name,
                        "type": goal_type.lower().replace(" ", "_"),
                        "target_amount": target_amount,
                        "timeframe_years": timeframe,
                        "priority": priority.lower(),
                        "current_saved": current_saved
                    }
                    st.session_state.goals.append(new_goal)
                    st.success(f"Added goal: {goal_name}")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Clear Goal Form", use_container_width=True, key="clear_goal"):
                st.rerun()
    
    # Display current goals
    if st.session_state.goals:
        st.markdown("### Your Financial Goals")
        
        for i, goal in enumerate(st.session_state.goals):
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{goal['name']}**")
                    st.caption(goal['type'].replace("_", " ").title())
                
                with col2:
                    st.write(f"₹{goal['target_amount']:,.0f}")
                    st.caption("Today's value")
                
                with col3:
                    st.write(f"{goal['timeframe_years']} years")
                
                with col4:
                    priority_color = {
                        "critical": "red",
                        "high": "orange",
                        "medium": "blue",
                        "low": "green"
                    }
                    st.markdown(f"<span style='color:{priority_color.get(goal['priority'], 'black')}'>{goal['priority'].title()}</span>", unsafe_allow_html=True)
                
                with col5:
                    progress = (goal['current_saved'] / goal['target_amount']) * 100 if goal['target_amount'] > 0 else 0
                    st.progress(min(100, progress) / 100)
                    st.caption(f"{progress:.1f}%")
                
                with col6:
                    if st.button("❌", key=f"delete_goal_{i}"):
                        st.session_state.goals.pop(i)
                        st.rerun()
    else:
        st.info("No financial goals added yet. Click 'Add a Financial Goal' to add your goals.")
    
    return st.session_state.goals

# ============================================================================
# SESSION PERSISTENCE FUNCTIONS
# ============================================================================

def save_session():
    """Save current session data to a file"""
    session_data = {
        'personal_info': st.session_state.get('personal_info'),
        'income_data': st.session_state.get('income_data'),
        'assets_data': st.session_state.get('assets_data'),
        'debts': st.session_state.get('debts', []),
        'insurance_data': st.session_state.get('insurance_data'),
        'goals': st.session_state.get('goals', []),
        'current_section': st.session_state.get('current_section', 0),
        'analysis_done': st.session_state.get('analysis_done', False),
        'analysis_results': st.session_state.get('analysis_results'),
        'saved_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to JSON string
    json_data = json.dumps(session_data, default=str, indent=2)
    
    # Create download link
    b64 = base64.b64encode(json_data.encode()).decode()
    
    # Get user name for filename
    user_name = "Financial_Plan"
    if session_data['personal_info'] and 'name' in session_data['personal_info']:
        user_name = session_data['personal_info']['name'].replace(" ", "_")
    
    filename = f"{user_name}_Financial_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="save-btn">💾 Save Session</a>'
    
    st.markdown(f"""
    <div style="text-align: center; margin: 10px 0;">
        {href}
    </div>
    <style>
        .save-btn {{
            display: inline-block;
            background-color: #9b59b6;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 600;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        .save-btn:hover {{
            background-color: #8e44ad;
            color: white;
            text-decoration: none;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    return session_data

def load_session():
    """Load session data from uploaded file"""
    uploaded_file = st.file_uploader("📁 Upload saved session file (.json)", type=['json'])
    
    if uploaded_file is not None:
        try:
            # Read and parse JSON data
            json_data = uploaded_file.read().decode('utf-8')
            session_data = json.loads(json_data)
            
            # Validate the session data
            required_keys = ['personal_info', 'income_data', 'assets_data', 'debts', 
                           'insurance_data', 'goals', 'current_section']
            
            if all(key in session_data for key in required_keys):
                # Restore session state
                st.session_state.personal_info = session_data.get('personal_info')
                st.session_state.income_data = session_data.get('income_data')
                st.session_state.assets_data = session_data.get('assets_data')
                st.session_state.debts = session_data.get('debts', [])
                st.session_state.insurance_data = session_data.get('insurance_data')
                st.session_state.goals = session_data.get('goals', [])
                st.session_state.current_section = session_data.get('current_section', 0)
                st.session_state.analysis_done = session_data.get('analysis_done', False)
                st.session_state.analysis_results = session_data.get('analysis_results')
                
                saved_time = session_data.get('saved_timestamp', 'Unknown')
                st.success(f"✅ Session loaded successfully! (Saved: {saved_time})")
                st.rerun()
            else:
                st.error("Invalid session file format. Missing required data.")
                
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid session file.")
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")

# ============================================================================
# ANALYSIS FUNCTIONS WITH NEW FEATURES
# ============================================================================

def run_analysis(personal_info, income_data, assets_data, debts, insurance_data, goals):
    """Run complete financial analysis with new features"""
    
    constants = FinancialConstants()
    
    # Initialize analyzers
    emergency_analyzer = EmergencyFundAnalyzer(constants)
    retirement_analyzer = RetirementAnalyzer(constants)
    debt_analyzer = DebtAnalyzer(constants)
    goal_analyzer = GoalAnalyzer(constants)
    health_score_calculator = FinancialHealthScore(constants)
    monte_carlo_simulator = MonteCarloSimulator(constants)
    
    # Calculate annual income
    annual_income = sum(income_data["income"].values()) * 12
    
    # Run analyses
    emergency_analysis = emergency_analyzer.analyze(
        monthly_expenses=income_data["monthly_expenses"],
        dependents=personal_info["dependents"],
        cash_and_bank=assets_data["cash_and_bank"]
    )
    
    retirement_analysis = retirement_analyzer.analyze(
        age=personal_info["age"],
        retirement_age=personal_info["retirement_age"],
        monthly_expenses=income_data["monthly_expenses"],
        current_corpus=assets_data["retirement_corpus"] + sum(inv["current_value"] for inv in assets_data["investments"]),
        monthly_investment=assets_data["monthly_investment"]
    )
    
    debt_analysis = debt_analyzer.analyze(debts, annual_income)
    
    # Add goal analysis
    goal_analysis = goal_analyzer.analyze(
        goals=goals,
        age=personal_info["age"]
    )
    
    # Calculate financial health score
    health_score = health_score_calculator.calculate_score(
        emergency_analysis=emergency_analysis,
        retirement_analysis=retirement_analysis,
        debt_analysis=debt_analysis,
        goal_analysis=goal_analysis,
        personal_info=personal_info,
        monthly_savings_rate=income_data.get("savings_rate", 0),
        insurance_coverage=insurance_data
    )
    
    # Run Monte Carlo simulations
    monte_carlo_results = {}
    
    # 1. Retirement simulation
    if personal_info["age"] < personal_info["retirement_age"]:
        # Get allocation based on risk appetite
        allocation = constants.AGE_BASED_ALLOCATION.get(personal_info["risk_appetite"], constants.AGE_BASED_ALLOCATION["moderate"])
        
        retirement_simulation = monte_carlo_simulator.simulate_retirement(
            current_age=personal_info["age"],
            retirement_age=personal_info["retirement_age"],
            current_corpus=assets_data["retirement_corpus"] + sum(inv["current_value"] for inv in assets_data["investments"]),
            monthly_investment=assets_data["monthly_investment"],
            target_corpus=retirement_analysis["corpus_needed"],
            equity_allocation=allocation["equity"],
            debt_allocation=allocation["debt"],
            gold_allocation=0.05  # Fixed gold allocation
        )
        monte_carlo_results["retirement"] = retirement_simulation
    
    # 2. Goal simulations
    goal_simulations = []
    for goal in goals:
        goal_sim = monte_carlo_simulator.simulate_goal(
            goal_name=goal["name"],
            current_saved=goal.get("current_saved", 0),
            monthly_saving=0,  # We'll calculate this properly
            target_amount=goal["target_amount"],
            years=goal["timeframe_years"],
            equity_allocation=0.6,  # Default for goals
            debt_allocation=0.35,
            gold_allocation=0.05
        )
        goal_simulations.append(goal_sim)
    
    monte_carlo_results["goals"] = goal_simulations
    
    # 3. Portfolio growth simulation
    portfolio_simulation = monte_carlo_simulator.simulate_portfolio_growth(
        initial_investment=assets_data["total_investments"],
        monthly_contribution=assets_data["monthly_investment"],
        years=10,  # 10-year projection
        equity_allocation=0.6,
        debt_allocation=0.35,
        gold_allocation=0.05
    )
    monte_carlo_results["portfolio"] = portfolio_simulation
    
    # Generate recommendations
    recommendations = generate_recommendations(
        emergency_analysis,
        retirement_analysis,
        debt_analysis,
        goal_analysis,
        personal_info,
        annual_income,
        insurance_data,
        health_score
    )
    
    return {
        "emergency": emergency_analysis,
        "retirement": retirement_analysis,
        "debt": debt_analysis,
        "goals": goal_analysis,
        "health_score": health_score,
        "monte_carlo": monte_carlo_results,
        "recommendations": recommendations,
        "summary": {
            "net_worth": (assets_data["cash_and_bank"] + 
                         sum(inv["current_value"] for inv in assets_data["investments"]) +
                         assets_data["retirement_corpus"] +
                         assets_data["real_estate_value"] +
                         assets_data["other_assets"]) - debt_analysis["total_debt"],
            "annual_income": annual_income,
            "monthly_savings": sum(income_data["income"].values()) - income_data["monthly_expenses"],
            "savings_rate": income_data.get("savings_rate", 0)
        }
    }

def generate_recommendations(emergency, retirement, debt, goals, personal_info, annual_income, insurance, health_score):
    """Generate personalized recommendations"""
    
    recommendations = []
    
    # Add health score-based recommendation
    health_category = health_score.get("category", "Fair")
    health_total = health_score.get("total_score", 50)
    
    if health_category == "Poor":
        recommendations.append({
            "title": "🚨 Urgent Financial Health Improvement Needed",
            "description": f"Your financial health score is {health_total}/100 ({health_category}). Immediate action is required in multiple areas.",
            "priority": "critical",
            "actions": [
                "Focus on building emergency fund first",
                "Create a debt repayment plan",
                "Review all financial areas systematically"
            ],
            "timeline": "3 months",
            "impact": "Improve financial health score by 20+ points"
        })
    elif health_category == "Fair":
        recommendations.append({
            "title": "📈 Improve Your Financial Health",
            "description": f"Your financial health score is {health_total}/100 ({health_category}). Focus on weakest areas to reach 'Good' status.",
            "priority": "high",
            "actions": [
                "Identify and address weakest component from health score",
                "Set specific improvement targets",
                "Track progress monthly"
            ],
            "timeline": "6 months",
            "impact": f"Reach 'Good' financial health (65+ score)"
        })
    
    # Emergency fund recommendations
    if emergency["priority"] in ["high", "critical"]:
        monthly_saving = emergency["shortfall"] / 6
        recommendations.append({
            "title": "🚨 Build Emergency Fund",
            "description": f"Your emergency fund covers only {emergency['months_coverage']:.1f} months. You need ₹{emergency['shortfall']:,.0f} more for {emergency['recommended_months']} months coverage.",
            "priority": "critical" if emergency["priority"] == "critical" else "high",
            "actions": [
                f"Save ₹{monthly_saving:,.0f} monthly for next 6 months",
                "Keep emergency fund in savings account or liquid funds",
                "Don't invest emergency fund in risky assets"
            ],
            "timeline": "6 months",
            "impact": "Financial security during emergencies"
        })
    
    # Debt recommendations
    if debt["high_interest_debt_count"] > 0:
        recommendations.append({
            "title": "💰 Pay Off High-Interest Debt",
            "description": f"You have {debt['high_interest_debt_count']} high-interest debts. Paying these off could save you ₹{debt['high_interest_savings_possible']:,.0f} in interest.",
            "priority": "high",
            "actions": [
                "List debts by interest rate (highest first)",
                "Allocate extra payments to highest interest debt",
                "Consider debt consolidation"
            ],
            "timeline": "12 months",
            "impact": f"Save ₹{debt['high_interest_savings_possible']:,.0f} in interest"
        })
    
    if debt["debt_to_income_ratio"] > 40:
        recommendations.append({
            "title": "📉 Reduce Debt Burden",
            "description": f"Your debt-to-income ratio is {debt['debt_to_income_ratio']:.1f}% ({debt['status']}). Target should be below 40%.",
            "priority": "high" if debt["debt_to_income_ratio"] > 60 else "medium",
            "actions": [
                "Avoid taking new debt",
                "Create budget for extra debt repayment",
                "Consider balance transfer options"
            ],
            "timeline": "6-12 months",
            "impact": "Improved cash flow"
        })
    
    # Retirement recommendations
    if retirement["priority"] in ["high", "critical"]:
        recommendations.append({
            "title": "🏖️ Boost Retirement Savings",
            "description": f"Retirement readiness is {retirement['readiness_percentage']:.1f}%. Need additional ₹{retirement['additional_monthly_saving_needed']:,.0f} monthly.",
            "priority": retirement["priority"],
            "actions": [
                f"Increase retirement contribution by ₹{retirement['additional_monthly_saving_needed']:,.0f} monthly",
                "Maximize tax-advantaged accounts (EPF, PPF, NPS)",
                "Review portfolio allocation annually"
            ],
            "timeline": f"{retirement['years_to_retirement']} years",
            "impact": f"Retirement corpus of ₹{retirement['corpus_needed']:,.0f}"
        })
    
    # Insurance recommendations
    required_life_insurance = annual_income * 10
    if insurance["life_insurance"] < required_life_insurance * 0.7:
        shortfall = required_life_insurance - insurance["life_insurance"]
        recommendations.append({
            "title": "🛡️ Increase Life Insurance",
            "description": f"Your life insurance coverage (₹{insurance['life_insurance']:,.0f}) is less than recommended (₹{required_life_insurance:,.0f}).",
            "priority": "medium",
            "actions": [
                f"Consider term insurance of ₹{shortfall:,.0f}",
                "Review existing policies",
                "Ensure coverage till retirement age"
            ],
            "timeline": "1 month",
            "impact": "Financial protection for dependents"
        })
    
    # Goal recommendations
    if goals["total_goals"] > 0:
        # Check if goals are feasible given current savings
        total_monthly_required = (
            (emergency["shortfall"] / 6 if emergency["priority"] in ["high", "critical"] else 0) +
            (retirement["additional_monthly_saving_needed"] if retirement["priority"] in ["high", "critical"] else 0) +
            goals["total_monthly_investment_needed"]
        )
        
        monthly_savings_capacity = annual_income / 12 * 0.3  # Assuming 30% savings capacity
        
        if total_monthly_required > monthly_savings_capacity:
            recommendations.append({
                "title": "🎯 Prioritize Financial Goals",
                "description": f"You have {goals['total_goals']} financial goals requiring ₹{goals['total_monthly_investment_needed']:,.0f}/month. This exceeds your savings capacity.",
                "priority": "high",
                "actions": [
                    "Review and prioritize goals by importance",
                    "Consider extending timeframes for less critical goals",
                    "Focus on 3-4 most important goals first"
                ],
                "timeline": "1 month",
                "impact": "Realistic goal achievement"
            })
        
        # Check for critical/high priority goals
        critical_goals = [g for g in goals["goals_details"] if g["priority"] in ["critical", "high"]]
        for goal in critical_goals[:3]:  # Top 3 critical/high priority goals
            if goal["progress_percentage"] < 40:
                recommendations.append({
                    "title": f"🚨 Focus on: {goal['name']}",
                    "description": f"{goal['name']} is only {goal['progress_percentage']:.1f}% funded. Needs ₹{goal['monthly_saving_needed']:,.0f}/month for {goal['timeframe_years']} years.",
                    "priority": goal["priority"],
                    "actions": [
                        f"Set up SIP of ₹{goal['monthly_saving_needed']:,.0f}/month",
                        "Review investment allocation for this goal",
                        f"Target completion: {goal['completion_year']}"
                    ],
                    "timeline": f"{goal['timeframe_years']} years",
                    "impact": f"Achieve {goal['name']} goal"
                })
    
    # Investment allocation recommendation
    recommendations.append({
        "title": "📊 Optimize Investment Allocation",
        "description": f"Based on your {personal_info['risk_appetite']} risk profile, review your investment allocation.",
        "priority": "medium",
        "actions": [
            "Review current portfolio allocation",
            "Rebalance to recommended allocation",
            "Use SIPs for regular investing"
        ],
        "timeline": "Quarterly",
        "impact": "Better risk-adjusted returns"
    })
    
    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))
    
    return recommendations

# ============================================================================
# NEW: DISPLAY FUNCTIONS FOR NEW FEATURES
# ============================================================================

def display_financial_health_score(health_score):
    """Display financial health score"""
    
    st.markdown('<h2 class="section-header">❤️ Financial Health Score</h2>', unsafe_allow_html=True)
    
    # Main score display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display score with color based on category
        score = health_score["total_score"]
        category = health_score["category"]
        color = health_score["color"]
        
        # Determine CSS class
        if score >= 80:
            css_class = "health-score-excellent"
        elif score >= 65:
            css_class = "health-score-good"
        elif score >= 50:
            css_class = "health-score-fair"
        else:
            css_class = "health-score-poor"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 3px solid {color};">
            <div class="{css_class}">{score}/100</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: {color}; margin-top: 10px;">{category}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**{health_score['description']}**")
    
    with col2:
        # Display component breakdown
        st.markdown("#### Component Breakdown")
        
        components = health_score["components"]
        
        for comp_name, comp_data in components.items():
            # Format component name
            display_name = comp_name.replace("_", " ").title()
            
            col_a, col_b, col_c = st.columns([2, 1, 3])
            
            with col_a:
                st.write(f"**{display_name}**")
            
            with col_b:
                score_val = comp_data["score"]
                # Simple progress bar
                progress = score_val / 100
                st.progress(progress)
                st.caption(f"{score_val:.1f}")
            
            with col_c:
                st.caption(comp_data["description"])
    
    # Health assessment
    st.markdown("---")
    st.markdown("#### Health Assessment")
    
    score_val = health_score["total_score"]
    if score_val >= 80:
        st.success("""
        **Excellent!** Your financial health is in great shape. Key strengths:
        - Strong emergency fund and savings rate
        - Healthy debt management
        - Good retirement planning
        - Keep maintaining this discipline!
        """)
    elif score_val >= 65:
        st.info("""
        **Good!** Your financial health is solid. Areas for improvement:
        - Review your weakest component from above
        - Consider increasing savings rate
        - Ensure adequate insurance coverage
        - You're close to excellent status!
        """)
    elif score_val >= 50:
        st.warning("""
        **Fair.** Your financial health needs attention. Priority actions:
        - Focus on emergency fund if inadequate
        - Reduce high-interest debt
        - Increase retirement contributions
        - Set specific financial goals
        """)
    else:
        st.error("""
        **Poor.** Immediate action required. Critical areas:
        - Build emergency fund immediately
        - Create debt repayment plan
        - Increase income or reduce expenses
        - Seek professional financial advice
        - Start with one small improvement today
        """)

def display_monte_carlo_results(monte_carlo_results, retirement_analysis, goals_analysis):
    """Display Monte Carlo simulation results"""
    
    st.markdown('<h2 class="section-header">🎲 Monte Carlo Simulations</h2>', unsafe_allow_html=True)
    
    st.info("""
    **What are Monte Carlo Simulations?**
    These are probability-based projections that show potential outcomes for your investments.
    They account for market volatility and uncertainty, giving you a range of possible results.
    """)
    
    # Retirement simulation
    if "retirement" in monte_carlo_results:
        retirement_sim = monte_carlo_results["retirement"]
        
        st.markdown("#### Retirement Projection Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_prob = retirement_sim.get("success_probability", 0)
            if success_prob >= 80:
                color = "green"
                status = "High Confidence"
            elif success_prob >= 60:
                color = "orange"
                status = "Moderate Confidence"
            else:
                color = "red"
                status = "Low Confidence"
            
            st.metric(
                "Success Probability",
                f"{success_prob:.1f}%",
                status,
                delta_color="normal" if success_prob >= 80 else "off"
            )
        
        with col2:
            median_value = retirement_sim["statistics"]["percentiles"]["p50"]
            required = retirement_sim.get("required_corpus", 0)
            if median_value >= required:
                status = "Above Target"
                delta_color = "normal"
            else:
                status = "Below Target"
                delta_color = "off"
            
            st.metric(
                "Median Projection",
                f"₹{median_value:,.0f}",
                status,
                delta_color=delta_color
            )
        
        with col3:
            range_low = retirement_sim["statistics"]["percentiles"]["p5"]
            range_high = retirement_sim["statistics"]["percentiles"]["p95"]
            st.metric(
                "90% Confidence Range",
                f"₹{range_low:,.0f} - ₹{range_high:,.0f}"
            )
        
        # Retirement projection chart
        if "yearly_paths" in retirement_sim:
            st.markdown("##### Retirement Corpus Projection Paths")
            
            # Create DataFrame for paths
            years = list(range(len(retirement_sim["yearly_paths"][0])))
            
            # Sample a few paths for visualization
            sample_paths = retirement_sim["yearly_paths"][:50]  # Show first 50 paths
            
            fig = go.Figure()
            
            # Add individual paths (transparent)
            for path in sample_paths:
                fig.add_trace(go.Scatter(
                    x=years,
                    y=path,
                    mode='lines',
                    line=dict(color='rgba(52, 152, 219, 0.1)', width=1),
                    showlegend=False
                ))
            
            # Add percentiles
            percentiles = retirement_sim["statistics"]["percentiles"]
            
            # Calculate percentiles across all paths
            all_paths_array = np.array(retirement_sim["yearly_paths"])
            p5_values = np.percentile(all_paths_array, 5, axis=0)
            p50_values = np.percentile(all_paths_array, 50, axis=0)
            p95_values = np.percentile(all_paths_array, 95, axis=0)
            
            fig.add_trace(go.Scatter(
                x=years,
                y=p5_values,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='5th Percentile (Worst 5%)'
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=p50_values,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Median (50th Percentile)'
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=p95_values,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='95th Percentile (Best 5%)'
            ))
            
            # Add target line if available
            if "required_corpus" in retirement_sim:
                target = retirement_sim["required_corpus"]
                fig.add_hline(
                    y=target,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text=f"Target: ₹{target:,.0f}",
                    annotation_position="bottom right"
                )
            
            fig.update_layout(
                title="Retirement Corpus Projection (Monte Carlo Simulation)",
                xaxis_title="Years from Now",
                yaxis_title="Portfolio Value (₹)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Goal simulations
    if "goals" in monte_carlo_results and monte_carlo_results["goals"]:
        st.markdown("#### Goal Achievement Probability")
        
        goal_data = []
        for goal_sim in monte_carlo_results["goals"]:
            prob = goal_sim["success_probability"]
            goal_data.append({
                "Goal": goal_sim["goal_name"],
                "Target": f"₹{goal_sim['target_amount']:,.0f}",
                "Probability": f"{prob:.1f}%",
                "Median Value": f"₹{goal_sim['median_value']:,.0f}",
                "Status": "Likely" if prob >= 70 else "Uncertain" if prob >= 40 else "Unlikely"
            })
        
        df_goals = pd.DataFrame(goal_data)
        
        # Display as table with color coding
        for _, row in df_goals.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{row['Goal']}**")
            
            with col2:
                st.write(row["Target"])
            
            with col3:
                prob = float(row["Probability"].replace("%", ""))
                if prob >= 70:
                    color = "green"
                elif prob >= 40:
                    color = "orange"
                else:
                    color = "red"
                
                st.markdown(f"<span style='color:{color}; font-weight:bold'>{row['Probability']}</span>", unsafe_allow_html=True)
            
            with col4:
                st.write(row["Median Value"])
            
            with col5:
                status = row["Status"]
                if status == "Likely":
                    st.success("✅ " + status)
                elif status == "Uncertain":
                    st.warning("⚠️ " + status)
                else:
                    st.error("❌ " + status)
    
    # Portfolio simulation
    if "portfolio" in monte_carlo_results:
        portfolio_sim = monte_carlo_results["portfolio"]
        
        st.markdown("#### 10-Year Portfolio Growth Projection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mean_value = portfolio_sim["statistics"]["mean"]
            st.metric("Average Projection", f"₹{mean_value:,.0f}")
        
        with col2:
            min_value = portfolio_sim["statistics"]["min"]
            max_value = portfolio_sim["statistics"]["max"]
            st.metric("Range", f"₹{min_value:,.0f} - ₹{max_value:,.0f}")
        
        with col3:
            std_dev = portfolio_sim["statistics"]["std"]
            st.metric("Volatility (Std Dev)", f"₹{std_dev:,.0f}")
        
        # Histogram of final values
        st.markdown("##### Distribution of Possible Outcomes")
        
        fig = px.histogram(
            x=portfolio_sim["final_values"],
            nbins=50,
            labels={'x': 'Final Portfolio Value (₹)', 'y': 'Frequency'},
            title="Distribution of Portfolio Values after 10 Years"
        )
        
        # Add vertical line for mean
        fig.add_vline(
            x=portfolio_sim["statistics"]["mean"],
            line_dash="dash",
            line_color="red",
            annotation_text="Mean"
        )
        
        # Add vertical line for median
        fig.add_vline(
            x=portfolio_sim["statistics"]["median"],
            line_dash="dash",
            line_color="blue",
            annotation_text="Median"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("##### Interpretation")
        
        st.info("""
        **Understanding the Results:**
        - **Success Probability**: The chance your investments will meet or exceed your target
        - **Median Projection**: The middle value (50% of simulations above, 50% below)
        - **Confidence Range**: Where 90% of outcomes fall (between 5th and 95th percentile)
        - **Distribution**: Shows all possible outcomes from 1,000 simulations
        
        **Recommendations based on results:**
        1. If success probability < 60%: Consider increasing savings or adjusting allocation
        2. If confidence range is wide: Your investments are volatile, consider more stable assets
        3. If median is below target: You need to save more or adjust your expectations
        """)

def display_results(analysis_results, personal_info):
    """Display analysis results with new features"""
    
    # Financial Health Score at the top
    if "health_score" in analysis_results:
        display_financial_health_score(analysis_results["health_score"])
        st.markdown("---")
    
    st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Key Metrics - Updated to include health score
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Financial Health Score
        if "health_score" in analysis_results:
            score = analysis_results["health_score"]["total_score"]
            category = analysis_results["health_score"]["category"]
            st.metric(
                "Financial Health",
                f"{score}/100",
                category,
                delta_color="normal" if score >= 65 else "off"
            )
        else:
            st.metric("Financial Health", "N/A")
    
    with col2:
        st.metric(
            "Emergency Fund Status",
            analysis_results["emergency"]["status"],
            delta=f"{analysis_results['emergency']['adequacy_percentage']:.1f}%",
            delta_color="normal" if analysis_results["emergency"]["adequacy_percentage"] >= 100 else "off"
        )
    
    with col3:
        st.metric(
            "Retirement Readiness",
            analysis_results["retirement"]["status"],
            delta=f"{analysis_results['retirement']['readiness_percentage']:.1f}%",
            delta_color="normal" if analysis_results["retirement"]["readiness_percentage"] >= 70 else "off"
        )
    
    with col4:
        st.metric(
            "Debt Situation",
            analysis_results["debt"]["status"],
            delta=f"{analysis_results['debt']['debt_to_income_ratio']:.1f}% DTI",
            delta_color="normal" if analysis_results["debt"]["debt_to_income_ratio"] <= 40 else "off"
        )
    
    with col5:
        st.metric(
            "Net Worth",
            f"₹{analysis_results['summary']['net_worth']:,.0f}",
            delta=f"₹{analysis_results['summary']['monthly_savings']:,.0f}/month savings"
        )
    
    # Detailed Analysis - Updated tabs
    tab_names = ["Health Score", "Monte Carlo", "Emergency Fund", "Retirement", "Debt", "Goals", "Recommendations"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Health Score tab
        if "health_score" in analysis_results:
            display_financial_health_score(analysis_results["health_score"])
        else:
            st.info("Health score analysis not available.")
    
    with tabs[1]:  # Monte Carlo tab
        if "monte_carlo" in analysis_results:
            display_monte_carlo_results(
                analysis_results["monte_carlo"],
                analysis_results["retirement"],
                analysis_results["goals"]
            )
        else:
            st.info("Monte Carlo simulations not available.")
    
    with tabs[2]:  # Emergency Fund
        st.markdown("### Emergency Fund Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart for emergency fund
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = analysis_results["emergency"]["adequacy_percentage"],
                title = {'text': "Adequacy Percentage"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': analysis_results["emergency"]["color"]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Key Metrics")
            st.metric("Current Emergency Fund", f"₹{analysis_results['emergency']['current_fund']:,.0f}")
            st.metric("Required Emergency Fund", f"₹{analysis_results['emergency']['required_fund']:,.0f}")
            st.metric("Months Coverage", f"{analysis_results['emergency']['months_coverage']:.1f} months")
            st.metric("Shortfall", f"₹{analysis_results['emergency']['shortfall']:,.0f}")
    
    with tabs[3]:  # Retirement
        st.markdown("### Retirement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Retirement Readiness Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=analysis_results["retirement"]["readiness_percentage"],
                title={'text': "Retirement Readiness", 'font': {'size': 20}},
                number={'font': {'size': 38}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': analysis_results["retirement"]["color"]},
                    'shape': "angular",
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"},
                    ]
                },
                domain={'x': [0, 1], 'y': [0, 0.75]}
            ))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Retirement Details")
            st.metric("Years to Retirement", analysis_results["retirement"]["years_to_retirement"])
            st.metric("Corpus Needed", f"₹{analysis_results['retirement']['corpus_needed']:,.0f}")
            st.metric("Projected Corpus", f"₹{analysis_results['retirement']['projected_corpus']:,.0f}")
            st.metric("Additional Monthly Saving", f"₹{analysis_results['retirement']['additional_monthly_saving_needed']:,.0f}")
    
    with tabs[4]:  # Debt
        st.markdown("### Debt Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Debt to income ratio
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = analysis_results["debt"]["debt_to_income_ratio"],
                title = {'text': "Debt-to-Income Ratio"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': analysis_results["debt"]["color"]},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Debt Summary")
            st.metric("Total Debt", f"₹{analysis_results['debt']['total_debt']:,.0f}")
            st.metric("Monthly EMI", f"₹{analysis_results['debt']['monthly_emi']:,.0f}")
            st.metric("High-Interest Debts", analysis_results["debt"]["high_interest_debt_count"])
            st.metric("Potential Interest Savings", f"₹{analysis_results['debt']['high_interest_savings_possible']:,.0f}")
    
    with tabs[5]:  # Goals
        st.markdown("### Financial Goals Analysis")
        
        if analysis_results["goals"]["total_goals"] > 0:
            # Display goals summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Goals", analysis_results["goals"]["total_goals"])
            
            with col2:
                st.metric("Total Target Amount (Today)", f"₹{analysis_results['goals']['total_target_amount_pv']:,.0f}")
                st.caption("Present Value")
            
            with col3:
                st.metric("Monthly Investment Needed", f"₹{analysis_results['goals']['total_monthly_investment_needed']:,.0f}")
            
            # Display inflation-adjusted amount
            st.info(f"💰 **Inflation Note**: Your goals will require ₹{analysis_results['goals']['total_target_amount_fv']:,.0f} in future value (after inflation adjustment)")
            
            # Display individual goals
            st.markdown("#### Individual Goal Analysis")
            
            for goal in analysis_results["goals"]["goals_details"]:
                with st.expander(f"{goal['name']} - {goal['status']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Target Amount (Today)", f"₹{goal['target_amount_pv']:,.0f}")
                        st.caption("Present Value")
                        st.metric("Target Amount (Future)", f"₹{goal['target_amount_fv']:,.0f}")
                        st.caption("Future Value (inflation-adjusted)")
                    
                    with col2:
                        st.metric("Timeframe", f"{goal['timeframe_years']} years")
                        st.metric("Priority", goal['priority'].title())
                    
                    with col3:
                        st.metric("Current Savings", f"₹{goal['current_saved']:,.0f}")
                        st.metric("Monthly Needed", f"₹{goal['monthly_saving_needed']:,.0f}")
                    
                    # Progress bar
                    progress = goal['progress_percentage'] / 100
                    st.progress(min(1.0, progress))
                    st.caption(f"Progress: {goal['progress_percentage']:.1f}%")
                    
                    # Completion year
                    st.info(f"Target completion year: {goal['completion_year']}")
        else:
            st.info("No financial goals have been set. Add goals in the Goals section.")
    
    with tabs[6]:  # Recommendations
        st.markdown("### Priority Recommendations")
        
        for i, rec in enumerate(analysis_results["recommendations"]):
            with st.container():
                st.markdown(f"#### {i+1}. {rec['title']}")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(rec["description"])
                    st.markdown("**Actions:**")
                    for action in rec["actions"]:
                        st.markdown(f"- {action}")
                
                with col2:
                    priority_color = {
                        "critical": "red",
                        "high": "orange",
                        "medium": "blue",
                        "low": "green"
                    }
                    st.markdown(
                        f"<div style='background-color:{priority_color.get(rec['priority'], 'gray')}; color:white; padding:10px; border-radius:5px; text-align:center;'>"
                        f"{rec['priority'].upper()}</div>",
                        unsafe_allow_html=True
                    )
                    st.write(f"**Timeline:** {rec['timeline']}")
                    st.write(f"**Impact:** {rec['impact']}")
                
                st.divider()

# ============================================================================
# 7. MAIN APPLICATION WITH ALL FEATURES
# ============================================================================

def main():
    """Main application function"""
    
    # Display welcome
    display_welcome()
    
    # Initialize session state with defaults
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 0
    if 'personal_info' not in st.session_state:
        st.session_state.personal_info = None
    if 'income_data' not in st.session_state:
        st.session_state.income_data = None
    if 'assets_data' not in st.session_state:
        st.session_state.assets_data = None
    if 'debts' not in st.session_state:
        st.session_state.debts = []
    if 'insurance_data' not in st.session_state:
        st.session_state.insurance_data = None
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    if 'show_save' not in st.session_state:
        st.session_state.show_save = False
    if 'show_load' not in st.session_state:
        st.session_state.show_load = False
    
    # Define sections
    sections = [
        ("Personal Info", create_personal_info_section),
        ("Income & Expenses", create_income_expenses_section),
        ("Assets", create_assets_section),
        ("Debts", create_debts_section),
        ("Insurance", create_insurance_section),
        ("Goals", create_goals_section),
        ("Analysis & Report", None)
    ]
    
    # Create sidebar for navigation and session management
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money-bag.png", width=100)
        st.title("Financial Planning Engine")
        
        # Session Management Section
        st.markdown("### 💾 Session Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True, help="Save current session to file"):
                st.session_state.show_save = True
                st.rerun()
        
        with col2:
            if st.button("📂 Load", use_container_width=True, help="Load saved session from file"):
                st.session_state.show_load = True
                st.rerun()
        
        st.divider()
        
        # Progress tracking
        current_section_name = sections[st.session_state.current_section][0]
        progress = (st.session_state.current_section + 1) / len(sections)
        
        st.progress(progress)
        st.caption(f"**Progress: {int(progress * 100)}%**")
        st.markdown(f"**Current: {current_section_name}**")
        
        st.divider()
        
        # Quick navigation
        st.markdown("### 🗺️ Quick Navigation")
        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.session_state.current_section < len(sections) - 1:
            if st.button("⏭️ Skip to Analysis", use_container_width=True):
                st.session_state.current_section = len(sections) - 1
                st.rerun()
        
        st.divider()
        
        # Section indicators
        st.markdown("### 📋 Sections:")
        for i, (section_name, _) in enumerate(sections):
            if i == st.session_state.current_section:
                st.markdown(f"▶ **{section_name}**")
            elif i < st.session_state.current_section:
                st.markdown(f"✓ {section_name}")
            else:
                st.markdown(f"○ {section_name}")
        
        # Show health score if available
        if st.session_state.analysis_done and st.session_state.analysis_results:
            st.divider()
            st.markdown("### ❤️ Your Financial Health")
            health_score = st.session_state.analysis_results.get("health_score", {})
            if health_score:
                score = health_score.get("total_score", 0)
                category = health_score.get("category", "N/A")
                color = health_score.get("color", "#000000")
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: {color};">{score}/100</div>
                    <div style="font-size: 1rem; color: {color};">{category}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Main content area
    # Show save/load section if triggered
    if st.session_state.get('show_save', False):
        st.markdown('<h2 class="section-header">💾 Save Current Session</h2>', unsafe_allow_html=True)
        
        # Check if there's data to save
        has_data = any([
            st.session_state.get('personal_info'),
            st.session_state.get('income_data'),
            st.session_state.get('assets_data'),
            st.session_state.get('debts'),
            st.session_state.get('insurance_data'),
            st.session_state.get('goals')
        ])
        
        if has_data:
            save_session()
        else:
            st.warning("No data to save yet. Complete at least one section.")
        
        if st.button("← Back to Main", use_container_width=True):
            st.session_state.show_save = False
            st.rerun()
        return
    
    if st.session_state.get('show_load', False):
        st.markdown('<h2 class="section-header">📂 Load Saved Session</h2>', unsafe_allow_html=True)
        load_session()
        
        if st.button("← Back to Main", use_container_width=True):
            st.session_state.show_load = False
            st.rerun()
        return
    
    # Navigation buttons at top
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.current_section > 0:
            if st.button("⬅ Previous", use_container_width=True):
                st.session_state.current_section -= 1
                st.rerun()
    
    with col2:
        st.markdown(f"**Step {st.session_state.current_section + 1} of {len(sections)}**")
    
    with col3:
        if st.session_state.current_section < len(sections) - 1:
            if st.button("Next ➡", use_container_width=True, type="primary"):
                # Validate before moving forward
                current_section_name = sections[st.session_state.current_section][0]
                
                if current_section_name == "Income & Expenses" and not st.session_state.personal_info:
                    st.error("Please complete Personal Information first!")
                    st.stop()
                elif current_section_name == "Assets" and not st.session_state.income_data:
                    st.error("Please complete Income & Expenses first!")
                    st.stop()
                elif current_section_name == "Debts" and not st.session_state.assets_data:
                    st.error("Please complete Assets section first!")
                    st.stop()
                elif current_section_name == "Insurance" and not st.session_state.debts:
                    st.error("Please complete Debts section first!")
                    st.stop()
                elif current_section_name == "Goals" and not st.session_state.insurance_data:
                    st.error("Please complete Insurance section first!")
                    st.stop()
                elif current_section_name == "Analysis & Report":
                    required = [st.session_state.personal_info, st.session_state.income_data, 
                               st.session_state.assets_data, st.session_state.insurance_data]
                    if not all(required) or len(st.session_state.goals) == 0:
                        st.error("Please complete all sections first!")
                        st.stop()
                
                st.session_state.current_section += 1
                st.rerun()
    
    with col4:
        if st.session_state.current_section == len(sections) - 2:
            if st.button("Analyze Now", use_container_width=True, type="secondary"):
                required = [st.session_state.personal_info, st.session_state.income_data, 
                           st.session_state.assets_data, st.session_state.insurance_data]
                if not all(required) or len(st.session_state.goals) == 0:
                    st.error("Please complete all sections first!")
                    st.stop()
                st.session_state.current_section = len(sections) - 1
                st.rerun()
    
    st.markdown("---")
    
    # Display current section
    current_section_name, section_function = sections[st.session_state.current_section]
    
    st.markdown(f'<div class="section-header">{current_section_name}</div>', unsafe_allow_html=True)
    
    if section_function:
        # Regular section
        result = section_function()
        
        # Store result
        if current_section_name == "Personal Info":
            st.session_state.personal_info = result
        elif current_section_name == "Income & Expenses":
            st.session_state.income_data = result
        elif current_section_name == "Assets":
            st.session_state.assets_data = result
        elif current_section_name == "Debts":
            st.session_state.debts = result
        elif current_section_name == "Insurance":
            st.session_state.insurance_data = result
        elif current_section_name == "Goals":
            st.session_state.goals = result
        
        st.success(f"✅ {current_section_name} completed!")
        
    else:
        # Analysis & Report section
        # Check if all data is available
        required = [st.session_state.personal_info, st.session_state.income_data, 
                   st.session_state.assets_data, st.session_state.insurance_data]
        
        if not all(required) or len(st.session_state.goals) == 0:
            st.error("⚠️ **Incomplete Data**: Please complete all sections first!")
            
            missing_sections = []
            if not st.session_state.personal_info:
                missing_sections.append("Personal Info")
            if not st.session_state.income_data:
                missing_sections.append("Income & Expenses")
            if not st.session_state.assets_data:
                missing_sections.append("Assets")
            if not st.session_state.insurance_data:
                missing_sections.append("Insurance")
            if len(st.session_state.goals) == 0:
                missing_sections.append("Goals")
            
            st.markdown(f"**Missing sections:** {', '.join(missing_sections)}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅ Go Back to Previous Section"):
                    st.session_state.current_section -= 1
                    st.rerun()
            st.stop()
        
        st.markdown("### 🚀 Ready to Analyze Your Finances?")
        st.markdown("""
        **New Features Available:**
        - ❤️ **Financial Health Score**: Single metric showing overall financial health
        - 🎲 **Monte Carlo Simulations**: Probability-based projections for retirement and goals
        - 📊 **Enhanced Analysis**: More detailed insights and recommendations
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if not st.session_state.analysis_done:
                if st.button("🚀 Run Financial Analysis", use_container_width=True, type="primary"):
                    with st.spinner("Running comprehensive analysis with Monte Carlo simulations..."):
                        analysis_results = run_analysis(
                            st.session_state.personal_info,
                            st.session_state.income_data,
                            st.session_state.assets_data,
                            st.session_state.debts,
                            st.session_state.insurance_data,
                            st.session_state.goals
                        )
                        
                        st.session_state.analysis_results = analysis_results
                        st.session_state.analysis_done = True
                        st.success("✅ Analysis completed with Monte Carlo simulations!")
                        st.balloons()
                        st.rerun()
            else:
                st.success("✅ Analysis already completed!")
        
        # Show results if analysis is done
        if st.session_state.analysis_done and st.session_state.analysis_results:
            display_results(st.session_state.analysis_results, st.session_state.personal_info)
            
            st.markdown("---")
            st.markdown("### 📄 Download Report")
            
            # Note: PDF report generation would need to be updated to include new features
            st.info("**Note**: The PDF report will include your Financial Health Score and key analysis results.")
            
            if st.button("Generate Enhanced PDF Report", use_container_width=True):
                st.info("Enhanced PDF report generation is being developed...")
                # For now, we'll use the existing PDF generation
                # In a full implementation, you would update create_pdf_report to include new features

# ============================================================================
# 8. RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
