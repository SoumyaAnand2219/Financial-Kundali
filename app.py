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
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from math import pow
import plotly.graph_objects as go
import plotly.express as px

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
        
        st.info("""
        🎯 **What this tool does:**
        - Analyzes your emergency fund adequacy
        - Calculates retirement corpus needed
        - Evaluates debt burden and suggests strategies
        - Provides actionable recommendations
        - Generates a detailed PDF report
        """)

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
        "monthly_investment": total_monthly_investment
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
            target_amount = st.number_input("Target Amount (₹)", min_value=0, value=0, step=10000, key="target_amount")
        
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
    
    return st.session_state.goals

def run_analysis(personal_info, income_data, assets_data, debts, insurance_data, goals):
    """Run complete financial analysis"""
    
    constants = FinancialConstants()
    
    # Initialize analyzers
    emergency_analyzer = EmergencyFundAnalyzer(constants)
    retirement_analyzer = RetirementAnalyzer(constants)
    debt_analyzer = DebtAnalyzer(constants)
    
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
    
    # Generate recommendations
    recommendations = generate_recommendations(
        emergency_analysis,
        retirement_analysis,
        debt_analysis,
        personal_info,
        annual_income,
        insurance_data
    )
    
    return {
        "emergency": emergency_analysis,
        "retirement": retirement_analysis,
        "debt": debt_analysis,
        "recommendations": recommendations,
        "summary": {
            "net_worth": (assets_data["cash_and_bank"] + 
                         sum(inv["current_value"] for inv in assets_data["investments"]) +
                         assets_data["retirement_corpus"] +
                         assets_data["real_estate_value"] +
                         assets_data["other_assets"]) - debt_analysis["total_debt"],
            "annual_income": annual_income,
            "monthly_savings": sum(income_data["income"].values()) - income_data["monthly_expenses"]
        }
    }

def generate_recommendations(emergency, retirement, debt, personal_info, annual_income, insurance):
    """Generate personalized recommendations"""
    
    recommendations = []
    
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

def display_results(analysis_results, personal_info):
    """Display analysis results"""
    
    st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Emergency Fund Status",
            analysis_results["emergency"]["status"],
            delta=f"{analysis_results['emergency']['adequacy_percentage']:.1f}%",
            delta_color="normal" if analysis_results["emergency"]["adequacy_percentage"] >= 100 else "off"
        )
    
    with col2:
        st.metric(
            "Retirement Readiness",
            analysis_results["retirement"]["status"],
            delta=f"{analysis_results['retirement']['readiness_percentage']:.1f}%",
            delta_color="normal" if analysis_results["retirement"]["readiness_percentage"] >= 70 else "off"
        )
    
    with col3:
        st.metric(
            "Debt Situation",
            analysis_results["debt"]["status"],
            delta=f"{analysis_results['debt']['debt_to_income_ratio']:.1f}% DTI",
            delta_color="normal" if analysis_results["debt"]["debt_to_income_ratio"] <= 40 else "off"
        )
    
    with col4:
        st.metric(
            "Net Worth",
            f"₹{analysis_results['summary']['net_worth']:,.0f}",
            delta=f"₹{analysis_results['summary']['monthly_savings']:,.0f}/month savings"
        )
    
    # Detailed Analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Emergency Fund", "Retirement", "Debt", "Recommendations"])
    
    with tab1:
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
    
    with tab2:
        st.markdown("### Retirement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Retirement Readiness Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=analysis_results["retirement"]["readiness_percentage"],
                    title={'text': "Retirement Readiness", 'font': {'size': 20}},
                    
                    number={'font': {'size': 38}},  # Number inside gauge

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

                    # Adjust gauge position
                    domain={'x': [0, 1], 'y': [0, 0.75]}
                ))

                fig.update_layout(
                    height=260,
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)


        
        with col2:
            st.markdown("#### Retirement Details")
            st.metric("Years to Retirement", analysis_results["retirement"]["years_to_retirement"])
            st.metric("Corpus Needed", f"₹{analysis_results['retirement']['corpus_needed']:,.0f}")
            st.metric("Projected Corpus", f"₹{analysis_results['retirement']['projected_corpus']:,.0f}")
            st.metric("Additional Monthly Saving", f"₹{analysis_results['retirement']['additional_monthly_saving_needed']:,.0f}")
    
    with tab3:
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
    
    with tab4:
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

def create_pdf_report(personal_info, income_data, assets_data, debts, insurance_data, goals, analysis_results):
    """Create PDF report"""
    
    # Create a PDF in memory
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=20
    ))
    
    # Build story
    story = []
    
    # Title
    story.append(Paragraph("Financial Planning Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Prepared for: {personal_info['name']}", styles["Normal"]))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
    story.append(Spacer(1, 30))
    
    # Client Information
    story.append(Paragraph("Client Information", styles["SectionTitle"]))
    client_data = [
        ["Name", personal_info["name"]],
        ["Age", str(personal_info["age"])],
        ["Retirement Age", str(personal_info["retirement_age"])],
        ["Dependents", str(personal_info["dependents"])],
        ["Risk Appetite", personal_info["risk_appetite"].title()]
    ]
    client_table = Table(client_data, colWidths=[150, 150])
    client_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    story.append(client_table)
    story.append(Spacer(1, 20))
    
    # Financial Summary
    story.append(Paragraph("Financial Summary", styles["SectionTitle"]))
    
    total_income = sum(income_data["income"].values()) * 12
    total_assets = (assets_data["cash_and_bank"] + 
                   sum(inv["current_value"] for inv in assets_data["investments"]) +
                   assets_data["retirement_corpus"] +
                   assets_data["real_estate_value"] +
                   assets_data["other_assets"])
    net_worth = total_assets - analysis_results["debt"]["total_debt"]
    
    summary_data = [
        ["Annual Income", f"₹{total_income:,.0f}"],
        ["Monthly Expenses", f"₹{income_data['monthly_expenses']:,.0f}"],
        ["Total Assets", f"₹{total_assets:,.0f}"],
        ["Total Debt", f"₹{analysis_results['debt']['total_debt']:,.0f}"],
        ["Net Worth", f"₹{net_worth:,.0f}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[150, 150])
    summary_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ecf0f1'))
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Analysis Results
    story.append(Paragraph("Analysis Results", styles["SectionTitle"]))
    
    analysis_data = [
        ["Emergency Fund", analysis_results["emergency"]["status"], f"{analysis_results['emergency']['adequacy_percentage']:.1f}%"],
        ["Retirement Readiness", analysis_results["retirement"]["status"], f"{analysis_results['retirement']['readiness_percentage']:.1f}%"],
        ["Debt Situation", analysis_results["debt"]["status"], f"{analysis_results['debt']['debt_to_income_ratio']:.1f}% DTI"]
    ]
    
    analysis_table = Table(analysis_data, colWidths=[120, 120, 60])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    story.append(analysis_table)
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Priority Recommendations", styles["SectionTitle"]))
    
    for i, rec in enumerate(analysis_results["recommendations"][:5], 1):
        story.append(Paragraph(f"{i}. {rec['title']}", ParagraphStyle(
            name='RecTitle',
            parent=styles['Normal'],
            fontSize=11,
            fontWeight='bold',
            spaceAfter=4
        )))
        story.append(Paragraph(rec["description"], styles["Normal"]))
        story.append(Paragraph("Actions:", ParagraphStyle(
            name='ActionTitle',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=2
        )))
        for action in rec["actions"]:
            story.append(Paragraph(f"• {action}", ParagraphStyle(
                name='ActionItem',
                parent=styles['Normal'],
                fontSize=8,
                leftIndent=20,
                spaceAfter=1
            )))
        story.append(Spacer(1, 10))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "Generated by Financial Planning Engine - For personalized advice, consult with a certified financial planner.",
        ParagraphStyle(
            name='Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        )
    ))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def create_download_button(pdf_bytes, filename):
    """Create a download button for the PDF"""
    
    # Encode PDF bytes to base64
    b64 = base64.b64encode(pdf_bytes).decode()
    
    # Create download link
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-btn">📥 Download PDF Report</a>'
    
    # Display download button
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        {href}
    </div>
    <style>
        .download-btn {{
            display: inline-block;
            background-color: #2ecc71;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 600;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        .download-btn:hover {{
            background-color: #27ae60;
            color: white;
            text-decoration: none;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 7. MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Display welcome
    display_welcome()
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Create sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money-bag.png", width=100)
        st.title("Navigation")
        
        section = st.radio(
            "Go to:",
            ["Personal Info", "Income & Expenses", "Assets", "Debts", "Insurance", "Goals", "Analysis & Report"],
            index=0
        )
        
        st.divider()
        
        st.markdown("### Quick Stats")
        if st.session_state.analysis_done and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.metric("Net Worth", f"₹{results['summary']['net_worth']:,.0f}")
            st.metric("Retirement Readiness", f"{results['retirement']['readiness_percentage']:.1f}%")
            st.metric("Debt-to-Income", f"{results['debt']['debt_to_income_ratio']:.1f}%")
        
        st.divider()
        
        st.markdown("### Need Help?")
        st.info("""
        Fill in all sections for accurate analysis.
        Use realistic estimates for best results.
        All data stays in your browser.
        """)
    
    # Main content area
    if section == "Personal Info":
        personal_info = create_personal_info_section()
        st.session_state.personal_info = personal_info
        
    elif section == "Income & Expenses":
        if 'personal_info' not in st.session_state:
            st.warning("Please fill Personal Information first!")
            st.stop()
        income_data = create_income_expenses_section()
        st.session_state.income_data = income_data
        
    elif section == "Assets":
        if 'income_data' not in st.session_state:
            st.warning("Please fill Income & Expenses first!")
            st.stop()
        assets_data = create_assets_section()
        st.session_state.assets_data = assets_data
        
    elif section == "Debts":
        if 'assets_data' not in st.session_state:
            st.warning("Please fill Assets section first!")
            st.stop()
        debts = create_debts_section()
        st.session_state.debts = debts
        
    elif section == "Insurance":
        if 'debts' not in st.session_state:
            st.warning("Please fill Debts section first!")
            st.stop()
        insurance_data = create_insurance_section()
        st.session_state.insurance_data = insurance_data
        
    elif section == "Goals":
        if 'insurance_data' not in st.session_state:
            st.warning("Please fill Insurance section first!")
            st.stop()
        goals = create_goals_section()
        st.session_state.goals = goals
        
    elif section == "Analysis & Report":
        # Check if all data is available
        required_sections = ['personal_info', 'income_data', 'assets_data', 'debts', 'insurance_data', 'goals']
        missing = [section for section in required_sections if section not in st.session_state]
        
        if missing:
            st.error(f"Please complete the following sections first: {', '.join(missing)}")
            st.stop()
        
        st.markdown('<h2 class="section-header">📈 Run Analysis & Generate Report</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Ready to Analyze Your Finances?")
            st.markdown("""
            Click the button below to run a comprehensive analysis of your financial situation.
            The analysis will cover:
            - Emergency fund adequacy
            - Retirement readiness
            - Debt burden analysis
            - Personalized recommendations
            
            After analysis, you can download a detailed PDF report.
            """)
        
        with col2:
            if st.button("🚀 Run Financial Analysis", use_container_width=True, type="primary"):
                with st.spinner("Analyzing your finances..."):
                    # Run analysis
                    analysis_results = run_analysis(
                        st.session_state.personal_info,
                        st.session_state.income_data,
                        st.session_state.assets_data,
                        st.session_state.debts,
                        st.session_state.insurance_data,
                        st.session_state.goals
                    )
                    
                    # Store results
                    st.session_state.analysis_results = analysis_results
                    st.session_state.analysis_done = True
                    
                    st.success("Analysis completed successfully!")
                    st.rerun()
        
        # Display results if analysis is done
        if st.session_state.analysis_done and st.session_state.analysis_results:
            display_results(st.session_state.analysis_results, st.session_state.personal_info)
            
            # Generate and display PDF download button
            st.markdown("---")
            st.markdown("### 📄 Download Report")
            
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = create_pdf_report(
                        st.session_state.personal_info,
                        st.session_state.income_data,
                        st.session_state.assets_data,
                        st.session_state.debts,
                        st.session_state.insurance_data,
                        st.session_state.goals,
                        st.session_state.analysis_results
                    )
                    
                    # Create download button
                    filename = f"Financial_Plan_{st.session_state.personal_info['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    create_download_button(pdf_bytes, filename)
                    
                    st.success("PDF report generated! Click the download button above.")
    
    # Progress indicator at bottom
    if section != "Analysis & Report":
        st.markdown("---")
        
        sections = ["Personal Info", "Income & Expenses", "Assets", "Debts", "Insurance", "Goals", "Analysis & Report"]
        current_index = sections.index(section)
        
        # Calculate progress
        progress = (current_index + 1) / len(sections)
        
        st.progress(progress)
        st.caption(f"Progress: {int(progress * 100)}% complete")

# ============================================================================
# 8. RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
