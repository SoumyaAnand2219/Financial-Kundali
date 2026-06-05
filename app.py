import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Financial Planning Engine",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# CORE FINANCIAL LOGIC
# -----------------------------

SECTORAL_INFLATION = {
    "General": 6.0,
    "Education": 10.0,
    "Healthcare": 12.0,
    "Retirement": 6.5,
    "Marriage": 8.0,
    "House": 7.0,
    "Vehicle": 5.0
}

RISK_RETURNS = {
    "Conservative": 8.0,
    "Moderate": 10.0,
    "Aggressive": 12.0
}

ASSET_ALLOCATION = {
    "Conservative": {
        "Equity": 30,
        "Debt": 55,
        "Gold": 10,
        "Liquid": 5
    },
    "Moderate": {
        "Equity": 55,
        "Debt": 30,
        "Gold": 10,
        "Liquid": 5
    },
    "Aggressive": {
        "Equity": 75,
        "Debt": 15,
        "Gold": 5,
        "Liquid": 5
    }
}


def future_value_with_inflation(present_value, years, inflation_rate):
    return present_value * ((1 + inflation_rate / 100) ** years)


def sip_required(future_value, years, expected_return, current_savings=0):
    months = years * 12
    monthly_rate = expected_return / 12 / 100

    future_value_of_current_savings = current_savings * ((1 + expected_return / 100) ** years)
    remaining_goal_amount = max(future_value - future_value_of_current_savings, 0)

    if monthly_rate == 0:
        return remaining_goal_amount / months

    sip = remaining_goal_amount * monthly_rate / (((1 + monthly_rate) ** months - 1) * (1 + monthly_rate))
    return max(sip, 0)


def lumpsum_required(future_value, years, expected_return):
    return future_value / ((1 + expected_return / 100) ** years)


def calculate_retirement_corpus(
    current_monthly_expense,
    current_age,
    retirement_age,
    life_expectancy,
    inflation_rate,
    post_retirement_return
):
    years_to_retirement = retirement_age - current_age
    retirement_years = life_expectancy - retirement_age

    monthly_expense_at_retirement = current_monthly_expense * ((1 + inflation_rate / 100) ** years_to_retirement)

    monthly_rate = post_retirement_return / 12 / 100
    months = retirement_years * 12

    if monthly_rate == 0:
        corpus = monthly_expense_at_retirement * months
    else:
        corpus = monthly_expense_at_retirement * ((1 - (1 + monthly_rate) ** (-months)) / monthly_rate)

    return corpus, monthly_expense_at_retirement


def emergency_fund_required(monthly_expense, income_stability, risk_profile):
    months = 6

    if income_stability == "Unstable":
        months += 3
    elif income_stability == "Stable":
        months -= 1

    if risk_profile == "Conservative":
        months += 2
    elif risk_profile == "Aggressive":
        months -= 1

    months = max(months, 3)

    return monthly_expense * months, months


def allocate_goals(goals_df, available_monthly_savings, risk_profile):
    expected_return = RISK_RETURNS[risk_profile]

    goals_df = goals_df.sort_values(by="Priority", ascending=True).copy()

    results = []
    remaining_savings = available_monthly_savings

    for _, goal in goals_df.iterrows():
        inflation = SECTORAL_INFLATION[goal["Inflation Sector"]]

        future_goal_value = future_value_with_inflation(
            goal["Current Cost"],
            goal["Years"],
            inflation
        )

        required_sip = sip_required(
            future_goal_value,
            goal["Years"],
            expected_return,
            goal["Current Savings"]
        )

        if required_sip <= remaining_savings:
            status = "Fully Funded"
            allocated = required_sip
            funding = 100
            shortfall = 0
            remaining_savings -= required_sip
        else:
            allocated = max(remaining_savings, 0)
            funding = (allocated / required_sip) * 100 if required_sip > 0 else 0
            shortfall = required_sip - allocated

            if funding >= 70:
                status = "Partially Funded"
            else:
                status = "Not Fully Achievable"

            remaining_savings = 0

        results.append({
            "Goal": goal["Goal Name"],
            "Priority": goal["Priority"],
            "Current Cost": goal["Current Cost"],
            "Inflation Sector": goal["Inflation Sector"],
            "Years": goal["Years"],
            "Future Cost": round(future_goal_value, 2),
            "Required SIP": round(required_sip, 2),
            "Allocated SIP": round(allocated, 2),
            "Funding %": round(funding, 2),
            "Shortfall": round(shortfall, 2),
            "Status": status
        })

    return pd.DataFrame(results), remaining_savings


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("Financial Planning Engine")
st.caption("Goal-based investment planning with risk profile, priority logic and sectoral inflation.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "User Profile",
    "Goal Planning",
    "Retirement",
    "Calculators",
    "Final Summary"
])

# -----------------------------
# TAB 1: USER PROFILE
# -----------------------------
with tab1:
    st.header("User Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Current Age", min_value=18, max_value=80, value=30)
        retirement_age = st.number_input("Retirement Age", min_value=40, max_value=75, value=60)

    with col2:
        life_expectancy = st.number_input("Life Expectancy", min_value=60, max_value=100, value=85)
        risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])

    with col3:
        monthly_income = st.number_input("Monthly Income ₹", min_value=0, value=100000, step=5000)
        monthly_expense = st.number_input("Monthly Expense ₹", min_value=0, value=50000, step=5000)

    income_stability = st.selectbox("Income Stability", ["Stable", "Moderate", "Unstable"])

    available_monthly_savings = max(monthly_income - monthly_expense, 0)

    st.success(f"Available Monthly Savings: ₹{available_monthly_savings:,.0f}")

    emergency_amount, emergency_months = emergency_fund_required(
        monthly_expense,
        income_stability,
        risk_profile
    )

    st.info(f"Recommended Emergency Fund: ₹{emergency_amount:,.0f} ({emergency_months} months of expenses)")

# -----------------------------
# TAB 2: GOAL PLANNING
# -----------------------------
with tab2:
    st.header("Goal Planning")

    st.write("Enter your financial goals below.")

    number_of_goals = st.number_input("Number of Goals", min_value=1, max_value=10, value=3)

    goals = []

    for i in range(number_of_goals):
        st.subheader(f"Goal {i + 1}")

        col1, col2, col3 = st.columns(3)

        with col1:
            goal_name = st.text_input(f"Goal Name {i + 1}", value=f"Goal {i + 1}")
            priority = st.number_input(f"Priority {i + 1}", min_value=1, max_value=10, value=i + 1)

        with col2:
            current_cost = st.number_input(f"Current Cost ₹ {i + 1}", min_value=0, value=1000000, step=50000)
            current_savings = st.number_input(f"Current Savings ₹ {i + 1}", min_value=0, value=0, step=50000)

        with col3:
            years = st.number_input(f"Years Left {i + 1}", min_value=1, max_value=50, value=10)
            inflation_sector = st.selectbox(
                f"Inflation Sector {i + 1}",
                list(SECTORAL_INFLATION.keys()),
                key=f"inflation_{i}"
            )

        goals.append({
            "Goal Name": goal_name,
            "Priority": priority,
            "Current Cost": current_cost,
            "Current Savings": current_savings,
            "Years": years,
            "Inflation Sector": inflation_sector
        })

    goals_df = pd.DataFrame(goals)

    if st.button("Generate Goal Plan"):
        result_df, remaining = allocate_goals(
            goals_df,
            available_monthly_savings,
            risk_profile
        )

        st.subheader("Goal Allocation Result")
        st.dataframe(result_df, use_container_width=True)

        st.success(f"Remaining Monthly Savings After Goal Allocation: ₹{remaining:,.0f}")

# -----------------------------
# TAB 3: RETIREMENT
# -----------------------------
with tab3:
    st.header("Retirement Planning")

    retirement_inflation = st.slider("Retirement Inflation %", 4.0, 12.0, 6.5)
    post_retirement_return = st.slider("Post-Retirement Return %", 4.0, 10.0, 7.0)

    corpus, retirement_expense = calculate_retirement_corpus(
        monthly_expense,
        age,
        retirement_age,
        life_expectancy,
        retirement_inflation,
        post_retirement_return
    )

    years_to_retirement = retirement_age - age
    expected_return = RISK_RETURNS[risk_profile]

    monthly_sip_for_retirement = sip_required(
        corpus,
        years_to_retirement,
        expected_return,
        current_savings=0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Monthly Expense at Retirement", f"₹{retirement_expense:,.0f}")
    col2.metric("Required Retirement Corpus", f"₹{corpus:,.0f}")
    col3.metric("Monthly SIP Needed", f"₹{monthly_sip_for_retirement:,.0f}")

# -----------------------------
# TAB 4: CALCULATORS
# -----------------------------
with tab4:
    st.header("Investment Calculators")

    calculator = st.selectbox("Choose Calculator", ["SIP Calculator", "Lumpsum Calculator", "SWP Calculator"])

    if calculator == "SIP Calculator":
        sip_amount = st.number_input("Monthly SIP ₹", value=10000, step=1000)
        sip_years = st.number_input("Investment Years", value=10)
        sip_return = st.slider("Expected Return %", 1.0, 20.0, 12.0)

        months = sip_years * 12
        monthly_rate = sip_return / 12 / 100

        fv = sip_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

        st.metric("Future Value", f"₹{fv:,.0f}")

    elif calculator == "Lumpsum Calculator":
        amount = st.number_input("Lumpsum Amount ₹", value=100000, step=10000)
        years = st.number_input("Years", value=10)
        expected_return = st.slider("Expected Return %", 1.0, 20.0, 12.0)

        fv = amount * ((1 + expected_return / 100) ** years)

        st.metric("Future Value", f"₹{fv:,.0f}")

    elif calculator == "SWP Calculator":
        corpus_amount = st.number_input("Corpus ₹", value=5000000, step=100000)
        withdrawal = st.number_input("Monthly Withdrawal ₹", value=30000, step=1000)
        swp_return = st.slider("Expected Return %", 1.0, 15.0, 8.0)
        swp_years = st.number_input("Withdrawal Years", value=20)

        balance = corpus_amount
        monthly_rate = swp_return / 12 / 100

        for month in range(swp_years * 12):
            balance = balance * (1 + monthly_rate) - withdrawal
            if balance <= 0:
                break

        if balance > 0:
            st.success(f"Corpus remaining after {swp_years} years: ₹{balance:,.0f}")
        else:
            st.error(f"Corpus may finish in approximately {month // 12} years and {month % 12} months.")

# -----------------------------
# TAB 5: FINAL SUMMARY
# -----------------------------
with tab5:
    st.header("Final Financial Summary")

    st.subheader("Risk-Based Asset Allocation")

    allocation = ASSET_ALLOCATION[risk_profile]
    allocation_df = pd.DataFrame({
        "Asset Class": list(allocation.keys()),
        "Allocation %": list(allocation.values())
    })

    st.dataframe(allocation_df, use_container_width=True)

    st.subheader("Key Recommendations")

    st.write(f"""
    - Risk Profile: **{risk_profile}**
    - Expected Portfolio Return: **{RISK_RETURNS[risk_profile]}%**
    - Emergency Fund Required: **₹{emergency_amount:,.0f}**
    - Available Monthly Savings: **₹{available_monthly_savings:,.0f}**
    - Retirement Corpus Required: **₹{corpus:,.0f}**
    - Retirement SIP Required: **₹{monthly_sip_for_retirement:,.0f}**
    """)

    st.warning(
        "Disclaimer: This is an educational financial planning tool. "
        "Mutual fund investments are subject to market risks. "
        "Please read all scheme-related documents carefully before investing."
    )
