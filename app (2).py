import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Predictive Retention Dashboard",
    layout="wide"
)

st.title("🔮 Predictive Retention & Churn Risk Explorer")
st.caption("Helping leadership teams make fast, data-driven retention decisions")

# --------------------------------------------------
# Tutorial / Onboarding
# --------------------------------------------------
with st.expander("👋 How to use this tool"):
    st.markdown("""
    **1. Select departments and time range**  
    **2. Adjust churn drivers** (salary & burnout)  
    **3. Simulate retention interventions**  
    **4. Review churn risk, alerts, and recommendations**  
    **5. Export insights for leadership meetings**
    """)

# --------------------------------------------------
# Mock Data Generator
# --------------------------------------------------
@st.cache_data
def generate_employee_data(n=800):
    np.random.seed(42)

    departments = [
        "Engineering", "Sales", "Marketing",
        "Finance", "HR", "Operations", "Support"
    ]
    
    # Department-specific bias to create varied churn risk
    dept_bias = {
        "Engineering": 0.0,    # medium
        "Sales": 0.2,          # high
        "Marketing": -0.05,    # low
        "Finance": 0.0,
        "HR": -0.1,            # low
        "Operations": 0.05,
        "Support": 0.25         # high
    }

    data = {
        "Employee ID": [f"E-{1000+i}" for i in range(n)],
        "Department": np.random.choice(departments, n),
        "Salary vs Market (%)": np.random.normal(-5, 15, n).clip(-30, 30),
        "Burnout Index": np.random.beta(2, 2, n),
        "Date": [
            datetime.today() - timedelta(days=int(x))
            for x in np.random.uniform(0, 365, n)
        ],
    }

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])

    # Base churn risk with department bias
    df["Base Churn Risk"] = (
        0.08
        + (-df["Salary vs Market (%)"] * 0.002)
        + (df["Burnout Index"] * 0.35)
        + df["Department"].map(dept_bias)
    ).clip(0, 0.6)

    return df

df = generate_employee_data()

# --------------------------------------------------
# Filters
# --------------------------------------------------
st.subheader("🔎 Filters")

col1, col2 = st.columns(2)

with col1:
    dept_filter = st.multiselect(
        "Departments",
        df["Department"].unique(),
        default=list(df["Department"].unique()),  # Default all
        help="Select one or more departments to analyze"
    )

with col2:
    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()

    date_range = st.slider(
        "Time Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

if not dept_filter:
    st.error("⚠️ Please select at least one department to proceed.")
    st.stop()

filtered_df = df[
    (df["Department"].isin(dept_filter)) &
    (df["Date"].between(*date_range))
]

# --------------------------------------------------
# Churn Driver Sliders
# --------------------------------------------------
st.subheader("🎛️ Churn Driver Assumptions")

c1, c2 = st.columns(2)

with c1:
    salary_adjustment = st.slider(
        "Salary Competitiveness Adjustment (%)",
        -20, 20, 0,
        help="Adjust perceived salary position vs market"
    )

with c2:
    burnout_adjustment = st.slider(
        "Burnout Environment Adjustment",
        -0.3, 0.3, 0.0,
        help="Reflects workload, culture, and manager effectiveness"
    )

# --------------------------------------------------
# Scenario Simulation
# --------------------------------------------------
st.subheader("🧪 Scenario Simulation")

sim_salary_increase = st.slider(
    "Proposed Salary Increase (%)",
    0, 15, 5,
    help="Simulate the impact of a salary increase on churn risk"
)

sim_df = filtered_df.copy()

# Adjusted columns (consistent names)
sim_df["Adj_Salary"] = sim_df["Salary vs Market (%)"] + salary_adjustment + sim_salary_increase
sim_df["Adj_Burnout"] = (sim_df["Burnout Index"] + burnout_adjustment).clip(0, 1)

def churn_risk(salary_vs_market, burnout):
    return (0.08 + (-salary_vs_market * 0.002) + (burnout * 0.35)).clip(0, 0.6)

sim_df["Simulated Churn Risk"] = churn_risk(sim_df["Adj_Salary"], sim_df["Adj_Burnout"])
avg_churn = sim_df["Simulated Churn Risk"].mean()

# --------------------------------------------------
# Visual Indicators
# --------------------------------------------------
def risk_label(risk):
    if risk < 0.15:
        return "🟢 Low Risk"
    elif risk < 0.25:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

st.subheader("📊 Churn Risk Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Average Churn Risk", f"{avg_churn:.1%}", risk_label(avg_churn))
m2.metric("Employees Analyzed", len(sim_df))
m3.metric("Estimated Annual Leavers", int(len(sim_df) * avg_churn))

# --------------------------------------------------
# Real-Time Alerts
# --------------------------------------------------
if avg_churn >= 0.25:
    st.error("🚨 High Churn Alert: Overall churn risk exceeds 25%. Immediate retention action recommended.")
elif avg_churn >= 0.18:
    st.warning("⚠️ Medium Churn Alert: Churn risk is rising. Consider proactive retention strategies.")
else:
    st.success("✅ Churn Risk Stable: Current conditions show manageable attrition levels.")

# --------------------------------------------------
# Department-Level Insights
# --------------------------------------------------
st.subheader("🏢 Department Churn Risk")

dept_risk = sim_df.groupby("Department")["Simulated Churn Risk"].mean().sort_values(ascending=False)
st.bar_chart(dept_risk)

st.markdown("**Department-level churn risk details:**")
for dept, risk in dept_risk.items():
    st.write(f"{dept}: {risk:.1%} — {risk_label(risk)}")

# --------------------------------------------------
# Actionable Recommendations (heading appears correctly)
# --------------------------------------------------
show_recommendations = (
    avg_churn > 0.25
    or sim_df["Adj_Burnout"].mean() > 0.6
    or dept_risk.iloc[0] > 0.25
)

if show_recommendations:
    st.markdown("### 💡 Actionable Recommendations")
    
    if avg_churn > 0.25 and sim_salary_increase < 5:
        st.success("• Increase salaries by ~5% to reduce churn risk by an estimated 15–20%.")
    if sim_df["Adj_Burnout"].mean() > 0.6:
        st.success("• Address burnout drivers such as workload, manager span, or on-call pressure.")
    
    top_dept = dept_risk.index[0]
    if dept_risk.iloc[0] > 0.25:
        st.success(f"• Prioritize retention initiatives in **{top_dept}**, which has the highest predicted churn.")

# --------------------------------------------------
# Export Insights
# --------------------------------------------------
st.subheader("📤 Export Insights")
export_df = dept_risk.reset_index()
export_df.columns = ["Department", "Predicted Churn Risk"]

st.download_button(
    "Download Retention Report (CSV)",
    export_df.to_csv(index=False),
    "retention_report.csv"
)

# --------------------------------------------------
# Metric Explanations with hover tooltips
# --------------------------------------------------
st.markdown("""
**Metric Explanations:**  
- **Salary vs Market (%)** <span title="Negative values indicate pay below market; positive means above market">ℹ️</span>  
- **Burnout Index** <span title="Higher values correlate with higher voluntary attrition">ℹ️</span>  
- **Churn Risk** <span title="Probability of leaving within ~12 months">ℹ️</span>  
""", unsafe_allow_html=True)
