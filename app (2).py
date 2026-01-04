import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Predictive Retention Dashboard",
    layout="wide"
)

st.title("🔮 Predictive Retention & Churn Risk Explorer")
st.caption("Helping leadership teams make fast, data-driven retention decisions")

# -----------------------------
# Onboarding
# -----------------------------
with st.expander("👋 How to use this tool"):
    st.markdown("""
    **1. Select departments and time range**  
    **2. Adjust churn drivers** (salary & burnout)  
    **3. Simulate retention interventions**  
    **4. Review churn risk, alerts, benchmarks, and recommendations**  
    **5. Export insights for leadership meetings**
    """)

# -----------------------------
# Mock Data Generator
# -----------------------------
@st.cache_data
def generate_employee_data(n=800):
    np.random.seed(42)

    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Support"]
    
    dept_bias = {
        "Engineering": 0.0,
        "Sales": 0.2,
        "Marketing": -0.05,
        "Finance": 0.0,
        "HR": -0.1,
        "Operations": 0.05,
        "Support": 0.25
    }

    data = {
        "Employee ID": [f"E-{1000+i}" for i in range(n)],
        "Department": np.random.choice(departments, n),
        "Salary vs Market (%)": np.random.normal(-5, 15, n).clip(-30, 30),
        "Burnout Index": np.random.beta(2, 2, n),
        "Date": [datetime.today() - timedelta(days=int(x)) for x in np.random.uniform(0, 365, n)],
    }

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    
    df["Dept Bias"] = df["Department"].map(dept_bias)
    
    df["Base Churn Risk"] = (
        0.08
        + (-df["Salary vs Market (%)"] * 0.002)
        + (df["Burnout Index"] * 0.35)
        + df["Dept Bias"]
    )
    df["Base Churn Risk"] = df["Base Churn Risk"].clip(0, 0.6)
    
    return df

# -----------------------------
# Dataset Selection / Upload
# -----------------------------
st.subheader("📂 Data Source")

use_default = st.radio(
    "Choose your dataset",
    ["Use default (mock) dataset", "Upload my own CSV"],
    horizontal=True
)

data_ready = False  # Flag to indicate dataset is ready

if use_default == "Use default (mock) dataset":
    df = generate_employee_data()
    data_ready = True
else:
    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"],
        help="CSV should include columns: Employee ID, Department, Salary vs Market (%), Burnout Index, Date"
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ["Employee ID", "Department", "Salary vs Market (%)", "Burnout Index", "Date"]
        if all(col in df.columns for col in required_cols):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # -----------------------------
            # Add Dept Bias if missing
            # -----------------------------
            if "Dept Bias" not in df.columns:
                dept_bias = {
                    "Engineering": 0.0,
                    "Sales": 0.2,
                    "Marketing": -0.05,
                    "Finance": 0.0,
                    "HR": -0.1,
                    "Operations": 0.05,
                    "Support": 0.25
                }
                df["Dept Bias"] = df["Department"].map(dept_bias).fillna(0)

            data_ready = True
        else:
            st.error(f"⚠️ CSV must include columns: {', '.join(required_cols)}")
    else:
        st.info("📌 Please upload your CSV to see results.")

# -----------------------------
# Main App Logic (only if data is ready)
# -----------------------------
if data_ready:
    # -----------------------------
    # Filters
    # -----------------------------
    st.subheader("🔎 Filters")
    col1, col2 = st.columns(2)
    with col1:
        dept_filter = st.multiselect(
            "Departments",
            df["Department"].unique(),
            default=list(df["Department"].unique()),
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

    # -----------------------------
    # Churn Driver Sliders
    # -----------------------------
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

    # -----------------------------
    # Scenario Simulation
    # -----------------------------
    st.subheader("🧪 Scenario Simulation")
    sim_salary_increase = st.slider(
        "Proposed Salary Increase (%)",
        0, 15, 5,
        help="Simulate the impact of a salary increase on churn risk"
    )

    sim_df = filtered_df.copy()
    sim_df["Adj_Salary"] = sim_df["Salary vs Market (%)"] + salary_adjustment + sim_salary_increase
    sim_df["Adj_Burnout"] = (sim_df["Burnout Index"] + burnout_adjustment).clip(0, 1)

    # -----------------------------
    # Churn Risk Function
    # -----------------------------
    def churn_risk(salary_vs_market, burnout, dept_bias):
        risk = 0.08 + (-salary_vs_market * 0.002) + (burnout * 0.35) + dept_bias
        return float(np.clip(risk, 0, 0.6))

    sim_df["Simulated Churn Risk"] = sim_df.apply(
        lambda row: churn_risk(row["Adj_Salary"], row["Adj_Burnout"], row["Dept Bias"]),
        axis=1
    )

    avg_churn = sim_df["Simulated Churn Risk"].mean()

    # -----------------------------
    # Benchmarking & Trend Context
    # -----------------------------
    prev_df = filtered_df[filtered_df["Date"] < date_range[0]]
    prev_churn = prev_df["Base Churn Risk"].mean() if not prev_df.empty else np.nan
    industry_avg = 0.15

    # -----------------------------
    # Visual Indicators
    # -----------------------------
    def risk_label(risk):
        if risk < 0.15:
            return "🟢 Low Risk"
        elif risk < 0.25:
            return "🟡 Medium Risk"
        else:
            return "🔴 High Risk"

    st.subheader("📊 Churn Risk Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Churn Risk", f"{avg_churn:.1%}", risk_label(avg_churn))
    m2.metric("Employees Analyzed", len(sim_df))
    m3.metric("Est. Annual Leavers", int(len(sim_df) * avg_churn))
    if not np.isnan(prev_churn):
        trend_arrow = "▲" if avg_churn > prev_churn else "▼" if avg_churn < prev_churn else "→"
        m4.metric("Change vs Last Quarter", f"{avg_churn - prev_churn:+.1%}", trend_arrow)

    # -----------------------------
    # Real-Time Alerts
    # -----------------------------
    if avg_churn >= 0.25:
        st.error("🚨 High Churn Alert: Overall churn risk exceeds 25%. Immediate retention action recommended.")
    elif avg_churn >= 0.18:
        st.warning("⚠️ Medium Churn Alert: Churn risk is rising. Consider proactive retention strategies.")
    else:
        st.success("✅ Churn Risk Stable: Current conditions show manageable attrition levels.")

    # -----------------------------
    # Department-Level Insights
    # -----------------------------
    st.subheader("🏢 Department Churn Risk")
    dept_stats = sim_df.groupby("Department").agg(
        Avg_Churn=("Simulated Churn Risk", "mean"),
        Employees=("Employee ID", "count")
    )
    dept_stats["Est_Leavers"] = (dept_stats["Avg_Churn"] * dept_stats["Employees"]).round().astype(int)
    dept_stats = dept_stats.sort_values("Avg_Churn", ascending=False)

    def risk_color(risk):
        if risk < 0.15:
            return "green"
        elif risk < 0.25:
            return "yellow"
        else:
            return "red"

    dept_stats = dept_stats.copy()
    dept_stats["Color"] = dept_stats["Avg_Churn"].apply(risk_color)

    chart = alt.Chart(dept_stats.reset_index()).mark_bar().encode(
        x='Department',
        y='Avg_Churn',
        color=alt.Color('Color:N', scale=None),
        tooltip=['Department', 'Avg_Churn', 'Employees', 'Est_Leavers']
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Department-level churn details:**")
    for idx, row in dept_stats.iterrows():
        st.write(f"{idx}: {row['Avg_Churn']:.1%} — {risk_label(row['Avg_Churn'])} — {row['Employees']} employees — Est. {row['Est_Leavers']} leavers")

    # -----------------------------
    # Actionable Recommendations
    # -----------------------------
    st.subheader("💡 Actionable Recommendations")
    high_risk_depts = dept_stats[dept_stats["Avg_Churn"] > 0.25]

    if high_risk_depts.empty:
        st.success("Overall churn is within manageable levels. Maintain current retention efforts.")
    else:
        for idx, row in high_risk_depts.iterrows():
            dept_df = sim_df[sim_df["Department"] == idx]
            salary_effect = ((dept_df["Adj_Salary"] + 5) - dept_df["Adj_Salary"]) * -0.002
            burnout_effect = -0.35 * 0.1
            est_churn_reduction = (salary_effect.mean() + burnout_effect) * 100
            st.success(f"• **{idx}**: Top drivers → Low salaries & high burnout. Estimated churn reduction: ~{abs(est_churn_reduction):.1f}% by increasing salaries 5–7% and reducing burnout by 0.1.")

    # -----------------------------
    # Export Insights
    # -----------------------------
    st.subheader("📤 Export Insights")
    export_df = dept_stats.reset_index()
    export_df.rename(columns={
        "Avg_Churn": "Predicted Churn Risk",
        "Employees": "Employee Count",
        "Est_Leavers": "Estimated Leavers"
    }, inplace=True)

    st.download_button(
        "Download Retention Report (CSV)",
        export_df.to_csv(index=False),
        "retention_report.csv"
    )

    # -----------------------------
    # Model Explanation
    # -----------------------------
    st.subheader("ℹ️ How the Model Works")
    st.markdown("""
    - **Key Drivers:** Salary vs market, burnout index, department-specific patterns  
    - **Scope:** Estimates voluntary churn risk over the next ~12 months  
    - **Outputs:** Directional estimates to guide leadership decisions, not precise predictions  
    - **Benchmarking:** Current churn is compared to prior quarter and industry averages  
    - **Actionable insights:** Highlight high-risk departments, top drivers, and expected impact of interventions
    """)
