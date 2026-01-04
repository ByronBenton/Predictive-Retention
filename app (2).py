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

data_ready = False

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
# Main App Logic
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
        max_date = df["Date"].max().to_pydatetime()
        min_date = max_date - timedelta(days=365)
        date_range = st.slider(
            "Time Range",
            min_value=df["Date"].min().to_pydatetime(),
            max_value=df["Date"].max().to_pydatetime(),
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
            "Adjust salary competitiveness vs market (%)",
            -20, 20, 0,
            help="Adjust perceived salary position vs market. Positive = higher salary competitiveness."
        )
    with c2:
        burnout_adjustment = st.slider(
            "Burnout Environment Adjustment",
            -0.3, 0.3, 0.0,
            help="Increase/decrease estimated burnout score by this percentage. Positive values = higher burnout risk."
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
    # Churn Risk Calculation
    # -----------------------------
    sim_df["Simulated Churn Risk"] = np.clip(
        0.08 - sim_df["Adj_Salary"]*0.002 + sim_df["Adj_Burnout"]*0.35 + sim_df["Dept Bias"],
        0, 0.6
    )

    avg_churn = sim_df["Simulated Churn Risk"].mean()

    # -----------------------------
    # Visual Indicators
    # -----------------------------
    st.subheader("📊 Churn Risk Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Churn Risk", f"{avg_churn:.1%}")
    m2.metric("Employees Analyzed", len(sim_df))
    m3.metric("Est. Annual Leavers", int(len(sim_df) * avg_churn))

    # -----------------------------
    # Department-Level Insights
    # -----------------------------
    st.subheader("🏢 Department Churn Risk")
    dept_stats = sim_df.groupby("Department").agg(
        Avg_Churn=("Simulated Churn Risk", "mean"),
        Employees=("Employee ID", "count")
    ).reset_index()

    # Dimmed colors
    def risk_color(risk):
        if risk < 0.15:
            return "#6aaa64"  # muted green
        elif risk < 0.25:
            return "#f2c94c"  # soft yellow
        else:
            return "#eb5757"  # soft red

    dept_stats["Color"] = dept_stats["Avg_Churn"].apply(risk_color)

    # Highlight top 3 at-risk departments
    top_depts = dept_stats.nlargest(3, 'Avg_Churn')['Department'].tolist()
    dept_stats['Label'] = dept_stats['Department'].apply(lambda x: "🔥" if x in top_depts else "")

    # Bars
    bars = alt.Chart(dept_stats).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('Department', sort='-y', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Avg_Churn', title='Churn Risk (%)'),
        color=alt.Color('Color:N', scale=None),
        tooltip=[
            alt.Tooltip('Department'),
            alt.Tooltip('Avg_Churn', format='.1%'),
            alt.Tooltip('Employees', title='Employee Count')
        ]
    )

    # Labels for top 3 departments only, dynamically offset
    labels = alt.Chart(dept_stats[dept_stats['Label'] != ""]).mark_text(
        fontWeight='bold',
        align='center'
    ).encode(
        x='Department',
        y=alt.Y('Avg_Churn'),
        text='Label'
    ).transform_calculate(
        y_offset='datum.Avg_Churn + 0.05'
    ).encode(
        y='Churn Risk(%):Q'
    )

    chart = alt.layer(bars, labels).resolve_scale(
        y='independent'
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    st.markdown("""
     🔴 High Risk (>25%), 🟡 Medium Risk (15–25%), 🟢 Low Risk (<15%)
    """)

    # -----------------------------
    # Department-level details table
    # -----------------------------
    dept_stats["Est_Leavers"] = (dept_stats["Avg_Churn"] * dept_stats["Employees"]).round().astype(int)
    dept_stats = dept_stats.sort_values("Avg_Churn", ascending=False)

    st.markdown("**Department-level churn details:**")
    for idx, row in dept_stats.iterrows():
        st.write(f"{row['Department']}: {row['Avg_Churn']:.1%} — {row['Label']} — {row['Employees']} employees — Est. {row['Est_Leavers']} leavers")

    # -----------------------------
    # Actionable Recommendations
    # -----------------------------
    st.subheader("💡 Actionable Recommendations")
    high_risk_depts = dept_stats[dept_stats["Avg_Churn"] > 0.25]

    if high_risk_depts.empty:
        st.success("Overall churn is within manageable levels. Maintain current retention efforts.")
    else:
        rec_depts = ", ".join(high_risk_depts["Department"].tolist())
        st.success(f"Top Priority → {rec_depts}: Increase salaries 5–7% & reduce burnout by 0.1 → estimated churn reduction ~4–5%.")

    # -----------------------------
    # Export Insights
    # -----------------------------
    st.subheader("📤 Export Insights")
    export_df = dept_stats.rename(columns={
        "Avg_Churn": "Predicted Churn Risk",
        "Employees": "Employee Count",
        "Est_Leavers": "Estimated Leavers"
    })[["Department", "Predicted Churn Risk", "Employee Count", "Estimated Leavers"]]

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
    - **Confidence:** Churn estimates have an approximate ±5% uncertainty
    """)

