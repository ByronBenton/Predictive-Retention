# 🔮 Predictive Retention & Churn Risk Dashboard

A Streamlit dashboard to explore predictive employee churn and simulate retention interventions.

---

## Features

- Analyze employee churn risk by **department** and **time range**
- Adjust key **churn drivers**: salary competitiveness & burnout index
- Simulate **salary increases and burnout interventions**
- Visualize **department-level churn**, risk alerts, and trends
- Export **retention insights** as CSV
- Use default mock dataset or **upload your own CSV**

---

## Default Dataset

The dashboard comes with a mock dataset (`sample_employee_data.csv`) for testing.  
- 100 rows with realistic variations in salaries, burnout, and department bias.  
- Can be downloaded directly from the dashboard.

---

## CSV Upload Format

If uploading your own CSV, it should include the following columns:

| Column | Description |
|--------|-------------|
| Employee ID | Unique identifier for each employee |
| Department | Department name (e.g., Engineering, Sales, Marketing, etc.) |
| Salary vs Market (%) | Salary competitiveness vs market (%) |
| Burnout Index | Value between 0 (low) and 1 (high) |
| Date | Employee data date (YYYY-MM-DD) |

> Optional column: `Dept Bias`. If missing, default department bias values are applied automatically.

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd <repo_folder>
