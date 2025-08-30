# app.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Optional Prophet; fallback to SARIMAX if not installed
PROPHET_AVAILABLE = True
try:
    from prophet import Prophet
except Exception:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="AI Scheduling Assistant", layout="wide")
DEFAULT_EXCEL = "Final power Plan (1).xlsx"   # change if you saved file under different name

# ------------------------------
# HELPERS
# ------------------------------
@st.cache_data(show_spinner=False)
def load_monthly_table_from_path(path: str) -> pd.DataFrame:
    """Read the first sheet of the Excel file and normalize the Month column."""
    df = pd.read_excel(path, sheet_name=0)
    # Ensure first column named Month
    if df.columns[0] != "Month":
        df = df.rename(columns={df.columns[0]: "Month"})
    # parse Month column
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month"])
    df = df.sort_values("Month").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_monthly_table_from_buffer(buffer) -> pd.DataFrame:
    df = pd.read_excel(buffer, sheet_name=0)
    if df.columns[0] != "Month":
        df = df.rename(columns={df.columns[0]: "Month"})
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month"])
    df = df.sort_values("Month").reset_index(drop=True)
    return df

def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide (Month + state cols) -> long format (Month, State, Consumption)."""
    long = df_wide.melt(id_vars="Month", var_name="State", value_name="Consumption")
    long["Consumption"] = pd.to_numeric(long["Consumption"], errors="coerce")
    long = long.dropna(subset=["Consumption"])
    return long

def forecast_series(monthly_series: pd.Series, horizon: int, model_choice: str):
    """
    monthly_series: pd.Series indexed by datetime (month start), values = numeric consumption
    Returns: forecast_df (Month, yhat, yhat_lower, yhat_upper), model_obj_or_result
    """
    s = monthly_series.asfreq("MS")  # monthly start
    s = s.astype(float)
    # Fill small gaps (lightly)
    s = s.fillna(method="ffill").fillna(method="bfill")

    if model_choice == "Prophet" and PROPHET_AVAILABLE:
        dfp = pd.DataFrame({"ds": s.index, "y": s.values})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon, freq="MS")
        fcst = m.predict(future)
        out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Month"})
        return out, m

    # fallback: SARIMAX baseline
    order = (1,1,1)
    seasonal_order = (1,1,1,12) if len(s) >= 24 else (0,1,1,12)
    model = SARIMAX(s, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    pred = res.get_forecast(steps=horizon)
    mean = pd.Series(pred.predicted_mean.values, index=idx)
    ci = pred.conf_int(alpha=0.2)  # 80% CI
    lower = pd.Series(ci.iloc[:,0].values, index=idx)
    upper = pd.Series(ci.iloc[:,1].values, index=idx)
    out = pd.DataFrame({"Month": idx, "yhat": mean.values, "yhat_lower": lower.values, "yhat_upper": upper.values})
    return out, res

def build_cluster_inputs(df_wide: pd.DataFrame):
    """Return X_scaled (ndarray), state_index (index), feature_df (states x months)."""
    feature_df = df_wide.set_index("Month").T
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    # Fill small gaps across months
    feature_df = feature_df.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)
    return X_scaled, feature_df.index, feature_df

def calc_state_insights(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby("State")["Consumption"]
    avg = g.mean()
    std = g.std(ddof=0)
    cov = (std / avg).replace([np.inf, -np.inf], np.nan)
    peak_avg = g.max() / avg
    predictability = (1 / cov).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        "Average": avg,
        "Seasonal_Variability(CoV)": cov,
        "Peak_to_Average": peak_avg,
        "Predictability(↑)": predictability
    }).sort_values("Predictability(↑)", ascending=False)
    return out

# ------------------------------
# UI: file selection / load
# ------------------------------
st.title("⚡ AI Scheduling Assistant — Forecasting • Clustering • Insights")

st.sidebar.header("Data & Controls")
uploaded = st.sidebar.file_uploader("Upload monthly Excel (Month + state columns)", type=["xlsx"])
use_default = False
if uploaded is None:
    # if default file exists, offer to load it
    if Path(DEFAULT_EXCEL).exists():
        use_default = st.sidebar.checkbox(f"Use local file `{DEFAULT_EXCEL}`", value=True)
    else:
        st.sidebar.info("Upload an Excel file or place the default file in the folder.")

if uploaded is None and not use_default:
    st.stop()

# Load data depending on choice
if uploaded is not None:
    df_wide = load_monthly_table_from_buffer(uploaded)
else:
    df_wide = load_monthly_table_from_path(DEFAULT_EXCEL)

# List states (columns except Month)
state_cols = [c for c in df_wide.columns if c != "Month"]
if len(state_cols) == 0:
    st.error("No state columns found. Ensure the first column is 'Month' and the rest are states.")
    st.stop()

# ------------------------------
# Sidebar controls for forecasts & clustering
# ------------------------------
st.sidebar.subheader("Forecasting Controls")
state_choice = st.sidebar.selectbox("Select state", state_cols, index=0)
horizon = st.sidebar.slider("Forecast horizon (months)", min_value=3, max_value=18, value=6, step=1)
model_options = ["SARIMAX"]
if PROPHET_AVAILABLE:
    model_options.insert(0, "Prophet")
model_choice = st.sidebar.selectbox("Forecast model", model_options)

st.sidebar.subheader("Clustering Controls")
k_clusters = st.sidebar.slider("Number of clusters (KMeans)", min_value=2, max_value=6, value=3, step=1)
do_save_clusters = st.sidebar.checkbox("Enable 'Save clusters' button", value=True)

# ------------------------------
# Tabs: Forecasting / Clustering / Insights
# ------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "🧭 Clustering", "💡 Insights"])

# ---- Forecasting tab ----
with tab1:
    st.header(f"Forecasting — {state_choice}")
    s = df_wide.set_index("Month")[state_choice].astype(float)
    if s.dropna().shape[0] < 3:
        st.warning("Not enough data points for reliable forecasting (need at least 3 months).")
    fcst_df, model_obj = forecast_series(s, horizon=horizon, model_choice=model_choice)

    # Plot actual + forecast
    hist_df = pd.DataFrame({"Month": s.index, "Actual": s.values})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Month"], y=hist_df["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=fcst_df["Month"], y=fcst_df["yhat"], mode="lines+markers", name="Forecast"))
    # Confidence
    fig.add_trace(go.Scatter(
        x=pd.concat([fcst_df["Month"], fcst_df["Month"][::-1]]),
        y=pd.concat([fcst_df["yhat_upper"], fcst_df["yhat_lower"][::-1]]),
        fill="toself", mode="lines", line=dict(width=0), name="Confidence", fillcolor="rgba(200,200,200,0.2)"
    ))
    fig.update_layout(height=500, xaxis_title="Month", yaxis_title="Energy", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast table (next months)")
    st.dataframe(fcst_df.set_index("Month").round(2))

    # Optionally download forecast CSV
    csv = fcst_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast (CSV)", data=csv, file_name=f"{state_choice}_forecast.csv", mime="text/csv")

# ---- Clustering tab ----
with tab2:
    st.header("Clustering (Load Profiles)")

    X_scaled, states_idx, feature_df = build_cluster_inputs(df_wide)
    kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    heat_df = feature_df.copy()
    heat_df["Cluster"] = labels
    heat_df_sorted = heat_df.sort_values("Cluster")

    st.subheader("Heatmap (states x months)")
    fig_hm = px.imshow(
        heat_df_sorted.drop(columns=["Cluster"]),
        labels=dict(x="Month", y="State", color="Consumption"),
        x=feature_df.columns.strftime("%b %Y"),
        y=feature_df.index,
        aspect="auto",
        color_continuous_scale="YlGnBu"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Cluster projection (PCA)")
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": pts[:,0], "PC2": pts[:,1], "State": states_idx, "Cluster": labels})
    fig_sc = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str), text="State", title="States in PCA space")
    fig_sc.update_traces(textposition="top center")
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Cluster assignments")
    cluster_table = pd.DataFrame({"State": states_idx, "Cluster": labels}).sort_values("Cluster").reset_index(drop=True)
    st.dataframe(cluster_table)

    if do_save_clusters:
        if st.button("Save cluster assignments to Excel"):
            out = heat_df.copy()
            out["Cluster"] = labels
            out.to_excel("clustered_states_output.xlsx")
            st.success("Saved clustered_states_output.xlsx to working folder")

# ---- Insights tab ----
with tab3:
    st.header("Insights")

    long_df = to_long(df_wide)
    insights = calc_state_insights(long_df)
    st.subheader("State-level metrics")
    st.dataframe(insights.style.format({
        "Average": "{:,.2f}",
        "Seasonal_Variability(CoV)": "{:.3f}",
        "Peak_to_Average": "{:.3f}",
        "Predictability(↑)": "{:.3f}"
    }), use_container_width=True)

    most_pred = insights["Predictability(↑)"].idxmax()
    most_var = insights["Seasonal_Variability(CoV)"].idxmax()
    highest_peak = insights["Peak_to_Average"].idxmax()

    st.markdown(f"""
    **Quick summary**
    - Most predictable: **{most_pred}**
    - Highest seasonal variability (CoV): **{most_var}**
    - Highest peak-to-average ratio: **{highest_peak}**
    """)

    st.download_button("Download insights (CSV)", data=insights.to_csv().encode("utf-8"), file_name="state_insights.csv", mime="text/csv")
