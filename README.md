# AI-Scheduling-assistant

An interactive Streamlit web app for energy demand forecasting, clustering of consumption patterns, and generating state-level insights. The assistant supports time series forecasting using either Facebook Prophet or SARIMAX, KMeans clustering with PCA visualization, and insightful metrics like predictability, seasonal variability, and peak-to-average ratios.

Demo Link: https://ishikayadav13-ai-scheduling-assistant-app-xxd35n.streamlit.app/

🚀 Features

📈 Forecasting

1.Forecast monthly energy demand for selected states. 2.Choose between Prophet (if available) and SARIMAX. 3.Configurable forecast horizon (3–18 months). 4.Interactive Plotly charts with confidence intervals. 5.Export forecast results as CSV.

🧭 Clustering 1.Cluster states by their consumption patterns using KMeans. 2.Visualize clusters using: 3.Heatmaps (consumption across months). 4.PCA scatter plots (2D projection of states). 5.Save cluster assignments to Excel for further analysis.

💡 Insights 1.State-level consumption metrics: 2.Average demand 3.Seasonal variability (Coefficient of Variation) 4.Peak-to-average ratio 5.Predictability score 6.Automatic summary of: 7.Most predictable state 8.Highest variability

Highest peak-to-average

Download insights as CSV
