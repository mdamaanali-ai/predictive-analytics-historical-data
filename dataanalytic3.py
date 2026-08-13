# ============================================================
# PREDICTIVE ANALYTICS USING HISTORICAL DATA
# ============================================================
# Features:
#   - Historical data upload
#   - Data cleaning & preprocessing
#   - Trend analysis
#   - Linear Regression forecasting
#   - Model evaluation
#   - Future prediction
#   - Interactive Streamlit dashboard
#   - Downloadable predictions
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)


# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            color: #2563EB;
            text-align: center;
        }

        .subtitle {
            text-align: center;
            color: #64748B;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .metric-card {
            padding: 15px;
            border-radius: 12px;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">📈 Predictive Analytics Using Historical Data</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze historical trends and forecast future values using Machine Learning</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Historical CSV Dataset",
    type=["csv"]
)

use_sample_data = st.sidebar.checkbox(
    "Use Sample Dataset",
    value=uploaded_file is None
)


forecast_periods = st.sidebar.slider(
    "Forecast Periods",
    min_value=5,
    max_value=100,
    value=30
)


# ------------------------------------------------------------
# SAMPLE DATA GENERATOR
# ------------------------------------------------------------

def generate_sample_data():

    np.random.seed(42)

    dates = pd.date_range(
        start="2024-01-01",
        periods=120,
        freq="D"
    )

    trend = np.linspace(100, 180, len(dates))

    seasonal_pattern = (
        10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
    )

    noise = np.random.normal(
        0,
        5,
        len(dates)
    )

    values = trend + seasonal_pattern + noise

    return pd.DataFrame({
        "Date": dates,
        "Value": values.round(2)
    })


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

if uploaded_file is not None and not use_sample_data:

    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully.")

    except Exception as error:

        st.error(f"Unable to read the dataset: {error}")
        st.stop()

else:

    df = generate_sample_data()

    st.info(
        "ℹ️ Sample historical dataset is being used. "
        "Upload your own CSV from the sidebar for real predictions."
    )


# ------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------

st.header("📋 Historical Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Records",
        len(df)
    )

with col2:
    st.metric(
        "Columns",
        len(df.columns)
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ------------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------------

st.header("🧹 Data Cleaning & Preprocessing")

df = df.copy()

# Remove duplicate records
df = df.drop_duplicates()

# Remove completely empty rows
df = df.dropna(how="all")

# Try to identify date column
date_column = None

for column in df.columns:

    if "date" in column.lower() or "time" in column.lower():

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        if converted.notna().sum() > 0:

            date_column = column
            df[column] = converted
            break


# If no date column exists, create one
if date_column is None:

    date_column = "Date"

    df[date_column] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="D"
    )


# ------------------------------------------------------------
# SELECT TARGET COLUMN
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


if not numeric_columns:

    st.error(
        "❌ No numerical column was found for prediction."
    )

    st.info(
        "Your CSV should contain at least one numerical "
        "column such as Sales, Revenue, Price, Temperature, etc."
    )

    st.stop()


target_column = st.sidebar.selectbox(
    "Select Target Column",
    numeric_columns
)


# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

model_data = df[
    [date_column, target_column]
].copy()

model_data[target_column] = pd.to_numeric(
    model_data[target_column],
    errors="coerce"
)

model_data = model_data.dropna()

model_data = model_data.sort_values(
    date_column
)

model_data = model_data.drop_duplicates(
    subset=[date_column]
)

model_data = model_data.reset_index(
    drop=True
)


if len(model_data) < 10:

    st.error(
        "❌ At least 10 valid historical records are required."
    )

    st.stop()


# ------------------------------------------------------------
# HISTORICAL TREND
# ------------------------------------------------------------

st.header("📊 Historical Trend Analysis")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    model_data[date_column],
    model_data[target_column],
    linewidth=2
)

ax.set_title(
    f"Historical {target_column} Trend"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    target_column
)

ax.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

data = model_data.copy()

# Convert date into numerical time index
data["Time_Index"] = np.arange(
    len(data)
)

# Additional useful time-based features
data["Year"] = data[date_column].dt.year

data["Month"] = data[date_column].dt.month

data["Day"] = data[date_column].dt.day

data["DayOfWeek"] = data[
    date_column
].dt.dayofweek


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X = data[
    [
        "Time_Index",
        "Year",
        "Month",
        "Day",
        "DayOfWeek"
    ]
]

y = data[target_column]


# Use chronological split
split_index = int(
    len(data) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


st.header("🎯 Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with metric2:

    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

with metric3:

    st.metric(
        "R² Score",
        f"{r2:.2%}"
    )


# ------------------------------------------------------------
# ACTUAL VS PREDICTED
# ------------------------------------------------------------

st.subheader(
    "Actual vs Predicted Values"
)

test_dates = model_data[
    date_column
].iloc[split_index:]


fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)

ax2.plot(
    test_dates,
    y_test.values,
    label="Actual",
    linewidth=2
)

ax2.plot(
    test_dates,
    y_pred,
    label="Predicted",
    linewidth=2
)

ax2.set_title(
    "Actual vs Predicted"
)

ax2.set_xlabel("Date")

ax2.set_ylabel(
    target_column
)

ax2.legend()

ax2.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig2)


# ------------------------------------------------------------
# FUTURE FORECAST
# ------------------------------------------------------------

st.header("🔮 Future Forecast")

last_date = model_data[
    date_column
].iloc[-1]

last_index = len(model_data) - 1


future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_periods,
    freq="D"
)


future_indices = np.arange(
    last_index + 1,
    last_index + 1 + forecast_periods
)


future_features = pd.DataFrame({

    "Time_Index": future_indices,

    "Year": future_dates.year,

    "Month": future_dates.month,

    "Day": future_dates.day,

    "DayOfWeek": future_dates.dayofweek
})


future_predictions = model.predict(
    future_features
)


forecast_df = pd.DataFrame({

    "Date": future_dates,

    "Predicted_Value": future_predictions.round(2)

})


# ------------------------------------------------------------
# FORECAST VISUALIZATION
# ------------------------------------------------------------

fig3, ax3 = plt.subplots(
    figsize=(12, 5)
)

ax3.plot(
    model_data[date_column],
    model_data[target_column],
    label="Historical",
    linewidth=2
)

ax3.plot(
    forecast_df["Date"],
    forecast_df["Predicted_Value"],
    label="Forecast",
    linestyle="--",
    linewidth=2
)

ax3.set_title(
    f"{target_column} - Future Forecast"
)

ax3.set_xlabel("Date")

ax3.set_ylabel(
    target_column
)

ax3.legend()

ax3.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig3)


# ------------------------------------------------------------
# FORECAST TABLE
# ------------------------------------------------------------

st.subheader(
    "📅 Forecasted Values"
)

st.dataframe(
    forecast_df,
    use_container_width=True
)


# ------------------------------------------------------------
# DOWNLOAD FORECAST
# ------------------------------------------------------------

csv_data = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv_data,
    file_name="future_forecast.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

st.header("📌 Analysis Summary")

average_value = model_data[
    target_column
].mean()

minimum_value = model_data[
    target_column
].min()

maximum_value = model_data[
    target_column
].max()

forecast_average = forecast_df[
    "Predicted_Value"
].mean()


summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.write(
        f"**Historical Average:** {average_value:.2f}"
    )

    st.write(
        f"**Historical Minimum:** {minimum_value:.2f}"
    )

    st.write(
        f"**Historical Maximum:** {maximum_value:.2f}"
    )


with summary_col2:

    st.write(
        f"**Forecast Period:** {forecast_periods} days"
    )

    st.write(
        f"**Forecast Average:** {forecast_average:.2f}"
    )

    st.write(
        f"**Model R² Score:** {r2:.2%}"
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Predictive Analytics Project | Historical Data Forecasting"
)