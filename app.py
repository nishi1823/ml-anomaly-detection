import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="ML Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ML Anomaly Detection System")
st.write(
    "Detect unusual transactions using "
    "Isolation Forest."
)


# -----------------------------
# Generate sample transaction data
# -----------------------------
@st.cache_data
def generate_data(n_samples=500):

    np.random.seed(42)

    amount = np.random.normal(
        500, 150, n_samples
    )

    frequency = np.random.normal(
        5, 2, n_samples
    )

    balance = np.random.normal(
        5000, 1500, n_samples
    )

    data = pd.DataFrame({
        "Transaction Amount": amount,
        "Transaction Frequency": frequency,
        "Account Balance": balance
    })

    # Add artificial anomalies
    anomaly_indices = np.random.choice(
        n_samples,
        15,
        replace=False
    )

    data.loc[
        anomaly_indices,
        "Transaction Amount"
    ] *= 5

    data.loc[
        anomaly_indices,
        "Transaction Frequency"
    ] *= 4

    return data


data = generate_data()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Model Settings")

contamination = st.sidebar.slider(
    "Expected Anomaly Rate",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.01
)


# -----------------------------
# Prepare data
# -----------------------------
features = [
    "Transaction Amount",
    "Transaction Frequency",
    "Account Balance"
]

X = data[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -----------------------------
# Isolation Forest
# -----------------------------
model = IsolationForest(
    contamination=contamination,
    random_state=42
)

predictions = model.fit_predict(X_scaled)

scores = model.decision_function(X_scaled)


data["Prediction"] = predictions
data["Anomaly Score"] = scores

data["Status"] = data["Prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})


# -----------------------------
# Metrics
# -----------------------------
normal_count = (
    data["Status"] == "Normal"
).sum()

anomaly_count = (
    data["Status"] == "Anomaly"
).sum()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        len(data)
    )

with col2:
    st.metric(
        "Normal Transactions",
        normal_count
    )

with col3:
    st.metric(
        "Detected Anomalies",
        anomaly_count
    )


# -----------------------------
# Visualization
# -----------------------------
st.subheader("📊 Transaction Analysis")

fig, ax = plt.subplots()

normal = data[
    data["Status"] == "Normal"
]

anomalies = data[
    data["Status"] == "Anomaly"
]

ax.scatter(
    normal["Transaction Amount"],
    normal["Account Balance"],
    label="Normal",
    alpha=0.6
)

ax.scatter(
    anomalies["Transaction Amount"],
    anomalies["Account Balance"],
    label="Anomaly",
    marker="x",
    s=80
)

ax.set_xlabel("Transaction Amount")
ax.set_ylabel("Account Balance")
ax.set_title("Normal vs Anomalous Transactions")

ax.legend()

st.pyplot(fig)


# -----------------------------
# Anomaly table
# -----------------------------
st.subheader("🚨 Detected Anomalies")

anomaly_table = data[
    data["Status"] == "Anomaly"
].sort_values(
    "Anomaly Score"
)

st.dataframe(
    anomaly_table[
        [
            "Transaction Amount",
            "Transaction Frequency",
            "Account Balance",
            "Anomaly Score",
            "Status"
        ]
    ],
    use_container_width=True
)