# 🔍 ML Anomaly Detection System

A machine learning-based anomaly detection system that identifies unusual transactions using the **Isolation Forest** algorithm and presents the results through an interactive **Streamlit dashboard**.

## 🚀 Features

- Detects unusual transaction behavior
- Uses Isolation Forest for unsupervised anomaly detection
- Standardizes numerical features using StandardScaler
- Generates anomaly scores
- Displays normal and anomalous transactions
- Interactive anomaly-rate control
- Transaction visualization
- Interactive anomaly results table
- Streamlit web interface

## 🧠 Machine Learning Approach

The project uses **Isolation Forest**, an unsupervised machine learning algorithm designed to identify observations that are significantly different from normal data patterns.

### Pipeline

```text
Transaction Data
       ↓
Data Generation
       ↓
Feature Selection
       ↓
StandardScaler
       ↓
Isolation Forest
       ↓
Anomaly Prediction
       ↓
Anomaly Score
       ↓
Visualization