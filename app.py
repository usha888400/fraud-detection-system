import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="wide")
st.title("🔍 Real-Time Fraud Detection System")
st.markdown("**AI-powered system to detect fraudulent transactions in real time**")

@st.cache_data
def generate_and_train():
    np.random.seed(42)
    n = 10000
    
    # Generate transaction data
    normal = pd.DataFrame({
        'amount': np.random.exponential(100, int(n*0.98)),
        'time_hour': np.random.randint(0, 24, int(n*0.98)),
        'distance_from_home': np.random.exponential(10, int(n*0.98)),
        'distance_from_last': np.random.exponential(5, int(n*0.98)),
        'ratio_to_median': np.random.normal(1, 0.3, int(n*0.98)),
        'online_order': np.random.choice([0, 1], int(n*0.98), p=[0.7, 0.3]),
        'used_chip': np.random.choice([0, 1], int(n*0.98), p=[0.3, 0.7]),
        'used_pin': np.random.choice([0, 1], int(n*0.98), p=[0.4, 0.6]),
        'fraud': 0
    })
    
    fraud = pd.DataFrame({
        'amount': np.random.exponential(500, int(n*0.02)),
        'time_hour': np.random.choice([0,1,2,3,4,22,23], int(n*0.02)),
        'distance_from_home': np.random.exponential(100, int(n*0.02)),
        'distance_from_last': np.random.exponential(50, int(n*0.02)),
        'ratio_to_median': np.random.normal(3, 1, int(n*0.02)),
        'online_order': np.random.choice([0, 1], int(n*0.02), p=[0.2, 0.8]),
        'used_chip': np.random.choice([0, 1], int(n*0.02), p=[0.8, 0.2]),
        'used_pin': np.random.choice([0, 1], int(n*0.02), p=[0.9, 0.1]),
        'fraud': 1
    })
    
    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42)
    
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }
    
    return df, model, scaler, X.columns.tolist(), metrics

df, model, scaler, features, metrics = generate_and_train()

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
with col2:
    st.metric("Precision", f"{metrics['precision']:.1%}")
with col3:
    st.metric("Recall", f"{metrics['recall']:.1%}")
with col4:
    st.metric("F1 Score", f"{metrics['f1']:.1%}")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Transaction Distribution")
    fig1 = px.pie(df, names='fraud', title='Fraud vs Normal Transactions',
                  color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("💰 Amount Distribution by Type")
    fig2 = px.histogram(df, x='amount', color='fraud',
                        barmode='overlay', nbins=50,
                        title='Transaction Amount vs Fraud',
                        color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Confusion Matrix
st.subheader("📈 Model Performance - Confusion Matrix")
cm = metrics['cm']
fig3 = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted Normal', 'Predicted Fraud'],
    y=['Actual Normal', 'Actual Fraud'],
    colorscale='Reds',
    text=cm,
    texttemplate="%{text}"
))
fig3.update_layout(title='Confusion Matrix')
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Real-time prediction
st.subheader("🎯 Real-Time Transaction Prediction")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", 0.0, 10000.0, 150.0)
    time_hour = st.slider("Transaction Hour", 0, 23, 14)
    distance_home = st.number_input("Distance from Home (km)", 0.0, 500.0, 5.0)
    distance_last = st.number_input("Distance from Last Transaction (km)", 0.0, 500.0, 2.0)
with col2:
    ratio_median = st.number_input("Ratio to Median Purchase", 0.0, 10.0, 1.0)
    online_order = st.selectbox("Online Order", ['No', 'Yes'])
    used_chip = st.selectbox("Used Chip", ['No', 'Yes'])
    used_pin = st.selectbox("Used PIN", ['No', 'Yes'])

if st.button("🔍 Detect Fraud", type="primary"):
    input_data = np.array([[
        amount, time_hour, distance_home, distance_last,
        ratio_median,
        1 if online_order == 'Yes' else 0,
        1 if used_chip == 'Yes' else 0,
        1 if used_pin == 'Yes' else 0
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED!")
        st.error(f"Fraud Probability: {probability:.1%}")
        st.warning("Actions: Block transaction, alert customer, flag account for review")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION")
        st.success(f"Fraud Probability: {probability:.1%}")
        st.info("Transaction approved and processed normally")
