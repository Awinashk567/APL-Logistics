import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Page configuration
st.set_page_config(page_title="APL Logistics Risk Dashboard", layout="wide")

# App title
st.title("📦 APL Logistics: Late Delivery Risk Analytics")
st.markdown("Predictive intelligence system for global supply chain operations.")

# Data aur Model load karne ke liye caching
@st.cache_data
def load_data():
    # Yahan path theek kar diya gaya hai (Cloud ke liye)
    df = pd.read_csv("APL_Logistics.csv", encoding='latin1')
    return df

@st.cache_resource
def load_model():
    with open('delivery_risk_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_columns.pkl', 'rb') as file:
        model_columns = pickle.load(file)
    return model, model_columns

# Load datasets and models
df = load_data()
model, model_columns = load_model()

# Sidebar Navigation (As per PDF requirements)
st.sidebar.title("Navigation Menu")
options = st.sidebar.radio(
    "Select Module:", 
    ["Delay Risk Overview", "Order-Level Risk Prediction", "Region & Mode Risk Analysis", "Operations Action Panel"]
)

# ---------------------------------------------
# MODULE 1: DELAY RISK OVERVIEW
# ---------------------------------------------
if options == "Delay Risk Overview":
    st.header("📋 Delay Risk Overview")
    
    # KPIs Calculation
    total_orders = len(df)
    late_orders = df['Late_delivery_risk'].sum()
    risk_rate = (late_orders / total_orders) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipment Orders", f"{total_orders:,}")
    col2.metric("High-Risk (Late) Orders", f"{late_orders:,}")
    col3.metric("Overall Late Risk Rate", f"{risk_rate:.2f}%")
    
    st.markdown("---")
    st.subheader("Overall Risk Distribution")
    
    # Graph: Risk Distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x='Late_delivery_risk', data=df, palette='Set2', ax=ax)
    ax.set_xticklabels(['On-Time / Ahead', 'Late Risk'])
    ax.set_xlabel("Delivery Risk Status")
    ax.set_ylabel("Order Count")
    st.pyplot(fig)

# ---------------------------------------------
# MODULE 2: ORDER-LEVEL RISK PREDICTION (Future Data Input)
# ---------------------------------------------
elif options == "Order-Level Risk Prediction":
    st.header("🔍 Order-Level Risk Prediction (New Order)")
    st.markdown("Naye ya future orders ki details niche daalein aur check karein ki delivery late hone ka risk hai ya nahi.")
    
    # Form layout for user inputs
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            shipping_mode = st.selectbox("Shipping Mode", df['Shipping Mode'].unique())
            market = st.selectbox("Market Region", df['Market'].unique())
            order_region = st.selectbox("Order Region", df['Order Region'].unique())
            customer_segment = st.selectbox("Customer Segment", df['Customer Segment'].unique())
            
        with col2:
            scheduled_days = st.number_input("Days for shipment (scheduled)", min_value=0, max_value=10, value=4)
            product_price = st.number_input("Product Price ($)", min_value=0.0, value=100.0)
            order_qty = st.number_input("Order Item Quantity", min_value=1, max_value=20, value=1)
            benefit = st.number_input("Benefit per order ($)", value=20.0)
            
        submit_button = st.form_submit_button(label="Predict Delivery Risk")
        
    if submit_button:
        # User input ko dataframe me convert karna
        input_data = pd.DataFrame([{
            'Shipping Mode': shipping_mode,
            'Market': market,
            'Order Region': order_region,
            'Customer Segment': customer_segment,
            'Days for shipment (scheduled)': scheduled_days,
            'Product Price': product_price,
            'Order Item Quantity': order_qty,
            'Benefit per order': benefit
        }])
        
        # Training wale format (One-Hot Encoding) me badalna
        input_encoded = pd.get_dummies(input_data)
        # Baki bache columns ko 0 set karna jo is single row me nahi hain
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Prediction karna
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]
        
        st.markdown("### 📊 Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ **High Risk!** Is order ke late hone ke chances high hain. (Probability: {probability*100:.2f}%)")
        else:
            st.success(f"✅ **Low Risk / On-Time!** Ye order time par deliver ho jayega. (Probability: {probability*100:.2f}%)")

# ---------------------------------------------
# MODULE 3: REGION & MODE RISK ANALYSIS
# ---------------------------------------------
elif options == "Region & Mode Analysis" or options == "Region & Mode Risk Analysis":
    st.header("🌎 Region & Shipping Mode Risk Analysis")
    
    # Filter capability as per PDF
    selected_market = st.selectbox("Select Market to Filter Graphs:", ["All"] + list(df['Market'].unique()))
    
    plot_df = df if selected_market == "All" else df[df['Market'] == selected_market]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk by Shipping Mode")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Shipping Mode', y='Late_delivery_risk', data=plot_df, estimator=np.mean, ci=None, palette='Blues_r', ax=ax1)
        ax1.set_ylabel("Average Late Risk Probability")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Risk by Order Region")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        region_risk = plot_df.groupby('Order Region')['Late_delivery_risk'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=region_risk.values, y=region_risk.index, palette='Oranges_r', ax=ax2)
        ax2.set_xlabel("Average Late Risk Probability")
        st.pyplot(fig2)

# ---------------------------------------------
# MODULE 4: OPERATIONS ACTION PANEL
# ---------------------------------------------
elif options == "Operations Action Panel":
    st.header("🚨 Operations Action Panel")
    st.markdown("Ye un orders ki list hai jinpar immediate attention ki zaroorat hai (High Risk Orders Queue).")
    
    # Filtering high risk orders to display as action queue
    high_risk_df = df[df['Late_delivery_risk'] == 1][['Shipping Mode', 'Market', 'Order Region', 'Product Name', 'Days for shipment (scheduled)']].head(20)
    st.dataframe(high_risk_df, use_container_width=True)
