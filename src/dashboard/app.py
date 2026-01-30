# src/dashboard/app.py
# PROFESSIONAL DARK TEAL CYBERSECURITY THEME

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# ==================== PATH SETUP ====================
SRC_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = SRC_DIR.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ==================== CUSTOM MODULES ====================
from auth.users import UserManager
from database.db_manager import DatabaseManager
from explainability.explainer import ModelExplainer

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PROFESSIONAL DARK TEAL THEME ====================
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main container - Dark professional background */
    .main {
        background: linear-gradient(135deg, #051F20 0%, #0B2B26 50%, #163832 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers with professional styling */
    h1 {
        color: #8EB69B !important;
        font-weight: 800 !important;
        font-size: 2.75rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 2px 8px rgba(142, 182, 155, 0.2);
    }
    
    h2 {
        color: #8EB69B !important;
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #8EB69B !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
    }
    
    /* Paragraph text */
    p, .stMarkdown {
        color: #8EB69B !important;
        line-height: 1.6;
    }
    
    /* Professional Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 56, 50, 0.95) 0%, rgba(11, 43, 38, 0.95) 100%);
        padding: 28px 24px;
        border-radius: 16px;
        border: 1px solid rgba(142, 182, 155, 0.2);
        box-shadow: 
            0 8px 32px rgba(5, 31, 32, 0.6),
            inset 0 1px 0 rgba(142, 182, 155, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #8EB69B 0%, #235347 100%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 12px 48px rgba(5, 31, 32, 0.8),
            0 0 0 1px rgba(142, 182, 155, 0.3),
            inset 0 1px 0 rgba(142, 182, 155, 0.2);
        border-color: rgba(142, 182, 155, 0.4);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #8EB69B;
        margin: 16px 0 8px 0;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
        text-shadow: 0 2px 12px rgba(142, 182, 155, 0.3);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: rgba(142, 182, 155, 0.8);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .metric-delta {
        font-size: 0.8125rem;
        color: rgba(142, 182, 155, 0.6);
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Role badge - Premium design */
    .role-badge {
        background: linear-gradient(135deg, #235347 0%, #163832 100%);
        padding: 10px 24px;
        border-radius: 24px;
        display: inline-block;
        font-weight: 700;
        color: #8EB69B;
        font-size: 0.8125rem;
        box-shadow: 
            0 4px 12px rgba(35, 83, 71, 0.4),
            inset 0 1px 0 rgba(142, 182, 155, 0.2);
        letter-spacing: 1.5px;
        border: 1px solid rgba(142, 182, 155, 0.2);
    }
    
    /* Sidebar - Professional dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #051F20 0%, #0B2B26 100%);
        border-right: 1px solid rgba(142, 182, 155, 0.15);
    }
    
    [data-testid="stSidebar"] * {
        color: #8EB69B !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #8EB69B !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        margin-bottom: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Navigation radio buttons */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(22, 56, 50, 0.3);
        padding: 14px 18px;
        border-radius: 10px;
        margin: 6px 0;
        transition: all 0.2s;
        border-left: 3px solid transparent;
        font-weight: 500;
        font-size: 0.9375rem;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(22, 56, 50, 0.6);
        border-left-color: #8EB69B;
        transform: translateX(4px);
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] [data-checked="true"] {
        background: linear-gradient(135deg, rgba(35, 83, 71, 0.6) 0%, rgba(22, 56, 50, 0.6) 100%);
        border-left-color: #8EB69B;
        box-shadow: 0 4px 12px rgba(35, 83, 71, 0.3);
    }
    
    /* Buttons - Professional gradient */
    .stButton button {
        background: linear-gradient(135deg, #235347 0%, #163832 100%);
        color: #8EB69B;
        font-weight: 700;
        border: 1px solid rgba(142, 182, 155, 0.3);
        padding: 14px 32px;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 16px rgba(35, 83, 71, 0.3),
            inset 0 1px 0 rgba(142, 182, 155, 0.2);
        font-size: 1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 8px 24px rgba(35, 83, 71, 0.5),
            0 0 0 1px rgba(142, 182, 155, 0.4),
            inset 0 1px 0 rgba(142, 182, 155, 0.3);
        background: linear-gradient(135deg, #2a6355 0%, #1d4a40 100%);
        border-color: rgba(142, 182, 155, 0.5);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(35, 83, 71, 0.3) 0%, rgba(22, 56, 50, 0.3) 100%) !important;
        border-left: 4px solid #8EB69B !important;
        color: #8EB69B !important;
        border-radius: 8px;
        padding: 16px !important;
        box-shadow: 0 4px 12px rgba(35, 83, 71, 0.2);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.15) 0%, rgba(180, 83, 9, 0.15) 100%) !important;
        border-left: 4px solid #f59e0b !important;
        color: #fbbf24 !important;
        border-radius: 8px;
        padding: 16px !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(22, 56, 50, 0.4) 0%, rgba(11, 43, 38, 0.4) 100%) !important;
        border-left: 4px solid #8EB69B !important;
        color: #8EB69B !important;
        border-radius: 8px;
        padding: 16px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(185, 28, 28, 0.15) 100%) !important;
        border-left: 4px solid #ef4444 !important;
        color: #fca5a5 !important;
        border-radius: 8px;
        padding: 16px !important;
    }
    
    /* Data tables - Professional styling */
    .dataframe {
        background: rgba(22, 56, 50, 0.6) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(142, 182, 155, 0.2) !important;
        box-shadow: 0 4px 16px rgba(5, 31, 32, 0.4) !important;
    }
    
    .dataframe thead tr th {
        background: rgba(35, 83, 71, 0.6) !important;
        color: #8EB69B !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.8125rem !important;
        padding: 14px 12px !important;
        border-bottom: 2px solid rgba(142, 182, 155, 0.3) !important;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid rgba(142, 182, 155, 0.1) !important;
        transition: background 0.2s;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(35, 83, 71, 0.3) !important;
    }
    
    .dataframe tbody tr td {
        color: #8EB69B !important;
        padding: 12px !important;
        font-size: 0.875rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(22, 56, 50, 0.4);
        border: 2px dashed rgba(142, 182, 155, 0.4);
        border-radius: 16px;
        padding: 32px 24px;
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(142, 182, 155, 0.6);
        background: rgba(22, 56, 50, 0.6);
    }
    
    [data-testid="stFileUploader"] label {
        color: #8EB69B !important;
        font-weight: 600;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(22, 56, 50, 0.8) !important;
        border: 1px solid rgba(142, 182, 155, 0.3) !important;
        border-radius: 10px !important;
        color: #8EB69B !important;
    }
    
    .stSelectbox option {
        background: #163832 !important;
        color: #8EB69B !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(22, 56, 50, 0.6) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(142, 182, 155, 0.2) !important;
        color: #8EB69B !important;
        font-weight: 600 !important;
        padding: 14px 18px !important;
        transition: all 0.2s;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(35, 83, 71, 0.6) !important;
        border-color: rgba(142, 182, 155, 0.4) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(11, 43, 38, 0.6);
        border: 1px solid rgba(142, 182, 155, 0.2);
        border-top: none;
        border-radius: 0 0 10px 10px;
    }
    
    /* Metric widgets */
    [data-testid="stMetricValue"] {
        color: #8EB69B !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(142, 182, 155, 0.8) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: rgba(142, 182, 155, 0.7) !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #235347 0%, #163832 100%);
        border: 1px solid rgba(142, 182, 155, 0.3);
    }
    
    /* Divider */
    hr {
        border-color: rgba(142, 182, 155, 0.2) !important;
        margin: 2rem 0 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #8EB69B !important;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(11, 43, 38, 0.4);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(142, 182, 155, 0.4);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(142, 182, 155, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==================== INIT MANAGERS ====================
def init_managers():
    return UserManager(), DatabaseManager()

user_manager, db_manager = init_managers()

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    rf_model = joblib.load(MODEL_DIR / "random_forest.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgboost.pkl")
    nn_model = keras.models.load_model(MODEL_DIR / "neural_network.h5")
    return rf_model, xgb_model, nn_model

@st.cache_data
def load_background_data():
    X_train = np.load(DATA_DIR / "processed" / "X_train.npy")
    return X_train[:100]

@st.cache_data
def load_results():
    return pd.read_csv(DATA_DIR / "model_comparison.csv")

# ==================== INITIALIZE ====================
try:
    rf_model, xgb_model, nn_model = load_models()
    background_data = load_background_data()
    results_df = load_results()
except Exception as e:
    st.error(f"❌ Startup error: {e}")
    st.stop()

# ==================== SESSION STATE INIT ====================
if 'logout' not in st.session_state:
    st.session_state['logout'] = False
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'total_predictions' not in st.session_state:
    st.session_state['total_predictions'] = 0
if 'total_attacks' not in st.session_state:
    st.session_state['total_attacks'] = 0
if 'total_normal' not in st.session_state:
    st.session_state['total_normal'] = 0
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []

# ==================== FEATURES ====================
FEATURE_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

# ==================== AUTH ====================
name, auth_status, username = user_manager.login()

if auth_status is False:
    st.error("❌ Invalid credentials")
    st.stop()
elif auth_status is None:
    st.warning("👆 Please login to access the system")
    st.info("**Demo Credentials:**\n- **Admin:** admin / admin123\n- **Analyst:** analyst1 / viewer123")
    st.stop()

user_role = user_manager.get_user_role(username)
user_manager.log_action(username, "login")

# ==================== HEADER ====================
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("# 🔒 Network Intrusion Detection System")
    st.markdown("*Enterprise-Grade AI Security Analytics Platform*")
with col2:
    st.markdown(f"**👤 {name}**")
    st.markdown(f'<div class="role-badge">ROLE: {user_role.upper()}</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎯 NAVIGATION")
    
    pages = [
        "📊 Dashboard",
        "🔍 Live Prediction",
        "📈 Historical Analytics",
        "📚 Model Comparison"
    ]
    
    if user_role == "admin":
        pages.append("📋 Audit Logs")
    
    page = st.radio("", pages, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚡ SYSTEM STATUS")
    st.success("✅ All Systems Online")
    st.info(f"🔐 Secure Session Active")
    st.markdown(f"🕐 **{datetime.now().strftime('%I:%M %p')}**")
    
    st.markdown("---")
    
    # FIXED LOGOUT BUTTON
    if st.button("🚪 LOGOUT", use_container_width=True, key="logout_btn"):
        # Clear authentication state
        st.session_state['name'] = None
        st.session_state['authentication_status'] = None
        st.session_state['username'] = None
        st.session_state['logout'] = True
        
        # Force page reload
        st.rerun()

# ==================== DASHBOARD PAGE ====================
if page == "📊 Dashboard":
    st.header("📊 System Overview Dashboard")
    
    # Calculate metrics
    total_preds = st.session_state['total_predictions']
    total_attacks = st.session_state['total_attacks']
    total_normal = st.session_state['total_normal']
    attack_rate = (total_attacks / total_preds * 100) if total_preds > 0 else 0
    
    # Metric Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 TOTAL PREDICTIONS</div>
            <div class="metric-value">{total_preds:,}</div>
            <div class="metric-delta">Lifetime Analytics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ THREATS DETECTED</div>
            <div class="metric-value" style="color: #fbbf24;">{total_attacks:,}</div>
            <div class="metric-delta">{attack_rate:.1f}% Attack Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✅ CLEAN TRAFFIC</div>
            <div class="metric-value" style="color: #8EB69B;">{total_normal:,}</div>
            <div class="metric-delta">{100-attack_rate:.1f}% Normal</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ OPTIMAL MODEL</div>
            <div class="metric-value" style="font-size: 2.2rem;">{best_model['Accuracy']:.1%}</div>
            <div class="metric-delta">{best_model['Model']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Performance Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Accuracy Comparison")
        fig = px.bar(
            results_df, 
            x="Model", 
            y="Accuracy",
            color="Accuracy",
            color_continuous_scale=[[0, "#163832"], [0.5, "#235347"], [1, "#8EB69B"]],
            text="Accuracy"
        )
        fig.update_traces(
            texttemplate='%{text:.2%}', 
            textposition='outside',
            marker_line_color='rgba(142, 182, 155, 0.3)',
            marker_line_width=1
        )
        fig.update_layout(
            plot_bgcolor='rgba(22, 56, 50, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8EB69B', size=13, family='Inter'),
            showlegend=False,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(142, 182, 155, 0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Training Time Analysis")
        fig = px.bar(
            results_df,
            x="Model",
            y="Training Time (s)",
            color="Training Time (s)",
            color_continuous_scale=[[0, "#163832"], [0.5, "#235347"], [1, "#8EB69B"]],
            text="Training Time (s)"
        )
        fig.update_traces(
            texttemplate='%{text:.2f}s', 
            textposition='outside',
            marker_line_color='rgba(142, 182, 155, 0.3)',
            marker_line_width=1
        )
        fig.update_layout(
            plot_bgcolor='rgba(22, 56, 50, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8EB69B', size=13, family='Inter'),
            showlegend=False,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(142, 182, 155, 0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Table
    st.subheader("📊 Comprehensive Performance Metrics")
    styled_df = results_df.style.format({
        'Accuracy': '{:.2%}',
        'ROC-AUC': '{:.4f}',
        'Training Time (s)': '{:.4f}'
    }).background_gradient(cmap='Greens', subset=['Accuracy', 'ROC-AUC'])
    
    st.dataframe(styled_df, use_container_width=True, height=200)
    
    # Recent Activity
    if st.session_state['prediction_history']:
        st.subheader("📜 Recent Prediction Activity")
        recent_df = pd.DataFrame(st.session_state['prediction_history'][-10:])
        st.dataframe(recent_df, use_container_width=True)

# ==================== LIVE PREDICTION PAGE ====================
elif page == "🔍 Live Prediction":
    st.header("🔍 Real-Time Threat Detection")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("### 📤 Upload Network Traffic Data")
        st.info("📄 Upload CSV file containing network traffic features for AI-powered threat analysis.")
    
    with col2:
        model_choice = st.selectbox(
            "🤖 AI Model Selection",
            ["Random Forest", "XGBoost", "Neural Network"],
            help="Choose the machine learning model for prediction analysis"
        )
    
    uploaded = st.file_uploader(
        "📁 Drag and drop or browse CSV files",
        type="csv",
        help="Upload network traffic data (CSV format, max 200MB)"
    )
    
    if uploaded:
        data = pd.read_csv(uploaded)
        
        st.success(f"✅ File uploaded successfully! **{len(data):,} samples** loaded for analysis.")
        
        with st.expander("👁️ Preview Data (First 10 Rows)"):
            st.dataframe(data.head(10), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🚀 RUN THREAT ANALYSIS", use_container_width=True):
                with st.spinner("🔄 Analyzing network patterns with AI..."):
                    # Get predictions
                    if model_choice == "Random Forest":
                        model = rf_model
                        probs = model.predict_proba(data)[:,1]
                    elif model_choice == "XGBoost":
                        model = xgb_model
                        probs = model.predict_proba(data)[:,1]
                    else:
                        model = nn_model
                        probs = model.predict(data).flatten()
                    
                    preds = (probs > 0.5).astype(int)
                    
                    # Update session state
                    attacks = sum(preds)
                    normal = len(preds) - attacks
                    st.session_state['total_predictions'] += len(preds)
                    st.session_state['total_attacks'] += attacks
                    st.session_state['total_normal'] += normal
                    
                    # Add to history
                    st.session_state['prediction_history'].append({
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Model': model_choice,
                        'Samples': len(preds),
                        'Attacks': attacks,
                        'Normal': normal
                    })
                    
                    result_df = pd.DataFrame({
                        "Sample ID": range(len(preds)),
                        "Prediction": ["🔴 Attack" if p else "🟢 Normal" for p in preds],
                        "Confidence": probs,
                        "Risk Level": ["High" if p > 0.8 else "Medium" if p > 0.5 else "Low" for p in probs]
                    })
                    
                    st.success("✅ **Analysis Complete!** Dashboard metrics have been updated with latest findings.")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⚠️ Threats Detected", attacks, f"{(attacks/len(preds)*100):.1f}%")
                    with col2:
                        st.metric("✅ Clean Traffic", normal, f"{(normal/len(preds)*100):.1f}%")
                    with col3:
                        avg_conf = probs.mean()
                        st.metric("🎯 Average Confidence", f"{avg_conf:.2%}")
                    
                    # Results table
                    st.subheader("📋 Detailed Analysis Results")
                    st.dataframe(
                        result_df.style.format({"Confidence": "{:.2%}"}),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 EXPORT RESULTS (CSV)",
                        csv,
                        f"threat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key='download-csv',
                        use_container_width=True
                    )
    else:
        st.warning("👆 Upload a CSV file to initiate threat detection analysis")
        
        # Sample data generator
        st.markdown("### 🧪 Need Test Data?")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("📦 GENERATE SAMPLE DATASET", use_container_width=True):
                sample_data = pd.DataFrame(
                    np.random.rand(20, len(FEATURE_NAMES)),
                    columns=FEATURE_NAMES
                )
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Sample CSV",
                    csv,
                    "sample_network_traffic.csv",
                    "text/csv",
                    use_container_width=True
                )

# ==================== HISTORICAL ANALYTICS PAGE ====================
elif page == "📈 Historical Analytics":
    st.header("📈 Historical Threat Intelligence")
    
    if st.session_state['prediction_history']:
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        
        st.success(f"📊 **{len(history_df):,} prediction sessions** recorded in system logs")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total = history_df['Samples'].sum()
            st.metric("Total Samples Analyzed", f"{total:,}")
        with col2:
            attacks = history_df['Attacks'].sum()
            st.metric("Cumulative Threats", f"{attacks:,}")
        with col3:
            rate = (attacks / total * 100) if total > 0 else 0
            st.metric("Overall Attack Rate", f"{rate:.2f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Time series chart
        st.subheader("📉 Threat Detection Timeline")
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['Timestamp'],
            y=history_df['Attacks'],
            mode='lines+markers',
            name='Threats Detected',
            line=dict(color='#fbbf24', width=3),
            marker=dict(size=10, line=dict(width=2, color='#163832')),
            fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=history_df['Timestamp'],
            y=history_df['Normal'],
            mode='lines+markers',
            name='Clean Traffic',
            line=dict(color='#8EB69B', width=3),
            marker=dict(size=10, line=dict(width=2, color='#163832')),
            fill='tozeroy',
            fillcolor='rgba(142, 182, 155, 0.1)'
        ))
        fig.update_layout(
            plot_bgcolor='rgba(22, 56, 50, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8EB69B', family='Inter', size=13),
            xaxis=dict(
                title="Timeline",
                showgrid=True,
                gridcolor='rgba(142, 182, 155, 0.1)'
            ),
            yaxis=dict(
                title="Sample Count",
                showgrid=True,
                gridcolor='rgba(142, 182, 155, 0.1)'
            ),
            hovermode='x unified',
            legend=dict(
                bgcolor='rgba(22, 56, 50, 0.8)',
                bordercolor='rgba(142, 182, 155, 0.3)',
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.subheader("📜 Complete Session History")
        st.dataframe(history_df, use_container_width=True)
        
    else:
        st.info("📊 **No historical data available yet.**\n\nInitiate threat detection in the **Live Prediction** module to populate analytics dashboard.")

# ==================== MODEL COMPARISON PAGE ====================
elif page == "📚 Model Comparison":
    st.header("📚 AI Model Performance Analysis")
    
    st.markdown("""
    ### 🎯 Model Selection Guide
    
    **Enterprise AI Models for Intrusion Detection:**
    
    - 🌲 **Random Forest**: Ensemble learning with balanced accuracy and computational efficiency
    - 🚀 **XGBoost**: Gradient boosting with highest accuracy (80.24%), production-recommended
    - 🧠 **Neural Network**: Deep learning architecture for complex threat pattern recognition
    """)
    
    # Performance table
    st.subheader("📊 Comparative Performance Metrics")
    st.dataframe(results_df, use_container_width=True)
    
    # Visual comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            title="Accuracy Benchmark",
            color="Accuracy",
            color_continuous_scale=[[0, "#163832"], [0.5, "#235347"], [1, "#8EB69B"]],
            text="Accuracy"
        )
        fig.update_traces(
            texttemplate='%{text:.2%}', 
            textposition='outside',
            marker_line_color='rgba(142, 182, 155, 0.3)',
            marker_line_width=1
        )
        fig.update_layout(
            plot_bgcolor='rgba(22, 56, 50, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8EB69B', family='Inter')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            results_df,
            x="Model",
            y="ROC-AUC",
            title="ROC-AUC Score Analysis",
            color="ROC-AUC",
            color_continuous_scale=[[0, "#163832"], [0.5, "#235347"], [1, "#8EB69B"]],
            text="ROC-AUC"
        )
        fig.update_traces(
            texttemplate='%{text:.4f}', 
            textposition='outside',
            marker_line_color='rgba(142, 182, 155, 0.3)',
            marker_line_width=1
        )
        fig.update_layout(
            plot_bgcolor='rgba(22, 56, 50, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8EB69B', family='Inter')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("🎲 Confusion Matrix Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(str(DATA_DIR / "Random_Forest_confusion_matrix.png"), caption="Random Forest", use_container_width=True)
    with col2:
        st.image(str(DATA_DIR / "XGBoost_confusion_matrix.png"), caption="XGBoost", use_container_width=True)
    with col3:
        st.image(str(DATA_DIR / "Neural_Network_confusion_matrix.png"), caption="Neural Network", use_container_width=True)
    
    # ROC Curves
    st.subheader("📈 ROC Curve Comparative Analysis")
    st.image(str(DATA_DIR / "roc_curves_comparison.png"), use_container_width=True)

# ==================== AUDIT LOGS (ADMIN ONLY) ====================
elif page == "📋 Audit Logs" and user_role == "admin":
    st.header("📋 System Audit Logs")
    
    logs = user_manager.get_audit_logs()
    
    if logs:
        logs_df = pd.DataFrame(logs)
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info("No audit logs recorded in system yet.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    f"<center style='color: rgba(142, 182, 155, 0.7); font-size: 0.875rem;'>"
    f"Logged in as <b style='color: #8EB69B;'>{name}</b> ({user_role.upper()}) | "
    f"Network Intrusion Detection System © 2026 | Enterprise Security Platform"
    f"</center>",
    unsafe_allow_html=True
)
