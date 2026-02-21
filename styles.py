import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* Main Background - Slate Theme */
    .stApp {
        background-color: #0F172A;
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar - Deep Indigo */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 2px solid #334155;
        padding-top: 2rem;
    }

    /* Sidebar Radio Buttons (Menu) */
    div[data-testid="stSidebarNav"] {
        display: none;
    }

    .st-emotion-cache-1647ebj {
        padding: 1rem 1.5rem;
    }

    /* Make Radio buttons look like prominent menu items */
    div[data-testid="stSidebar"] .stRadio > div {
        gap: 10px;
    }

    div[data-testid="stSidebar"] .stRadio label {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
        padding: 15px 20px;
        border-radius: 12px;
        color: #CBD5E1;
        font-size: 1.1rem !important;
        font-weight: 500;
        transition: all 0.3s ease;
        margin-bottom: 8px;
        cursor: pointer;
        display: block;
        width: 100%;
    }

    div[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: #6366F1;
        color: white;
    }

    div[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: #6366F1 !important;
        color: white !important;
        border-color: #818CF8 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    /* Glassmorphism Cards - High Contrast */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }

    /* KPI Cards - Larger & Bolder */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 40px;
    }

    .kpi-card {
        flex: 1;
        background: rgba(99, 102, 241, 0.08);
        border: 2px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        transform: scale(1.05);
        border-color: #6366F1;
        background: rgba(99, 102, 241, 0.15);
    }

    .kpi-value {
        font-size: 3rem; /* Increased size */
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 10px;
    }

    .kpi-label {
        font-size: 1.1rem; /* Increased size */
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Headings - Prominent */
    h1 {
        font-size: 3.5rem !important;
        color: white !important;
        font-weight: 900 !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 2rem !important;
        color: #818CF8 !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
    }

    /* Insight Box */
    .insight-box {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2) 0%, rgba(15, 23, 42, 0) 100%);
        border-left: 6px solid #6366F1;
        padding: 25px;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-size: 1.15rem;
        line-height: 1.6;
        color: #E2E8F0;
    }

    .insight-box b {
        color: #818CF8;
        font-size: 1.25rem;
    }

    /* Status Badges */
    .status-badge {
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 700;
    }

    /* Hide standard sidebar radio circle */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Bigger text for everything */
    p, li {
        font-size: 1.1rem;
        color: #CBD5E1;
    }
    </style>
    """, unsafe_allow_html=True)

def kpi_card(label, value, delta=None):
    delta_html = ""
    if delta:
        color = "#22C55E" if delta.startswith("+") else "#EF4444"
        delta_html = f'<div style="color: {color}; font-size: 1rem; font-weight: bold; margin-top: 5px;">{delta} ↑</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def insight_panel(text):
    st.markdown(f'<div class="insight-box">💡 <b>AI Intel:</b> {text}</div>', unsafe_allow_html=True)
