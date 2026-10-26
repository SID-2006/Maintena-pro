import os
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

def render_html(html_str):
    """Safely renders HTML without markdown treating indented lines as code blocks."""
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Maintena Pro | Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Luxury CSS
css_path = Path("assets") / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & Models Initialization
# ---------------------------------------------------------
models = utils.load_models()
summary = utils.load_summary_metrics()
df = utils.load_dataset()

# ---------------------------------------------------------
# Global Sidebar
# ---------------------------------------------------------
with st.sidebar:
    render_html("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="background: linear-gradient(135deg, #10B981 0%, #00F0FF 100%); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 18px rgba(0, 240, 255, 0.35);">⚙️</div>
            <div>
                <h3 style="margin: 0; font-size: 1.4rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">Maintena Pro</h3>
                <span style="font-size: 0.72rem; color: #00F0FF; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">PREDICTIVE TELEMETRY</span>
            </div>
        </div>
    """)
    st.markdown("---")

    st.markdown("#### 🤖 Predictive Classifier:")
    model_choices = {
        "Tuned Random Forest": "Random Forest (Tuned • 97.4% Acc)",
        "Random Forest": "Random Forest (100 Trees)",
        "Bagging Classifier": "Bagging Classifier (50 Trees)",
        "Decision Tree": "Decision Tree Classifier",
        "K-Nearest Neighbors (k=5)": "K-Nearest Neighbors (KNN, k=5)",
        "Logistic Regression": "Logistic Regression (Linear)",
        "Gaussian Naive Bayes": "Naive Bayes (Probabilistic)"
    }

    selected_model_key = st.selectbox(
        "Active Model for Telemetry Inference:",
        options=list(model_choices.keys()),
        format_func=lambda k: model_choices[k],
        index=0,
        help="Select which trained classification algorithm evaluates machine failure risk."
    )

    comp_list = summary.get("classification_comparison", [])
    model_meta = next((m for m in comp_list if m["Model"] == selected_model_key), None)
    if model_meta:
        render_html(f"""
            <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 14px 16px; margin-top: 8px; border: 1px solid rgba(255,255,255,0.08); box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">F1 Score:</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #00F0FF; font-family: 'JetBrains Mono';">{model_meta['F1 Score']:.3f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                    <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Test Accuracy:</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #10B981; font-family: 'JetBrains Mono';">{model_meta['Accuracy']*100:.1f}%</span>
                </div>
            </div>
        """)

    st.markdown("---")
    st.markdown("#### 🏭 Fleet Status Snapshot")
    ds_stats = summary.get("dataset_stats", {})
    render_html(f"""
        <div style="background: rgba(13,19,33,0.7); border-radius: 12px; padding: 16px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.84rem; line-height: 1.6; color: #CBD5E1;">
            • <b>Total Machines:</b> {ds_stats.get('total_records', 10000):,} units<br>
            • <b>Healthy Units:</b> {ds_stats.get('normal_count', 9661):,} (96.6%)<br>
            • <b>Failures Tracked:</b> {ds_stats.get('failure_count', 339):,} (3.4%)<br>
            • <b>Failure Modes:</b> TWF, HDF, PWF, OSF, RNF<br>
            • <b>Regression Target:</b> Process Temp ($R^2 = 0.76$)
        </div>
    """)

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
render_html("""
    <div class="maintena-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <h1 class="maintena-title">Smart Machine Health Predictor</h1>
                <p class="maintena-subtitle">Industrial Predictive Maintenance & Diagnostics Platform • 10,000 CNC Telemetry Records</p>
            </div>
            <div>
                <div class="badge-live">
                    <span class="pulse-dot"></span>
                    TELEMETRY ENGINE ONLINE
                </div>
            </div>
        </div>
    </div>
""")

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "🎯 1. Live Machine Health Predictor",
    "📊 2. Exploratory Data Analysis (EDA)",
    "📁 3. Batch Fleet Diagnostics (CSV)",
    "🤖 4. ML Models Leaderboard & Benchmarks",
    "🌐 5. Machine Operating Regimes (Clustering)"
])

# =========================================================
# TAB 1: LIVE MACHINE HEALTH PREDICTOR
# =========================================================
with tabs[0]:
    render_html("""
        <div class="friendly-card" style="border-left: 4px solid #00F0FF; padding: 16px 20px; margin-bottom: 20px;">
            <b style="color: #00F0FF; font-size: 1.05rem;">💡 Interactive Diagnostic Simulator:</b>
            <p style="margin: 4px 0 0 0; color: #CBD5E1; font-size: 0.92rem;">
                Select a <b>Quick Scenario</b> below to test a pre-configured operating condition, or adjust the sliders manually to simulate live machine conditions and see real-time predictive diagnostics.
            </p>
        </div>
    """)

    # Initialize session state for inputs
    if "p_type" not in st.session_state:
        st.session_state["p_type"] = "M"
    if "p_air_c" not in st.session_state:
        st.session_state["p_air_c"] = 25.0
    if "p_proc_c" not in st.session_state:
        st.session_state["p_proc_c"] = 35.5
    if "p_speed" not in st.session_state:
        st.session_state["p_speed"] = 1550
    if "p_torque" not in st.session_state:
        st.session_state["p_torque"] = 40.0
    if "p_wear" not in st.session_state:
        st.session_state["p_wear"] = 35

    def apply_preset(ptype, air, proc, spd, trq, wear):
        st.session_state["p_type"] = ptype
        st.session_state["p_air_c"] = air
        st.session_state["p_proc_c"] = proc
        st.session_state["p_speed"] = spd
        st.session_state["p_torque"] = trq
        st.session_state["p_wear"] = wear

    # Friendly Presets
    st.markdown("##### ⚡ Quick Scenarios (Click to test):")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.button("🟢 Nominal Steady-State (Safe)", on_click=apply_preset, args=("M", 25.0, 35.5, 1550, 40.0, 35), use_container_width=True)
    with col_p2:
        st.button("🟠 Thermal Dissipation Warning", on_click=apply_preset, args=("L", 30.5, 37.0, 1320, 52.0, 110), use_container_width=True)
    with col_p3:
        st.button("🔴 Tool Blade Fatigue (Terminal Wear)", on_click=apply_preset, args=("L", 28.0, 37.5, 1350, 65.0, 220), use_container_width=True)
    with col_p4:
        st.button("⚡ High Power Surge (Motor Overload)", on_click=apply_preset, args=("H", 26.0, 36.0, 2700, 38.0, 80), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Sliders
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        render_html("""
            <div class="category-badge">
                🌡️ 1. Thermal & Material Specifications
            </div>
        """)

        type_map = {
            "L": "L — Standard (High volume, budget tier)",
            "M": "M — Medium (General industrial tier)",
            "H": "H — Heavy Duty (Precision & high durability tier)"
        }
        product_type = st.selectbox(
            "Machine Quality Variant (Type):",
            options=["L", "M", "H"],
            key="p_type",
            format_func=lambda x: type_map[x],
            help="Quality class of the milling machine."
        )

        air_temp_c = st.slider(
            "Room / Ambient Temperature (°C):",
            min_value=22.0,
            max_value=32.0,
            key="p_air_c",
            step=0.5,
            format="%.1f °C",
            help="Ambient environment temperature surrounding the equipment."
        )
        air_temp_k = air_temp_c + 273.15
        st.caption(f"ℹ️ Scientific Units: **{air_temp_k:.2f} K** &nbsp;|&nbsp; **{air_temp_c * 9/5 + 32:.1f} °F**")

        proc_temp_c = st.slider(
            "Machine Operating Temperature (°C):",
            min_value=32.0,
            max_value=42.0,
            key="p_proc_c",
            step=0.5,
            format="%.1f °C",
            help="Internal process operating temperature measured at the cutter/workpiece contact."
        )
        proc_temp_k = proc_temp_c + 273.15
        st.caption(f"ℹ️ Scientific Units: **{proc_temp_k:.2f} K** (Thermal Delta: **{proc_temp_c - air_temp_c:.1f} °C**)")

    with col_in2:
        render_html("""
            <div class="category-badge">
                ⚙️ 2. Mechanical & Tool Dynamics
            </div>
        """)

        rot_speed = st.slider(
            "Spindle Rotational Speed (RPM):",
            min_value=1100,
            max_value=2900,
            key="p_speed",
            step=25,
            help="Spindle rotational speed. Nominal steady-state is around 1400 - 1600 RPM."
        )

        torque = st.slider(
            "Spindle Resistance Force (Torque in Nm):",
            min_value=5.0,
            max_value=80.0,
            key="p_torque",
            step=1.0,
            format="%.1f Nm",
            help="Twisting resistance force applied on the cutter during machining."
        )

        tool_wear = st.slider(
            "Accumulated Tool Usage Time (Minutes):",
            min_value=0,
            max_value=250,
            key="p_wear",
            step=5,
            help="Accumulated working minutes of the cutting tool bit. Tools begin failing dangerously after 200 minutes."
        )
        wear_pct = (tool_wear / 240) * 100
        st.caption(f"ℹ️ Tool Lifespan Consumed: **{tool_wear} / 240 minutes** ({wear_pct:.0f}% of lifespan)")

    # Run Prediction using backend
    inputs = {
        "Type": product_type,
        "Air temperature [K]": air_temp_k,
        "Process temperature [K]": proc_temp_k,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }
    pred_res = utils.predict_single(inputs, selected_model_name=selected_model_key)
    fail_prob = pred_res["failure_probability"]
    fail_state = pred_res["failure_prediction"]
    physics = pred_res["physics"]

    st.markdown("---")
    st.markdown("### 📋 Predictive Health Diagnostic Result")

    # High-impact verdict banner
    if fail_prob >= 0.50 or any(r["severity"] == "CRITICAL" for r in physics["active_risks"]):
        banner_class = "result-danger"
        verdict_icon = "🚨"
        verdict_title = "CRITICAL FAILURE IMMINENT!"
        verdict_msg = "The predictive model indicates this machine is under extreme operational strain or has breached safe mechanical tolerances. Immediate inspection or shutdown recommended."
        gauge_color = "#FF3B5C"
        badge_text = "CRITICAL RISK"
    elif fail_prob >= 0.25 or len(physics["active_risks"]) > 0:
        banner_class = "result-warning"
        verdict_icon = "⚠️"
        verdict_title = "WARNING: Elevated Stress & Degradation"
        verdict_msg = "Operating parameters are approaching critical physical thresholds. Schedule maintenance check soon."
        gauge_color = "#F59E0B"
        badge_text = "MODERATE RISK"
    else:
        banner_class = "result-safe"
        verdict_icon = "✅"
        verdict_title = "ALL SYSTEMS NOMINAL: Equipment Operating Healthy"
        verdict_msg = "Sensory telemetry indicates normal operating temperatures, safe rotational speeds, standard torque, and healthy tool life."
        gauge_color = "#10B981"
        badge_text = "SAFE & HEALTHY"

    render_html(f"""
        <div class="{banner_class}">
            <div style="font-size: 1.35rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.01em;">
                {verdict_icon} {verdict_title} &nbsp; 
                <span style="font-size: 0.76rem; background: rgba(0,0,0,0.35); padding: 4px 12px; border-radius: 9999px; font-weight: 700; letter-spacing: 0.05em;">{badge_text}</span>
            </div>
            <p style="margin: 0; font-size: 0.94rem; color: #E2E8F0; line-height: 1.5;">{verdict_msg}</p>
        </div>
    """)

    # Result Gauge and Quick Summary Box
    col_res1, col_res2 = st.columns([1, 1.3])

    with col_res1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fail_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#FFFFFF", "family": "Plus Jakarta Sans"}},
            title={"text": "Failure Risk Probability", "font": {"size": 15, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": gauge_color, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.04)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [25, 50], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [50, 100], "color": "rgba(255, 59, 92, 0.2)"}
                ]
            }
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col_res2:
        pred_proc_c = pred_res["predicted_process_temp"] - 273.15
        render_html(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;">
                <div class="metric-box">
                    <div class="metric-box-title">Equipment Status</div>
                    <div class="metric-box-value" style="color: {gauge_color}; font-size: 1.3rem;">{badge_text}</div>
                    <span style="font-size: 0.76rem; color: #94A3B8;">Predicted by {model_choices[selected_model_key].split('(')[0].strip()}</span>
                </div>
                <div class="metric-box">
                    <div class="metric-box-title">Failure Probability</div>
                    <div class="metric-box-value" style="color: {gauge_color}; font-size: 1.3rem;">{fail_prob*100:.1f}%</div>
                    <span style="font-size: 0.76rem; color: #94A3B8;">Model confidence risk score</span>
                </div>
                <div class="metric-box">
                    <div class="metric-box-title">Expected Machine Temp</div>
                    <div class="metric-box-value" style="color: #00F0FF; font-size: 1.3rem;">{pred_proc_c:.1f} °C</div>
                    <span style="font-size: 0.76rem; color: #94A3B8;">Estimated by Linear Regression</span>
                </div>
                <div class="metric-box">
                    <div class="metric-box-title">Tool Health Status</div>
                    <div class="metric-box-value" style="color: {'#FF3B5C' if tool_wear >= 200 else '#10B981'}; font-size: 1.3rem;">{240 - tool_wear} min left</div>
                    <span style="font-size: 0.76rem; color: #94A3B8;">Out of 240 mins nominal lifespan</span>
                </div>
            </div>
        """)

    # Plain Explanations and Actions
    st.markdown("<br>", unsafe_allow_html=True)
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("#### 🧐 Physical Anomaly Diagnostic")
        if physics["active_risks"]:
            for r in physics["active_risks"]:
                color = "#FF3B5C" if r["severity"] == "CRITICAL" else "#F59E0B"
                friendly_names = {
                    "TWF": "Tool Wear Failure Limit Reached",
                    "HDF": "Heat Dissipation Deficiency (Overheating)",
                    "PWF": "Abnormal Motor Power Surge",
                    "OSF": "Tool Overstrain (Excessive Cutting Torque)"
                }
                title = friendly_names.get(r["code"], r["name"])
                render_html(f"""
                    <div class="friendly-card" style="border-left: 4px solid {color}; padding: 16px 20px;">
                        <b style="color: {color}; font-size: 1rem;">⚠️ {title}</b>
                        <p style="margin: 6px 0 0 0; font-size: 0.88rem; color: #E2E8F0; line-height: 1.5;">
                            {r['reason']}
                        </p>
                    </div>
                """)
        else:
            render_html("""
                <div class="friendly-card" style="border-left: 4px solid #10B981; padding: 16px 20px;">
                    <b style="color: #10B981; font-size: 1rem;">✅ All Physical Constraints Satisfied</b>
                    <p style="margin: 6px 0 0 0; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                        Motor power, thermal differential, tool wear duration, and overstrain parameters are all within normal engineering bounds.
                    </p>
                </div>
            """)

    with col_exp2:
        st.markdown("#### 🛠️ Recommended Operator Action")
        rec_items = "".join([
            f'<div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;">'
            f'<div style="background:rgba(0,240,255,0.18);color:#00F0FF;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;">{idx}</div>'
            f'<div style="font-size:0.9rem;color:#E2E8F0;line-height:1.5;">{rec}</div>'
            f'</div>'
            for idx, rec in enumerate(pred_res["recommendations"], 1)
        ])
        render_html(f'<div class="friendly-card">{rec_items}</div>')

# =========================================================
# TAB 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tabs[1]:
    st.markdown("### 📊 Exploratory Data Analysis (EDA) & Sensor Analytics")
    st.markdown("""
        In-depth statistical exploration and visualizations of the **10,000 factory milling machines** in the benchmark dataset.
        Explore failure distributions, bivariate physical relationships, sensor distributions, outlier boxplots, and feature correlations.
    """)

    if df is not None:
        # EDA Sub-View Selector
        eda_view = st.radio(
            "Select EDA View:",
            options=[
                "1. Fleet Health & Failure Modes Breakdown",
                "2. Sensor Distributions & Outlier Box Plots",
                "3. Physical Failure Boundaries & Bivariate Scatters",
                "4. Correlation Matrix & Heatmap",
                "5. Raw Data Explorer & Summary Statistics"
            ],
            horizontal=True
        )

        st.markdown("---")

        # ---------------------------------------------------------
        # SUB-VIEW 1: Fleet Health & Failure Modes Breakdown
        # ---------------------------------------------------------
        if eda_view == "1. Fleet Health & Failure Modes Breakdown":
            st.markdown("#### 1. Machine Health & Quality Distribution")
            
            # Top KPI Summary Cards
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                render_html("""
                    <div class="metric-box">
                        <div class="metric-box-title">Total Monitored Units</div>
                        <div class="metric-box-value">10,000</div>
                        <span style="font-size: 0.78rem; color: #94A3B8;">CNC Milling Machines</span>
                    </div>
                """)
            with col_k2:
                render_html("""
                    <div class="metric-box">
                        <div class="metric-box-title">Healthy Machines</div>
                        <div class="metric-box-value" style="color: #10B981;">9,661</div>
                        <span style="font-size: 0.78rem; color: #10B981;">96.61% Normal Operational</span>
                    </div>
                """)
            with col_k3:
                render_html("""
                    <div class="metric-box">
                        <div class="metric-box-title">Machine Failures</div>
                        <div class="metric-box-value" style="color: #FF3B5C;">339</div>
                        <span style="font-size: 0.78rem; color: #FF3B5C;">3.39% Minority Class</span>
                    </div>
                """)
            with col_k4:
                render_html("""
                    <div class="metric-box">
                        <div class="metric-box-title">Primary Failure Mode</div>
                        <div class="metric-box-value" style="color: #F59E0B; font-size: 1.3rem;">Heat Dissipation</div>
                        <span style="font-size: 0.78rem; color: #94A3B8;">115 Incidents (33.9%)</span>
                    </div>
                """)

            st.markdown("<br>", unsafe_allow_html=True)

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("##### 🚨 Failure Modes Breakdown (339 Total Failures)")
                ft = summary.get("dataset_stats", {}).get("failure_types", {})
                if ft:
                    fig_ft_pie = px.pie(
                        names=list(ft.keys()),
                        values=list(ft.values()),
                        hole=0.55,
                        color_discrete_sequence=["#FF3B5C", "#F59E0B", "#3B82F6", "#8B5CF6", "#EC4899"],
                        title="Distribution of Machine Failure Causes"
                    )
                    fig_ft_pie.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=360,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_ft_pie, use_container_width=True)

            with col_chart2:
                st.markdown("##### 🏷️ Failure Rate by Machine Quality Tier (L, M, H)")
                quality_grp = df.groupby("Type")["Machine failure"].agg(
                    Total="count",
                    Failures="sum",
                    Rate="mean"
                ).reset_index()
                quality_grp["Rate (%)"] = quality_grp["Rate"] * 100

                fig_type = px.bar(
                    quality_grp,
                    x="Type",
                    y="Rate (%)",
                    text_auto=".2f",
                    color="Type",
                    color_discrete_map={"L": "#FF3B5C", "M": "#F59E0B", "H": "#10B981"},
                    title="Failure Rate by Product Quality Type"
                )
                fig_type.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13, 19, 33, 0.6)",
                    yaxis=dict(title="Failure Rate (%)"),
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_type, use_container_width=True)

            render_html("""
                <div class="friendly-card" style="font-size: 0.88rem; line-height: 1.6; color: #CBD5E1;">
                    <b>Key Observation:</b> Type <b>L (Low Quality / High Volume)</b> machines exhibit the highest failure rate of <b>~3.92%</b>, 
                    while Type <b>H (High Quality Heavy Duty)</b> machines fail at only <b>~1.53%</b>. This confirms higher build quality directly correlates with lower downtime.
                </div>
            """)

        # ---------------------------------------------------------
        # SUB-VIEW 2: Sensor Distributions & Outlier Box Plots
        # ---------------------------------------------------------
        elif eda_view == "2. Sensor Distributions & Outlier Box Plots":
            st.markdown("#### 2. Sensor Distributions & Outlier Analysis")
            st.markdown("Analyze the distribution of sensor telemetry and examine how outliers correlate with machine breakdown.")

            col_sel1, col_sel2 = st.columns([1.5, 1])
            with col_sel1:
                selected_sensor = st.selectbox(
                    "Choose Sensor Feature to Inspect:",
                    options=[
                        "Torque [Nm]",
                        "Rotational speed [rpm]",
                        "Tool wear [min]",
                        "Process temperature [K]",
                        "Air temperature [K]"
                    ],
                    index=0
                )

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown(f"##### Histogram: {selected_sensor}")
                fig_hist = px.histogram(
                    df,
                    x=selected_sensor,
                    color="Machine failure",
                    color_discrete_map={0: "#3B82F6", 1: "#FF3B5C"},
                    barmode="overlay",
                    nbins=40,
                    title=f"Distribution of {selected_sensor} (Normal vs Failure)"
                )
                fig_hist.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13, 19, 33, 0.6)",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(title="Machine Failure", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_d2:
                st.markdown(f"##### Boxplot & Outliers: {selected_sensor}")
                fig_box = px.box(
                    df,
                    x="Machine failure",
                    y=selected_sensor,
                    color="Machine failure",
                    color_discrete_map={0: "#3B82F6", 1: "#FF3B5C"},
                    points="outliers",
                    title=f"Outlier Comparison for {selected_sensor}"
                )
                fig_box.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13, 19, 33, 0.6)",
                    xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Healthy (0)", "Failed (1)"]),
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown(f"##### 📋 Statistical Summary for `{selected_sensor}`")
            stat_summary = df.groupby("Machine failure")[selected_sensor].describe().T
            stat_summary.columns = ["Healthy Machines (0)", "Failed Machines (1)"]
            st.dataframe(stat_summary.style.format("{:.2f}").highlight_max(axis=1, color="#1E3A8A"), use_container_width=True)

        # ---------------------------------------------------------
        # SUB-VIEW 3: Physical Failure Boundaries & Bivariate Scatters
        # ---------------------------------------------------------
        elif eda_view == "3. Physical Failure Boundaries & Bivariate Scatters":
            st.markdown("#### 3. Physical Failure Boundaries & Engineering Insights")
            st.markdown("CNC tool failure is caused by physical parameter boundaries. Below are the 3 major physical relationships documented in the benchmark dataset:")

            biv_choice = st.radio(
                "Select Bivariate Relationship:",
                options=[
                    "Torque vs Rotational Speed (Power Hyperbola)",
                    "Tool Wear vs Torque (Overstrain Failure Boundary)",
                    "Air Temp vs Process Temp (Thermal Dissipation Boundary)"
                ],
                horizontal=True
            )

            # Downsample for snappy browser rendering
            sample_sub = df.sample(n=3000, random_state=42)

            if biv_choice == "Torque vs Rotational Speed (Power Hyperbola)":
                st.markdown("##### ⚡ Mechanical Shaft Power: $P = \\tau \\cdot \\omega$")
                st.caption("Power Failure (PWF) occurs when power exits the safe window of **[3,500 W - 9,000 W]**.")
                fig_biv = px.scatter(
                    sample_sub,
                    x="Rotational speed [rpm]",
                    y="Torque [Nm]",
                    color="Machine failure",
                    color_discrete_map={0: "#3B82F6", 1: "#FF3B5C"},
                    opacity=0.65,
                    title="Torque vs Rotational Speed (Showing Power Failure Boundaries)"
                )
                fig_biv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13, 19, 33, 0.6)",
                    height=450
                )
                st.plotly_chart(fig_biv, use_container_width=True)

            elif biv_choice == "Tool Wear vs Torque (Overstrain Failure Boundary)":
                st.markdown("##### 🔴 Overstrain Load Boundary: $\\text{Torque} \\times \\text{Wear}$")
                st.caption("Overstrain Failure (OSF) triggers when the product of Torque and Tool Wear exceeds threshold (e.g. 11,000 min·Nm for Type L).")
                fig_biv = px.scatter(
                    sample_sub,
                    x="Tool wear [min]",
                    y="Torque [Nm]",
                    color="Machine failure",
                    color_discrete_map={0: "#3B82F6", 1: "#FF3B5C"},
                    opacity=0.65,
                    title="Tool Wear vs Torque (Overstrain Failure Clusters in Top-Right)"
                )
                fig_biv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13, 19, 33, 0.6)",
                    height=450
                )
                st.plotly_chart(fig_biv, use_container_width=True)

            else:
                st.markdown("##### 🌡️ Thermal Dissipation Boundary: $\\Delta T = T_{\\text{proc}} - T_{\\text{air}}$")
                st.caption("Heat Dissipation Failure (HDF) occurs when process temperature fails to rise at least **8.6 K** above ambient temperature at low spindle speeds.")
                sample_sub_temp = sample_sub.copy()
                sample_sub_temp["Temp_Delta"] = sample_sub_temp["Process temperature [K]"] - sample_sub_temp["Air temperature [K]"]
                fig_biv = px.scatter(
                    sample_sub_temp,
                    x="Air temperature [K]",
                    y="Process temperature [K]",
                    color="Machine failure",
                    color_discrete_map={0: "#3B82F6", 1: "#FF3B5C"},
                    hover_data=["Temp_Delta", "Rotational speed [rpm]"],
                    opacity=0.65,
                    title="Air Temperature vs Process Temperature"
                )
                fig_biv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13, 19, 33, 0.6)",
                    height=450
                )
                st.plotly_chart(fig_biv, use_container_width=True)

        # ---------------------------------------------------------
        # SUB-VIEW 4: Correlation Matrix & Heatmap
        # ---------------------------------------------------------
        elif eda_view == "4. Correlation Matrix & Heatmap":
            st.markdown("#### 4. Sensor Correlation Heatmap")
            st.markdown("Reveals linear dependencies among all numerical features in the dataset.")

            num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]
            corr_mat = df[num_cols].corr()

            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation Heatmap Between Machine Sensors"
            )
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=480,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            render_html("""
                <div class="friendly-card" style="font-size: 0.88rem; line-height: 1.6; color: #CBD5E1;">
                    <b>Key Correlation Insights:</b><br>
                    • <b>Air Temp & Process Temp:</b> Strong positive correlation (<b>r = +0.88</b>), which forms the basis of our Linear Regression model.<br>
                    • <b>Torque & Rotational Speed:</b> Strong negative correlation (<b>r = -0.88</b>), reflecting mechanical conservation of power ($P = \\tau \\cdot \\omega$).<br>
                    • <b>Torque & Machine Failure:</b> Highest linear correlation with failure (<b>r = +0.19</b>), making Torque the single most influential predictor.
                </div>
            """)

        # ---------------------------------------------------------
        # SUB-VIEW 5: Raw Data Explorer & Summary Statistics
        # ---------------------------------------------------------
        else:
            st.markdown("#### 5. Raw Dataset Explorer & Descriptive Statistics")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_fail = st.selectbox("Filter by Machine Status:", options=["All Machines", "Healthy Only (0)", "Failed Only (1)"])
            with col_f2:
                filter_type = st.multiselect("Filter by Quality Tier:", options=["L", "M", "H"], default=["L", "M", "H"])

            filtered_df = df.copy()
            if filter_fail == "Healthy Only (0)":
                filtered_df = filtered_df[filtered_df["Machine failure"] == 0]
            elif filter_fail == "Failed Only (1)":
                filtered_df = filtered_df[filtered_df["Machine failure"] == 1]
            filtered_df = filtered_df[filtered_df["Type"].isin(filter_type)]

            st.caption(f"Displaying **{len(filtered_df):,}** matching records out of 10,000:")
            st.dataframe(filtered_df.head(200), use_container_width=True, height=360)

            st.markdown("##### 📋 Complete Statistical Table (`df.describe().T`)")
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

# =========================================================
# TAB 3: BATCH FLEET DIAGNOSTICS (CSV)
# =========================================================
with tabs[2]:
    st.markdown("### 📁 Batch Fleet Diagnostics with CSV")
    st.markdown("""
        Test an entire fleet of machines in a single batch. Upload your operational CSV telemetry or use our pre-packaged **20-machine sample batch**.
    """)

    col_csv1, col_csv2 = st.columns([2, 1])
    with col_csv1:
        uploaded_file = st.file_uploader("Upload Fleet Telemetry CSV File:", type=["csv"])
    with col_csv2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_sample_btn = st.button("📥 Load Sample 20 Machines", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample_btn:
        if Path("sample_batch.csv").exists():
            batch_df = pd.read_csv("sample_batch.csv")

    if batch_df is not None:
        try:
            with st.spinner("Processing batch with Machine Learning classifier..."):
                results = utils.process_batch(batch_df, selected_model_name=selected_model_key)

            st.success(f"Successfully processed {len(results)} machines!")

            tot = len(results)
            crit = int((results["Risk_Level"] == "CRITICAL").sum())
            warn = int((results["Risk_Level"] == "WARNING").sum())
            norm = int((results["Risk_Level"] == "NORMAL").sum())

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Total Tested Units", f"{tot}")
            with b2:
                st.metric("Healthy Machines", f"{norm}", delta=f"{norm/tot*100:.0f}%")
            with b3:
                st.metric("Warning Status", f"{warn}", delta=f"{warn/tot*100:.0f}%", delta_color="inverse")
            with b4:
                st.metric("Critical Failures", f"{crit}", delta=f"{crit/tot*100:.0f}%", delta_color="inverse")

            def color_risk(val):
                if val == "CRITICAL":
                    return "background-color: rgba(255, 59, 92, 0.3); color: #FCA5A5; font-weight: bold;"
                elif val == "WARNING":
                    return "background-color: rgba(245, 158, 11, 0.3); color: #FDE68A; font-weight: bold;"
                return "background-color: rgba(16, 185, 129, 0.2); color: #A7F3D0;"

            st.dataframe(
                results.style.map(color_risk, subset=["Risk_Level"]),
                use_container_width=True,
                height=380
            )

            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Enriched Diagnostic Report as CSV",
                data=csv_data,
                file_name="machine_health_prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error testing batch: {str(e)}")
    else:
        st.info("💡 Tip: Click **'Load Sample 20 Machines'** above to test fleet diagnostics instantly!")

# =========================================================
# TAB 4: ML MODELS LEADERBOARD & BENCHMARKS
# =========================================================
with tabs[3]:
    st.markdown("### 🤖 Machine Learning Model Benchmarks & Leaderboard")
    st.markdown("""
        Comprehensive performance evaluation across all 7 evaluated classification algorithms, cross-validation metrics, and feature importance rankings.
    """)

    comp_data = summary.get("classification_comparison", [])
    if comp_data:
        m_df = pd.DataFrame(comp_data)
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=m_df["Model"],
            y=m_df["Accuracy"] * 100,
            name="Accuracy (%)",
            marker_color="#3B82F6"
        ))
        fig_m.add_trace(go.Bar(
            x=m_df["Model"],
            y=m_df["F1 Score"] * 100,
            name="F1 Score (%)",
            marker_color="#10B981"
        ))
        fig_m.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13, 19, 33, 0.6)",
            title="Algorithm Comparison: Accuracy vs F1 Score",
            yaxis=dict(title="Score (%)", range=[0, 105]),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("#### 🏆 Performance Leaderboard")
        display_m = m_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].copy()
        st.dataframe(
            display_m.style.format({
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1 Score": "{:.3f}"
            }).highlight_max(subset=["F1 Score", "Recall", "Precision"], color="#065F46"),
            use_container_width=True
        )

    st.markdown("---")
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("#### 🔲 Confusion Matrix Inspector")
        cm_model_choice = st.selectbox(
            "Select Model to Inspect Confusion Matrix:",
            options=[m["Model"] for m in comp_data],
            index=0
        )
        selected_m = next((m for m in comp_data if m["Model"] == cm_model_choice), None)
        if selected_m and "Confusion Matrix" in selected_m:
            cm = np.array(selected_m["Confusion Matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal (0)", "Failure (1)"],
                y=["Normal (0)", "Failure (1)"],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.markdown("#### 🎯 Feature Importance Rankings (Tuned Random Forest)")
        f_imp = summary.get("feature_importances", [])
        if f_imp:
            imp_df = pd.DataFrame(f_imp).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Predictive Impact of Sensors on Machine Failure"
            )
            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13, 19, 33, 0.6)",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# =========================================================
# TAB 5: MACHINE OPERATING REGIMES (CLUSTERING)
# =========================================================
with tabs[4]:
    st.markdown("### 🌐 Machine Operating Regimes & Clustering")
    st.markdown("""
        Unsupervised **K-Means Clustering** & **Gaussian Mixture Models (GMM)** categorize machinery into 3 distinct operational modes without manual labels.
    """)

    if df is not None:
        cluster_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        scaled_data = models["cluster_scaler"].transform(df[cluster_features])
        pca_coords = models["pca_model"].transform(scaled_data)
        km_labels = models["kmeans_model"].predict(scaled_data)

        sample_idx = np.random.RandomState(42).choice(len(df), size=2500, replace=False)
        pca_df = pd.DataFrame({
            "PC1": pca_coords[sample_idx, 0],
            "PC2": pca_coords[sample_idx, 1],
            "Cluster": [f"Regime {l}" for l in km_labels[sample_idx]],
            "Speed": df.loc[sample_idx, "Rotational speed [rpm]"].values,
            "Torque": df.loc[sample_idx, "Torque [Nm]"].values,
            "Wear": df.loc[sample_idx, "Tool wear [min]"].values
        })

        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            color_discrete_map={"Regime 0": "#00F0FF", "Regime 1": "#3B82F6", "Regime 2": "#F59E0B"},
            hover_data=["Speed", "Torque", "Wear"],
            opacity=0.65,
            title="PCA 2D Mapping of Fleet Operating Regimes"
        )
        fig_pca.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13, 19, 33, 0.6)",
            height=450
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        col_reg0, col_reg1, col_reg2 = st.columns(3)
        with col_reg0:
            render_html("""
                <div class="friendly-card" style="border-top: 3px solid #00F0FF;">
                    <h4 style="color: #00F0FF; margin: 0 0 6px 0;">Regime 0: Heavy Thermal Load</h4>
                    <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.5;">
                        Elevated temperatures (~311 K) and moderate torque (~43 Nm). Represents warm factory environments or heavy summer cycle shifts.
                    </p>
                </div>
            """)
        with col_reg1:
            render_html("""
                <div class="friendly-card" style="border-top: 3px solid #3B82F6;">
                    <h4 style="color: #3B82F6; margin: 0 0 6px 0;">Regime 1: Standard Cool Steady-State</h4>
                    <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.5;">
                        Nominal baseline state (~298 K air, 1474 rpm). Exhibits the lowest baseline failure probabilities.
                    </p>
                </div>
            """)
        with col_reg2:
            render_html("""
                <div class="friendly-card" style="border-top: 3px solid #F59E0B;">
                    <h4 style="color: #F59E0B; margin: 0 0 6px 0;">Regime 2: High-Speed Light-Torque</h4>
                    <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.5;">
                        High rotational velocity (~1797 rpm) with low torque (~26.5 Nm). Typical of light finishing cuts.
                    </p>
                </div>
            """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
render_html("""
    <div style="display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 0.8rem; padding: 12px 0;">
        <div>Maintena Pro • Industrial Predictive Maintenance Intelligence System</div>
        <div>Built with Streamlit, Scikit-Learn & Plotly • 10,000 CNC Telemetry Records</div>
    </div>
""")
