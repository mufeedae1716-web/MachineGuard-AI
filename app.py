import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import timedelta
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE ----------------
st.set_page_config(page_title="MachineGuard AI", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [BASE_DIR, BASE_DIR / "archive (5)", BASE_DIR / "dataset", BASE_DIR / "data"]

DATA_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
    "maintenance": "PdM_maint.csv",
    "errors": "PdM_errors.csv",
}
MODEL_PATH = BASE_DIR / "machine_failure_model_final.keras"
SCALER_PATH = BASE_DIR / "machine_failure_scaler_final.pkl"
FEATURES_PATH = BASE_DIR / "machine_failure_features_final.pkl"

# ---------------- STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.stApp {background: radial-gradient(circle at 70% 0%, #101d3c 0%, #050b19 42%, #020611 100%); color:#eef5ff;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#031329 0%,#06152d 65%,#031021 100%); border-right:1px solid #18304d;}
[data-testid="stSidebar"] * {color:#eaf4ff;}
.block-container {padding-top:1.2rem; padding-bottom:2rem; max-width:1600px;}
h1,h2,h3 {letter-spacing:-.5px;}
h1 {color:#38e5ed;}
h2,h3 {color:#45e8ee;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#0b1b32,#071326); border:1px solid #1b3553; padding:17px 18px; border-radius:15px;}
[data-testid="stMetricLabel"] {color:#c8d7ef !important;}
[data-testid="stMetricValue"] {color:#36dce9 !important;}
.panel {background:linear-gradient(145deg,rgba(10,27,49,.97),rgba(5,16,33,.97));border:1px solid #1a3554;border-radius:16px;padding:20px 22px;margin-bottom:14px;}
.small-muted {color:#9eb5d4;font-size:13px;}
.kicker {color:#45e8ee;font-weight:700;font-size:13px;letter-spacing:1.3px;text-transform:uppercase;}
.hero {background:linear-gradient(100deg,rgba(3,14,31,.98) 0%,rgba(6,28,54,.93) 50%,rgba(7,27,48,.65) 100%);border:1px solid #1b4264;border-radius:18px;padding:28px;min-height:225px;}
.hero-title {font-size:27px;font-weight:800;line-height:1.3;color:#f4f8ff;}
.hero-accent {color:#40e4ec;}
div.stButton > button {background:linear-gradient(90deg,#16c9d6,#1689ef);color:#02111f;border:0;border-radius:9px;font-weight:700;min-height:42px;}
div.stButton > button:hover {background:linear-gradient(90deg,#45edf0,#50a7ff);color:#03111e;}
[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label {color:#dbeaff;}
hr {border-color:#1b3552;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=False)
def load_csv(filename):
    for folder in DATA_DIR_CANDIDATES:
        path = folder / filename
        if path.exists():
            df = pd.read_csv(path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            return df
    return pd.DataFrame()

@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH) if MODEL_PATH.exists() else None
    scaler = None
    features = None
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)
    if features is not None:
        features = list(features)
    return model, scaler, features

telemetry = load_csv(DATA_FILES["telemetry"])
failures = load_csv(DATA_FILES["failures"])
machines = load_csv(DATA_FILES["machines"])
maintenance = load_csv(DATA_FILES["maintenance"])
errors = load_csv(DATA_FILES["errors"])

# Normalize common column names
for df in [telemetry, failures, machines, maintenance, errors]:
    if not df.empty:
        df.columns = [str(c).strip() for c in df.columns]

model = scaler = feature_names = None
artifact_error = None
try:
    model, scaler, feature_names = load_artifacts()
except Exception as exc:
    artifact_error = str(exc)

# ---------------- HELPERS ----------------
def col(df, choices):
    for name in choices:
        if name in df.columns:
            return name
    return None

def pretty_machine(mid):
    return f"MACHINE_{int(mid):03d}"

def prepare_machine_frame(mid):
    if telemetry.empty:
        return pd.DataFrame()
    idcol = col(telemetry, ["machineID", "machineId", "MachineID"])
    timecol = col(telemetry, ["datetime", "date", "timestamp"])
    if idcol is None:
        return pd.DataFrame()
    d = telemetry[telemetry[idcol] == mid].copy()
    if timecol:
        d = d.sort_values(timecol)
    return d

def make_feature_row(history, feature_names):
    """Build a feature row in the saved training-feature order."""
    if history.empty:
        raise ValueError("No telemetry history exists for this machine.")
    sensor_aliases = {
        "volt": ["volt", "voltage", "Voltage (V)"],
        "rotate": ["rotate", "rotation", "Rotational speed [rpm]"],
        "pressure": ["pressure", "Pressure"],
        "vibration": ["vibration", "Vibration"],
    }
    history = history.copy()
    history.columns = [str(c).strip() for c in history.columns]
    latest = history.iloc[-1]
    # Base values plus common engineered features
    vals = {}
    for canonical, aliases in sensor_aliases.items():
        actual = next((a for a in aliases if a in history.columns), None)
        if actual:
            series = pd.to_numeric(history[actual], errors="coerce")
            vals[canonical] = float(series.iloc[-1])
            vals[f"{canonical}_mean_6h"] = float(series.tail(6).mean())
            vals[f"{canonical}_change_1h"] = float(series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else 0.0
            vals[f"{canonical}_mean_24h"] = float(series.tail(24).mean())
    # Provide alternate names sometimes used by training notebooks
    alias_map = {
        "voltage": "volt", "rotation": "rotate",
        "Air temperature [K]": "volt", "Process temperature [K]": "rotate",
    }
    for target, source in alias_map.items():
        if source in vals:
            vals[target] = vals[source]
    # Preserve any directly present numeric columns
    for name in history.columns:
        if name in latest.index:
            try:
                value = pd.to_numeric(pd.Series([latest[name]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    vals[name] = float(value)
            except Exception:
                pass
    # Fill only expected feature names, preserving exact order
    row = {}
    missing = []
    for feat in feature_names:
        if feat in vals:
            row[feat] = vals[feat]
        elif feat.endswith("_mean_6h") and feat[:-8] in vals:
            base = feat[:-8]
            row[feat] = vals[base]
        elif feat in history.columns:
            row[feat] = float(pd.to_numeric(pd.Series([latest[feat]]), errors="coerce").fillna(0).iloc[0])
        else:
            # Common one-hour change aliases
            base = feat.replace("_change", "_change_1h")
            if base in vals:
                row[feat] = vals[base]
            else:
                missing.append(feat)
                row[feat] = 0.0
    return pd.DataFrame([row], columns=feature_names), missing

def get_machine_list():
    idcol = col(telemetry, ["machineID", "machineId", "MachineID"])
    if idcol:
        return sorted(pd.to_numeric(telemetry[idcol], errors="coerce").dropna().astype(int).unique().tolist())
    return []

def latest_sensor_values(history):
    if history.empty:
        return {}
    last = history.iloc[-1]
    names = {"volt": "Voltage", "rotate": "Rotation", "pressure": "Pressure", "vibration": "Vibration"}
    out = {}
    for c, label in names.items():
        if c in history.columns:
            out[label] = float(last[c])
    return out

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 2px 18px">
      <div style="font-size:25px;font-weight:800;line-height:1.15">⚙️ MACHINE<span style="color:#39e5ed">GUARD AI</span></div>
      <div style="font-size:12px;color:#a9c0df;margin-top:7px">Predictive Maintenance Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="border:1px solid #173d61;border-radius:14px;padding:14px;background:linear-gradient(145deg,#071a32,#041023);text-align:center;margin-bottom:18px">
    """, unsafe_allow_html=True)
    machine_image = BASE_DIR / "machine_banner.png"
    if machine_image.exists():
        st.image(str(machine_image), use_container_width=True)
    else:
        st.markdown("<div style='font-size:76px;text-align:center'>🏭</div>", unsafe_allow_html=True)
    st.markdown("""
      <div style="color:#42e4ec;font-weight:700;text-align:center">SMART FACTORY</div>
      <div style="color:#a4bad7;font-size:12px;text-align:center">Sensor-based early warning</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("NAVIGATION", [
        "Overview", "Machine Identification", "Failure Prediction",
        "Sensor Analytics", "Maintenance History", "Alerts & Notifications",
        "Reports", "Project Details", "About System"
    ], label_visibility="visible")
    st.markdown("---")
    st.caption("AI • SENSOR ANALYTICS • MAINTENANCE")

# ---------------- HEADER / DATA STATUS ----------------
st.markdown('<div class="kicker">PREDICTIVE MAINTENANCE SYSTEM</div>', unsafe_allow_html=True)
st.title("MachineGuard AI")
st.markdown('<div class="small-muted">Machine health monitoring and early failure warning</div>', unsafe_allow_html=True)

missing_data = [name for name, fn in DATA_FILES.items() if load_csv(fn).empty]
if missing_data:
    st.warning("Some CSV files were not found or are empty: " + ", ".join(missing_data) + ". Keep the PdM CSV files in the same folder as app.py or inside archive (5).")
if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURES_PATH.exists():
    st.warning("Model files are missing. Keep machine_failure_model_final.keras, machine_failure_scaler_final.pkl, and machine_failure_features_final.pkl beside app.py.")
if artifact_error:
    st.error("Could not load model artifacts: " + artifact_error)

machine_ids = get_machine_list()
if not machine_ids:
    st.error("Telemetry data or machineID column was not found. Check the PdM_telemetry.csv file location and column names.")
    st.stop()

# shared machine selector
selected_mid = st.selectbox("Select Machine ID", machine_ids, format_func=pretty_machine, key="global_machine")
history = prepare_machine_frame(selected_mid)
timecol = col(history, ["datetime", "date", "timestamp"]) if not history.empty else None
fail_idcol = col(failures, ["machineID", "machineId", "MachineID"]) if not failures.empty else None
maint_idcol = col(maintenance, ["machineID", "machineId", "MachineID"]) if not maintenance.empty else None

# latest sensor cards
sensor_values = latest_sensor_values(history)
risk_score = None
prediction_error = None
if model is not None and scaler is not None and feature_names and not history.empty:
    try:
        X, missing_features = make_feature_row(history, feature_names)
        Xs = scaler.transform(X)
        pred = np.asarray(model.predict(Xs, verbose=0)).reshape(-1)
        risk_score = float(pred[0])
        if risk_score > 1:
            risk_score = risk_score / 100.0
    except Exception as exc:
        prediction_error = str(exc)

# ---------------- PAGES ----------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machine Status", "MONITORING", "Telemetry available")
    if risk_score is not None:
        c2.metric("Failure Risk Score", f"{risk_score*100:.2f}%", "Model estimate")
    else:
        c2.metric("Failure Risk Score", "N/A", "Check model/features")
    warning_count = 0
    if model is not None and scaler is not None and feature_names:
        for mid in machine_ids:
            try:
                h = prepare_machine_frame(mid)
                x, _ = make_feature_row(h, feature_names)
                z = scaler.transform(x)
                score = float(np.asarray(model.predict(z, verbose=0)).reshape(-1)[0])
                if score > 1: score /= 100
                warning_count += score >= 0.90
            except Exception:
                pass
    c3.metric("High-Warning Machines", f"{warning_count:02d}", "Score ≥ 90%")
    maint_count = len(maintenance[maint_idcol].dropna().unique()) if maint_idcol else 0
    c4.metric("Machines with Maintenance Records", str(maint_count), "From maintenance CSV")
    st.markdown("""
    <div class="hero">
      <div class="kicker">SMART MONITORING</div>
      <div class="hero-title">Predict Failures.<br><span class="hero-accent">Support Better Maintenance Decisions.</span></div>
      <p style="color:#c5d8ef;max-width:600px;line-height:1.7">Use telemetry readings and a trained deep-learning model to estimate early-warning scores for individual machines.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns([0.95, 1.05, 1.5])
    with a:
        st.markdown('<div class="panel"><h3>Latest Sensor Readings</h3>', unsafe_allow_html=True)
        units = {"Voltage":"", "Rotation":" rpm", "Pressure":"", "Vibration":""}
        if sensor_values:
            for k, v in sensor_values.items():
                st.markdown(f"**{k}** <span style='float:right;color:#43e5ed;font-weight:700'>{v:.3f}{units.get(k,'')}</span>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)
        else:
            st.info("Sensor columns were not recognized in telemetry.")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><h3>Failure Risk Prediction</h3>', unsafe_allow_html=True)
        if risk_score is not None:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_score*100,
                number={"suffix":"%","font":{"color":"#3fe3ed","size":32}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#b8c9e4"},
                       "bar":{"color":"#28d6df"},
                       "bgcolor":"#07172b","bordercolor":"#234361",
                       "steps":[{"range":[0,30],"color":"#0b6c65"},{"range":[30,70],"color":"#9a6b12"},{"range":[70,100],"color":"#8d2739"}],
                       "threshold":{"line":{"color":"#ffb347","width":4},"thickness":0.8,"value":90}}))
            fig.update_layout(height=220, margin=dict(l=8,r=8,t=8,b=5), paper_bgcolor="rgba(0,0,0,0)", font_color="#eaf4ff")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.markdown("**" + ("HIGH WARNING" if risk_score >= .90 else "BELOW WARNING THRESHOLD") + "**")
            st.caption("Model score is an estimate, not a guarantee of failure or safety.")
        else:
            st.warning("Prediction unavailable. Check the saved model and training features.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><h3>Sensor Trends</h3>', unsafe_allow_html=True)
        if not history.empty and timecol:
            sensors = [s for s in ["volt","rotate","pressure","vibration"] if s in history.columns]
            if sensors:
                trend = history.tail(48)[[timecol] + sensors].melt(id_vars=timecol, var_name="Sensor", value_name="Reading")
                fig = px.line(trend, x=timecol, y="Reading", color="Sensor", template="plotly_dark")
                fig.update_layout(height=290, margin=dict(l=5,r=5,t=5,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            else: st.info("No standard sensor columns found.")
        else: st.info("Timestamp data is not available for trend chart.")
        st.markdown("</div>", unsafe_allow_html=True)
    d, e = st.columns([1.2, 1])
    with d:
        st.markdown('<div class="panel"><h3>Recent Failure Records</h3>', unsafe_allow_html=True)
        if not failures.empty:
            f = failures.copy()
            if fail_idcol: f = f[f[fail_idcol] == selected_mid]
            st.dataframe(f.tail(8), use_container_width=True, hide_index=True)
        else: st.info("Failure records not available.")
        st.markdown("</div>", unsafe_allow_html=True)
    with e:
        st.markdown('<div class="panel"><h3>Maintenance Records</h3>', unsafe_allow_html=True)
        if not maintenance.empty:
            m = maintenance.copy()
            if maint_idcol: m = m[m[maint_idcol] == selected_mid]
            st.dataframe(m.tail(8), use_container_width=True, hide_index=True)
        else: st.info("Maintenance records not available.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Machine Identification":
    st.header("Machine Identification")
    st.markdown("View telemetry and maintenance records for the selected machine.")
    st.subheader(pretty_machine(selected_mid))
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel"><h3>Machine information</h3>', unsafe_allow_html=True)
        if not machines.empty and "machineID" in machines.columns:
            info = machines[machines["machineID"] == selected_mid]
            st.dataframe(info, use_container_width=True, hide_index=True)
        else: st.info("Machine metadata not available.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><h3>Latest telemetry</h3>', unsafe_allow_html=True)
        st.dataframe(history.tail(10), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Failure Prediction":
    st.header("Interactive Machine Risk Assessment")
    st.caption("Choose a machine and enter the latest sensor readings.")
    defaults = latest_sensor_values(history)
    with st.form("risk_form"):
        left, right = st.columns(2)
        with left:
            volt = st.number_input("Voltage (V)", value=float(defaults.get("Voltage", 170.0)), format="%.3f")
            rotate = st.number_input("Rotation (rpm)", value=float(defaults.get("Rotation", 1500.0)), format="%.3f")
        with right:
            pressure = st.number_input("Pressure", value=float(defaults.get("Pressure", 100.0)), format="%.3f")
            vibration = st.number_input("Vibration", value=float(defaults.get("Vibration", 40.0)), format="%.3f")
        submitted = st.form_submit_button("⚡ Predict Machine Failure", use_container_width=True)
    if submitted:
        if model is None or scaler is None or not feature_names:
            st.error("Model, scaler, or feature list could not be loaded.")
        elif history.empty:
            st.error("No history is available for this machine.")
        else:
            try:
                # Update latest row with user-entered values, then regenerate expected features.
                h = history.copy()
                mapping = {"volt": volt, "rotate": rotate, "pressure": pressure, "vibration": vibration}
                for sensor, value in mapping.items():
                    if sensor in h.columns:
                        h.loc[h.index[-1], sensor] = value
                X, missing = make_feature_row(h, feature_names)
                Xs = scaler.transform(X)
                score = float(np.asarray(model.predict(Xs, verbose=0)).reshape(-1)[0])
                if score > 1: score /= 100
                st.markdown('<div class="panel"><h2> Prediction Result</h2>', unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                r1.metric("Machine ID", pretty_machine(selected_mid))
                r2.metric("Early-Warning Score", f"{score*100:.2f}%")
                if score >= .90:
                    st.error(f"{pretty_machine(selected_mid)} has crossed the warning threshold.")
                    st.markdown("**Suggested next step:** inspect the machine and verify its sensor readings.")
                else:
                    st.success(f"{pretty_machine(selected_mid)} is below the warning threshold for these inputs.")
                    st.markdown("**Suggested next step:** continue routine monitoring.")
                st.caption("This is a model estimate, not a guarantee of failure or safety. Input feature calculations must match training.")
                if missing:
                    st.warning("Some expected features were not recognized and were set to 0: " + ", ".join(missing))
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as exc:
                st.error("Prediction failed: " + str(exc))
                st.info("Send this error and the training notebook's feature-engineering cell so the prediction calculations can be aligned.")
    st.markdown('<div class="panel"><h3>How to interpret the score</h3><p>Scores at or above 90% trigger the current warning threshold. This threshold is a project setting and should be validated against test data before operational use.</p></div>', unsafe_allow_html=True)

elif page == "Sensor Analytics":
    st.header("Sensor Analytics")
    if not history.empty:
        st.dataframe(history.tail(100), use_container_width=True, hide_index=True)
        if timecol:
            sensor_cols = [c for c in ["volt","rotate","pressure","vibration"] if c in history.columns]
            for s in sensor_cols:
                fig = px.line(history.tail(200), x=timecol, y=s, title=f"{s.title()} trend", template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
    else: st.info("No telemetry found for selected machine.")

elif page == "Maintenance History":
    st.header("Maintenance History")
    if not maintenance.empty:
        m = maintenance.copy()
        if maint_idcol: m = m[m[maint_idcol] == selected_mid]
        st.dataframe(m, use_container_width=True, hide_index=True)
    else: st.info("PdM_maint.csv was not found.")

elif page == "Alerts & Notifications":
    st.header("Alerts & Notifications")
    if risk_score is not None and risk_score >= .90:
        st.error(f"⚠️ {pretty_machine(selected_mid)} has a high model warning score ({risk_score*100:.2f}%). Verify readings and arrange inspection.")
    else:
        st.success("No selected-machine score currently exceeds the 90% warning threshold.")
    if not failures.empty:
        st.subheader("Recorded failures")
        st.dataframe(failures.tail(50), use_container_width=True, hide_index=True)
    if not errors.empty:
        st.subheader("Recorded error events")
        st.dataframe(errors.tail(50), use_container_width=True, hide_index=True)

elif page == "Reports":
    st.header("Reports")
    report = history.copy()
    st.subheader(f"Telemetry report — {pretty_machine(selected_mid)}")
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.download_button("Download selected machine telemetry CSV", report.to_csv(index=False).encode("utf-8"), file_name=f"{pretty_machine(selected_mid)}_telemetry.csv", mime="text/csv")

elif page == "Project Details":
    st.header("Project Details")
    st.markdown("""
    <div class="panel">
      <h2>AI-Based Predictive Maintenance and Early Failure Warning System</h2>
      <p><b>Project name:</b> MachineGuard AI</p>
      <p><b>Objective:</b> Estimate early-warning scores from machine telemetry and support machine-wise maintenance decisions.</p>
      <p><b>Dataset:</b> Predictive Maintenance CSV files (telemetry, failures, machines, errors, maintenance).</p>
      <p><b>Model:</b> Saved deep-learning model loaded from the project folder.</p>
      <p><b>Important limitation:</b> The score is a model estimate. Validate the model on held-out data and align all feature engineering with training before making operational claims.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.header("About MachineGuard AI")
    st.markdown("""
    <div class="panel">
      <h3>Predictive Maintenance Intelligence</h3>
      <p>This prototype demonstrates telemetry exploration, machine-level risk scoring, sensor trend visualization, and maintenance-history review.</p>
      <p>It is not connected to live factory equipment. CSV data is historical, and predictions depend on the saved model and the feature calculations used during training.</p>
    </div>
    """, unsafe_allow_html=True)

if prediction_error:
    with st.expander("Prediction diagnostic"):
        st.error(prediction_error)

st.markdown("<div style='text-align:center;color:#7189a9;padding:20px 0 5px;font-size:12px'>MACHINEGUARD AI • PREDICTIVE MAINTENANCE INTELLIGENCE</div>", unsafe_allow_html=True)