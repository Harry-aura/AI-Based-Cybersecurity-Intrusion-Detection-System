import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CyberGuard-IDS Console", layout="wide")
st.title("🛡️ CyberGuard AI - Intrusion Detection Dashboard")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"Backend Status: {health.get('status', 'Online')}")
except Exception:
    st.sidebar.error("Backend Offline - Ensure FastAPI server is running on :8000")

st.subheader("Network Packet Simulation")
col1, col2, col3 = st.columns(3)

with col1:
    duration = st.number_input("Duration (seconds)", min_value=0, value=0)
    protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
    service = st.selectbox("Service", ["http", "smtp", "ftp", "private", "other"])

with col2:
    flag = st.selectbox("Connection Flag", ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"])
    src_bytes = st.number_input("Source Bytes", min_value=0, value=215)
    dst_bytes = st.number_input("Destination Bytes", min_value=0, value=45000)

with col3:
    count = st.number_input("Connection Count", min_value=0, value=1)
    srv_count = st.number_input("Service Count", min_value=0, value=1)
    same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)

if st.button("Analyze Traffic", type="primary"):
    payload = {
        "duration": int(duration),
        "protocol_type": str(protocol_type).lower(),
        "service": str(service).lower(),
        "flag": str(flag).strip().upper(),
        "src_bytes": int(src_bytes),
        "dst_bytes": int(dst_bytes),
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1 if service == "http" else 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": int(count),
        "srv_count": int(srv_count),
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": float(same_srv_rate),
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            is_attack = data.get("is_attack", data.get("binary_prediction", 0) == 1)
            prob = data.get("attack_probability", data.get("probability", 0.0))
            category = data.get("multiclass_prediction", data.get("category", "Normal"))

            if is_attack:
                st.error(f"⚠️ THREAT DETECTED: {str(category).upper()} (Confidence: {float(prob)*100:.2f}%)")
            else:
                st.success(f"✅ NORMAL TRAFFIC (Benign Confidence: {(1 - float(prob))*100:.2f}%)")
            st.json(data)
        else:
            st.error(f"Server Error {res.status_code}: {res.text}")
    except Exception as exc:
        st.error(f"Failed to reach API: {exc}")