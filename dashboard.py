import streamlit as st
import pickle
import pandas as pd
import time
import random
import re
import pydeck as pdk

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="SMS Spam Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS STYLING
st.markdown("""
<style>
    /* GLOBAL HIDES */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* BACKGROUND */
    .stApp {
        background-color: #0E1117;
        background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* LOGIN BOX */
    .login-box {
        background-color: #0E1117;
        border: 1px solid rgba(37, 99, 235, 0.5);
        border-radius: 12px;
        padding: 50px;
        text-align: center;
        margin-top: 80px;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.1);
    }
    
    /* INPUT FIELDS */
    .stTextArea textarea {
        background-color: #16181F; 
        color: #fff;
        border: 1px solid #444;
        font-family: 'Consolas', monospace;
        font-size: 16px; 
    }
    .stTextArea textarea:focus {
        border-color: #2563EB;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.3);
    }
    
    /* BUTTONS */
    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 12px 20px;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 1px;
        font-weight: 600;
        width: 100%;
        transition: 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        border-color: rgba(255, 255, 255, 0.3);
        transform: translateY(-1px);
    }
    
    /* RESULT CARDS */
    .safe-card {
        background-color: rgba(6, 78, 59, 0.2);
        border: 1px solid #059669;
        border-left: 6px solid #059669;
        padding: 30px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    .threat-card {
        background-color: rgba(127, 29, 29, 0.2);
        border: 1px solid #DC2626;
        border-left: 6px solid #DC2626;
        padding: 30px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    
    /* DATAFRAME FIX */
    div[data-testid="stDataFrame"] {
        background-color: #15171e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* METRIC LABELS */
    div[data-testid="stMetricLabel"] {color: #888; font-size: 14px;}
    div[data-testid="stMetricValue"] {color: #fff; font-size: 28px; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# 3. STATE MANAGEMENT
if 'page' not in st.session_state:
    st.session_state['page'] = 'landing'
if 'history' not in st.session_state:
    st.session_state['history'] = [] 
if 'threat_count' not in st.session_state:
    st.session_state['threat_count'] = 0
if 'safe_count' not in st.session_state:
    st.session_state['safe_count'] = 0
if 'input_buffer' not in st.session_state:
    st.session_state['input_buffer'] = ""
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None 

# 4. LOGIC
@st.cache_resource
def load_brain():
    try:
        with open('spam_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("Error: Model file missing. Please run 'train_model.py'.")
        st.stop()

def get_location_data(text):
    # RiskLevel: 1 = Normal, 2 = High Risk (Common Spoofing Hubs)
    country_db = {
        r'\+1':   (37.09, -95.71, 'USA/Canada (+1)', 2), 
        r'\+91':  (20.59, 78.96, 'India (+91)', 1),
        r'\+44':  (55.37, -3.43, 'UK (+44)', 2), 
        r'\+971': (25.20, 55.27, 'UAE/Dubai (+971)', 2),
        r'\+86':  (35.86, 104.19, 'China (+86)', 2),
        r'\+62':  (-0.78, 113.92, 'Indonesia (+62)', 2),
        r'\+92':  (30.37, 69.34, 'Pakistan (+92)', 2),
        r'\+234': (9.08, 8.67, 'Nigeria (+234)', 2),
        r'\+84':  (14.05, 108.27, 'Vietnam (+84)', 2),
        r'\+251': (9.14, 40.48, 'Ethiopia (+251)', 2),
        r'\+254': (-1.29, 36.82, 'Kenya (+254)', 2),
    }
    
    sorted_codes = sorted(country_db.keys(), key=len, reverse=True)
    for code in sorted_codes:
        if re.search(code, text):
            data = country_db[code]
            return {'lat': data[0], 'lon': data[1], 'loc': data[2], 'risk': data[3]}

    return {'lat': random.uniform(10, 60), 'lon': random.uniform(-100, 100), 'loc': 'Unknown Region', 'risk': 0}

def analyze_risk_hybrid(text, model):
    # 1. Standard Model Prediction
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    confidence = max(prob) * 100
    
    # 2. HEURISTIC ENGINE (The Safety Net)
    text_lower = text.lower()
    geo_data = get_location_data(text)
    
    # THREAT INTELLIGENCE DICTIONARIES 
    greed_triggers = ['youtube', 'hiring', 'part-time', 'earn', 'salary', 'daily', 'profit', 'investment', 'lottery', 'winner', 'prize', 'bank', 'blocked', 'kyc', 'pan card', 'update']
    arrest_triggers = ['police', 'arrest', 'warrant', 'cbi', 'narcotics', 'ncb', 'crime', 'branch', 'court', 'jail', 'fedex', 'customs', 'seized', 'drug', 'illegal', 'money laundering', 'case file', 'fir']
    harass_triggers = ['video', 'footage', 'leak', 'viral', 'expose', 'face', 'recording', 'call', 'upload', 'social media', 'shame', 'friends', 'family']
    cyber_triggers = ['encrypt', 'decrypt', 'files', 'locked', 'key', 'bitcoin', 'btc', 'wallet', 'recover', 'access', 'hacked', 'trojan', 'malware']
    indian_context = ['mumbai', 'delhi', 'bangalore', 'india', 'rbi', 'cbi', 'sbi', 'tax', 'income', 'aadhaar']
    
    # LOGIC CHECKS
    all_threat_words = greed_triggers + arrest_triggers + harass_triggers + cyber_triggers
    has_threat_word = any(word in text_lower for word in all_threat_words)
    is_international = "India" not in geo_data['loc']
    
    # RULE 1: High Risk Geo + Any Threat Word = AUTOMATIC SPAM
    if geo_data['risk'] == 2 and has_threat_word:
        return "spam", 99.9, geo_data
        
    # RULE 2: Digital Arrest Mismatch
    has_arrest_context = any(word in text_lower for word in arrest_triggers + indian_context)
    if is_international and has_arrest_context:
        return "spam", 99.9, geo_data
        
    # RULE 3: Ransomware/Harassment is ALWAYS High Risk
    if any(word in text_lower for word in harass_triggers + cyber_triggers):
        if confidence < 80: 
            return "spam", 95.0, geo_data

    return pred, confidence, geo_data

def highlight_threats(text):
    triggers = ['urgent', 'free', 'winner', 'cash', 'prize', 'code', 'loan', 'claim', 'call', 'txt', 'win', 'youtube', 'hiring', 'earn', 'drug', 'seized', 'police', 'arrest', 'customs']
    for word in triggers:
        text = re.sub(f"({word})", r'<span style="background-color: #450a0a; color: #ff6b6b; padding: 0 4px; border-bottom: 1px solid red;">\1</span>', text, flags=re.IGNORECASE)
    return text

# 5. LANDING PAGE
if st.session_state['page'] == 'landing':
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div class="login-box">
            <h2 style="color:white; margin-bottom: 15px; letter-spacing: 1px;">SMS SPAM DETECTION SYSTEM</h2>
            <p style="color:#888; font-size: 14px; margin-bottom: 30px;">Machine Learning Final Project</p>
            <div style="text-align: left; color: #BBB; font-size: 13px; line-height: 2.2; border-top: 1px solid #333; padding-top: 20px;">
                <div>[+] Algorithm: Multinomial Naive Bayes</div>
                <div>[+] Dataset: UCI SMS Spam Collection</div>
                <div>[+] Features: Real-time Analysis & Geo-Tracing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("LAUNCH DASHBOARD"):
            with st.spinner("Initializing Secure Environment..."):
                time.sleep(1.5)
                st.session_state['page'] = 'app'
                st.rerun()

# 6. DASHBOARD
elif st.session_state['page'] == 'app':
    
    model = load_brain()
    
    # Header
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown("### SPAMSHIELD")
        st.caption("System Status: Active | Model: Naive Bayes + Heuristic Engine v3.0")
    with c2:
        if st.button("LOGOUT"):
            st.session_state['page'] = 'landing'
            st.rerun()
            
    st.divider()

    # ROW 1: CENTERED INPUT 
    c_space_l, c_input, c_space_r = st.columns([1, 4, 1])
    
    with c_input:
        st.markdown("#### MESSAGE ANALYSIS CONSOLE")
        input_text = st.text_area("Input Stream", value=st.session_state.get('input_buffer', ''), height=100, label_visibility="collapsed", placeholder="Enter SMS text to scan for threats...")
        
        if st.button("RUN SECURITY SCAN"):
            if input_text:
                st.session_state['input_buffer'] = input_text
                
                with st.spinner("Decrypting & Analyzing Payload..."):
                    time.sleep(1.0)
                    
                    # RUN HYBRID ANALYSIS
                    pred, confidence, geo = analyze_risk_hybrid(input_text, model)
                    timestamp = time.strftime("%H:%M:%S")
                    
                    if pred == 'ham':
                        st.session_state['safe_count'] += 1
                        result_data = {"type": "safe", "conf": confidence, "msg": input_text, "geo": geo}
                    else:
                        st.session_state['threat_count'] += 1
                        result_data = {"type": "threat", "conf": confidence, "msg": input_text, "geo": geo}
                    
                    st.session_state['last_result'] = result_data
                    
                    # Log
                    new_log = {"Time": timestamp, "Result": "SAFE" if pred == 'ham' else "SPAM", "Message": input_text, "Confidence": f"{confidence:.1f}%"}
                    st.session_state['history'].insert(0, new_log)
                    st.rerun() 

    st.write("") 

    # ROW 2: INTEL LAYER 
    col_intel_left, col_intel_right = st.columns([1, 1])

    with col_intel_left:
        # Display Result
        if st.session_state.get('last_result'):
            res = st.session_state['last_result']
            if res['type'] == 'safe':
                st.markdown(f"""
                <div class="safe-card">
                    <h2 style="margin:0; color:#34D399; font-size: 28px;">[ VERIFIED SAFE ]</h2>
                    <p style="margin:10px 0 0 0; font-size:16px; color:#fff;">Confidence Score: <b>{res['conf']:.1f}%</b></p>
                    <p style="margin:5px 0 0 0; font-size:14px; color:#aaa;">No malicious patterns detected in content.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="threat-card">
                    <h2 style="margin:0; color:#F87171; font-size: 28px;">[ THREAT DETECTED ]</h2>
                    <p style="margin:10px 0 0 0; font-size:16px; color:#fff;">Confidence Score: <b>{res['conf']:.1f}%</b></p>
                    <p style="margin:5px 0 0 0; font-size:14px; color:#aaa;">Recommendation: <b>Block Sender Immediately.</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(" **Forensic Highlight:**")
                st.markdown(f"> {highlight_threats(res['msg'])}", unsafe_allow_html=True)
        else:
            st.info("System Ready. Awaiting Input from Console...")
            
        st.write("---")
        m1, m2 = st.columns(2)
        m1.metric("Threats Blocked", st.session_state['threat_count'])
        m2.metric("Safe Messages", st.session_state['safe_count'])

    with col_intel_right:
        # NEW: PRECISION GLOW MAP (Layered PyDeck) 
        current_text = st.session_state.get('input_buffer', '')
        
        if st.session_state.get('last_result'):
            geo = st.session_state['last_result']['geo']
            loc_label = geo['loc']
            lat, lon = geo['lat'], geo['lon']
            
            is_threat = st.session_state['last_result']['type'] == 'threat'
            
            # Define Colors (RGB)
            threat_color = [255, 50, 50] # Neon Red
            safe_color = [50, 255, 100]  # Neon Green
            base_color = threat_color if is_threat else safe_color

            # LAYER 1: The Core Dot (Small, solid, bright)
            core_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
                get_position='[lon, lat]',
                get_color=base_color + [255], # Solid opacity
                get_radius=30000, # 30km radius (small pin)
                pickable=True,
            )
            
            # LAYER 2: The Outer Glow (Large, highly transparent)
            glow_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
                get_position='[lon, lat]',
                get_color=base_color + [80], # High transparency for glow effect
                get_radius=100000, # 100km radius (large halo)
                pickable=False,
            )
            
            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=3, pitch=0)
            
            r = pdk.Deck(
                # Stack the layers: Glow first, then Core on top
                layers=[glow_layer, core_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip={"text": loc_label}
            )
            
            st.markdown(f"**ORIGIN TRACE:** `{loc_label}`")
            st.pydeck_chart(r)
            
        else:
            st.markdown("**ORIGIN TRACE:** `Waiting...`")
            st.map(pd.DataFrame({'lat': [20.59], 'lon': [78.96]}), zoom=1, use_container_width=True, height=280)

    # ROW 3: LOGS
    st.write("---")
    st.markdown("#### AUDIT LOGS")
    
    if st.session_state['history']:
        df = pd.DataFrame(st.session_state['history'])
        event = st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            selection_mode="multi-row",
            on_select="rerun",
            height=200 
        )
        
        selected_rows = event.selection.rows
        c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
        
        with c1:
            if len(selected_rows) == 1:
                if st.button("RELOAD"):
                    idx = selected_rows[0]
                    st.session_state['input_buffer'] = st.session_state['history'][idx]['Message']
                    st.rerun()
            else:
                st.button("RELOAD", disabled=True)

        with c2:
            if len(selected_rows) > 0:
                if st.button(f"DELETE ({len(selected_rows)})"):
                    for idx in sorted(selected_rows, reverse=True):
                        st.session_state['history'].pop(idx)
                    st.rerun()
            else:
                st.button("DELETE", disabled=True)

        with c3:
            if st.button("CLEAR ALL"):
                st.session_state['history'] = []
                st.session_state['threat_count'] = 0
                st.session_state['safe_count'] = 0
                st.rerun()
    else:
        st.caption("No recent activity recorded.")

        