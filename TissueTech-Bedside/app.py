import os
import requests
import gradio as gr

# ==========================================
# 1. CLOUD SETTINGS & SECURITY INITIALIZATION
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile" 

# ==========================================
# 2. EXTENSIVE DATA TEMPLATE (Deep Research Source)
# ==========================================
ASAL_RESEARCH_PAPER = {
    "background_layer": """
    RESEARCH TITLE: Towards a Low Cost Multi Parameter Monitoring Framework for Pressure Ulcer Prevention in Resource Limited Healthcare Settings
    CLINICAL BURDEN & SYSTEM FOCUS:
    - Pressure ulcers (PUs), or bedsores, are severe localized injuries to the skin and deep tissue structures caused by prolonged mechanical loading.
    - WHO Global Statistics (2023): Affects 1 in 10 hospitalized patients worldwide, skyrocketing to 33% within highly critical Intensive Care Units (ICUs).
    - Financial & Operational Crisis: Treating a single full-thickness Stage IV pressure ulcer drains hospital reserves by $20,000 to $150,000 USD.
    """,
    "findings_layer": """
    EMPIRICAL INSIGHTS & CLINICAL PATHOLOGY:
    - Finding 1: Most intelligent commercial mattresses focus strictly on positional body classification, failing to perform predictive tracking.
    - Finding 2: Comprehensive multi-parameter sensory tracking drastically increases preventative clinical value. Sharp localized temperature spikes across specific high-pressure target zones act as an early biological marker of severe tissue ischemia and acute inflammation well before visible dermal damage occurs.
    - Finding 4: Decentralized AI notification loops effectively mitigate caregiver burnout, accelerating average nursing repositioning intervention times.
    """,
    "architecture_layer": """
    LOW-COST HARDWARE ARRAY & ALGORITHMIC FRAMEWORK ($20–$45 BUDGET):
    1. Continuous Sensing Fabric: Piezoresistive Force Sensitive Resistors (FSR402, ~$5-10) to map localized pressure; ultra-thin micro-thermistors (~$1-3) for continuous thermodynamic skin scanning; and capacitive hygrometer arrays (~$3-5) to monitor moisture accumulation.
    2. Edge Microcontroller Node: Managed by an ESP32 Development Module (~$3-8) embedded into TPU-Coated Medical Fabric with Copper Conductive Thread.
    3. Mathematical Risk Index (RI) Protocol: Computes live values locally using the validated pathophysiology equation:
       RI = (0.50 * Pressure Score) + (0.30 * Temperature Score) + (0.20 * Moisture Score)
    """
}

def calculate_system_metrics(p, t, m):
    p_factor = float(p) / 100.0
    t_min, t_max = 30.0, 42.0
    t_factor = (float(t) - t_min) / (t_max - t_min)
    t_factor = max(0.0, min(1.0, t_factor))
    m_factor = float(m) / 100.0
    
    calculated_ri = (0.50 * p_factor) + (0.30 * t_factor) + (0.20 * m_factor)
    return round(calculated_ri, 3)

def practical_simulation_engine(pressure, temp, moisture):
    score = calculate_system_metrics(pressure, temp, moisture)
    
    if score < 0.42:
        zone_status = "🟢 LOW OPERATIONAL RISK STATE"
        context_block = f"{ASAL_RESEARCH_PAPER['background_layer']}\n{ASAL_RESEARCH_PAPER['findings_layer']}"
    elif score < 0.70:
        zone_status = "🟡 MODERATE CLINICAL ALERT STATE"
        context_block = f"{ASAL_RESEARCH_PAPER['findings_layer']}\n{ASAL_RESEARCH_PAPER['architecture_layer']}"
    else:
        zone_status = "🔴 CRITICAL HIGH-RISK EMERGENCY"
        context_block = f"{ASAL_RESEARCH_PAPER['background_layer']}\n{ASAL_RESEARCH_PAPER['architecture_layer']}\n{ASAL_RESEARCH_PAPER['findings_layer']}"

    if not GROQ_API_KEY:
        return "⚠️ **Groq API Key Error**: Please open the Space Settings tab, find 'Secrets', and add your `GROQ_API_KEY`."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are a Principal Embedded Biomedical AI Systems Engineer. Generate an extensive, "
        "rigorous medical-technical evaluation report based on live bedside telemetry. "
        "Provide deep scientific explanations, specific hardware price breakdowns, and explicit clinical mechanics."
    )
    
    user_prompt = f"""
    [LIVE BEDSIDE STREAMING TELEMETRY DATA]:
    - Calculated Risk Index (RI): {score}
    - Operational Zone Evaluation: {zone_status}
    - Sensor Inputs -> Mechanical Pressure Load: {pressure}%, Core Dermal Temperature Node: {temp}°C, Relative Epidermal Moisture: {moisture}%
    
    [VERIFIED RESEARCH CONTEXT]:
    {context_block}
    
    Generate an extensive technical report structured under these exact headers:
    ### I. ADVANCED SYSTEM STATUS & PATHOPHYSIOLOGICAL ANALYSIS
    ### II. RIGOROUS RESEARCH PAPER CROSS-REFERENCE & VALUE PROPOSITION
    ### III. CRITICAL BEDSIDE NURSING PROTOCOLS & CLINICAL INTERVENTIONS
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.25,
        "max_tokens": 1200
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content']
        elif res.status_code == 429:
            return "⏳ **Groq API Rate Limit Hit (429)**: Please wait 5-8 seconds and click transmit again to refresh the token window."
        else:
            return f"❌ Groq API Communication Failure ({res.status_code}): {res.text}"
    except Exception as e:
        return f"❌ Server Timeout During Complex Compilation: {str(e)}"

attractive_css = """
body, .gradio-container {
    background-color: #060913 !important;
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
}
.main-title {
    text-align: center;
    padding: 30px 0 15px 0;
}
.main-title h1 {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #9b51e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
}
.main-title p {
    color: #94a3b8 !important;
    font-size: 1.15rem;
}
.control-panel {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%) !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    border-radius: 16px !important;
    padding: 25px !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6) !important;
    backdrop-filter: blur(16px) !important;
}
.output-panel {
    background: linear-gradient(145deg, rgba(10, 15, 30, 0.9) 0%, rgba(15, 23, 42, 0.7) 100%) !important;
    border: 1px solid rgba(155, 81, 224, 0.25) !important;
    border-radius: 16px !important;
    padding: 25px !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.7) !important;
    backdrop-filter: blur(16px) !important;
}
input[type="range"] {
    accent-color: #00f2fe !important;
}
.action-btn {
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #9b51e0 100%) !important;
    color: #04060d !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    text-transform: uppercase;
    padding: 14px 20px !important;
    border-radius: 10px !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.4) !important;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.action-btn:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 0 35px rgba(0, 242, 254, 0.75), 0 0 15px rgba(155, 81, 224, 0.5) !important;
    color: #ffffff !important;
}
.clinical-output {
    background: rgba(10, 15, 30, 0.5) !important;
    border-left: 4px solid #9b51e0 !important;
    padding: 22px !important;
    border-radius: 8px;
    color: #e2e8f0 !important;
}
.clinical-output h3 {
    color: #00f2fe !important;
    font-weight: 700 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"), css=attractive_css) as app:
    gr.HTML(
        "<div class='main-title'>"
        "<h1>🏥 TissueTech Bedside RAG Architecture</h1>"
        "<p>⚡ Real-Time Clinical Telemetry & Multi-Parameter Simulation Dashboard</p>"
        "</div>"
    )
    with gr.Row():
        with gr.Column(scale=2, elem_classes="control-panel"):
            gr.Markdown("### 📡 Live Microcontroller Node Emulation")
            p_slider = gr.Slider(0, 100, value=90, label="🚨 Pressure Load Matrix (FSR402) %")
            t_slider = gr.Slider(30.0, 42.0, value=39.1, step=0.1, label="🌡️ Dermal Thermal Node (DHT11) °C")
            m_slider = gr.Slider(0, 100, value=85, label="💧 Epidermal Moisture Index %")
            gr.Markdown("---")
            run_btn = gr.Button("Transmit Matrix to Cloud RAG", elem_classes="action-btn")
        with gr.Column(scale=3, elem_classes="output-panel"):
            gr.Markdown("### 📋 Peer-Reviewed AI Diagnostic Analysis")
            output_markdown = gr.Markdown(elem_classes="clinical-output", value="_Adjust sliders on the left and trigger compilation._")
            
    run_btn.click(fn=practical_simulation_engine, inputs=[p_slider, t_slider, m_slider], outputs=[output_markdown])

app.launch()