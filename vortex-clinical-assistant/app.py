import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr

# ==========================================
# 1. API KEY SETUP & INITIALIZATION
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"

# ==========================================
# 2. DATA SOURCE (RESEARCH PAPER DATA)
# ==========================================
RESEARCH_PAPER = """
Towards a Low Cost Multi Parameter Monitoring Framework for Pressure Ulcer Prevention in Resource Limited Healthcare Settings
Team: Neurotech | Project Q2 — 72-Hour Research Hackathon

Section 1: Title & Research Question
Title: Towards a Low Cost Multi Parameter Monitoring Framework for Pressure Ulcer Prevention in Resource Limited Healthcare Settings
Research Question: Can an affordable, sensor integrated, AI driven monitoring system continuously and effectively detect pressure ulcer risk factors in low resource hospital environments where current solutions remain inaccessible or inadequate?

Section 2: Hypothesis
If a low-cost monitoring system integrating pressure, temperature, and moisture sensors with an AI-driven alert mechanism is deployed in resource-limited healthcare settings, then it may improve early detection of pressure ulcer risk factors and reduce preventable ulcer development through continuous monitoring and timely caregiver intervention.

Section 3: Background
Pressure ulcers (PUs), also known as bedsores or pressure injuries, are localized injuries to the skin and underlying tissue that develop when sustained pressure restricts blood flow to a region of the body, most commonly over bony prominences such as the sacrum, heels, and hips [1]. They are classified into four stages of severity, with Stage IV ulcers involving full thickness tissue loss extending to muscle and bone injuries that are often irreversible and life threatening.
The global burden of pressure ulcers is substantial. According to the World Health Organization, PUs affect approximately 1 in 10 hospitalized patients worldwide, with prevalence rates in intensive care units reaching as high as 33% in some settings (WHO, 2023) [3]. In low and middle income countries, the problem is compounded by understaffed wards, poor patient nutrition, and limited access to preventive technologies. The financial cost is equally staggering in the United States alone, treating a single full thickness pressure ulcer can cost between $20,000 and $150,000 USD (Bluestein & Javaheri, 2008) [4].
The primary risk factor for pressure ulcer development is prolonged immobility. Bedridden patients particularly the elderly, post-surgical patients, and those with neurological conditions are most vulnerable because they cannot reposition themselves independently. Standard of care dictates that nursing staff reposition at risk patients every two hours [5]; however, in resource limited settings with high patient to nurse ratios, this protocol is frequently missed. Additional contributing factors include excess skin moisture, elevated skin temperature due to impaired perfusion, and patient specific intrinsic factors such as malnutrition, advanced age, and comorbidities like diabetes.

Section 4: Key Findings
Finding 1: Current intelligent monitoring systems focus on posture classification rather than holistic risk prediction. A 2022 systematic review by Silva et al. found that the majority of systems were primarily concerned with detecting lying position using pressure sensor arrays [5]. Of the 21 studies reviewed, only two incorporated any form of predictive modelling for ulcer occurrence risk.
Finding 2: Multi parameter sensing significantly improves clinical utility. Systems combining pressure data with temperature and moisture sensors provided caregivers with richer, more actionable information. One reviewed study demonstrated that temperature changes at high pressure zones serve as an early biological indicator of tissue ischemia before visible skin damage appears.
Finding 3: Real world deployment exposes major limitations of existing solutions. Several studies reported significant performance degradation when systems were deployed in real ward conditions versus controlled laboratory settings due to bed cushion use, electrical interference, and variable patient body habitus.
Finding 4: AI driven alert systems reduce caregiver workload and improve response times. Automated notifications sent to caregivers' mobile devices resulted in faster repositioning response times compared to standard manual observation protocols.
Finding 5: The absence of real world clinical datasets is a major barrier to predictive model development. Silva et al. identified the scarcity of multi parameter patient datasets as the primary reason predictive modelling remains underdeveloped.

Section 5: Insight and Proposed Idea
Based on the evidence reviewed, we propose the development of a low-cost, multi-parameter pressure ulcer monitoring framework: a sensor-integrated, AI-assisted system specifically designed for deployment in resource-limited healthcare settings.
The proposed framework consists of three components:
1. Low cost sensing layer: Piezoresistive pressure sensors (FSR402 Force Sensitive Resistor ~$5-10), low-resolution temperature sensors (DHT11 or DS18B20 ~$1-3), and capacitive moisture sensors (~$3-5) embedded beneath a standard hospital mattress cover using an ESP32 Development Board (~$3-8), TPU-Coated Medical Fabric (~$2-4), Copper Conductive Thread (~$1-2), and a 5V USB power pack (~$5-10). Total estimated hardware cost is $20–45 USD per bed.
2. Lightweight AI-assisted alert module: Runs on a low-power microcontroller continuously calculating a unified Multi-Parameter Risk Index (RI):
Risk Index (RI) = (W1 * Pressure) + (W2 * Temperature) + (W3 * Moisture)
Based on current pressure ulcer pathophysiology research, weight coefficients are: 50% pressure (W1=0.50), 30% temperature (W2=0.30), and 20% moisture (W3=0.20). Communication operates offline using ESP-NOW or local Wi-Fi mesh networking.
3. Passive longitudinal data logging system: Stores timestamped multimodal measurements, caregiver responses, and intervention outcomes to enable future machine learning model updates.

Section 6: Conclusion
This paper examined whether an affordable, sensor-integrated, AI-assisted monitoring framework could effectively reduce pressure ulcer risk in resource-limited healthcare settings. By integrating low-cost pressure, temperature, and moisture sensors into a unified risk-scoring framework (50% pressure, 30% temperature, 20% moisture), the system offers a clinically meaningful approach to early detection while supporting nursing judgment over unstable local networks.
"""

# Vector Pipeline Processing
print("Loading Local Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=["\nSection ", "\nFinding ", "\n\n", "\n", " "]
)
docs = text_splitter.create_documents([RESEARCH_PAPER])

print("Building Local FAISS Index Map...")
vector_store = FAISS.from_documents(docs, embeddings)
print("Processing Completed!")

# Mathematical Calculations
def calculate_risk_index(pressure, temperature, moisture):
    p_score = float(pressure) / 100.0
    t_min, t_max = 30.0, 42.0
    t_score = (float(temperature) - t_min) / (t_max - t_min)
    t_score = max(0.0, min(1.0, t_score))
    m_score = float(moisture) / 100.0
    
    ri = (0.50 * p_score) + (0.30 * t_score) + (0.20 * m_score)
    return round(ri, 3)

# Direct HTTP API Request Function
def raw_groq_api_call(system_prompt, user_prompt):
    if not GROQ_API_KEY:
        return "⚠️ **System Secret Error**: Please add 'GROQ_API_KEY' under your Spaces Settings tab."
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"❌ Groq API Error Status {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Network Error: {str(e)}"

# Generation Execution Engine
def clinical_rag_engine(user_query, mode="Research Q&A Mode", pressure=0, temp=36.5, moisture=0):
    search_results = vector_store.similarity_search(user_query, k=2)
    context_text = "\n---\n".join([doc.page_content for doc in search_results])
    
    if mode == "Patient Simulation Analyzer":
        ri_score = calculate_risk_index(pressure, temp, moisture)
        zone = "LOW RISK" if ri_score < 0.45 else "MODERATE RISK" if ri_score < 0.75 else "HIGH RISK ZONE"
        
        system_prompt = "You are an advanced clinical decision system assistant. Formulate your response using professional markdown."
        user_prompt = f"""
        [LIVE PATIENT METRICS]:
        - Risk Index Score: {ri_score}
        - Clinical Status: {zone}
        - Sensor Inputs -> Pressure: {pressure}%, Temp: {temp}°C, Moisture: {moisture}%
        
        [RESEARCH PAPER CONTEXT]:
        {context_text}
        
        Provide structured bedside nurse instructions utilizing strict bullet points and clear alert highlights matching the paper text.
        """
    else:
        system_prompt = "Answer academic queries strictly using the provided context blocks. Format beautifully. If not found, say 'This information is not available in the provided research paper.'"
        user_prompt = f"[CONTEXT]:\n{context_text}\n\n[QUERY]:\n{user_query}"
        
    return raw_groq_api_call(system_prompt, user_prompt)

# ==========================================
# 3. HIGH-END CYBER LUXE CUSTOM CSS
# ==========================================
custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Space Grotesk', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.neon-title {
    text-align: center;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(0, 242, 254, 0.3);
    margin-bottom: 5px !important;
}
.neon-subtitle {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 1.1rem;
    margin-bottom: 25px !important;
}
.cyber-panel {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(0, 242, 254, 0.2) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    backdrop-filter: blur(8px) !important;
}
.tabs {
    border-bottom: 1px solid rgba(0, 242, 254, 0.1) !important;
}
.tabitem {
    background: transparent !important;
}
button.primary-btn {
    background: linear-gradient(90deg, #00d2ff 0%, #0066ff 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 0 15px rgba(0, 210, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 25px rgba(0, 210, 255, 0.7) !important;
}
button.action-btn {
    background: linear-gradient(90deg, #ff007f 0%, #7f00ff 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 0 15px rgba(255, 0, 127, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 25px rgba(255, 0, 127, 0.7) !important;
}
.output-box {
    background: rgba(30, 41, 59, 0.5) !important;
    border-left: 4px solid #00f2fe !important;
    border-radius: 4px !important;
    padding: 15px !important;
}
"""

# Interface Setup Layout with Soft/Slate Base + Custom CSS Overrides
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"), css=custom_css) as app:
    
    gr.HTML("<h1 class='neon-title'>🏥 Neurotech Clinical RAG Assistant</h1>")
    gr.HTML("<p class='neon-subtitle'>AI decision framework for resource-limited hospital wards</p>")
    
    with gr.Tabs(elem_classes="tabs"):
        
        with gr.TabItem("📋 Academic Q&A Map"):
            with gr.Row(elem_classes="cyber-panel"):
                with gr.Column():
                    qa_input = gr.Textbox(
                        label="Ask about the research paper:", 
                        placeholder="e.g., What are the three hardware layers and their exact pricing metrics?",
                        lines=2
                    )
                    qa_submit = gr.Button("Query Knowledge Base", elem_classes="primary-btn")
                with gr.Column():
                    gr.Markdown("### 🔍 Verified Insights Database Output")
                    qa_output = gr.Markdown(elem_classes="output-box")
            
            qa_submit.click(
                fn=lambda q: clinical_rag_engine(q, mode="Research Q&A Mode"), 
                inputs=[qa_input], 
                outputs=[qa_output]
            )
            
        with gr.TabItem("🎛️ Bedside Patient Simulator"):
            with gr.Row(elem_classes="cyber-panel"):
                with gr.Column():
                    gr.Markdown("### 📡 Real-time Sensor Controls")
                    slider_p = gr.Slider(0, 100, value=85, label="Pressure Load Matrix (%)")
                    slider_t = gr.Slider(30.0, 42.0, value=39.5, step=0.1, label="Skin Temperature Node (°C)")
                    slider_m = gr.Slider(0, 100, value=75, label="Epidermal Moisture Level (%)")
                    sim_submit = gr.Button("Evaluate Patient Trajectory & Calculate RI", elem_classes="action-btn")
                with gr.Column():
                    gr.Markdown("### ⚡ Live Clinical Alert Module")
                    sim_output = gr.Markdown(elem_classes="output-box")
                    
            sim_submit.click(
                fn=lambda p, t, m: clinical_rag_engine("Simulate", mode="Patient Simulation Analyzer", pressure=p, temp=t, moisture=m),
                inputs=[slider_p, slider_t, slider_m],
                outputs=[sim_output]
            )

# Launching Application
app.launch()