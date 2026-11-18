"""
Micro-Merchant Informal Credit Assessment Engine - Streamlit Cloud Web Application.
"""
import os
import sys
import tempfile

# 1. Path setup MUST be first
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 2. Inject Streamlit secrets into environment
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ[key] = value
except Exception:
    pass

# 3. Third-party and local imports
import streamlit as st
from PIL import Image
from src.pipeline.orchestrator import CreditAuditPipeline

st.set_page_config(
    page_title="Micro-Merchant Credit Assessment Engine",
    page_icon="🏪",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return CreditAuditPipeline(
        onnx_model_path="models/best.onnx",
        whisper_size="tiny",
        slm_model="qwen2.5:1.5b"
    )

pipeline = load_pipeline()

st.title("🏪 Micro-Merchant Informal Credit Assessment Engine")
st.markdown("### *Edge Vision + Voice Multi-Modal Underwriting for Financial Inclusion*")
st.write("---")

col_input, col_output = st.columns(2, gap="large")

with col_input:
    st.subheader("📥 Field Officer Telemetry Ingestion")
    
    uploaded_image = st.file_uploader(
        "Upload Store Shelf Inspection Image / Video", 
        type=["jpg", "jpeg", "png", "mp4"]
    )
    
    uploaded_audio = st.file_uploader(
        "Upload Spoken Merchant Interview Audio", 
        type=["wav", "mp3", "m4a"]
    )
    
    run_audit_btn = st.button("⚡ Run AI Micro-Audit", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Multi-Modal Underwriting Output")
    
    if run_audit_btn:
        if not uploaded_image or not uploaded_audio:
            st.warning("⚠️ Please upload both a shelf inspection image/video and an interview recording.")
        else:
            with st.spinner("Executing ONNX Vision, Whisper STT, and Financial Cross-Verification..."):
                img_suffix = "." + uploaded_image.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=img_suffix) as tmp_img:
                    tmp_img.write(uploaded_image.read())
                    tmp_img_path = tmp_img.name

                audio_suffix = "." + uploaded_audio.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_audio:
                    tmp_audio.write(uploaded_audio.read())
                    tmp_audio_path = tmp_audio.name

                try:
                    results = pipeline.run_full_audit(media_path=tmp_img_path, audio_path=tmp_audio_path)
                    decision = results["decision"]

                    st.image(results["annotated_image"], caption=f"YOLOv8 INT8 Detection: {results['detected_items']} items found ({results['shelf_fill_pct']:.0f}% fill)", use_container_width=True)
                    st.text_area("🎤 Whisper Audio Transcript", results["transcript"], height=100)

                    score = decision.credit_score
                    if score >= 700:
                        score_color = "#22c55e"
                        badge_label = "LOW RISK - RECOMMENDED APPROVAL"
                    elif score >= 600:
                        score_color = "#eab308"
                        badge_label = "MODERATE RISK - MANUAL REVIEW"
                    else:
                        score_color = "#ef4444"
                        badge_label = "HIGH / SEVERE RISK - CAUTION"

                    engine_badge = getattr(decision, "engine_source", "Multi-Tier AI Underwriting")

                    st.markdown(f"""
                    <div style="background-color: #1e293b; color: white; padding: 20px; border-radius: 12px; margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 10px;">
                            <h3 style="margin: 0; color: #f8fafc;">{decision.merchant_name}</h3>
                            <span style="background: {score_color}; padding: 4px 12px; border-radius: 15px; font-weight: bold; color: black; font-size: 12px;">{badge_label}</span>
                        </div>
                        <div style="display: flex; gap: 15px; margin-top: 15px;">
                            <div style="flex: 1; background: #0f172a; padding: 12px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 11px; color: #94a3b8;">CREDIT SCORE</div>
                                <div style="font-size: 36px; font-weight: 800; color: {score_color};">{score}</div>
                                <div style="font-size: 11px; color: #64748b;">Scale: 300 - 850</div>
                            </div>
                            <div style="flex: 1; background: #0f172a; padding: 12px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 11px; color: #94a3b8;">RECOMMENDED CEILING</div>
                                <div style="font-size: 28px; font-weight: 800; color: #38bdf8; margin-top: 5px;">₹{decision.recommended_loan_limit:,.0f}</div>
                                <div style="font-size: 11px; color: #64748b;">Confidence: {decision.confidence_score * 100:.0f}%</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; background: #0f172a; padding: 12px; border-radius: 8px; font-size: 13px;">
                            <b>Turnover Rate:</b> {decision.financial_metrics.inventory_turnover_rate:.2f}x/mo ({decision.financial_metrics.inventory_to_sales_ratio})<br>
                            <b>Underwriting Rationale:</b> {decision.underwriting_rationale}
                        </div>
                        <div style="margin-top: 10px; display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8;">
                            <span>Engine: <b style="color: #38bdf8;">{engine_badge}</b></span>
                            <span>Audit Latency: {results['total_latency_sec']}s</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during audit evaluation: {str(e)}")
                finally:
                    if os.path.exists(tmp_img_path): os.remove(tmp_img_path)
                    if os.path.exists(tmp_audio_path): os.remove(tmp_audio_path)
