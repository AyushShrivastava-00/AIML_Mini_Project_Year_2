"""
Micro-Merchant Informal Credit Assessment Engine - Streamlit Web Application.
Hardened for Localhost & Cloud. Uses textwrap.dedent to ensure clean HTML card rendering.
"""
import os
import sys
import tempfile
import textwrap
import streamlit as st
from src.pipeline.pdf_report import generate_pdf_credit_report


# 1. Streamlit configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Micro-Merchant Credit Assessment Engine",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Path Setup
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 3. Secrets Injection
try:
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ[key] = value
except Exception:
    pass

from PIL import Image
from src.pipeline.orchestrator import CreditAuditPipeline

# CSS Styling (Clean, zero indentation conflicts)
st.markdown(textwrap.dedent("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    .badge-item {
        background: #1e293b;
        color: #94a3b8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid #334155;
    }
    .custom-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .kpi-box {
        background: #0f172a;
        border: 1px solid #334155;
        padding: 14px;
        border-radius: 8px;
        text-align: center;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-val {
        font-size: 26px;
        font-weight: 800;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
    }
    .transcript-bubble {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 6px;
        color: #e2e8f0;
        font-size: 13px;
        line-height: 1.5;
        font-style: italic;
    }
</style>
"""), unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    absolute_onnx_path = os.path.join(ROOT_DIR, "models", "best.onnx")
    return CreditAuditPipeline(
        onnx_model_path=absolute_onnx_path,
        whisper_size="tiny",
        slm_model="allam-2-7b"
    )

pipeline = load_pipeline()

st.markdown("# 🏪 Micro-Merchant Informal Credit Assessment Engine")
st.markdown("##### *Edge Vision + Voice Multi-Modal Underwriting for Financial Inclusion*")

st.markdown(textwrap.dedent("""
<div class="badge-container">
    <span class="badge-item">⚡ Sub-Second Inference</span>
    <span class="badge-item">🔒 100% On-Device Privacy</span>
    <span class="badge-item">👁️ YOLOv8-Nano INT8</span>
    <span class="badge-item">🎙️ Whisper Speech-to-Text</span>
    <span class="badge-item">🧠 Multi-Tier SLM Underwriter</span>
</div>
"""), unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1.15], gap="large")

with col_input:
    st.markdown("### 📥 Field Telemetry Ingestion")
    st.caption("Upload store shelf inspection footage and borrower interview recording.")
    
    uploaded_image = st.file_uploader(
        "1. Store Shelf Footage (MP4 Video / JPG Image)", 
        type=["jpg", "jpeg", "png", "mp4"]
    )
    
    uploaded_audio = st.file_uploader(
        "2. Merchant Spoken Interview (WAV / MP3)", 
        type=["wav", "mp3", "m4a"]
    )
    
    st.write("")
    run_audit_btn = st.button("⚡ Execute Multi-Modal Micro-Audit", type="primary", use_container_width=True)

with col_output:
    st.markdown("### 📊 Underwriting Assessment Output")
    
    if not run_audit_btn:
        st.info("👈 Upload the inspection media and audio interview, then click **Execute Multi-Modal Micro-Audit** to begin evaluation.")
    else:
        if not uploaded_image or not uploaded_audio:
            st.warning("⚠️ Please provide both a store shelf image/video and an interview recording.")
        else:
            with st.spinner("Processing telemetry: Sampling video frames, running Whisper transcription, and executing financial risk models..."):
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
                    score = decision.credit_score

                    if score >= 700:
                        theme_color = "#10b981"
                        badge_label = "LOW RISK • APPROVED"
                    elif score >= 600:
                        theme_color = "#f59e0b"
                        badge_label = "MODERATE RISK • MANUAL REVIEW"
                    else:
                        theme_color = "#f43f5e"
                        badge_label = "SEVERE RISK • CAUTION / DECLINED"

                    # 1. Image Display
                    st.image(
                        results["annotated_image"], 
                        caption=f"Computer Vision Audit: {results['detected_items']} items localized ({results['shelf_fill_pct']:.0f}% capacity)", 
                        use_container_width=True
                    )

                    # 2. Executive Scorecard Rendered Without 4-space Indentation
                    engine_badge = getattr(decision, "engine_source", "Multi-Tier AI Engine")
                    
                    card_html = textwrap.dedent(f"""
<div class="custom-card" style="border-top: 4px solid {theme_color};">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 10px;">
<div>
<h2 style="margin: 0; color: #f8fafc; font-size: 20px;">{decision.merchant_name}</h2>
<span style="font-size: 12px; color: #94a3b8;">Informal Enterprise Credit Evaluation</span>
</div>
<span style="background: {theme_color}; padding: 4px 12px; border-radius: 16px; font-weight: 700; color: #000; font-size: 11px;">
{badge_label}
</span>
</div>
<div class="kpi-container">
<div class="kpi-box">
<div class="kpi-title">Credit Score</div>
<div class="kpi-val" style="color: {theme_color};">{score}</div>
<div class="kpi-sub">Scale: 300 - 850</div>
</div>
<div class="kpi-box">
<div class="kpi-title">Recommended Ceiling</div>
<div class="kpi-val" style="color: #38bdf8;">₹{decision.recommended_loan_limit:,.0f}</div>
<div class="kpi-sub">Confidence: {decision.confidence_score * 100:.0f}%</div>
</div>
<div class="kpi-box">
<div class="kpi-title">Turnover Velocity</div>
<div class="kpi-val" style="color: #cbd5e1;">{decision.financial_metrics.inventory_turnover_rate:.2f}x</div>
<div class="kpi-sub">Ratio: {decision.financial_metrics.inventory_to_sales_ratio}</div>
</div>
<div class="kpi-box">
<div class="kpi-title">Observed Stock</div>
<div class="kpi-val" style="color: #a78bfa;">{results['detected_items']}</div>
<div class="kpi-sub">{results['shelf_fill_pct']:.0f}% Shelf Fill</div>
</div>
</div>
<div style="background: #0f172a; padding: 12px 16px; border-radius: 8px; margin-top: 10px;">
<div style="font-weight: 600; color: #f8fafc; font-size: 12px; margin-bottom: 4px;">📑 Underwriting Rationale:</div>
<p style="margin: 0; font-size: 13px; color: #cbd5e1; line-height: 1.5;">{decision.underwriting_rationale}</p>
</div>
<div style="margin-top: 12px; display: flex; justify-content: space-between; font-size: 11px; color: #64748b;">
<span>Engine: <b style="color: #38bdf8;">{engine_badge}</b></span>
<span>Turnaround Latency: {results['total_latency_sec']}s</span>
</div>
</div>
""")
                    st.markdown(card_html, unsafe_allow_html=True)

                    # 3. Audio Transcript
                    st.markdown("##### 🎤 Spoken Interview Transcript (Whisper.cpp):")
                    transcript_html = textwrap.dedent(f"""
<div class="transcript-bubble">
"{results['transcript']}"
</div>
""")
                    st.markdown(transcript_html, unsafe_allow_html=True)
                    st.write("")

                    # 4. Forensic Telemetry
                    with st.expander("🔍 Inspect Deep Forensic Telemetry & Ratios"):
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Stated Daily Revenue", f"₹{decision.financial_metrics.estimated_daily_revenue:,.0f}")
                        m2.metric("Monthly Revenue", f"₹{decision.financial_metrics.estimated_monthly_revenue:,.0f}")
                        m3.metric("Discrepancy Flags", len(decision.discrepancy_flags))
                        if decision.discrepancy_flags:
                            st.error(f"Flags Triggered: {', '.join(decision.discrepancy_flags)}")
                        else:
                            st.success("No financial discrepancies detected. Inventory aligns with cash flow.")

                                        # ── ONE-CLICK EXPORTABLE AUDIT REPORTS ──────────────────────
                    st.write("")
                    col_pdf, col_json = st.columns(2)
                    
                    # 1. Executive PDF Memo (Primary Banking Deliverable)
                    pdf_bytes = generate_pdf_credit_report(decision, results)
                    clean_name = decision.merchant_name.replace(' ', '_').lower()
                    
                    with col_pdf:
                        st.download_button(
                            label="📄 Download Official PDF Assessment Memo",
                            data=pdf_bytes,
                            file_name=f"credit_memo_{clean_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    # 2. Raw JSON Telemetry (Developer Data Contract)
                    with col_json:
                        report_json = decision.model_dump_json(indent=2)
                        st.download_button(
                            label="💾 Export Raw JSON Audit Vector",
                            data=report_json,
                            file_name=f"telemetry_{clean_name}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    # ───────────────────────────────────────────────────────────

                except Exception as e:
                    st.error(f"Error during audit evaluation: {str(e)}")
                finally:
                    if os.path.exists(tmp_img_path): os.remove(tmp_img_path)
                    if os.path.exists(tmp_audio_path): os.remove(tmp_audio_path)
