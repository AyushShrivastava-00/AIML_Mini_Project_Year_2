"""
Micro-Merchant Informal Credit Assessment Engine - Web Application.
Interactive field loan officer dashboard built using Gradio Blocks.
"""
import os
import gradio as gr
from src.pipeline.orchestrator import CreditAuditPipeline

# Initialize the pipeline
pipeline = CreditAuditPipeline(
    onnx_model_path="models/best.onnx",
    whisper_size="tiny",
    slm_model="qwen2.5:1.5b"
)

def evaluate_merchant_application(image_input, audio_input):
    if image_input is None or audio_input is None:
        return None, "⚠️ Please upload both a shelf inspection image/video and an interview audio clip.", ""

    try:
        # Run orchestrator
        results = pipeline.run_full_audit(media_path=image_input, audio_path=audio_input)
        
        decision = results["decision"]
        annotated_img = results["annotated_image"]

        # Color-coded badge based on score
        score = decision.credit_score
        if score >= 700:
            score_color = "#22c55e"  # Green
            badge_text = "LOW RISK - RECOMMENDED FOR APPROVAL"
        elif score >= 600:
            score_color = "#eab308"  # Yellow
            badge_text = "MODERATE RISK - MANUAL REVIEW"
        else:
            score_color = "#ef4444"  # Red
            badge_text = "HIGH / SEVERE RISK - DECLINE / CAUTION"

        # HTML Scorecard
        scorecard_html = f"""
        <div style="background-color: #1e293b; color: white; padding: 20px; border-radius: 12px; font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px;">
                <div>
                    <h2 style="margin: 0; color: #f8fafc;">{decision.merchant_name}</h2>
                    <span style="font-size: 14px; color: #94a3b8;">Informal Retail Credit Audit</span>
                </div>
                <div style="background: {score_color}; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; color: black;">
                    {badge_text}
                </div>
            </div>

            <div style="display: flex; gap: 20px; margin-top: 18px;">
                <div style="flex: 1; background: #0f172a; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 12px; color: #94a3b8;">CREDIT SCORE</div>
                    <div style="font-size: 42px; font-weight: 800; color: {score_color}; margin: 4px 0;">{score}</div>
                    <div style="font-size: 12px; color: #64748b;">Scale: 300 - 850</div>
                </div>

                <div style="flex: 1; background: #0f172a; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 12px; color: #94a3b8;">RECOMMENDED LOAN CEILING</div>
                    <div style="font-size: 32px; font-weight: 800; color: #38bdf8; margin: 10px 0;">₹{decision.recommended_loan_limit:,.0f}</div>
                    <div style="font-size: 12px; color: #64748b;">Confidence: {decision.confidence_score * 100:.0f}%</div>
                </div>
            </div>

            <div style="margin-top: 18px; background: #0f172a; padding: 14px; border-radius: 8px;">
                <div style="font-weight: 600; color: #38bdf8; margin-bottom: 6px;">Telemetry Cross-Validation:</div>
                <div style="display: flex; justify-content: space-between; font-size: 14px; color: #cbd5e1;">
                    <span>Detected Stock: <b>{results['detected_items']} items</b> ({results['shelf_fill_pct']:.0f}% full)</span>
                    <span>Turnover Rate: <b>{decision.financial_metrics.inventory_turnover_rate:.2f}x/mo</b></span>
                    <span>Ratio: <b>{decision.financial_metrics.inventory_to_sales_ratio}</b></span>
                </div>
            </div>

            <div style="margin-top: 14px; background: #0f172a; padding: 14px; border-radius: 8px;">
                <div style="font-weight: 600; color: #f8fafc; margin-bottom: 4px;">Underwriting Rationale:</div>
                <p style="margin: 0; font-size: 14px; color: #cbd5e1; line-height: 1.5;">{decision.underwriting_rationale}</p>
            </div>
            
            <div style="margin-top: 12px; text-align: right; font-size: 12px; color: #64748b;">
                Audit Latency: {results['total_latency_sec']}s (Edge CPU Native)
            </div>
        </div>
        """

        transcript_output = f"🎤 Transcribed Merchant Interview:\n\n\"{results['transcript']}\""
        return annotated_img, scorecard_html, transcript_output

    except Exception as e:
        return None, f"<div style='color: red;'>Error during evaluation: {str(e)}</div>", ""

# Build Gradio Interface
custom_css = """
.container { max-width: 1050px; margin: auto; padding-top: 15px; }
"""

with gr.Blocks(title="Micro-Merchant Credit Engine", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏪 Micro-Merchant Informal Credit Assessment Engine
    ### *Edge Vision + Voice Multi-Modal Underwriting for Financial Inclusion*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Field Officer Telemetry Upload")
            image_input = gr.Image(label="Store Shelf Inspection Image / Video Frame", type="filepath")
            audio_input = gr.Audio(label="Spoken Merchant Interview (Hindi / English)", type="filepath")
            audit_button = gr.Button("⚡ Run AI Micro-Audit", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Computer Vision Audit")
            annotated_output = gr.Image(label="Inventory Item Detections (YOLOv8 INT8)")
            transcript_box = gr.Textbox(label="Audio Transcript (Whisper.cpp)", lines=3)

    gr.Markdown("### 📑 Automated Credit Underwriting Decision")
    scorecard_output = gr.HTML()

    # Link button trigger
    audit_button.click(
        fn=evaluate_merchant_application,
        inputs=[image_input, audio_input],
        outputs=[annotated_output, scorecard_output, transcript_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
