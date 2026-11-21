"""
Micro-Merchant Informal Credit Assessment Engine - Gradio Application.
Interactive loan officer dashboard supporting category selection and multi-frame pooling.
"""
import os
import sys
import gradio as gr

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.orchestrator import CreditAuditPipeline

pipeline = CreditAuditPipeline(
    onnx_model_path=os.path.join(ROOT_DIR, "models", "best.onnx"),
    whisper_size="tiny",
    local_slm="qwen2.5:1.5b",
    cloud_slm="llama-3.1-8b-instant"
)


def evaluate_merchant_application(image_input, audio_input, category_input):
    if image_input is None or audio_input is None:
        return None, "<div style='color: orange;'>⚠️ Please upload both shelf footage and an interview audio clip.</div>", ""

    try:
        results = pipeline.run_full_audit(
            media_path=image_input,
            audio_path=audio_input,
            store_category=category_input
        )
        decision = results["decision"]
        metrics = decision.financial_metrics
        annotated_img = results["annotated_image"]

        score = decision.credit_score
        if score >= 700:
            score_color = "#22c55e"
            badge_text = "LOW RISK - RECOMMENDED FOR APPROVAL"
        elif score >= 600:
            score_color = "#eab308"
            badge_text = "MODERATE RISK - MANUAL REVIEW"
        else:
            score_color = "#ef4444"
            badge_text = "HIGH / SEVERE RISK - DECLINE / CAUTION"

        scorecard_html = f"""
        <div style="background-color: #1e293b; color: white; padding: 20px; border-radius: 12px; font-family: sans-serif;">
          <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px;">
            <div>
              <h2 style="margin: 0; color: #f8fafc;">{decision.merchant_name}</h2>
              <span style="font-size: 14px; color: #94a3b8;">{category_input} Credit Evaluation</span>
            </div>
            <div style="background: {score_color}; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; color: black;">
              {badge_text}
            </div>
          </div>

          <div style="display: flex; gap: 15px; margin-top: 18px;">
            <div style="flex: 1; background: #0f172a; padding: 14px; border-radius: 8px; text-align: center;">
              <div style="font-size: 11px; color: #94a3b8;">CREDIT SCORE</div>
              <div style="font-size: 38px; font-weight: 800; color: {score_color}; margin: 4px 0;">{score}</div>
              <div style="font-size: 11px; color: #64748b;">Scale: 300 - 850</div>
            </div>

            <div style="flex: 1; background: #0f172a; padding: 14px; border-radius: 8px; text-align: center;">
              <div style="font-size: 11px; color: #94a3b8;">APPROVED LOAN LIMIT</div>
              <div style="font-size: 28px; font-weight: 800; color: #38bdf8; margin: 6px 0;">Rs. {decision.recommended_loan_limit:,.0f}</div>
              <div style="font-size: 11px; color: #64748b;">Tenor: {metrics.recommended_tenor_days} Days</div>
            </div>

            <div style="flex: 1; background: #0f172a; padding: 14px; border-radius: 8px; text-align: center;">
              <div style="font-size: 11px; color: #94a3b8;">SAFE DAILY EMI</div>
              <div style="font-size: 28px; font-weight: 800; color: #a78bfa; margin: 6px 0;">Rs. {metrics.safe_daily_emi:,.0f}</div>
              <div style="font-size: 11px; color: #64748b;">DSCR Capped (25%)</div>
            </div>
          </div>

          <div style="margin-top: 16px; background: #0f172a; padding: 12px; border-radius: 8px;">
            <div style="font-weight: 600; color: #38bdf8; margin-bottom: 4px;">Telemetry Cross-Validation:</div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #cbd5e1;">
              <span>Stock: <b>{results['detected_items']} items</b> ({results['shelf_fill_pct']:.0f}%)</span>
              <span>Velocity: <b>{metrics.inventory_turnover_rate:.2f}x/mo</b></span>
              <span>Status: <b>{metrics.inventory_to_sales_ratio}</b></span>
            </div>
          </div>

          <div style="margin-top: 14px; background: #0f172a; padding: 12px; border-radius: 8px;">
            <div style="font-weight: 600; color: #f8fafc; margin-bottom: 4px;">Underwriting Rationale:</div>
            <p style="margin: 0; font-size: 13px; color: #cbd5e1; line-height: 1.5;">{decision.underwriting_rationale}</p>
          </div>

          <div style="margin-top: 12px; display: flex; justify-content: space-between; font-size: 11px; color: #64748b;">
            <span>Engine: <b>{getattr(decision, 'engine_source', 'Multi-Tier AI Engine')}</b></span>
            <span>Audit Latency: {results['total_latency_sec']}s</span>
          </div>
        </div>
        """

        transcript_output = f"Transcribed Merchant Interview:\n\n\"{results['transcript']}\""
        return annotated_img, scorecard_html, transcript_output

    except Exception as e:
        return None, f"<div style='color: red;'>Error during evaluation: {str(e)}</div>", ""


with gr.Blocks(title="Micro-Merchant Credit Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏪 Micro-Merchant Informal Credit Assessment Engine
    ### *Edge Vision + Voice Multi-Modal Underwriting for Financial Inclusion*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Field Officer Telemetry Upload")
            category_input = gr.Dropdown(
                label="Store Category",
                choices=[
                    "Grocery / Kirana (FMCG)",
                    "Mobile / Electronics Kiosk",
                    "Apparel / Footwear Retail",
                    "General Retail / Kiosk"
                ],
                value="Grocery / Kirana (FMCG)"
            )
            image_input = gr.Image(label="Store Shelf Inspection Image / Frame", type="filepath")
            audio_input = gr.Audio(label="Spoken Merchant Interview (Hindi / English)", type="filepath")
            audit_button = gr.Button("⚡ Run AI Micro-Audit", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Computer Vision Audit")
            annotated_output = gr.Image(label="Inventory Item Detections (YOLOv8 INT8)")
            transcript_box = gr.Textbox(label="Audio Transcript (Whisper.cpp)", lines=3)

            gr.Markdown("### 📑 Automated Credit Underwriting Decision")
            scorecard_output = gr.HTML()

        audit_button.click(
            fn=evaluate_merchant_application,
            inputs=[image_input, audio_input, category_input],
            outputs=[annotated_output, scorecard_output, transcript_box]
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
