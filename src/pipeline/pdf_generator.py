"""
Institutional PDF Credit Assessment Memo Generator.
Fully hardened against Latin-1 encoding crashes by using standard typography.
Incorporates DSCR repayment tenors and daily debt-service limits.
"""
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_credit_report(decision, results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=17, leading=21, textColor=colors.HexColor('#0f172a'), fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor('#475569'), fontName='Helvetica')
    section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=11, leading=15, textColor=colors.HexColor('#1e293b'), fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#334155'), fontName='Helvetica')
    rationale_style = ParagraphStyle('RationaleBody', parent=styles['Normal'], fontSize=8.5, leading=13, textColor=colors.HexColor('#1e293b'), fontName='Helvetica-Oblique')

    story = []

    # Header
    story.append(Paragraph("MICRO-FINANCE INSTITUTIONAL CREDIT MEMO", title_style))
    story.append(Paragraph("Alternative Telemetry Underwriting | Computer Vision & Voice Cash-Flow Audit", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb'), spaceBefore=6, spaceAfter=10))

    # Metadata
    timestamp_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    doc_id = f"MFI-AUD-{abs(hash(decision.merchant_name + timestamp_str)) % 1000000:06d}"

    meta_data = [
        [
            Paragraph(f"<b>Borrower / Shop:</b> {decision.merchant_name}", body_style),
            Paragraph(f"<b>Category:</b> {decision.financial_metrics.store_category}", body_style)
        ],
        [
            Paragraph(f"<b>Audit Reference:</b> {doc_id}", body_style),
            Paragraph(f"<b>Engine:</b> {getattr(decision, 'engine_source', 'Multi-Tier AI Engine')}", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Executive Decision
    story.append(Paragraph("1. EXECUTIVE CREDIT UNDERWRITING DECISION", section_heading))

    score = decision.credit_score
    badge_bg = colors.HexColor('#dcfce7') if score >= 700 else (colors.HexColor('#fef3c7') if score >= 600 else colors.HexColor('#ffe4e6'))
    badge_text_color = colors.HexColor('#166534') if score >= 700 else (colors.HexColor('#92400e') if score >= 600 else colors.HexColor('#9f1239'))

    decision_data = [
        [
            Paragraph("<b>CREDIT RISK SCORE</b>", body_style),
            Paragraph("<b>APPROVED LOAN LIMIT</b>", body_style),
            Paragraph("<b>REPAYMENT TENOR</b>", body_style),
            Paragraph("<b>RECOMMENDED DAILY EMI</b>", body_style)
        ],
        [
            Paragraph(f"<font size='14'><b>{score}</b></font> / 850", body_style),
            Paragraph(f"<font size='13'><b>Rs. {decision.recommended_loan_limit:,.0f}</b></font>", body_style),
            Paragraph(f"<b>{decision.financial_metrics.recommended_tenor_days} Days</b>", body_style),
            Paragraph(f"<b>Rs. {decision.financial_metrics.safe_daily_emi:,.0f} / day</b>", body_style)
        ]
    ]
    dec_table = Table(decision_data, colWidths=[130, 140, 120, 130])
    dec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,1), (-1,1), badge_bg),
        ('TEXTCOLOR', (0,1), (-1,1), badge_text_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(dec_table)
    story.append(Spacer(1, 8))

    # Telemetry Breakdown
    story.append(Paragraph("2. SENSORY TELEMETRY CROSS-VALIDATION", section_heading))

    telemetry_data = [
        [
            Paragraph("<b>Telemetry Channel</b>", body_style),
            Paragraph("<b>Observed Reading</b>", body_style),
            Paragraph("<b>Financial Implication</b>", body_style)
        ],
        [
            Paragraph("Physical Stock (YOLOv8 INT8)", body_style),
            Paragraph(f"{results.get('detected_items', 0)} units ({results.get('shelf_fill_pct', 0):.0f}% fill)", body_style),
            Paragraph("Multi-frame max-pooling verifies physical shelf presence", body_style)
        ],
        [
            Paragraph("Stated Daily Sales (Whisper)", body_style),
            Paragraph(f"Rs. {decision.financial_metrics.estimated_daily_revenue:,.0f} / day", body_style),
            Paragraph(f"Implied Rs. {decision.financial_metrics.estimated_monthly_revenue:,.0f} monthly gross", body_style)
        ],
        [
            Paragraph("Net Monthly Margin (DSCR)", body_style),
            Paragraph(f"Rs. {decision.financial_metrics.net_monthly_margin:,.0f} / month", body_style),
            Paragraph("Free operating cash flow available for debt service", body_style)
        ],
        [
            Paragraph("Turnover Rate", body_style),
            Paragraph(f"{decision.financial_metrics.inventory_turnover_rate:.2f}x / month", body_style),
            Paragraph(f"Status: <b>{decision.financial_metrics.inventory_to_sales_ratio}</b>", body_style)
        ]
    ]
    tele_table = Table(telemetry_data, colWidths=[150, 160, 210])
    tele_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(tele_table)
    story.append(Spacer(1, 8))

    # Forensic Rationale
    story.append(Paragraph("3. UNDERWRITING RATIONALE", section_heading))
    safe_rationale = decision.underwriting_rationale.replace('₹', 'Rs. ')
    rationale_p = Paragraph(f'"{safe_rationale}"', rationale_style)
    rationale_table = Table([[rationale_p]], colWidths=[520])
    rationale_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#94a3b8')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(rationale_table)
    story.append(Spacer(1, 8))

    # Evidence & Flags
    story.append(Paragraph("4. INTERVIEW EVIDENCE & DISCREPANCY FLAGS", section_heading))
    safe_transcript = results.get('transcript', 'No transcript recorded.').replace('₹', 'Rs. ')
    story.append(Paragraph(f'<b>Verbatim Audio Transcript:</b> "{safe_transcript}"', body_style))
    story.append(Spacer(1, 6))

    if decision.discrepancy_flags:
        flags_text = "<br/>".join([f"- {flag.replace('₹', 'Rs. ')}" for flag in decision.discrepancy_flags])
        story.append(Paragraph(f"<font color='#dc2626'><b>[DISCREPANCY FLAGS TRIGGERED]:</b><br/>{flags_text}</font>", body_style))
    else:
        story.append(Paragraph("<font color='#16a34a'><b>[AUDIT PASSED]:</b> No revenue inflation or inventory turnover flags triggered.</font>", body_style))

    story.append(Spacer(1, 14))

    # Sign-Off
    signoff_data = [
        [Paragraph("<b>Field Auditing Officer:</b>", body_style), Paragraph("<b>Branch Credit Underwriter:</b>", body_style)],
        [Paragraph("<br/><br/>__________________________________<br/>Signature & Staff ID", body_style), Paragraph("<br/><br/>__________________________________<br/>Approval / Rejection Signature", body_style)]
    ]
    signoff_table = Table(signoff_data, colWidths=[260, 260])
    signoff_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(KeepTogether([signoff_table]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
