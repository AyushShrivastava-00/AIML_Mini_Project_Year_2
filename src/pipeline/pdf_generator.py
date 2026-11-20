"""
Institutional PDF Credit Assessment Memo Generator.
Generates an executive credit risk underwriting package using ReportLab.
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
    """
    Generates a publication-grade credit underwriting PDF memo in memory.
    Returns:
        pdf_bytes (bytes): Binary PDF data ready for Streamlit st.download_button.
    """
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
    
    # Custom Professional Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        fontName='Helvetica'
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        fontName='Helvetica'
    )
    rationale_style = ParagraphStyle(
        'RationaleBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica-Oblique'
    )

    story = []

    # ── HEADER & INSTITUTIONAL BRANDING ──────────────────────────────────────
    story.append(Paragraph("MICRO-FINANCE UNDERWRITING AUDIT MEMO", title_style))
    story.append(Paragraph("Edge Multi-Modal Telemetry Verification • Informal Sector Credit Assessment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb'), spaceBefore=8, spaceAfter=12))

    # ── METADATA BAR ─────────────────────────────────────────────────────────
    timestamp_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    doc_id = f"MFI-AUD-{abs(hash(decision.merchant_name + timestamp_str)) % 1000000:06d}"
    
    meta_data = [
        [
            Paragraph(f"<b>Borrower / Shop:</b> {decision.merchant_name}", body_style),
            Paragraph(f"<b>Audit Date:</b> {timestamp_str}", body_style)
        ],
        [
            Paragraph(f"<b>Audit Reference ID:</b> {doc_id}", body_style),
            Paragraph(f"<b>Engine Provenance:</b> {getattr(decision, 'engine_source', 'Multi-Tier AI Engine')}", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── PRIMARY CREDIT DECISION TILES ────────────────────────────────────────
    story.append(Paragraph("1. EXECUTIVE CREDIT UNDERWRITING DECISION", section_heading))
    
    score = decision.credit_score
    if score >= 700:
        badge_bg = colors.HexColor('#dcfce7')
        badge_text_color = colors.HexColor('#166534')
        verdict = "RECOMMENDED FOR APPROVAL (LOW RISK)"
    elif score >= 600:
        badge_bg = colors.HexColor('#fef3c7')
        badge_text_color = colors.HexColor('#92400e')
        verdict = "SUBJECT TO MANUAL REVIEW (MODERATE RISK)"
    else:
        badge_bg = colors.HexColor('#ffe4e6')
        badge_text_color = colors.HexColor('#9f1239')
        verdict = "DECLINED / HIGH DISCREPANCY (SEVERE RISK)"

    decision_data = [
        [
            Paragraph("<b>CREDIT RISK SCORE</b>", body_style),
            Paragraph("<b>RECOMMENDED LOAN CEILING</b>", body_style),
            Paragraph("<b>RISK CLASSIFICATION</b>", body_style),
            Paragraph("<b>CONFIDENCE INDEX</b>", body_style)
        ],
        [
            Paragraph(f"<font size='16'><b>{score}</b></font> / 850", body_style),
            Paragraph(f"<font size='14'><b>₹{decision.recommended_loan_limit:,.0f}</b></font>", body_style),
            Paragraph(f"<b>{decision.risk_tier}</b>", body_style),
            Paragraph(f"<b>{decision.confidence_score * 100:.0f}%</b>", body_style)
        ]
    ]
    dec_table = Table(decision_data, colWidths=[130, 140, 130, 120])
    dec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,1), (-1,1), badge_bg),
        ('TEXTCOLOR', (0,1), (-1,1), badge_text_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(dec_table)
    story.append(Spacer(1, 10))

    # ── TELEMETRY CROSS-VALIDATION BREAKDOWN ─────────────────────────────────
    story.append(Paragraph("2. SENSORY TELEMETRY CROSS-VALIDATION", section_heading))
    
    telemetry_data = [
        [
            Paragraph("<b>Telemetry Channel</b>", body_style),
            Paragraph("<b>Observed Metric</b>", body_style),
            Paragraph("<b>Financial Implication</b>", body_style)
        ],
        [
            Paragraph("Physical Stock (YOLOv8 INT8)", body_style),
            Paragraph(f"{results.get('detected_items', 0)} items ({results.get('shelf_fill_pct', 0):.0f}% fill)", body_style),
            Paragraph("Verified asset backing on store shelves", body_style)
        ],
        [
            Paragraph("Stated Daily Revenue (Whisper)", body_style),
            Paragraph(f"₹{decision.financial_metrics.estimated_daily_revenue:,.0f} / day", body_style),
            Paragraph(f"Implied ₹{decision.financial_metrics.estimated_monthly_revenue:,.0f} monthly gross", body_style)
        ],
        [
            Paragraph("Monthly Turnover Rate", body_style),
            Paragraph(f"{decision.financial_metrics.inventory_turnover_rate:.2f}x / month", body_style),
            Paragraph(f"Status: <b>{decision.financial_metrics.inventory_to_sales_ratio}</b> (Norm: 1.5x - 3.0x)", body_style)
        ],
        [
            Paragraph("Verification Latency", body_style),
            Paragraph(f"{results.get('total_latency_sec', 0.0)} seconds", body_style),
            Paragraph("Edge CPU inference turnaround SLA met", body_style)
        ]
    ]
    tele_table = Table(telemetry_data, colWidths=[150, 160, 210])
    tele_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(tele_table)
    story.append(Spacer(1, 10))

    # ── UNDERWRITING RATIONALE ───────────────────────────────────────────────
    story.append(Paragraph("3. FORENSIC UNDERWRITING RATIONALE", section_heading))
    rationale_p = Paragraph(f"\"{decision.underwriting_rationale}\"", rationale_style)
    rationale_table = Table([[rationale_p]], colWidths=[520])
    rationale_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#94a3b8')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(rationale_table)
    story.append(Spacer(1, 10))

    # ── DISCREPANCY FLAGS & AUDIT TRANSCRIPT ─────────────────────────────────
    story.append(Paragraph("4. SPOKEN INTERVIEW EVIDENCE", section_heading))
    transcript_p = Paragraph(f"<b>Verbatim Audio Transcript:</b> \"{results.get('transcript', 'No transcript recorded.')}\"", body_style)
    story.append(transcript_p)
    story.append(Spacer(1, 8))

    if decision.discrepancy_flags:
        flags_text = "<br/>• ".join(decision.discrepancy_flags)
        flag_p = Paragraph(f"<font color='#dc2626'><b>⚠️ Discrepancy Flags Triggered:</b><br/>• {flags_text}</font>", body_style)
        story.append(flag_p)
    else:
        story.append(Paragraph("<font color='#16a34a'><b>✅ Telemetry Consistency:</b> No revenue inflation or inventory turnover flags triggered.</font>", body_style))
    
    story.append(Spacer(1, 16))

    # ── SIGN-OFF & COMPLIANCE FOOTER ─────────────────────────────────────────
    signoff_data = [
        [
            Paragraph("<b>Field Auditing Officer:</b>", body_style),
            Paragraph("<b>Branch Credit Underwriter:</b>", body_style)
        ],
        [
            Paragraph("<br/><br/>__________________________________<br/>Signature & Staff ID", body_style),
            Paragraph("<br/><br/>__________________________________<br/>Approval / Rejection Signature", body_style)
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[260, 260])
    signoff_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    
    story.append(KeepTogether([signoff_table]))

    # Build PDF into memory
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
