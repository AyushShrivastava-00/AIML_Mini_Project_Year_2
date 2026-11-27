import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.slm.reasoner import CreditUnderwritingEngine

print("=" * 60)
print("⚡ DIRECT REASONER TEST")
print("=" * 60)

engine = CreditUnderwritingEngine()

transcript = (
    "Namaste sir, my name is Suresh Patel. I run this grocery store. "
    "My daily sales are around 4000 rupees. Every Monday, I pay my wholesale supplier "
    "15000 rupees to restock rice, pulses, and oil. My shop rent is 3000 rupees per month."
)

decision = engine.evaluate_merchant(
    visual_items_count=95,
    shelf_fill_pct=100.0,
    interview_transcript=transcript,
    store_category="Grocery / Kirana (FMCG)"
)

print(f"\n🎯 ENGINE ACTIVE: {decision.engine_source}")
print(f"📝 CREDIT SCORE : {decision.credit_score} / 850")
print(f"💰 LOAN LIMIT   : Rs. {decision.recommended_loan_limit:,.0f}")
print(f"📑 RATIONALE    :\n{decision.underwriting_rationale}")
print("=" * 60)
