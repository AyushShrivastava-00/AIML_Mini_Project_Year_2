"""
Verifies the SLM Underwriting Engine across two test scenarios:
Scenario A: Authentic, healthy Kirana merchant.
Scenario B: Fraudulent/inflated claim (empty shelves vs claimed ₹10,000/day).
"""
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.slm.reasoner import CreditUnderwritingEngine

def run_underwriting_suite():
    print("=" * 70)
    print("  SLM GENERATIVE UNDERWRITING ENGINE - TEST SUITE")
    print("=" * 70)

    engine = CreditUnderwritingEngine(model_name="qwen2.5:1.5b")

    # -------------------------------------------------------------
    # Test Case 1: Healthy Kirana Merchant (Our actual Day 3 & Day 4 outputs!)
    # -------------------------------------------------------------
    print("\n[TEST CASE 1] Authentic Merchant (Suresh Patel)")
    print("Inputs: 146 items on shelf (100% fill), Stated revenue: ₹2,500/day, ₹10k restock")
    
    interview_1 = (
        "Namaste Sir, my name is Suresh Patel. I run this grocery store. "
        "My daily sales are around 2500 rupees. Every Monday, I pay my wholesale supplier "
        "10000 rupees to restock rice, pulses, and oil. My shop rent is 3000 rupees per month."
    )
    
    decision_1 = engine.evaluate_merchant(
        visual_items_count=146,
        shelf_fill_pct=100.0,
        interview_transcript=interview_1
    )

    print("-" * 50)
    print(f"Merchant Name:          {decision_1.merchant_name}")
    print(f"Credit Score:           {decision_1.credit_score} / 850")
    print(f"Risk Tier:              {decision_1.risk_tier}")
    print(f"Recommended Loan Limit: ₹{decision_1.recommended_loan_limit:,.2f}")
    print(f"Turnover Ratio:         {decision_1.financial_metrics.inventory_turnover_rate:.2f}x ({decision_1.financial_metrics.inventory_to_sales_ratio})")
    print(f"Discrepancy Flags:      {decision_1.discrepancy_flags if decision_1.discrepancy_flags else 'None (Clean Audit)'}")
    print(f"Rationale:              {decision_1.underwriting_rationale}")

    # -------------------------------------------------------------
    # Test Case 2: Fraudulent / Inflated Claim
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[TEST CASE 2] High-Risk / Discrepant Merchant")
    print("Inputs: Only 12 items on shelf (15% fill), but Claims ₹15,000 daily sales!")
    
    interview_2 = (
        "Hello sir, I have a massive shop. I make 15000 rupees every single day. "
        "I need a 200000 rupee loan right now to expand."
    )
    
    decision_2 = engine.evaluate_merchant(
        visual_items_count=12,
        shelf_fill_pct=15.0,
        interview_transcript=interview_2
    )

    print("-" * 50)
    print(f"Merchant Name:          {decision_2.merchant_name}")
    print(f"Credit Score:           {decision_2.credit_score} / 850")
    print(f"Risk Tier:              {decision_2.risk_tier}")
    print(f"Recommended Loan Limit: ₹{decision_2.recommended_loan_limit:,.2f}")
    print(f"Turnover Ratio:         {decision_2.financial_metrics.inventory_turnover_rate:.2f}x ({decision_2.financial_metrics.inventory_to_sales_ratio})")
    print(f"Discrepancy Flags:      {decision_2.discrepancy_flags}")
    print(f"Rationale:              {decision_2.underwriting_rationale}")
    print("=" * 70)

if __name__ == "__main__":
    run_underwriting_suite()
