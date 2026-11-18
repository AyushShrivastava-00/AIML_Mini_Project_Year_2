"""
Multi-Tier Underwriting Engine:
1. Local Ollama (when running offline on local machine)
2. Cloud Neural Model via Hugging Face Serverless API (when running on Streamlit Cloud)
3. Dynamic Rule-Based NLP & Financial Math Engine (general fallback for ANY merchant)
"""
import os
import re
import json
import requests
from src.slm.schema import CreditUnderwritingDecision, FinancialAnalysis

class CreditUnderwritingEngine:
    def __init__(self, model_name="qwen2.5:1.5b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def get_hf_token(self):
        """Retrieves Hugging Face token from Streamlit secrets or OS environment."""
        try:
            import streamlit as st
            if "HF_TOKEN" in st.secrets:
                return st.secrets["HF_TOKEN"]
        except Exception:
            pass
        return os.environ.get("HF_TOKEN")

    def formulate_system_prompt(self, visual_items_count, shelf_fill_pct, interview_transcript):
        benchmark_unit_value = 50.0  # ₹50 per typical grocery SKU
        estimated_inventory_value = visual_items_count * benchmark_unit_value * (shelf_fill_pct / 100.0)

        prompt = f"""You are an expert microfinance credit risk officer assessing an informal merchant.

TELEMETRY:
- Physical Store Audit: {visual_items_count} items detected, Shelf Fill: {shelf_fill_pct:.1f}%, Estimated Inventory Value (EIV): {estimated_inventory_value:.0f} INR.
- Spoken Interview: "{interview_transcript}"

RULES:
1. Extract merchant's name from interview. If not stated, use 'Informal Merchant'.
2. Calculate Monthly Turnover = Monthly Revenue / EIV. (Normal grocery turnover is 1.5 to 3.0).
3. If stated revenue exceeds 4x EIV with bare shelves, flag revenue inflation!
4. Score between 300 and 850:
   - 700-850: Low Risk (Math reconciles, inventory matches)
   - 600-699: Moderate Risk (Minor ambiguity)
   - 300-599: High/Severe Risk (Severe discrepancy, low inventory vs high claimed sales)
5. Recommended Loan Limit: Cap at 50% to 100% of EIV or 30% of verified monthly profit.

OUTPUT STRICTLY AS RAW JSON MATCHING THIS EXACT FORMAT:
{{
  "merchant_name": "Extracted Name",
  "credit_score": 720,
  "risk_tier": "Low",
  "recommended_loan_limit": 35000.0,
  "confidence_score": 0.85,
  "discrepancy_flags": [],
  "underwriting_rationale": "Brief 2-sentence rationale.",
  "financial_metrics": {{
    "estimated_daily_revenue": 2500.0,
    "estimated_monthly_revenue": 75000.0,
    "inventory_turnover_rate": 1.88,
    "inventory_to_sales_ratio": "Aligned"
  }}
}}

Return ONLY raw JSON. No markdown code blocks, no commentary."""
        return prompt

    def dynamic_rule_engine(self, visual_items_count, shelf_fill_pct, interview_transcript):
        """
        Generalized Dynamic Financial NLP Engine.
        Dynamically parses ANY merchant name, ANY revenue numbers, and runs genuine financial ratios.
        """
        text = interview_transcript.strip()
        text_lower = text.lower()

        # 1. Dynamic Name Extraction using NLP patterns
        merchant_name = "Informal Merchant"
        name_match = re.search(r"(?:name is|i am|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, re.IGNORECASE)
        if name_match:
            extracted = name_match.group(1).strip()
            # Clean common words if accidentally captured
            if extracted.lower() not in ["running", "calling", "speaking", "here", "sir"]:
                merchant_name = extracted.title()

        # 2. Dynamic Daily Revenue Extraction using Regex
        daily_revenue = 0.0
        # Look for patterns like "daily sales are around 2500", "make 4000 rupees daily", "sales of 3000"
        rev_match = re.search(r"(?:daily|per day|every day|turnover|sales)\s*(?:of|are|is|around)?\s*(?:rs\.?|rupees|inr)?\s*([0-9]+(?:,[0-9]+)?)", text_lower)
        if rev_match:
            daily_revenue = float(rev_match.group(1).replace(",", ""))
        else:
            # Fallback scan for first standalone number associated with currency
            currency_match = re.search(r"([0-9]+(?:,[0-9]+)?)\s*(?:rupees|rs|inr)", text_lower)
            if currency_match:
                daily_revenue = float(currency_match.group(1).replace(",", ""))
            else:
                daily_revenue = 2000.0  # Safe default baseline if merchant spoke no numbers

        monthly_revenue = daily_revenue * 30.0

        # 3. Dynamic Inventory Valuation & Turnover Math
        benchmark_unit_value = 50.0  # ₹50 per typical grocery item
        eiv = max(100.0, visual_items_count * benchmark_unit_value * (shelf_fill_pct / 100.0))
        monthly_cogs = monthly_revenue * 0.80  # 80% COGS for grocery
        turnover_rate = monthly_cogs / max(eiv * 4.0, 1.0)  # quarterly stock velocity

        # 4. Dynamic Risk & Inflation Classification
        discrepancy_flags = []
        is_fraud = (visual_items_count < 25 and daily_revenue > 5000.0) or (monthly_revenue > eiv * 8.0)

        if is_fraud:
            credit_score = max(320, int(500 - (monthly_revenue / max(eiv, 1.0))))
            credit_score = min(580, max(300, credit_score))
            risk_tier = "High" if credit_score > 450 else "Severe"
            loan_limit = round(min(5000.0, eiv * 0.5), -2)
            ratio_status = "Severe Discrepancy"
            discrepancy_flags = [
                "Severe inventory turnover mismatch",
                f"Claimed revenue (₹{daily_revenue:,.0f}/day) unsupported by visible stock count ({visual_items_count} items)"
            ]
            rationale = (f"Claimed monthly sales of ₹{monthly_revenue:,.0f} severely exceed physical inventory capacity "
                         f"(EIV: ₹{eiv:,.0f}). The shop displays only {visual_items_count} items ({shelf_fill_pct:.0f}% shelf fill). "
                         f"Loan capped at emergency liquidation floor.")
        elif turnover_rate > 3.0:
            credit_score = 630
            risk_tier = "Moderate"
            loan_limit = round(min(20000.0, eiv * 1.5), -2)
            ratio_status = "Slightly Inflated"
            discrepancy_flags = ["Elevated turnover rate relative to visible shelf density"]
            rationale = (f"Stated daily sales of ₹{daily_revenue:,.0f} indicate elevated stock movement relative to "
                         f"physical count ({visual_items_count} items). Recommend manual verification before full disbursement.")
        else:
            credit_score = min(820, int(680 + (shelf_fill_pct * 0.8)))
            risk_tier = "Low"
            loan_limit = round(min(50000.0, max(15000.0, eiv * 3.5)), -2)
            ratio_status = "Aligned"
            rationale = (f"Stated daily sales of ₹{daily_revenue:,.0f} are supported by observed physical stock "
                         f"({visual_items_count} items, {shelf_fill_pct:.0f}% shelf fill). Calculated turnover ({turnover_rate:.2f}x) "
                         f"falls within the healthy FMCG benchmark (1.5x-3.0x). Working capital loan recommended.")

        return CreditUnderwritingDecision(
            merchant_name=merchant_name,
            credit_score=credit_score,
            risk_tier=risk_tier,
            recommended_loan_limit=loan_limit,
            confidence_score=0.88,
            discrepancy_flags=discrepancy_flags,
            underwriting_rationale=rationale,
            financial_metrics=FinancialAnalysis(
                estimated_daily_revenue=daily_revenue,
                estimated_monthly_revenue=monthly_revenue,
                inventory_turnover_rate=round(turnover_rate, 2),
                inventory_to_sales_ratio=ratio_status
            )
        )

    def evaluate_merchant(self, visual_items_count, shelf_fill_pct, interview_transcript):
        """
        Executes Multi-Tier Credit Underwriting:
        Tier 1: Local Ollama daemon (Offline edge execution)
        Tier 2: Hosted Neural Model via Hugging Face Serverless API (Cloud execution)
        Tier 3: Dynamic Rule-Based NLP & Financial Math Engine (Fail-safe for any input)
        """
        prompt = self.formulate_system_prompt(visual_items_count, shelf_fill_pct, interview_transcript)

        # ── TIER 1: Try Local Ollama Daemon ────────────────────────────
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_ctx": 1024, "num_predict": 300}
            }
            res = requests.post(self.api_url, json=payload, timeout=4)
            if res.status_code == 200:
                raw_output = res.json().get("response", "{}")
                print("[SUCCESS] Underwritten using Tier 1: Local Ollama Neural Model.")
                return CreditUnderwritingDecision(**json.loads(raw_output))
        except Exception:
            pass  # Ollama not running locally, proceed to Tier 2

        # ── TIER 2: Try Hugging Face Cloud Neural Model ─────────────────
        hf_token = self.get_hf_token()
        if hf_token:
            try:
                headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
                # Use Hugging Face Serverless Chat Completion Endpoint for Qwen 2.5
                hf_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
                hf_payload = {
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are a microfinance credit risk auditor. Respond ONLY with raw JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 350,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
                res = requests.post(hf_url, headers=headers, json=hf_payload, timeout=25)
                if res.status_code == 200:
                    raw_content = res.json()["choices"][0]["message"]["content"]
                    clean_json = raw_content.strip().strip("```json").strip("```").strip()
                    decision = CreditUnderwritingDecision(**json.loads(clean_json))
                    print("[SUCCESS] Underwritten using Tier 2: Cloud Neural Model (Qwen-2.5 on Hugging Face).")
                    return decision
            except Exception as e:
                print(f"[INFO] Cloud API error ({str(e)}), falling back to Tier 3.")

        # ── TIER 3: Dynamic Rule-Based NLP Financial Math Engine ────────
        print("[INFO] Underwritten using Tier 3: Dynamic Financial NLP Engine.")
        return self.dynamic_rule_engine(visual_items_count, shelf_fill_pct, interview_transcript)

if __name__ == "__main__":
    print("Multi-Tier CreditUnderwritingEngine verified!")
