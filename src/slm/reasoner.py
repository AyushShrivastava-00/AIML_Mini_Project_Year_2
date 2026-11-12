"""
Local SLM Credit Underwriting Engine.
Communicates with Ollama daemon on localhost:11434, executes few-shot prompts,
and enforces Pydantic JSON schema validation.
Optimized for high-speed CPU edge inference.
"""
import json
import requests
from src.slm.schema import CreditUnderwritingDecision

class CreditUnderwritingEngine:
    def __init__(self, model_name="qwen2.5:1.5b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def formulate_system_prompt(self, visual_items_count, shelf_fill_pct, interview_transcript):
        """
        Constructs a concise, high-speed forensic underwriting prompt.
        """
        benchmark_unit_value = 50.0  # ₹50 per typical grocery SKU
        estimated_inventory_value = visual_items_count * benchmark_unit_value * (shelf_fill_pct / 100.0)

        prompt = f"""You are an expert microfinance credit risk officer assessing an informal merchant.

TELEMETRY:
- Physical Store Audit: {visual_items_count} items detected, Shelf Fill: {shelf_fill_pct:.1f}%, Estimated Inventory Value (EIV): {estimated_inventory_value:.0f} INR.
- Spoken Interview: "{interview_transcript}"

RULES:
1. Calculate Monthly Turnover = Monthly Revenue / EIV. (Normal grocery turnover is 1.5 to 3.0).
2. If stated revenue exceeds 4x EIV with bare shelves, flag revenue inflation!
3. Score between 300 and 850:
   - 700-850: Low Risk (Math reconciles, inventory matches)
   - 600-699: Moderate Risk (Minor ambiguity)
   - 300-599: High/Severe Risk (Severe discrepancy, low inventory vs high claimed sales)
4. Recommended Loan Limit: Cap at 50% to 100% of EIV or 30% of verified monthly profit.

OUTPUT STRICTLY AS RAW JSON MATCHING THIS EXACT FORMAT:
{{
  "merchant_name": "Merchant Name",
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

No codeblocks, no commentary, only raw JSON.
"""
        return prompt

    def evaluate_merchant(self, visual_items_count, shelf_fill_pct, interview_transcript):
        """
        Calls local Ollama SLM with CPU token speed optimizations and validates output.
        """
        prompt = self.formulate_system_prompt(visual_items_count, shelf_fill_pct, interview_transcript)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,      # Deterministic financial reasoning
                "num_ctx": 1024,          # Compact context for CPU speed
                "num_predict": 300        # Caps generation to stop runaway loops
            }
        }

        try:
            # 120s is plenty of headroom with num_predict=300
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            raw_output = response_json.get("response", "{}")
            
            parsed_data = json.loads(raw_output)
            decision = CreditUnderwritingDecision(**parsed_data)
            return decision

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama daemon is not running! Ensure 'ollama serve' is running.")
        except Exception as e:
            raise ValueError(f"SLM Underwriting parsing error: {str(e)}")

if __name__ == "__main__":
    print("CreditUnderwritingEngine ready!")
