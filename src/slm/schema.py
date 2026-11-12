"""
Pydantic Data Schemas for Micro-Merchant Informal Credit Assessment.
Defines the strictly enforced, resilient JSON schema for the SLM Underwriting Engine.
"""
from typing import List, Union
from pydantic import BaseModel, Field, field_validator

class FinancialAnalysis(BaseModel):
    estimated_daily_revenue: float = Field(default=0.0, description="Estimated daily revenue in local currency")
    estimated_monthly_revenue: float = Field(default=0.0, description="Estimated 30-day gross revenue")
    inventory_turnover_rate: float = Field(default=2.0, description="Calculated monthly inventory turnover ratio number")
    inventory_to_sales_ratio: str = Field(default="Aligned", description="Assessment: 'Aligned', 'Slightly Inflated', or 'Severe Discrepancy'")

    @field_validator('inventory_turnover_rate', mode='before')
    @classmethod
    def parse_turnover(cls, v):
        """Coerces strings to float if model returns text like '2.5x' or swaps fields."""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = v.replace('x', '').replace('X', '').strip()
            try:
                return float(clean)
            except ValueError:
                return 2.0  # safe default fallback
        return 2.0

    @field_validator('estimated_daily_revenue', 'estimated_monthly_revenue', mode='before')
    @classmethod
    def parse_revenue(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = v.replace('₹', '').replace(',', '').replace('Rs', '').replace('rupees', '').strip()
            try:
                return float(clean)
            except ValueError:
                return 0.0
        return 0.0

class CreditUnderwritingDecision(BaseModel):
    merchant_name: str = Field(default="Informal Merchant", description="Name of the merchant")
    credit_score: int = Field(default=600, ge=300, le=850, description="Credit score between 300 and 850")
    risk_tier: str = Field(default="Moderate", description="Risk Classification: 'Low', 'Moderate', 'High', or 'Severe'")
    recommended_loan_limit: float = Field(default=10000.0, ge=0.0, description="Recommended maximum loan principal")
    confidence_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Confidence in assessment (0.0 to 1.0)")
    discrepancy_flags: List[str] = Field(default_factory=list, description="List of identified fraud or mismatch flags")
    underwriting_rationale: str = Field(default="", description="Detailed explanation of credit decision")
    financial_metrics: FinancialAnalysis

    @field_validator('credit_score', mode='before')
    @classmethod
    def parse_score(cls, v):
        try:
            val = int(float(str(v).strip()))
            return max(300, min(850, val))
        except (ValueError, TypeError):
            return 600

    @field_validator('recommended_loan_limit', mode='before')
    @classmethod
    def parse_loan(cls, v):
        try:
            return float(str(v).replace('₹', '').replace(',', '').strip())
        except (ValueError, TypeError):
            return 10000.0
