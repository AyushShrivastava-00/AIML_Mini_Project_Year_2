"""
Pydantic Data Schemas for Micro-Merchant Informal Credit Assessment.
Defines strictly enforced, resilient JSON schemas with DSCR and EMI parameters.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class FinancialAnalysis(BaseModel):
    model_config = ConfigDict(extra="allow")

    store_category: str = Field(default="Grocery / Kirana (FMCG)", description="Retail enterprise category")
    estimated_daily_revenue: float = Field(default=0.0, description="Estimated daily revenue in INR")
    estimated_monthly_revenue: float = Field(default=0.0, description="Estimated 30-day gross revenue in INR")
    estimated_monthly_cogs: float = Field(default=0.0, description="Estimated cost of goods sold per month")
    net_monthly_margin: float = Field(default=0.0, description="Estimated net profit after COGS and rent")
    inventory_turnover_rate: float = Field(default=2.5, description="Monthly inventory turnover velocity")
    inventory_to_sales_ratio: str = Field(default="Aligned", description="Turnover status assessment")
    recommended_tenor_days: int = Field(default=120, description="Recommended loan repayment horizon in days")
    safe_daily_emi: float = Field(default=0.0, description="Maximum recommended daily debt service payment")

    @field_validator('inventory_turnover_rate', mode='before')
    @classmethod
    def parse_turnover(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = v.replace('x', '').replace('X', '').strip()
            try:
                return float(clean)
            except ValueError:
                return 2.5
        return 2.5

    @field_validator('estimated_daily_revenue', 'estimated_monthly_revenue', 'estimated_monthly_cogs', 'net_monthly_margin', 'safe_daily_emi', mode='before')
    @classmethod
    def parse_currency(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = v.replace('₹', '').replace(',', '').replace('Rs', '').replace('rupees', '').replace('रुपये', '').strip()
            try:
                return float(clean)
            except ValueError:
                return 0.0
        return 0.0

    @field_validator('inventory_to_sales_ratio', mode='before')
    @classmethod
    def parse_ratio_string(cls, v):
        return str(v).strip() if v else "Aligned"


class CreditUnderwritingDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    merchant_name: str = Field(default="Informal Merchant", description="Name of the merchant")
    credit_score: int = Field(default=600, ge=300, le=850, description="Credit score between 300 and 850")
    risk_tier: str = Field(default="Moderate", description="Risk Tier: 'Low', 'Moderate', 'High', or 'Severe'")
    recommended_loan_limit: float = Field(default=10000.0, ge=0.0, description="Recommended maximum loan principal")
    confidence_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Confidence in assessment (0.0 to 1.0)")
    discrepancy_flags: List[str] = Field(default_factory=list, description="List of fraud or mismatch flags")
    underwriting_rationale: str = Field(default="", description="Detailed underwriting narrative")
    financial_metrics: FinancialAnalysis
    engine_source: Optional[str] = Field(default="Multi-Tier AI Underwriting Engine", description="Source engine used")

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
            clean = str(v).replace('₹', '').replace(',', '').replace('Rs', '').strip()
            return max(0.0, float(clean))
        except (ValueError, TypeError):
            return 10000.0

    @field_validator('confidence_score', mode='before')
    @classmethod
    def parse_confidence(cls, v):
        try:
            if isinstance(v, str):
                v = v.replace('%', '').strip()
            val = float(v)
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.88

    @field_validator('discrepancy_flags', mode='before')
    @classmethod
    def parse_flags(cls, v):
        if isinstance(v, list):
            return [str(item).strip() for item in v if item]
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ["none", "[]", "no flags", "clean"]:
                return []
            return [clean]
        return []

    @field_validator('risk_tier', mode='before')
    @classmethod
    def parse_risk_tier(cls, v):
        if not v:
            return "Moderate"
        tier_str = str(v).strip().title()
        for valid in ["Low", "Moderate", "High", "Severe"]:
            if valid.lower() in tier_str.lower():
                return valid
        return "Moderate"
