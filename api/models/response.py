# api/models/response.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

class WorkClassification(BaseModel):
    is_work: bool
    confidence: ConfidenceLevel
    reasoning: str

class TopicClassification(BaseModel):
    topic: str
    sub_topic: str
    confidence: ConfidenceLevel

class IntentClassification(BaseModel):
    intent: str  # ASKING, DOING, EXPRESSING
    confidence: ConfidenceLevel
    reasoning: str

class PIIDetection(BaseModel):
    """Détection d'informations personnelles identifiables"""
    has_pii: bool
    pii_types: List[str] = Field(default_factory=list)
    entities_found: List[Dict] = Field(default_factory=list)
    risk_level: RiskLevel
    recommendations: List[str] = Field(default_factory=list)

class QualityScore(BaseModel):
    """Score de qualité de la demande"""
    overall_score: float = Field(..., ge=0, le=10)
    clarity_score: float = Field(..., ge=0, le=10)
    context_score: float = Field(..., ge=0, le=10)
    precision_score: float = Field(..., ge=0, le=10)
    
    has_clear_role: bool
    has_context: bool
    has_clear_goal: bool
    
    word_count: int
    
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

class RiskAnalysis(BaseModel):
    """Analyse des risques"""
    overall_risk: RiskLevel
    risk_factors: List[Dict] = Field(default_factory=list)
    
    data_leak_risk: RiskLevel
    compliance_risk: RiskLevel
    hallucination_risk: RiskLevel
    
    alerts: List[str] = Field(default_factory=list)
    mitigation_actions: List[str] = Field(default_factory=list)

class MessageClassificationResponse(BaseModel):
    """Réponse complète de classification"""
    
    message_id: Optional[str]
    user_id: str
    organization_id: str
    processed_at: datetime
    
    # Classifications de base
    work: WorkClassification
    topic: TopicClassification
    intent: IntentClassification
    
    # Analyses avancées
    pii_detection: Optional[PIIDetection] = None
    quality_score: Optional[QualityScore] = None
    risk_analysis: Optional[RiskAnalysis] = None
    
    # Métadonnées
    processing_time_ms: float
    model_used: str