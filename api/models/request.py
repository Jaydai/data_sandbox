# api/models/request.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class MessageClassificationRequest(BaseModel):
    """Requête de classification d'un message"""
    
    message_id: Optional[str] = Field(None, description="ID unique du message")
    user_id: str = Field(..., description="ID de l'utilisateur")
    organization_id: str = Field(..., description="ID de l'organisation")
    content: str = Field(..., description="Contenu du message à analyser")
    context: Optional[List[str]] = Field(None, description="Messages précédents pour contexte")
    model: Optional[str] = Field("gpt-4", description="Modèle IA utilisé")
    created_at: Optional[datetime] = Field(None, description="Date de création")
    
    # Options d'analyse
    detect_pii: bool = Field(True, description="Détecter les infos confidentielles")
    assess_quality: bool = Field(True, description="Évaluer la qualité de la demande")
    analyze_risks: bool = Field(True, description="Analyser les risques")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "organization_id": "org_456",
                "content": "Peux-tu analyser les données de ventes du Q4 ?",
                "context": ["Bonjour", "J'ai besoin d'aide"],
                "model": "gpt-4"
            }
        }


class BulkClassificationRequest(BaseModel):
    """Requête de classification en batch"""
    
    organization_id: str
    messages: List[MessageClassificationRequest]
    async_processing: bool = Field(False, description="Traitement asynchrone")