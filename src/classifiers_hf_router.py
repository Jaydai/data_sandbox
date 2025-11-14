import os
import json
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

class HFRouterClassifier:
    """
    Classifier utilisant Hugging Face Router (format OpenAI)
    Compatible avec tous les modèles disponibles via le router
    """
    
    # Modèles disponibles via HF Router
    AVAILABLE_MODELS = {
        "gemma-9b": "google/gemma-2-9b-it:nebius",
        "llama-8b": "meta-llama/llama-3.1-8b-instruct:nebius",
        "mistral-7b": "mistralai/mistral-7b-instruct-v0.3:nebius",
        "qwen-7b": "qwen/qwen-2.5-7b-instruct:nebius",
    }
    
    def __init__(
        self, 
        model: str = "gemma-9b",
        api_key: Optional[str] = None
    ):
        """
        Args:
            model: Alias du modèle ou nom complet
            api_key: Token HF (ou dans .env sous HF_TOKEN)
        """
        load_dotenv()
        
        # Résoudre le token
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HF_TOKEN ou HUGGINGFACE_API_KEY doit être défini dans .env")
        
        # Résoudre le modèle
        if model in self.AVAILABLE_MODELS:
            self.model_name = self.AVAILABLE_MODELS[model]
        else:
            self.model_name = model
        
        # Client OpenAI configuré pour HF Router
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_key
        )
        
        logger.info(f"✅ HF Router initialisé avec {self.model_name}")
    
    def _call_model(
        self, 
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.1
    ) -> str:
        """
        Appel au modèle via HF Router (format OpenAI)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"❌ Erreur HF Router: {e}")
            return ""
    
    def _extract_json(self, text: str) -> str:
        """Extraire le JSON de la réponse"""
        if not text:
            return "{}"
        
        text = text.strip()
        
        # Enlever markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
        
        text = text.strip()
        
        # Extraire entre accolades
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        
        return text
    
    def classify_work_related(self, content: str, context: str = "") -> Dict:
        """Classifier Work / Non-Work"""
        
        messages = [
            {
                "role": "system",
                "content": """Tu es un expert en analyse de messages envoyés à des IA.
Détermine si un message est lié au TRAVAIL PROFESSIONNEL ou non.

TRAVAIL : activité professionnelle rémunérée, emails pros, rapports, analyse de données pour le travail, code professionnel.
NON-TRAVAIL : questions personnelles, loisirs, recettes, apprentissage personnel, divertissement.

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :
{"is_work": true, "confidence": "high", "reasoning": "explication brève"}"""
            },
            {
                "role": "user",
                "content": f"MESSAGE : {content[:1000]}\n{'CONTEXTE : ' + context if context else ''}\n\nRéponds uniquement avec le JSON :"
            }
        ]
        
        response = self._call_model(messages)
        json_str = self._extract_json(response)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Erreur JSON: {json_str[:200]}")
            return {
                "is_work": False, 
                "confidence": "low", 
                "reasoning": "Erreur de classification"
            }
    
    def classify_topic(self, content: str, context: str = "") -> Dict:
        """Classifier le topic"""
        
        messages = [
            {
                "role": "system",
                "content": """Tu es un expert en analyse de messages.
Classifie le SUJET PRINCIPAL parmi ces catégories :

WRITING, PRACTICAL_GUIDANCE, SEEKING_INFORMATION, TECHNICAL_HELP, MULTIMEDIA, SELF_EXPRESSION, OTHER

Réponds UNIQUEMENT avec un JSON valide :
{"topic": "WRITING", "sub_topic": "description", "confidence": "high"}"""
            },
            {
                "role": "user",
                "content": f"MESSAGE : {content[:1000]}\n{'CONTEXTE : ' + context if context else ''}\n\nJSON uniquement :"
            }
        ]
        
        response = self._call_model(messages)
        json_str = self._extract_json(response)
        
        try:
            return json.loads(json_str)
        except:
            logger.error(f"Erreur JSON: {json_str[:200]}")
            return {
                "topic": "OTHER", 
                "sub_topic": "unknown", 
                "confidence": "low"
            }
    
    def classify_intent(self, content: str, context: str = "") -> Dict:
        """Classifier l'intention"""
        
        messages = [
            {
                "role": "system",
                "content": """Tu es un expert en analyse d'intentions.

ASKING : Cherche des informations ou conseils pour prendre une décision
DOING : Demande de PRODUIRE quelque chose (texte, code, etc.)
EXPRESSING : Expression de sentiments sans attente d'action

Réponds UNIQUEMENT avec un JSON valide :
{"intent": "ASKING", "confidence": "high", "reasoning": "explication"}"""
            },
            {
                "role": "user",
                "content": f"MESSAGE : {content[:1000]}\n{'CONTEXTE : ' + context if context else ''}\n\nJSON uniquement :"
            }
        ]
        
        response = self._call_model(messages)
        json_str = self._extract_json(response)
        
        try:
            return json.loads(json_str)
        except:
            logger.error(f"Erreur JSON: {json_str[:200]}")
            return {
                "intent": "EXPRESSING", 
                "confidence": "low", 
                "reasoning": "Erreur"
            }
    
    def classify_complete(self, content: str, context: str = "") -> Dict:
        """Classification complète"""
        
        logger.info(f"🔍 Classification HF Router ({self.model_name}): {len(content)} caractères")
        
        # Tronquer si trop long
        content_truncated = content[:5000]
        context_truncated = context[:2000] if context else ""
        
        results = {
            "work": self.classify_work_related(content_truncated, context_truncated),
            "topic": self.classify_topic(content_truncated, context_truncated),
            "intent": self.classify_intent(content_truncated, context_truncated)
        }
        
        logger.info("✅ Classification terminée")
        return results


# Fonction helper
def classify_message_hf_router(
    content: str, 
    context: str = "", 
    model: str = "gemma-9b"
) -> Dict:
    """Classification rapide avec HF Router"""
    classifier = HFRouterClassifier(model=model)
    return classifier.classify_complete(content, context)