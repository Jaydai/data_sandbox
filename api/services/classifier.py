from typing import Dict, Optional, List
import logging

from src.classification import HFRouterEngine, MessageClassifier

logger = logging.getLogger(__name__)

class ClassifierService:
    """Service wrapper pour la classification"""
    
    def __init__(self, model: str = "gemma-9b", api_key: Optional[str] = None):
        """
        Args:
            model: Modèle à utiliser (gemma-9b, llama-8b, mistral-7b, qwen-7b)
        """
        engine = HFRouterEngine(model=model, api_key=api_key)
        self.engine = engine
        self.classifier = MessageClassifier(engine=engine)
        self.model_name = self.classifier.engine_name
        logger.info(f"✅ ClassifierService initialisé avec {self.model_name}")
    
    def classify(
        self, 
        content: str, 
        context: Optional[List[str]] = None
    ) -> Dict:
        """
        Classifier un message
        
        Args:
            content: Contenu du message
            context: Messages précédents (liste de strings)
        
        Returns:
            Dict avec work, topic, intent
        """
        # Convertir context en string
        context_str = " | ".join(context) if context else ""
        return self.classifier.classify(content, context_str)
