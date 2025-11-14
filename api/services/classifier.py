from typing import Dict, Optional, List
from src.classifiers_hf_router import HFRouterClassifier
import logging

logger = logging.getLogger(__name__)

class ClassifierService:
    """Service wrapper pour la classification"""
    
    def __init__(self, model: str = "gemma-9b"):
        """
        Args:
            model: Modèle à utiliser (gemma-9b, llama-8b, mistral-7b, qwen-7b)
        """
        self.classifier = HFRouterClassifier(model=model)
        self.model_name = self.classifier.model_name
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
        
        return self.classifier.classify_complete(content, context_str)