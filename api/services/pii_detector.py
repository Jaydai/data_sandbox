from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)

class PIIDetector:
    """Détecteur d'informations personnelles identifiables"""
    
    def __init__(self):
        try:
            self.analyzer = AnalyzerEngine()
            self._add_french_patterns()
            logger.info("✅ PII Detector initialisé")
        except Exception as e:
            logger.warning(f"⚠️  PII Detector non disponible: {e}")
            self.analyzer = None
    
    def _add_french_patterns(self):
        """Ajouter patterns français"""
        
        # Numéro sécurité sociale français
        secu_pattern = Pattern(
            name="french_secu",
            regex=r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
            score=0.85
        )
        secu_recognizer = PatternRecognizer(
            supported_entity="FR_SECU",
            patterns=[secu_pattern]
        )
        self.analyzer.registry.add_recognizer(secu_recognizer)
        
        # IBAN français
        iban_pattern = Pattern(
            name="french_iban",
            regex=r"\bFR\d{2}\s?(\d{4}\s?){5}\d{3}\b",
            score=0.85
        )
        iban_recognizer = PatternRecognizer(
            supported_entity="FR_IBAN",
            patterns=[iban_pattern]
        )
        self.analyzer.registry.add_recognizer(iban_recognizer)
    
    def detect(self, text: str) -> Dict:
        """Détecter les PII dans un texte"""
        
        if not self.analyzer:
            return {
                "has_pii": False,
                "pii_types": [],
                "entities": [],
                "risk_level": "unknown"
            }
        
        try:
            # Analyser avec Presidio
            results = self.analyzer.analyze(
                text=text,
                language="fr",
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                    "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
                    "FR_SECU", "FR_IBAN"
                ]
            )
            
            # Extraire les types
            pii_types = list(set([r.entity_type for r in results]))
            
            # Créer liste des entités
            entities = [
                {
                    "type": r.entity_type,
                    "text": text[r.start:r.end],
                    "score": round(r.score, 2)
                }
                for r in results
            ]
            
            # Calculer le risque
            risk_level = self._calculate_risk(pii_types)
            
            return {
                "has_pii": len(results) > 0,
                "pii_types": pii_types,
                "entities": entities,
                "risk_level": risk_level,
                "recommendations": self._get_recommendations(pii_types)
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur détection PII: {e}")
            return {
                "has_pii": False,
                "pii_types": [],
                "entities": [],
                "risk_level": "error"
            }
    
    def _calculate_risk(self, pii_types: List[str]) -> str:
        """Calculer le niveau de risque"""
        if not pii_types:
            return "none"
        
        critical = {"CREDIT_CARD", "FR_SECU", "FR_IBAN"}
        if any(pii in critical for pii in pii_types):
            return "critical"
        
        high_risk = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}
        if len([p for p in pii_types if p in high_risk]) >= 2:
            return "high"
        
        if len(pii_types) >= 1:
            return "medium"
        
        return "low"
    
    def _get_recommendations(self, pii_types: List[str]) -> List[str]:
        """Générer des recommandations"""
        recs = []
        
        if "CREDIT_CARD" in pii_types or "FR_IBAN" in pii_types:
            recs.append("⚠️ Données bancaires détectées - Ne jamais partager")
        
        if "FR_SECU" in pii_types:
            recs.append("⚠️ Numéro sécurité sociale - Hautement confidentiel")
        
        if "PERSON" in pii_types:
            recs.append("💡 Noms de personnes - Considérez l'anonymisation")
        
        if "EMAIL_ADDRESS" in pii_types or "PHONE_NUMBER" in pii_types:
            recs.append("💡 Coordonnées personnelles - Vérifiez le consentement")
        
        return recs