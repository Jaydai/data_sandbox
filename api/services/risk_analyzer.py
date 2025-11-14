from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Analyseur de risques liés à l'utilisation de l'IA"""
    
    def __init__(self):
        logger.info("✅ Risk Analyzer initialisé")
    
    def analyze(
        self, 
        content: str,
        classification: Dict,
        pii_detection: Dict,
        quality_score: Dict
    ) -> Dict:
        """
        Analyser les risques d'un message
        
        Args:
            content: Contenu du message
            classification: Résultat de classification
            pii_detection: Résultat détection PII
            quality_score: Score de qualité
        
        Returns:
            Analyse de risques complète
        """
        
        risk_factors = []
        alerts = []
        mitigation_actions = []
        
        # 1. Risque de fuite de données
        data_leak_risk = "none"
        if pii_detection.get("has_pii"):
            pii_risk = pii_detection.get("risk_level", "medium")
            if pii_risk in ["critical", "high"]:
                data_leak_risk = pii_risk
                risk_factors.append({
                    "type": "data_leak",
                    "severity": pii_risk,
                    "description": f"Informations sensibles détectées: {', '.join(pii_detection.get('pii_types', []))}"
                })
                alerts.append("🚨 Données sensibles dans le message")
                mitigation_actions.append("Anonymiser les données avant envoi")
        
        # 2. Risque de conformité
        compliance_risk = "low"
        if classification.get("work", {}).get("is_work"):
            if pii_detection.get("has_pii"):
                compliance_risk = "high"
                risk_factors.append({
                    "type": "compliance",
                    "severity": "high",
                    "description": "Données professionnelles sensibles - RGPD/confidentialité"
                })
                alerts.append("⚠️ Risque de non-conformité RGPD")
                mitigation_actions.append("Vérifier les politiques de confidentialité")
        
        # 3. Risque d'hallucination
        hallucination_risk = "low"
        if quality_score.get("precision_score", 10) < 5:
            hallucination_risk = "medium"
            risk_factors.append({
                "type": "hallucination",
                "severity": "medium",
                "description": "Demande imprécise - risque de réponse erronée"
            })
            mitigation_actions.append("Préciser davantage la demande")
        
        if not quality_score.get("has_context"):
            if hallucination_risk == "medium":
                hallucination_risk = "high"
            risk_factors.append({
                "type": "hallucination",
                "severity": "medium",
                "description": "Manque de contexte - risque d'interprétation erronée"
            })
            mitigation_actions.append("Fournir plus de contexte")
        
        # 4. Risque global
        overall_risk = self._calculate_overall_risk(
            data_leak_risk,
            compliance_risk,
            hallucination_risk
        )
        
        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "data_leak_risk": data_leak_risk,
            "compliance_risk": compliance_risk,
            "hallucination_risk": hallucination_risk,
            "alerts": alerts,
            "mitigation_actions": mitigation_actions
        }
    
    def _calculate_overall_risk(
        self, 
        data_leak: str, 
        compliance: str, 
        hallucination: str
    ) -> str:
        """Calculer le risque global"""
        
        risk_levels = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        
        max_risk = max(
            risk_levels.get(data_leak, 0),
            risk_levels.get(compliance, 0),
            risk_levels.get(hallucination, 0)
        )
        
        for level, value in risk_levels.items():
            if value == max_risk:
                return level
        
        return "low"