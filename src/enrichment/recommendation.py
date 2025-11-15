"""LLM helper to detect recommendation-oriented messages."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from src.classification.engines import ClassificationEngine

logger = logging.getLogger(__name__)


class RecommendationIntentEvaluator:
    """Use an LLM to decide if messages involve recommendations."""

    # Prompt pour les messages UTILISATEUR (questions)
    USER_MESSAGE_PROMPT = """Détecte si ce message utilisateur demande une RECOMMANDATION, CONSEIL ou SUGGESTION de produit/service/solution.

🎯 INCLURE si l'utilisateur:
- Demande des suggestions (recommande, suggère, conseille, propose, idées)
- Compare des options (vs, ou, meilleur, top, différence entre)
- Cherche un produit/service (quel, quelle, je cherche, j'ai besoin)
- Veut des avis (vaut le coup, est-ce bien, opinion sur)
- Liste des options (top 5, meilleures, alternatives)

📋 DOMAINES (inclure):
- Tech: logiciels, apps, outils, frameworks, CRM
- Voyage: destinations, hôtels, restaurants, activités
- Culture: livres, films, séries, musique
- Shopping: produits, marques, services
- Lifestyle: recettes, cadeaux, mode, sport
- Pro: formations, méthodes, ressources

❌ EXCLURE:
- Questions factuelles (météo, dates, définitions)
- Théorie pure (concepts, histoire)
- Debug code sans mention d'outil
- Résumés ou analyses
- Politique/actualités

⚡ STRATÉGIE: En cas de doute → INCLURE

Message: {content}

JSON (sans markdown):
{{"is_recommendation_request": true/false, "confidence": "high/medium/low", "reason": "court"}}
"""

    # Prompt pour les messages ASSISTANT (réponses)
    ASSISTANT_MESSAGE_PROMPT = """Détecte si ce message assistant FOURNIT des RECOMMANDATIONS ou SUGGESTIONS concrètes.

🎯 INCLURE si le message:
- Recommande des produits/services spécifiques
- Liste des options avec avantages/inconvénients
- Compare plusieurs solutions
- Suggère des choix avec justifications
- Donne des avis sur des produits/services
- Propose des alternatives ou options

📋 SIGNES de recommandation:
- Noms de produits/marques/services mentionnés
- Listes numérotées ou à puces d'options
- "Je recommande", "Je suggère", "Tu pourrais essayer"
- Comparaisons: "X est mieux pour", "Y convient si"
- Critères de choix expliqués
- "Voici quelques options", "Voici des suggestions"

❌ EXCLURE:
- Explications théoriques sans recommandation
- Réponses factuelles pures
- Code sans mention d'outils spécifiques
- Définitions ou concepts généraux
- Refus de recommander

⚡ STRATÉGIE: En cas de doute → INCLURE

Message (premiers 500 caractères): {content}

JSON (sans markdown):
{{"contains_recommendation": true/false, "confidence": "high/medium/low", "reason": "court"}}
"""

    def __init__(self, engine: ClassificationEngine) -> None:
        self.engine = engine

    def evaluate_user_message(self, content: str) -> Dict:
        """
        Évalue si un message utilisateur demande des recommandations.
        
        Returns:
            Dict avec is_recommendation_request, confidence, reason
        """
        if not content or len(content.strip()) < 10:
            return {
                "is_recommendation_request": False,
                "confidence": "high",
                "reason": "message trop court"
            }
        
        # Prend les 500 premiers caractères pour l'analyse
        snippet = content[:500]
        prompt = self.USER_MESSAGE_PROMPT.format(content=snippet)
        
        try:
            response = self.engine.complete(
                system_prompt="Réponds uniquement en JSON valide. Sois permissif.",
                user_message=prompt,
                temperature=0.2,
                max_tokens=150,
            )
            
            cleaned = self._clean_json_response(response)
            result = json.loads(cleaned)
            
            # Validation
            if not isinstance(result, dict) or 'is_recommendation_request' not in result:
                raise ValueError("Format JSON invalide")
            
            logger.debug("USER: %s | conf=%s | %s", 
                        "✅" if result.get('is_recommendation_request') else "❌",
                        result.get('confidence'),
                        result.get('reason'))
            
            return result
            
        except Exception as exc:
            logger.warning("⚠️ Erreur eval user message: %s", str(exc)[:100])
            # Permissif en cas d'erreur
            return {
                "is_recommendation_request": True,
                "confidence": "low",
                "reason": "erreur - inclus"
            }

    def evaluate_assistant_message(self, content: str) -> Dict:
        """
        Évalue si un message assistant contient des recommandations.
        
        Returns:
            Dict avec contains_recommendation, confidence, reason
        """
        if not content or len(content.strip()) < 20:
            return {
                "contains_recommendation": False,
                "confidence": "high",
                "reason": "message trop court"
            }
        
        # Prend les 1000 premiers caractères pour l'analyse
        snippet = content[:1000]
        prompt = self.ASSISTANT_MESSAGE_PROMPT.format(content=snippet)
        
        try:
            response = self.engine.complete(
                system_prompt="Réponds uniquement en JSON valide. Sois permissif.",
                user_message=prompt,
                temperature=0.2,
                max_tokens=150,
            )
            
            cleaned = self._clean_json_response(response)
            result = json.loads(cleaned)
            
            # Validation
            if not isinstance(result, dict) or 'contains_recommendation' not in result:
                raise ValueError("Format JSON invalide")
            
            logger.debug("ASSISTANT: %s | conf=%s | %s", 
                        "✅" if result.get('contains_recommendation') else "❌",
                        result.get('confidence'),
                        result.get('reason'))
            
            return result
            
        except Exception as exc:
            logger.warning("⚠️ Erreur eval assistant message: %s", str(exc)[:100])
            # Permissif en cas d'erreur
            return {
                "contains_recommendation": True,
                "confidence": "low",
                "reason": "erreur - inclus"
            }

    def evaluate(self, content: str, messages: List[Dict[str, str]] = None, role: str = 'user') -> Dict:
        """
        Méthode de compatibilité avec l'ancien code.
        Redirige vers evaluate_user_message ou evaluate_assistant_message.
        """
        if role and role.lower() == 'assistant':
            result = self.evaluate_assistant_message(content)
            # Normalise la clé pour compatibilité
            return {
                "is_recommendation": result.get('contains_recommendation', False),
                "confidence": result.get('confidence'),
                "reason": result.get('reason')
            }
        else:
            result = self.evaluate_user_message(content)
            return {
                "is_recommendation": result.get('is_recommendation_request', False),
                "confidence": result.get('confidence'),
                "reason": result.get('reason')
            }

    @staticmethod
    def _clean_json_response(response: str) -> str:
        """Nettoie la réponse du LLM pour extraire le JSON."""
        cleaned = response.strip()
        
        # Retire les blocs markdown
        if '```' in cleaned:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            else:
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
        
        # Extrait le JSON s'il y a du texte autour
        json_match = re.search(r'\{[^{}]*(?:"is_recommendation_request"|"contains_recommendation")[^{}]*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)
        
        return cleaned.strip()