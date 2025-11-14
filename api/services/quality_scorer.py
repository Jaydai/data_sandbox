import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class QualityScorer:
    """Évaluateur de qualité des demandes"""
    
    def __init__(self):
        # Mots-clés pour détecter les patterns
        self.role_keywords = [
            "tu es", "vous êtes", "agis comme", "joue le rôle",
            "en tant que", "comme un", "expert en"
        ]
        
        self.context_keywords = [
            "contexte", "background", "sachant que", "étant donné",
            "dans le cadre de", "pour le projet"
        ]
        
        self.goal_keywords = [
            "je veux", "j'ai besoin", "objectif", "but",
            "pour que", "afin de", "dans le but de"
        ]
        
        logger.info("✅ Quality Scorer initialisé")
    
    def score(self, text: str, context: List[str] = None) -> Dict:
        """Évaluer la qualité d'une demande"""
        
        text_lower = text.lower()
        
        # Scores individuels
        clarity_score = self._score_clarity(text)
        context_score, has_context = self._score_context(text_lower, context)
        precision_score = self._score_precision(text)
        
        # Détection de patterns
        has_clear_role = any(kw in text_lower for kw in self.role_keywords)
        has_clear_goal = any(kw in text_lower for kw in self.goal_keywords)
        
        # Score global
        overall_score = (
            clarity_score * 0.3 +
            context_score * 0.3 +
            precision_score * 0.4
        )
        
        word_count = len(text.split())
        
        # Forces et faiblesses
        strengths = self._identify_strengths(
            clarity_score, context_score, precision_score,
            has_clear_role, has_clear_goal, has_context
        )
        
        weaknesses = self._identify_weaknesses(
            clarity_score, context_score, precision_score,
            has_clear_role, has_clear_goal, has_context
        )
        
        suggestions = self._generate_suggestions(weaknesses)
        
        return {
            "overall_score": round(overall_score, 1),
            "clarity_score": round(clarity_score, 1),
            "context_score": round(context_score, 1),
            "precision_score": round(precision_score, 1),
            "has_clear_role": has_clear_role,
            "has_context": has_context,
            "has_clear_goal": has_clear_goal,
            "word_count": word_count,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions
        }
    
    def _score_clarity(self, text: str) -> float:
        """Score de clarté (0-10)"""
        score = 5.0
        
        word_count = len(text.split())
        if 10 <= word_count <= 100:
            score += 2.0
        elif word_count < 5:
            score -= 2.0
        elif word_count > 200:
            score -= 1.0
        
        if any(p in text for p in ['.', '?', '!']):
            score += 1.0
        
        if text[0].isupper():
            score += 0.5
        
        if not re.search(r'\b(lol|mdr|stp|svp|pk)\b', text.lower()):
            score += 0.5
        
        return min(10.0, max(0.0, score))
    
    def _score_context(self, text_lower: str, context: List[str]) -> tuple:
        """Score de contexte"""
        score = 5.0
        has_context = False
        
        if any(kw in text_lower for kw in self.context_keywords):
            score += 3.0
            has_context = True
        
        if context and len(context) > 0:
            score += 2.0
            has_context = True
        
        return min(10.0, score), has_context
    
    def _score_precision(self, text: str) -> float:
        """Score de précision"""
        score = 5.0
        
        if re.search(r'\d+', text):
            score += 1.5
        
        if re.search(r'\b[A-Z][a-z]+\b', text):
            score += 1.5
        
        generic_words = ['chose', 'truc', 'machin', 'ça']
        generic_count = sum(1 for w in generic_words if w in text.lower())
        
        if generic_count == 0:
            score += 2.0
        elif generic_count > 2:
            score -= 2.0
        
        return min(10.0, max(0.0, score))
    
    def _identify_strengths(self, clarity, context, precision, 
                           has_role, has_goal, has_context) -> List[str]:
        strengths = []
        
        if clarity >= 8:
            strengths.append("✅ Demande claire et bien structurée")
        if context >= 8:
            strengths.append("✅ Contexte bien fourni")
        if precision >= 8:
            strengths.append("✅ Demande précise et détaillée")
        if has_role:
            strengths.append("✅ Rôle explicite pour l'IA")
        if has_goal:
            strengths.append("✅ Objectif clairement exprimé")
        
        return strengths
    
    def _identify_weaknesses(self, clarity, context, precision,
                            has_role, has_goal, has_context) -> List[str]:
        weaknesses = []
        
        if clarity < 5:
            weaknesses.append("⚠️ Manque de clarté")
        if context < 5 and not has_context:
            weaknesses.append("⚠️ Contexte insuffisant")
        if precision < 5:
            weaknesses.append("⚠️ Manque de précision")
        if not has_role:
            weaknesses.append("💡 Pas de rôle défini")
        if not has_goal:
            weaknesses.append("💡 Objectif pas explicite")
        
        return weaknesses
    
    def _generate_suggestions(self, weaknesses: List[str]) -> List[str]:
        suggestions = []
        
        for weakness in weaknesses:
            if "clarté" in weakness:
                suggestions.append("💡 Structurez en phrases complètes")
            if "Contexte" in weakness:
                suggestions.append("💡 Ajoutez du contexte")
            if "précision" in weakness:
                suggestions.append("💡 Soyez plus spécifique")
            if "rôle" in weakness:
                suggestions.append("💡 Définissez un rôle pour l'IA")
            if "Objectif" in weakness:
                suggestions.append("💡 Énoncez votre objectif")
        
        return list(set(suggestions))