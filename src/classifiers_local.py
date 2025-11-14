import ollama
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class LocalMessageClassifier:
    """
    Classifier utilisant des modèles open-source locaux via Ollama
    """
    
    def __init__(self, model: str = "qwen2.5:14b"):
        """
        Args:
            model: Nom du modèle Ollama à utiliser
                   Options: "llama3.3:70b", "llama3.1:8b", "qwen2.5:14b"
        """
        self.model = model
        
        try:
            # Test simple : essayer de faire un appel au modèle
            logger.info(f"🔍 Vérification du modèle {model}...")
            
            test_response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': 'Test'}],
                options={'num_predict': 5}
            )
            
            logger.info(f"✅ Classifier initialisé avec {model}")
            
        except ollama.ResponseError as e:
            if "model" in str(e).lower() and "not found" in str(e).lower():
                logger.warning(f"⚠️  Modèle {model} non trouvé.")
                logger.info(f"⬇️  Téléchargement de {model}... (cela peut prendre quelques minutes)")
                ollama.pull(model)
                logger.info(f"✅ Modèle {model} téléchargé et prêt")
            else:
                logger.error(f"❌ Erreur: {e}")
                raise
        
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            raise
    
    def _call_ollama(
        self, 
        system_prompt: str, 
        user_message: str, 
        temperature: float = 0.0
    ) -> str:
        """
        Appel au modèle Ollama local
        
        Args:
            system_prompt: Instructions système
            user_message: Message à classifier
            temperature: Créativité (0 = déterministe)
        
        Returns:
            Réponse du modèle
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': user_message
                    }
                ],
                options={
                    'temperature': temperature,
                    'num_predict': 500  # Équivalent de max_tokens
                }
            )
            
            return response['message']['content']
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'appel au modèle: {e}")
            return ""
    
    def _clean_json_response(self, response: str) -> str:
        """Nettoie la réponse pour extraire le JSON"""
        response = response.strip()
        
        # Enlever les balises markdown
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        # Si pas de JSON trouvé, chercher entre accolades
        if not response.startswith('{'):
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                response = response[start:end+1]
        
        return response
    
    def classify_work_related(self, content: str, context: str = "") -> Dict:
        """
        Classifier Work / Non-Work
        """
        system_prompt = """Tu es un expert en analyse de messages envoyés à des IA.
Ta tâche est de déterminer si un message est lié au TRAVAIL PROFESSIONNEL ou non.

DÉFINITION DE "TRAVAIL" :
- Messages liés à une activité professionnelle rémunérée
- Emails professionnels, rapports, présentations
- Analyse de données, code pour le travail
- Communication avec collègues/clients
- Recherche d'informations pour des projets professionnels

PAS DU TRAVAIL :
- Questions personnelles (santé, loisirs, recettes)
- Apprentissage personnel sans lien avec le travail
- Divertissement, jeux, conversations sociales
- Aide aux devoirs (sauf si l'utilisateur est enseignant)

IMPORTANT : Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après.
Format exact :
{
    "is_work": true,
    "confidence": "high",
    "reasoning": "explication brève en une phrase"
}"""

        full_message = f"""MESSAGE À CLASSIFIER :
{content}

{f"CONTEXTE (messages précédents) : {context}" if context else ""}

Réponds uniquement avec le JSON, rien d'autre."""

        response = self._call_ollama(system_prompt, full_message)
        response = self._clean_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {response[:200]}")
            return {"is_work": False, "confidence": "low", "reasoning": "Erreur de classification"}
    
    def classify_topic(self, content: str, context: str = "") -> Dict:
        """
        Classifier le sujet principal du message
        """
        system_prompt = """Tu es un expert en analyse de messages envoyés à des IA.
Ta tâche est de classifier le SUJET PRINCIPAL du message.

CATÉGORIES PRINCIPALES (choisis UNE seule) :

1. WRITING : Rédaction, édition, traduction de texte
2. PRACTICAL_GUIDANCE : Conseils pratiques, tutorat, idées créatives
3. SEEKING_INFORMATION : Recherche d'informations factuelles
4. TECHNICAL_HELP : Programmation, maths, analyse de données
5. MULTIMEDIA : Création/analyse d'images ou autres médias
6. SELF_EXPRESSION : Conversations sociales, réflexions personnelles
7. OTHER : Autre ou ambigü

IMPORTANT : Réponds UNIQUEMENT avec un JSON valide.
Format exact :
{
    "topic": "WRITING",
    "sub_topic": "description plus précise",
    "confidence": "high"
}"""

        full_message = f"""MESSAGE À CLASSIFIER :
{content}

{f"CONTEXTE : {context}" if context else ""}

Réponds uniquement avec le JSON."""

        response = self._call_ollama(system_prompt, full_message)
        response = self._clean_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            logger.error(f"Erreur de parsing JSON: {response[:200]}")
            return {"topic": "OTHER", "sub_topic": "unknown", "confidence": "low"}
    
    def classify_intent(self, content: str, context: str = "") -> Dict:
        """
        Classifier l'intention : Asking / Doing / Expressing
        """
        system_prompt = """Tu es un expert en analyse d'intentions dans les messages.

DÉFINITIONS :

1. ASKING : Cherche des informations ou conseils pour prendre une décision
   Exemples : "Comment faire X?", "Quelle est la différence entre Y et Z?"

2. DOING : Demande de PRODUIRE quelque chose (texte, code, etc.)
   Exemples : "Écris un email", "Crée un tableau", "Traduis ce texte"

3. EXPRESSING : Expression de sentiments sans attente d'action
   Exemples : "Bonjour!", "Je suis content", salutations

IMPORTANT : Réponds UNIQUEMENT avec un JSON valide.
Format exact :
{
    "intent": "ASKING",
    "confidence": "high",
    "reasoning": "brève explication"
}"""

        full_message = f"""MESSAGE À CLASSIFIER :
{content}

{f"CONTEXTE : {context}" if context else ""}

Réponds uniquement avec le JSON."""

        response = self._call_ollama(system_prompt, full_message)
        response = self._clean_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            logger.error(f"Erreur de parsing JSON: {response[:200]}")
            return {"intent": "EXPRESSING", "confidence": "low", "reasoning": "Erreur"}
    
    def classify_complete(self, content: str, context: str = "") -> Dict:
        """
        Classification complète d'un message
        """
        logger.info(f"🔍 Classification complète du message ({len(content)} caractères)...")
        
        # Tronquer si trop long
        content_truncated = content[:5000] if len(content) > 5000 else content
        context_truncated = context[:2000] if len(context) > 2000 else context
        
        results = {
            "work": self.classify_work_related(content_truncated, context_truncated),
            "topic": self.classify_topic(content_truncated, context_truncated),
            "intent": self.classify_intent(content_truncated, context_truncated)
        }
        
        logger.info(f"✅ Classification terminée")
        return results


# Fonction helper
def classify_message_local(content: str, context: str = "", model: str = "qwen2.5:14b") -> Dict:
    """Classification rapide avec modèle local"""
    classifier = LocalMessageClassifier(model=model)
    return classifier.classify_complete(content, context)