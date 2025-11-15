"""Prompt templates shared by every classification engine."""
from typing import Optional

WORK_CLASSIFICATION_PROMPT = """Tu es un expert en analyse de messages envoyés à des IA.
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

TOPIC_CLASSIFICATION_PROMPT = """Tu es un expert en analyse de messages envoyés à des IA.
Ta tâche est de classifier le SUJET PRINCIPAL du message.

CATÉGORIES PRINCIPALES (choisis UNE seule) :

1. WRITING : Rédaction, édition, traduction de texte
2. PRACTICAL_GUIDANCE : Conseils pratiques, tutorat, idées créatives
3. SEEKING_INFORMATION : Recherche d'informations factuelles
4. TECHNICAL_HELP : Programmation, maths, analyse de données
5. MULTIMEDIA : Création/analyse d'images ou autres médias
6. SELF_EXPRESSION : Conversations sociales, réflexions personnelles
7. OTHER : Autre ou ambigu

IMPORTANT : Réponds UNIQUEMENT avec un JSON valide.
Format exact :
{
    "topic": "WRITING",
    "sub_topic": "description plus précise",
    "confidence": "high"
}"""

INTENT_CLASSIFICATION_PROMPT = """Tu es un expert en analyse d'intentions dans les messages.

DÉFINITIONS :

1. ASKING : Cherche des informations ou conseils pour prendre une décision
2. DOING : Demande de PRODUIRE quelque chose (texte, code, etc.)
3. EXPRESSING : Expression de sentiments sans attente d'action

IMPORTANT : Réponds UNIQUEMENT avec un JSON valide.
Format exact :
{
    "intent": "ASKING",
    "confidence": "high",
    "reasoning": "brève explication"
}"""


def build_user_message(content: str, context: Optional[str] = None) -> str:
    """Generate a normalized user message shared by every prompt."""
    context_section = f"\nCONTEXTE : {context.strip()}" if context else ""
    return (
        "MESSAGE À CLASSIFIER :\n"
        f"{content.strip()}"
        f"{context_section}\n\n"
        "Réponds uniquement avec le JSON, rien d'autre."
    )
