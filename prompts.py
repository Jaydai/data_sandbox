"""Example prompts for apply_prompts.py"""
PROMPT_1 = """Tu es un classifieur PII.

Analyse le message et détecte s'il contient des informations permettant d’identifier une personne.

Catégories :
- DIRECT_IDENTIFIER : nom complet réel, email, téléphone, adresse, identifiants de compte, numéro officiel, pseudo unique lié à une plateforme.
- INDIRECT_IDENTIFIER : combinaison précise pouvant identifier quelqu’un (poste rare + entreprise + localisation, date de naissance complète + localisation, etc.).
- NO_PERSONAL_DATA : rien d’identifiable.

Tâches :
1. Dire si le message contient des données personnelles.
2. Identifier les types : DIRECT_IDENTIFIER, INDIRECT_IDENTIFIER ou vide.
3. Donner un niveau de risque : NONE, LOW, MEDIUM, HIGH.
4. Donner un confidence_score (0–1) + confidence_label (LOW, MEDIUM, HIGH).
5. Extraire les éléments pertinents.

Réponds UNIQUEMENT avec ce JSON :

{
  "contains_personal_data": boolean,
  "personal_data_types": string[],
  "identification_risk": "NONE" | "LOW" | "MEDIUM" | "HIGH",
  "confidence_score": number,
  "confidence_label": "LOW" | "MEDIUM" | "HIGH",
  "extracted_elements": [
    {
      "text": string,
      "type": "DIRECT_IDENTIFIER" | "INDIRECT_IDENTIFIER",
      "reason": string
    }
  ],
  "global_reasoning": string
}

Si aucun élément identifiable : contains_personal_data=false, risk=NONE.
Message : {content}"""


PROMPT_2 = """Tu es un classifieur de contenus sensibles ("touchy").

Détecte si le message aborde un sujet délicat ou sensible.

Catégories :
- Psychologie : anxiété, dépression, traumatisme, relations toxiques, thérapie.
- Sexe : sexualité, consentement, intimité.
- Crime & illégal : vol, fraude, hacking, drogue, police, violence, armes.
- Violence : harcèlement, agressions, conflits.
- Santé sensible : maladies graves, examens intimes, traitements lourds.
- Ethique : manipulation, mensonges, dilemmes moraux.

Niveau :
- LOW : mention générale.
- MEDIUM : mention personnelle explicite.
- HIGH : sujet sensible avec détresse, danger ou situation complexe.

Réponds UNIQUEMENT avec ce JSON :

{
  "is_touchy": boolean,
  "touchy_categories": string[],
  "sensitivity_level": "LOW" | "MEDIUM" | "HIGH",
  "confidence_score": number,
  "confidence_label": "LOW" | "MEDIUM" | "HIGH",
  "extracted_elements": [
    {
      "text": string,
      "category": string,
      "reason": string
    }
  ],
  "global_reasoning": string
}

Si rien de sensible : is_touchy=false.
Message : {content}"""

