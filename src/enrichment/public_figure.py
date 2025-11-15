"""LLM-powered helper to decide if a person is a public figure."""
from __future__ import annotations

import functools
import logging
from typing import Optional

from src.classification.engines import ClassificationEngine

logger = logging.getLogger(__name__)


class PublicFigureChecker:
    """Use an LLM to reason about person names (real/private vs public)."""

    def __init__(self, engine: ClassificationEngine) -> None:
        self.engine = engine

    @functools.lru_cache(maxsize=512)
    def is_plausible_person(self, name: str) -> bool:
        question = (
            "Réponds par TRUE ou FALSE uniquement."
            "Le texte suivant ressemble-t-il à un nom complet de personne réelle (prénom + nom) ?\n"
            f"Nom : {name}\nRéponse :"
        )
        return self._ask_boolean(question)

    @functools.lru_cache(maxsize=512)
    def is_public_figure(self, name: str) -> bool:
        question = (
            "Réponds par TRUE ou FALSE uniquement."
            "La personne suivante est-elle largement reconnue comme personnalité publique (politique, artiste, athlète, dirigeant d'entreprise connu) ?\n"
            f"Nom : {name}\nRéponse :"
        )
        return self._ask_boolean(question)

    def _ask_boolean(self, prompt: str) -> bool:
        try:
            response = self.engine.complete(
                system_prompt="Tu es un vérificateur factuel.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=10,
            ).strip().lower()
        except Exception as exc:
            logger.warning("⚠️ Vérification impossible : %s", exc)
            return False

        if "true" in response:
            return True
        if "false" in response:
            return False
        return False
