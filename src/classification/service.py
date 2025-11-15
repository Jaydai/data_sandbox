"""High level classification service shared by the API and batch pipeline."""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from .engines import ClassificationEngine
from .prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    TOPIC_CLASSIFICATION_PROMPT,
    WORK_CLASSIFICATION_PROMPT,
    build_user_message,
)

logger = logging.getLogger(__name__)


class MessageClassifier:
    """Aggregate all classification tasks behind a single interface."""

    def __init__(
        self,
        engine: ClassificationEngine,
        *,
        max_content_chars: int = 5000,
        max_context_chars: int = 2000,
    ):
        self.engine = engine
        self.max_content_chars = max_content_chars
        self.max_context_chars = max_context_chars

    @property
    def engine_name(self) -> str:
        return self.engine.model_name

    def classify(self, content: str, context: Optional[str] = None) -> Dict:
        """Return work/topic/intent predictions in a single payload."""
        if not content:
            raise ValueError("content ne peut pas être vide")

        trimmed_content = self._trim(content, self.max_content_chars)
        trimmed_context = self._trim(context, self.max_context_chars) if context else ""

        logger.debug(
            "Classification request (model=%s, len=%s)",
            self.engine_name,
            len(trimmed_content),
        )

        return {
            "work": self._classify_with_prompt(
                WORK_CLASSIFICATION_PROMPT,
                trimmed_content,
                trimmed_context,
                default={"is_work": False, "confidence": "low", "reasoning": "Non classé"},
            ),
            "topic": self._classify_with_prompt(
                TOPIC_CLASSIFICATION_PROMPT,
                trimmed_content,
                trimmed_context,
                default={"topic": "OTHER", "sub_topic": "unknown", "confidence": "low"},
            ),
            "intent": self._classify_with_prompt(
                INTENT_CLASSIFICATION_PROMPT,
                trimmed_content,
                trimmed_context,
                default={"intent": "EXPRESSING", "confidence": "low", "reasoning": "Non classé"},
            ),
        }

    def _classify_with_prompt(
        self,
        system_prompt: str,
        content: str,
        context: str,
        *,
        default: Dict,
    ) -> Dict:
        user_message = build_user_message(content, context)
        raw = self.engine.complete(system_prompt, user_message)
        parsed = self._parse_response(raw)
        if not parsed:
            logger.warning("Réponse vide ou invalide pour le prompt %s", system_prompt.split("\n", 1)[0])
            return default
        return parsed

    @staticmethod
    def _trim(text: Optional[str], limit: int) -> str:
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit]

    @staticmethod
    def _parse_response(response: str) -> Dict:
        if not response:
            return {}

        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Impossible de parser la réponse: %s", cleaned[:200])
            return {}
