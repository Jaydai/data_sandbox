"""LLM-based evaluator for assistant/user chats."""
from __future__ import annotations

import json
import logging
from typing import Dict, List

from src.classification.engines import ClassificationEngine

logger = logging.getLogger(__name__)


class ChatQualityEvaluator:
    """Ask an LLM to rate conversation quality."""

    def __init__(self, engine: ClassificationEngine) -> None:
        self.engine = engine

    def evaluate(self, conversation: List[Dict[str, str]]) -> Dict:
        transcript = self._format_conversation(conversation)
        prompt = (
            "Tu es un auditeur qualité pour des conversations IA."
            "Analyse la discussion suivante et retourne un JSON avec ce format exact :\n"
            "{\n"
            "  \"assistant_fulfilled\": true/false,\n"
            "  \"clarification_needed\": true/false,\n"
            "  \"quality\": \"high|medium|low\",\n"
            "  \"notes\": \"phrase courte\"\n"
            "}\n"
            "Conversation :\n"
            f"{transcript}\n"
            "Réponds uniquement avec le JSON."
        )
        try:
            response = self.engine.complete(
                system_prompt="Tu évalues la qualité des chats.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=200,
            )
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip('`')
            return json.loads(cleaned)
        except Exception as exc:
            logger.warning("⚠️ Impossible d'obtenir l'évaluation du chat: %s", exc)
            return {
                "assistant_fulfilled": False,
                "clarification_needed": False,
                "quality": "unknown",
                "notes": "evaluation_failed",
            }

    @staticmethod
    def _format_conversation(conversation: List[Dict[str, str]]) -> str:
        lines = []
        for turn in conversation[:12]:  # limiter
            role = turn.get('role', 'user')
            content = str(turn.get('content', ''))
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)
