"""Heuristic evaluation helpers for assistant responses."""
from __future__ import annotations

import re
from typing import Dict


class AssistantResponseEvaluator:
    """Evaluate assistant answers with simple heuristics."""

    disclaimer_pattern = re.compile(r"\b(as an ai|cannot comply|can't assist|not able to)", re.I)
    refusal_pattern = re.compile(r"\b(i cannot|i'm unable|i will not|cannot help)\b", re.I)

    def evaluate(self, prompt: str, response: str) -> Dict:
        prompt_len = len(prompt)
        response_len = len(response)
        length_ratio = response_len / prompt_len if prompt_len else 1.0

        contains_code = "```" in response or "def " in response
        bullet_count = response.count("\n-") + response.count("\n*")
        numbered_steps = len(re.findall(r"\n\d+\.\s", response))
        sections = response.lower().count("\n\n")

        actionability_score = min(
            1.0,
            (bullet_count + numbered_steps) * 0.2 + (1 if contains_code else 0) * 0.3 + sections * 0.1,
        )

        disclaimer = bool(self.disclaimer_pattern.search(response))
        refusal = bool(self.refusal_pattern.search(response))

        overlap = self._word_overlap(prompt, response)

        return {
            "prompt_length": prompt_len,
            "response_length": response_len,
            "length_ratio": round(length_ratio, 2),
            "contains_code": contains_code,
            "bullet_points": bullet_count,
            "numbered_steps": numbered_steps,
            "actionability_score": round(actionability_score, 2),
            "disclaimer_present": disclaimer,
            "potential_refusal": refusal,
            "keyword_overlap": round(overlap, 3),
        }

    @staticmethod
    def _word_overlap(prompt: str, response: str) -> float:
        prompt_words = {w for w in re.findall(r"[\w']+", prompt.lower()) if len(w) > 3}
        response_words = {w for w in re.findall(r"[\w']+", response.lower()) if len(w) > 3}
        if not prompt_words or not response_words:
            return 0.0
        intersect = prompt_words & response_words
        union = prompt_words | response_words
        return len(intersect) / len(union)
