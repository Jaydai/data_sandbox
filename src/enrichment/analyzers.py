"""Lightweight enrichment analyzers for domain, risk, and sentiment."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List


class DomainKeywordClassifier:
    """Map messages to business domains using keyword heuristics."""

    def __init__(self) -> None:
        self.domain_keywords: Dict[str, List[str]] = {
            "finance": ["budget", "invoice", "p&l", "forecast", "revenue", "expenses", "roi"],
            "legal": ["contract", "nda", "compliance", "regulation", "gdpr", "privacy"],
            "hr": ["recruitment", "performance review", "payroll", "benefits", "onboarding"],
            "engineering": ["deploy", "api", "bug", "refactor", "architecture", "database"],
            "marketing": ["campaign", "seo", "branding", "newsletter", "content plan"],
            "sales": ["pipeline", "prospect", "crm", "quota", "demo", "opportunity"],
            "operations": ["process", "logistics", "supply", "sla", "vendor"],
        }

    def classify(self, text: str) -> Dict:
        text_lower = text.lower()
        matches: Dict[str, int] = {}
        hits: Dict[str, List[str]] = {}
        for domain, keywords in self.domain_keywords.items():
            keywords_found = [kw for kw in keywords if kw in text_lower]
            if keywords_found:
                matches[domain] = len(keywords_found)
                hits[domain] = keywords_found
        if not matches:
            return {"domain": "other", "keywords": []}
        best_domain = max(matches, key=matches.get)
        return {"domain": best_domain, "keywords": hits.get(best_domain, [])}


class RiskHeuristicsAnalyzer:
    """Raise soft alerts for potentially sensitive business data."""

    def __init__(self) -> None:
        self.patterns = {
            "financial": re.compile(r"\b(?:\d+[\.,]?\d*)\s?(?:€|eur|usd|k|m|million)\b", re.I),
            "confidential": re.compile(r"\b(confidential|strictly private|internal use)\b", re.I),
            "customer": re.compile(r"\b(client|customer|account)\s?(?:id|number)\b", re.I),
        }
        self.keywords = [
            "roadmap",
            "merger",
            "acquisition",
            "salary",
            "compensation",
            "tariff",
            "pricing",
        ]

    def analyze(self, text: str, has_pii: bool = False) -> Dict:
        flags: List[str] = []
        for label, pattern in self.patterns.items():
            if pattern.search(text):
                flags.append(label)
        hits = [kw for kw in self.keywords if kw in text.lower()]
        flags.extend(hits)
        if has_pii:
            flags.append("pii_detected")

        unique_flags = list(dict.fromkeys(flags))
        if not unique_flags:
            level = "low"
        elif has_pii or any(flag in {"financial", "confidential"} for flag in unique_flags):
            level = "high"
        else:
            level = "medium"

        return {"risk_level": level, "flags": unique_flags}


class SentimentAnalyzer:
    """Simple lexicon-based sentiment scoring."""

    def __init__(self) -> None:
        self.positive = {
            "great",
            "thanks",
            "helpful",
            "awesome",
            "love",
            "perfect",
            "excellent",
            "happy",
            "appreciate",
        }
        self.negative = {
            "issue",
            "problem",
            "stuck",
            "error",
            "frustrated",
            "bad",
            "fail",
            "broken",
            "urgent",
        }

    def score(self, text: str) -> Dict:
        tokens = re.findall(r"[\w']+", text.lower())
        counts = Counter(tokens)
        pos = sum(counts[word] for word in self.positive)
        neg = sum(counts[word] for word in self.negative)
        total = pos + neg
        if total == 0:
            return {"label": "neutral", "score": 0.0}
        score = (pos - neg) / total
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(score, 3)}
