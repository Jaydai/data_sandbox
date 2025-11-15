from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class PIIDetector:
    """Détecteur d'informations personnelles identifiables"""

    def __init__(self, public_figure_checker: Optional[object] = None):
        try:
            self.analyzer = AnalyzerEngine()
            self.french_recognizers = self._build_french_patterns()
            logger.info("✅ PII Detector initialisé")
        except Exception as exc:
            logger.warning(f"⚠️  PII Detector non disponible: {exc}")
            self.analyzer = None
            self.french_recognizers = []
        self.public_figure_checker = public_figure_checker
        self.name_pattern = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b")
        self.common_first_names = self._build_common_first_names()

    def _build_french_patterns(self) -> List[PatternRecognizer]:
        """Construire des recognizers regex indépendants du pipeline FR."""
        recognizers: List[PatternRecognizer] = []

        secu_pattern = Pattern(
            name="french_secu",
            regex=r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
            score=0.85,
        )
        recognizers.append(
            PatternRecognizer(
                supported_entity="FR_SECU",
                patterns=[secu_pattern],
            )
        )

        iban_pattern = Pattern(
            name="french_iban",
            regex=r"\bFR\d{2}\s?(\d{4}\s?){5}\d{3}\b",
            score=0.85,
        )
        recognizers.append(
            PatternRecognizer(
                supported_entity="FR_IBAN",
                patterns=[iban_pattern],
            )
        )

        return recognizers

    @staticmethod
    def _build_common_first_names() -> set:
        return {
            "alex", "alice", "andrew", "anna", "antoine", "arthur", "ben", "benjamin",
            "camille", "caroline", "charles", "charlotte", "claire", "daniel", "david",
            "emilie", "emma", "ethan", "eva", "florian", "francois", "gabriel", "george",
            "helene", "henry", "isabelle", "jacob", "jean", "jeanne", "jessica", "john",
            "joseph", "julia", "julien", "laura", "louis", "lucas", "marc", "marie",
            "martin", "maxime", "michael", "michel", "nathan", "nicole", "olivier", "paul",
            "peter", "philippe", "roger", "sarah", "simon", "sophia", "sophie", "stephane",
            "thomas", "victor", "vincent", "william",
        }

    def detect(self, text: str) -> Dict:
        """Détecter les PII dans un texte"""

        if not self.analyzer:
            return {
                "has_pii": False,
                "pii_types": [],
                "entities": [],
                "risk_level": "unknown",
            }

        try:
            general_entities = [
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "IBAN_CODE",
                "IP_ADDRESS",
            ]
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=general_entities,
            )

            for recognizer in self.french_recognizers:
                fr_results = recognizer.analyze(
                    text=text,
                    entities=recognizer.supported_entities,
                    nlp_artifacts=None,
                )
                results.extend(fr_results)

            private_names, public_names = self._classify_names(text)
            for name, start, end in private_names:
                results.append(
                    RecognizerResult(
                        entity_type="PERSON_PRIVATE",
                        start=start,
                        end=end,
                        score=0.7,
                    )
                )

            pii_types = list({r.entity_type for r in results})
            entities = [
                {
                    "type": r.entity_type,
                    "text": text[r.start : r.end],
                    "score": round(r.score, 2),
                }
                for r in results
            ]

            risk_level = self._calculate_risk(pii_types)

            return {
                "has_pii": bool(results),
                "pii_types": pii_types,
                "entities": entities,
                "risk_level": risk_level,
                "private_names": [n for n, _, _ in private_names],
                "public_names": [n for n in public_names],
                "recommendations": self._get_recommendations(pii_types),
            }

        except Exception as exc:
            logger.error(f"❌ Erreur détection PII: {exc}")
            return {
                "has_pii": False,
                "pii_types": [],
                "entities": [],
                "risk_level": "error",
            }

    def _calculate_risk(self, pii_types: List[str]) -> str:
        if not pii_types:
            return "none"

        critical = {"CREDIT_CARD", "FR_SECU", "FR_IBAN"}
        if any(entity in critical for entity in pii_types):
            return "critical"

        high_risk = {"EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON_PRIVATE"}
        if len([entity for entity in pii_types if entity in high_risk]) >= 2:
            return "high"

        if pii_types:
            return "medium"

        return "low"

    def _get_recommendations(self, pii_types: List[str]) -> List[str]:
        recs: List[str] = []

        if "CREDIT_CARD" in pii_types or "FR_IBAN" in pii_types:
            recs.append("⚠️ Données bancaires détectées - Ne jamais partager")

        if "FR_SECU" in pii_types:
            recs.append("⚠️ Numéro sécurité sociale - Hautement confidentiel")

        if "PERSON_PRIVATE" in pii_types:
            recs.append("💡 Identité privée détectée - anonymiser avant partage")

        if "EMAIL_ADDRESS" in pii_types or "PHONE_NUMBER" in pii_types:
            recs.append("💡 Coordonnées personnelles - Vérifiez le consentement")

        return recs

    def _classify_names(self, text: str):
        private = []
        public = []
        if not self.public_figure_checker:
            return private, public

        for match in self.name_pattern.finditer(text):
            name = match.group(0)
            first = name.split()[0].lower()
            if first not in self.common_first_names:
                continue
            if not self.public_figure_checker.is_plausible_person(name):
                continue
            if self.public_figure_checker.is_public_figure(name):
                public.append(name)
            else:
                private.append((name, match.start(), match.end()))
        return private, public
