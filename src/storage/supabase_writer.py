"""Utility to push enriched results to a Supabase SQL table."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseResultWriter:
    """Persist enriched messages inside a dedicated Supabase table."""

    def __init__(
        self,
        *,
        table_name: Optional[str] = None,
        batch_size: int = 500,
        url_env: str = "SUPABASE_RESULTS_URL",
        key_env: str = "SUPABASE_RESULTS_KEY",
    ) -> None:
        load_dotenv()
        self.url = os.getenv(url_env)
        self.key = os.getenv(key_env)
        if not self.url or not self.key:
            raise ValueError(
                "SupabaseResultWriter: définissez SUPABASE_RESULTS_URL et SUPABASE_RESULTS_KEY"
            )

        self.table_name = table_name or os.getenv("SUPABASE_RESULTS_TABLE", "messages")
        self.batch_size = batch_size
        self.client: Client = create_client(self.url, self.key)
        logger.info("✅ SupabaseResultWriter prêt pour la table %s", self.table_name)

    def upsert_messages(self, records: Iterable[Dict[str, Any]]) -> None:
        """Upsert records in batches."""
        batch: List[Dict[str, Any]] = []
        for record in records:
            batch.append(record)
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []

        if batch:
            self._flush(batch)

    def _flush(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            logger.info("⬆️  Envoi de %s lignes vers %s", len(batch), self.table_name)
            self.client.table(self.table_name).upsert(batch).execute()
            logger.info("✅ Batch écrit avec succès")
        except Exception as exc:
            logger.error("❌ Impossible d'écrire dans Supabase: %s", exc)
            raise
