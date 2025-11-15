#!/usr/bin/env python3
"""Select chats likely about recommendations and copy them into another Supabase table."""
import argparse
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from supabase import Client, create_client
from tqdm import tqdm

from src.classification import HFRouterEngine, OllamaEngine
from src.enrichment import RecommendationIntentEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filtrer les chats contenant des recommandations produits/services')
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--source-key', required=True)
    parser.add_argument('--dest-url', required=True)
    parser.add_argument('--dest-key', required=True)
    parser.add_argument('--source-table', default='chats')
    parser.add_argument('--dest-table', default='interesting_chats')
    parser.add_argument('--engine', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--model', default='mistral-small:latest')
    parser.add_argument('--select-batch-size', type=int, default=1000)
    parser.add_argument('--insert-batch-size', type=int, default=200)
    parser.add_argument('--max-rows', type=int, default=None)
    return parser.parse_args()


def create_engine(engine_name: str, model: str):
    engine_name = engine_name.lower()
    if engine_name == 'hf':
        return HFRouterEngine(model=model)
    if engine_name == 'ollama':
        return OllamaEngine(model=model)
    raise ValueError("engine doit être 'hf' ou 'ollama'")


def sanitize_record(record: Dict) -> Dict:
    sanitized = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, (np.bool_, np.bool8)):
            sanitized[key] = bool(value)
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def get_chat_title(row: Dict) -> str:
    for key in ('chat_name', 'name', 'title'):
        value = row.get(key)
        if value:
            return str(value)
    metadata = row.get('metadata')
    if isinstance(metadata, dict):
        return metadata.get('name') or metadata.get('title') or ''
    if isinstance(metadata, str):
        try:
            meta = json.loads(metadata)
            return meta.get('name') or meta.get('title') or ''
        except Exception:
            return ''
    return ''


def fetch_batch(client: Client, table: str, start: int, end: int) -> List[Dict]:
    response = client.table(table).select('*').range(start, end).execute()
    return response.data or []


def upsert_records(client: Client, table: str, records: List[Dict]) -> None:
    if not records:
        return
    client.table(table).upsert(records).execute()


def main() -> None:
    args = parse_args()
    source_client = create_client(args.source_url, args.source_key)
    dest_client = create_client(args.dest_url, args.dest_key)

    engine = create_engine(args.engine, args.model)
    evaluator = RecommendationIntentEvaluator(engine)

    insert_buffer: List[Dict] = []
    total_processed = 0
    start = 0
    batch_size = args.select_batch_size

    logger.info('🚀 Démarrage - source=%s, destination=%s', args.source_table, args.dest_table)
    while True:
        end = start + batch_size - 1
        rows = fetch_batch(source_client, args.source_table, start, end)
        if not rows:
            break
        logger.info('📥 Bloc de %s chats (range %s-%s)', len(rows), start, end)
        for row in rows:
            total_processed += 1
            if args.max_rows and total_processed > args.max_rows:
                rows = []
                break
            title = get_chat_title(row)
            if not title:
                continue
            preview = [{'role': 'SYSTEM', 'content': title}]
            intent = evaluator.evaluate(title, preview)
            if not intent.get('is_recommendation'):
                logger.info('❌ Chat %s ignoré (raison=%s)', row.get('id'), row.get('title'))
                continue
            logger.info('✅ Chat %s marqué comme recommendation (%s)', row.get('id'), intent.get('reason'))
            sanitized = sanitize_record(row)
            sanitized['is_recommendation'] = True
            sanitized['recommendation_reason'] = intent.get('reason')
            insert_buffer.append(sanitized)
            if len(insert_buffer) >= args.insert_batch_size:
                logger.info('⬆️  Envoi de %s chats sélectionnés vers %s', len(insert_buffer), args.dest_table)
                upsert_records(dest_client, args.dest_table, insert_buffer)
                insert_buffer = []
        if args.max_rows and total_processed >= args.max_rows:
            break
        start += batch_size

    if insert_buffer:
        logger.info('⬆️  Envoi final de %s chats sélectionnés', len(insert_buffer))
        upsert_records(dest_client, args.dest_table, insert_buffer)

    logger.info('🏁 Terminé - Chats traités: %s', total_processed)


if __name__ == '__main__':
    main()
