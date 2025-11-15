#!/usr/bin/env python3
"""Select assistant recommendations, newest-first, and store them in Supabase."""
import argparse
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from supabase import Client, create_client

from src.classification import HFRouterEngine, OllamaEngine
from src.enrichment import RecommendationIntentEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filtrer les réponses assistants avec recommandations (ordre décroissant).')
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--source-key', required=True)
    parser.add_argument('--dest-url', required=True)
    parser.add_argument('--dest-key', required=True)
    parser.add_argument('--source-table', default='messages')
    parser.add_argument('--dest-table', default='assistant_recommendations')
    parser.add_argument('--start-date', help='Filtrer à partir de cette date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Filtrer jusqu\'à cette date (YYYY-MM-DD)')
    parser.add_argument('--engine', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--model', default='mistral-small:latest')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--insert-batch-size', type=int, default=50)
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


def get_message_text(row: Dict) -> str:
    content = row.get('content')
    if content:
        return str(content)
    metadata = row.get('metadata')
    if isinstance(metadata, dict):
        return metadata.get('content') or ''
    if isinstance(metadata, str):
        try:
            meta = json.loads(metadata)
            return meta.get('content') or ''
        except Exception:
            return ''
    return ''


def fetch_blocks_desc(client: Client, table: str, batch_size: int, max_rows: int = None,
                      start_date: str = None, end_date: str = None):
    start = 0
    fetched = 0
    while True:
        query = (
            client.table(table)
            .select('*')
            .eq('role', 'assistant')
            .order('created_at', desc=True)
            .range(start, start + batch_size - 1)
        )
        if start_date:
            query = query.gte('created_at', start_date)
        if end_date:
            query = query.lte('created_at', end_date)
        rows = query.execute().data
        if not rows:
            break
        yield rows
        fetched += len(rows)
        if max_rows and fetched >= max_rows:
            break
        start += batch_size


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

    buffer: List[Dict] = []
    total_processed = 0
    selected_chats = set()

    logger.info('🚀 Analyse des messages assistants (ordre décroissant) vers %s', args.dest_table)

    for rows in fetch_blocks_desc(
        source_client,
        args.source_table,
        args.batch_size,
        args.max_rows,
        args.start_date,
        args.end_date,
    ):
        logger.info('📥 Bloc de %s messages', len(rows))
        for row in rows:
            total_processed += 1
            chat_id = row.get('chat_provider_id')
            if chat_id in selected_chats:
                continue
            text = get_message_text(row)
            if not text:
                continue
            preview = [{'role': 'assistant', 'content': text[:2000]}]
            intent = evaluator.evaluate(text[:100], preview)
            if not intent.get('is_recommendation'):
                logger.debug('❌ Message %s ignoré (%s)', row.get('message_provider_id'), intent.get('reason'))
                continue
            logger.info('✅ Message %s sélectionné (%s)', row.get('message_provider_id'), intent.get('reason'))
            sanitized = sanitize_record(row)
            sanitized['is_recommendation'] = True
            sanitized['recommendation_reason'] = intent.get('reason')
            buffer.append(sanitized)
            if chat_id:
                selected_chats.add(chat_id)
            if len(buffer) >= args.insert_batch_size:
                logger.info('⬆️  Envoi de %s messages vers %s', len(buffer), args.dest_table)
                upsert_records(dest_client, args.dest_table, buffer)
                buffer = []
        if args.max_rows and total_processed >= args.max_rows:
            break

    if buffer:
        logger.info('⬆️  Envoi final de %s messages vers %s', len(buffer), args.dest_table)
        upsert_records(dest_client, args.dest_table, buffer)

    logger.info('🏁 Terminé - Messages analysés: %s', total_processed)


if __name__ == '__main__':
    main()
