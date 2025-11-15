#!/usr/bin/env python3
"""Select recommendation-related user or assistant messages and copy them to Supabase tables."""
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
    parser = argparse.ArgumentParser(description='Filtrer les messages utilisateurs et assistants liés aux recommandations')
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--source-key', required=True)
    parser.add_argument('--dest-url', required=True)
    parser.add_argument('--dest-key', required=True)
    parser.add_argument('--source-table', default='messages')
    parser.add_argument('--dest-table-user', default='user_recommendation_requests')
    parser.add_argument('--dest-table-assistant', default='assistant_recommendations')
    parser.add_argument('--engine', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--model', default='mistral-small:latest')
    parser.add_argument('--select-batch-size', type=int, default=1000)
    parser.add_argument('--insert-batch-size', type=int, default=200)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--process-user', action='store_true', help='Analyser les messages utilisateurs')
    parser.add_argument('--process-assistant', action='store_true', help='Analyser les messages assistants')
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


def fetch_batch(client: Client, table: str, start: int, end: int) -> List[Dict]:
    resp = client.table(table).select('*').range(start, end).execute()
    return resp.data or []


def upsert_records(client: Client, table: str, records: List[Dict]) -> None:
    if not records:
        return
    client.table(table).upsert(records).execute()


def evaluate_message(text: str, evaluator: RecommendationIntentEvaluator, role: str) -> Dict:
    prefix = 'Utilisateur' if role == 'user' else 'Assistant'
    preview = [{'role': prefix, 'content': text[:2000]}]
    return evaluator.evaluate(text[:100], preview)


def main() -> None:
    args = parse_args()
    if not args.process_user and not args.process_assistant:
        args.process_user = args.process_assistant = True

    source_client = create_client(args.source_url, args.source_key)
    dest_client = create_client(args.dest_url, args.dest_key)
    engine = create_engine(args.engine, args.model)
    evaluator = RecommendationIntentEvaluator(engine)

    insert_user: List[Dict] = []
    insert_assistant: List[Dict] = []
    total_processed = 0
    start = 0
    batch_size = args.select_batch_size

    logger.info('🚀 Démarrage analyse messages (roles: user=%s assistant=%s)', args.process_user, args.process_assistant)

    while True:
        end = start + batch_size - 1
        rows = fetch_batch(source_client, args.source_table, start, end)
        if not rows:
            break
        logger.info('📥 Bloc %s-%s (%s lignes)', start, end, len(rows))
        for row in rows:
            total_processed += 1
            if args.max_rows and total_processed > args.max_rows:
                rows = []
                break
            role = str(row.get('role', '')).lower()
            text = get_message_text(row)
            if not text:
                continue

            if role == 'user' and args.process_user:
                result = evaluate_message(text, evaluator, role)
                if result.get('is_recommendation'):
                    logger.info('✅ USER %s (%s)', row.get('message_provider_id'), result.get('reason'))
                    sanitized = sanitize_record(row)
                    sanitized['is_recommendation'] = True
                    sanitized['recommendation_reason'] = result.get('reason')
                    insert_user.append(sanitized)
                    if len(insert_user) >= args.insert_batch_size:
                        logger.info('⬆️  Envoi de %s user messages vers %s', len(insert_user), args.dest_table_user)
                        upsert_records(dest_client, args.dest_table_user, insert_user)
                        insert_user = []
            elif role == 'assistant' and args.process_assistant:
                result = evaluate_message(text, evaluator, role)
                if result.get('is_recommendation'):
                    logger.info('✅ ASSISTANT %s (%s)', row.get('message_provider_id'), result.get('reason'))
                    sanitized = sanitize_record(row)
                    sanitized['is_recommendation'] = True
                    sanitized['recommendation_reason'] = result.get('reason')
                    insert_assistant.append(sanitized)
                    if len(insert_assistant) >= args.insert_batch_size:
                        logger.info('⬆️  Envoi de %s assistant messages vers %s', len(insert_assistant), args.dest_table_assistant)
                        upsert_records(dest_client, args.dest_table_assistant, insert_assistant)
                        insert_assistant = []
        if args.max_rows and total_processed >= args.max_rows:
            break
        start += batch_size

    if insert_user:
        logger.info('⬆️  Envoi final de %s user messages vers %s', len(insert_user), args.dest_table_user)
        upsert_records(dest_client, args.dest_table_user, insert_user)
    if insert_assistant:
        logger.info('⬆️  Envoi final de %s assistant messages vers %s', len(insert_assistant), args.dest_table_assistant)
        upsert_records(dest_client, args.dest_table_assistant, insert_assistant)

    logger.info('🏁 Terminé - Messages scannés: %s', total_processed)


if __name__ == '__main__':
    main()
