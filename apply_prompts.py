#!/usr/bin/env python3
"""Apply arbitrary LLM prompts to Supabase rows and upsert results into a table."""
import argparse
import importlib
import logging
from typing import Dict, List, Optional, Set

import pandas as pd
from supabase import Client, create_client

from src.classification import HFRouterEngine, OllamaEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Appliquer des prompts LLM et enregistrer les résultats dans Supabase')
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--source-key', required=True)
    parser.add_argument('--source-table', default='messages')
    parser.add_argument('--dest-table', default='interesting_messages')
    parser.add_argument('--engine', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--model', default='qwen2.5:14b')
    parser.add_argument('--prompts', nargs='+', required=True, help='Noms de variables dans prompts.py (ex: PROMPT_1)')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--insert-batch-size', type=int, default=50)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--text-field', default='content', help='Champ texte à analyser')
    parser.add_argument('--start-date', help='Filtrer à partir de cette date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Filtrer jusqu\'à cette date (YYYY-MM-DD)')
    return parser.parse_args()


def create_engine(engine_name: str, model: str):
    engine_name = engine_name.lower()
    if engine_name == 'hf':
        return HFRouterEngine(model=model)
    if engine_name == 'ollama':
        return OllamaEngine(model=model)
    raise ValueError("engine doit être 'hf' ou 'ollama'")


def load_prompts(prompt_names: List[str]) -> Dict[str, str]:
    module = importlib.import_module('prompts')
    prompts = {}
    for name in prompt_names:
        if not hasattr(module, name):
            raise ValueError(f"Prompt {name} introuvable dans prompts.py")
        prompts[name] = getattr(module, name)
    return prompts


def fetch_existing_ids(client: Client, table: str) -> Set[str]:
    ids = set()
    offset = 0
    page_size = 1000
    while True:
        resp = client.table(table).select('id').range(offset, offset + page_size - 1).execute()
        data = resp.data or []
        if not data:
            break
        ids.update(str(row['id']) for row in data if row.get('id'))
        if len(data) < page_size:
            break
        offset += page_size
    return ids


def fetch_batch(
    client: Client,
    table: str,
    offset: int,
    limit: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict]:
    query = client.table(table).select('*').order('created_at', desc=False)
    if start_date:
        query = query.gte('created_at', start_date)
    if end_date:
        query = query.lte('created_at', end_date)
    resp = query.range(offset, offset + limit - 1).execute()
    return resp.data or []


def render_prompt(template: str, content: str) -> str:
    if '{content}' in template:
        return template.replace('{content}', content)
    return f"{template}\n\n{content}"


def call_prompt(engine, text: str, template: str) -> str:
    rendered = render_prompt(template, text)
    response = engine.complete(
        system_prompt="Tu es un assistant qui répond uniquement au format demandé.",
        user_message=rendered[:4000],
        temperature=0.0,
        max_tokens=400,
    )
    return response.strip()


def upsert_records(client: Client, table: str, records: List[Dict]) -> None:
    if not records:
        return
    client.table(table).upsert(records).execute()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    engine = create_engine(args.engine, args.model)
    client = create_client(args.source_url, args.source_key)

    existing_ids = fetch_existing_ids(client, args.dest_table)
    logger.info('IDs déjà présents dans %s: %s', args.dest_table, len(existing_ids))

    offset = 0
    processed = 0
    pending: List[Dict] = []

    while True:
        batch = fetch_batch(client, args.source_table, offset, args.batch_size, args.start_date, args.end_date)
        if not batch:
            break
        logger.info('Bloc %s-%s', offset, offset + len(batch) - 1)
        for row in batch:
            row_id = str(row.get('id'))
            if not row_id or row_id in existing_ids:
                continue
            text = str(row.get(args.text_field) or '')
            if not text.strip():
                continue
            result_data = {}
            for prompt_name, template in prompts.items():
                try:
                    result_data[prompt_name] = call_prompt(engine, text, template)
                except Exception as exc:
                    logger.warning('Prompt %s échoué sur id=%s: %s', prompt_name, row_id, exc)
                    result_data[prompt_name] = 'ERROR'
            record = {
                'id': row_id,
                'created_at': row.get('created_at'),
                'chat_provider_id': row.get('chat_provider_id'),
                args.text_field: text,
            }
            record.update(result_data)
            pending.append(record)
            existing_ids.add(row_id)
            processed += 1
            if len(pending) >= args.insert_batch_size:
                logger.info('⬆️  Upsert de %s lignes dans %s', len(pending), args.dest_table)
                upsert_records(client, args.dest_table, pending)
                pending = []
            if args.max_rows and processed >= args.max_rows:
                break
        if args.max_rows and processed >= args.max_rows:
            break
        offset += args.batch_size

    if pending:
        logger.info('⬆️  Upsert final de %s lignes dans %s', len(pending), args.dest_table)
        upsert_records(client, args.dest_table, pending)

    logger.info('Terminé. Nouvelles lignes traitées: %s', processed)


if __name__ == '__main__':
    main()
