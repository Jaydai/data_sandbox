#!/usr/bin/env python3
"""Detect recommendation-oriented chats and persist enriched chats/messages."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.classification import HFRouterEngine, OllamaEngine
from src.data_loader import SupabaseDataLoader
from src.enrichment import RecommendationIntentEvaluator
from src.storage import SupabaseResultWriter


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def create_engine(engine_name: str, model: str):
    engine_name = engine_name.lower()
    if engine_name == "hf":
        return HFRouterEngine(model=model)
    if engine_name == "ollama":
        return OllamaEngine(model=model)
    raise ValueError("engine doit être 'hf' ou 'ollama'")


def detect_garbage(first_message: Optional[str]) -> Dict[str, str]:
    if not first_message:
        return {"is_garbage": True, "reason": "empty_message"}
    text = first_message.strip()
    reasons: List[str] = []
    if len(text) < 8:
        reasons.append("too_short")
    if not any(c.isalpha() for c in text):
        reasons.append("no_letters")
    if text.endswith("...") or text.endswith(","):
        reasons.append("incomplete_sentence")
    return {"is_garbage": bool(reasons), "reason": ",".join(reasons)}


def normalize_timestamp(value):
    if value is None or pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).isoformat()
    except Exception:
        return str(value)


def get_chat_name(chat_row: pd.Series) -> str:
    for key in ("chat_name", "name", "title"):
        value = chat_row.get(key)
        if value:
            return str(value)
    metadata = chat_row.get('metadata')
    if isinstance(metadata, dict):
        return metadata.get('name') or metadata.get('title') or ''
    if isinstance(metadata, str):
        try:
            meta_dict = json.loads(metadata)
            return meta_dict.get('name') or meta_dict.get('title') or ''
        except Exception:
            return ''
    return ''


def conversation_preview(messages: pd.DataFrame) -> List[Dict[str, str]]:
    preview = []
    for _, row in messages.head(6).iterrows():
        preview.append({'role': row.get('role', ''), 'content': str(row.get('content', ''))})
    return preview


def serialize_chat_record(record: Dict) -> Dict:
    serialized = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            serialized[key] = normalize_timestamp(value)
        else:
            serialized[key] = value
    return serialized


def serialize_message_row(row: pd.Series, chat_record: Dict) -> Dict:
    return {
        'id': row.get('id'),
        'created_at': normalize_timestamp(row.get('created_at')),
        'user_id': row.get('user_id'),
        'chat_provider_id': row.get('chat_provider_id'),
        'message_provider_id': row.get('message_provider_id'),
        'role': row.get('role'),
        'model': row.get('model'),
        'parent_message_provider_id': row.get('parent_message_provider_id'),
        'tools': row.get('tools'),
        'content': row.get('content'),
        'recommendation_chat_id': chat_record.get('chat_id'),
    }


def load_chats_by_mode(loader: SupabaseDataLoader, args: argparse.Namespace) -> pd.DataFrame:
    if args.mode == 'date':
        if not args.date:
            raise SystemExit('--date est requis en mode date')
        folder = f"date={args.date}"
        df = loader.load_chats_in_folder(folder, subfolder=args.chats_subfolder)
        if args.chat_fraction < 1.0 and not df.empty:
            df = df.sample(frac=args.chat_fraction, random_state=42)
        return df
    if args.mode == 'sample':
        return load_sample_chats(loader, args.sample_size, args.chats_subfolder)
    if args.mode == 'user':
        if not args.user_id:
            raise SystemExit('--user-id requis en mode user')
        return load_user_chats(loader, args.user_id, args.chats_subfolder)
    return pd.DataFrame()


def load_sample_chats(loader: SupabaseDataLoader, sample_size: int, subfolder: str) -> pd.DataFrame:
    folders = loader.list_date_folders(subfolder)
    collected: List[pd.DataFrame] = []
    for folder in folders:
        df = loader.load_chats_in_folder(folder, subfolder=subfolder)
        if df.empty:
            continue
        collected.append(df)
        combined = pd.concat(collected, ignore_index=True)
        if len(combined) >= sample_size:
            return combined.sample(n=sample_size, random_state=42)
    if not collected:
        return pd.DataFrame()
    combined = pd.concat(collected, ignore_index=True)
    return combined.sample(n=min(sample_size, len(combined)), random_state=42)


def load_user_chats(loader: SupabaseDataLoader, user_id: str, subfolder: str) -> pd.DataFrame:
    folders = loader.list_date_folders(subfolder)
    data = []
    for folder in folders:
        df = loader.load_chats_in_folder(folder, subfolder=subfolder)
        if df.empty or 'user_id' not in df.columns:
            continue
        subset = df[df['user_id'] == user_id]
        if subset.empty:
            continue
        data.append(subset)
    if not data:
        return pd.DataFrame()
    return pd.concat(data, ignore_index=True)


def process_chat_batch(
    chats_df: pd.DataFrame,
    loader: SupabaseDataLoader,
    args: argparse.Namespace,
    evaluator: RecommendationIntentEvaluator,
    chats_writer: Optional[SupabaseResultWriter],
    messages_writer: Optional[SupabaseResultWriter],
    output_dir: Path,
):
    if chats_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: List[Dict] = []
    total = args.max_chats or len(chats_df)
    iter_df = chats_df.head(total)
    for _, chat_row in tqdm(iter_df.iterrows(), total=len(iter_df), desc=f'Chats {output_dir.name}'):
        chat_id = chat_row.get('chat_provider_id') or chat_row.get('id')
        date_folder = chat_row.get('date_folder')
        if not chat_id:
            continue
        messages_df = loader.load_chat_messages(chat_id, date_folder, subfolder=args.messages_subfolder)
        if messages_df.empty:
            continue
        chat_name = get_chat_name(chat_row)
        intent = evaluator.evaluate(chat_name, conversation_preview(messages_df))
        if not intent.get('is_recommendation'):
            continue
        first_user = messages_df[messages_df['role'] == 'user'].head(1)
        first_content = str(first_user.iloc[0]['content']) if not first_user.empty else ''
        garbage = detect_garbage(first_content)
        if garbage['is_garbage']:
            continue
        chat_record = {
            'chat_id': chat_row.get('id'),
            'chat_provider_id': chat_id,
            'user_id': chat_row.get('user_id'),
            'chat_name': chat_name,
            'started_at': normalize_timestamp(chat_row.get('created_at')),
            'messages_count': len(messages_df),
            'is_recommendation': True,
            'recommendation_reason': intent.get('reason'),
            'first_user_message': first_content,
            'is_garbage': False,
        }
        accepted.append(chat_record)
        if chats_writer:
            chats_writer.upsert_messages([serialize_chat_record(chat_record)])
        if messages_writer:
            payload = [serialize_message_row(row, chat_record) for _, row in messages_df.iterrows()]
            if payload:
                messages_writer.upsert_messages(payload)

    if not accepted:
        print(f"Aucun chat compatible pour {output_dir.name}")
        return
    df = pd.DataFrame(accepted)
    output_path = output_dir / 'recommendation_chats.parquet'
    if args.overwrite or not output_path.exists():
        df.to_parquet(output_path, index=False)
        print(f"Résultats sauvegardés dans {output_path}")
    else:
        print(f"Fichier déjà présent pour {output_dir.name} (utiliser --overwrite)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Détecter les chats de recommandation')
    parser.add_argument('--mode', choices=['date', 'sample', 'user', 'all_dates'], default='date')
    parser.add_argument('--date', help='Date YYYY-MM-DD (mode date)')
    parser.add_argument('--chat-fraction', type=float, default=1.0)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--user-id', help='Utilisateur ciblé (mode user)')
    parser.add_argument('--model', default='mistral-small:latest')
    parser.add_argument('--engine', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--chats-subfolder', default='chats')
    parser.add_argument('--messages-subfolder', default='messages')
    parser.add_argument('--max-chats', type=int, default=None)
    parser.add_argument('--store-supabase', action='store_true')
    parser.add_argument('--chats-table', default='enriched_chats')
    parser.add_argument('--messages-table', default='enriched_messages')
    parser.add_argument('--supabase-batch-size', type=int, default=100)
    parser.add_argument('--output-dir', default='data/processed/recommendations')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    loader = SupabaseDataLoader()
    engine = create_engine(args.engine, args.model)
    evaluator = RecommendationIntentEvaluator(engine)

    chats_writer = SupabaseResultWriter(table_name=args.chats_table, batch_size=args.supabase_batch_size) if args.store_supabase else None
    messages_writer = SupabaseResultWriter(table_name=args.messages_table, batch_size=args.supabase_batch_size) if args.store_supabase else None

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == 'all_dates':
        folders = loader.list_date_folders(args.chats_subfolder)
        if not folders:
            print('Aucun dossier de chats trouvé')
            return
        for folder in folders:
            chats_df = loader.load_chats_in_folder(folder, subfolder=args.chats_subfolder)
            if chats_df.empty:
                continue
            if args.chat_fraction < 1.0:
                chats_df = chats_df.sample(frac=args.chat_fraction, random_state=42)
            process_chat_batch(
                chats_df,
                loader,
                args,
                evaluator,
                chats_writer,
                messages_writer,
                output_root / folder,
            )
        return

    chats_df = load_chats_by_mode(loader, args)
    if chats_df.empty:
        print('Aucun chat trouvé pour les paramètres fournis')
        return

    if args.mode == 'date':
        output_dir = output_root / f"date={args.date}"
    elif args.mode == 'user':
        output_dir = output_root / 'users' / (args.user_id or 'unknown')
    else:
        output_dir = output_root / 'samples'

    process_chat_batch(
        chats_df,
        loader,
        args,
        evaluator,
        chats_writer,
        messages_writer,
        output_dir,
    )


if __name__ == '__main__':
    main()
