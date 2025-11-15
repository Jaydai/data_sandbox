#!/usr/bin/env python3
"""Analyse et enrichissement des conversations complètes (chat level)."""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.classification import HFRouterEngine, OllamaEngine
from src.data_loader import SupabaseDataLoader
from src.enrichment import ChatQualityEvaluator
from src.storage import SupabaseResultWriter


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def create_engine(engine_name: str, model: str):
    engine_name = engine_name.lower()
    if engine_name == "hf":
        return HFRouterEngine(model=model)
    if engine_name == "ollama":
        return OllamaEngine(model=model)
    raise ValueError("engine doit être 'hf' ou 'ollama'")


def detect_garbage(first_message: Optional[str]) -> Dict:
    if not first_message:
        return {"is_garbage": True, "reason": "empty_message"}
    text = first_message.strip()
    reasons = []
    if len(text) < 8:
        reasons.append("too_short")
    if not any(c.isalpha() for c in text):
        reasons.append("no_letters")
    if text.endswith("...") or text.endswith(","):
        reasons.append("incomplete_sentence")
    return {"is_garbage": bool(reasons), "reason": ",".join(reasons) if reasons else ""}


def detect_clarification(messages: pd.DataFrame) -> bool:
    keywords = [
        "clarify",
        "more detail",
        "provide more",
        "could you explain",
        "i need more",
        "précise",
        "peux-tu détailler",
    ]
    assistant_msgs = messages[messages['role'] != 'user']
    for _, row in assistant_msgs.head(3).iterrows():
        content = str(row.get('content', '')).lower()
        if any(keyword in content for keyword in keywords):
            return True
    return False


def conversation_to_list(messages: pd.DataFrame) -> List[Dict[str, str]]:
    convo = []
    for _, row in messages.iterrows():
        convo.append({
            'role': row.get('role', ''),
            'content': str(row.get('content', '')),
        })
    return convo


def evaluate_chat_record(
    chat_row: pd.Series,
    messages_df: pd.DataFrame,
    quality_evaluator: ChatQualityEvaluator,
    heuristic_clarification: bool,
) -> Dict:
    conversation = conversation_to_list(messages_df)
    llm_eval = quality_evaluator.evaluate(conversation)

    first_user = messages_df[messages_df['role'] == 'user'].head(1)
    first_content = str(first_user.iloc[0]['content']) if not first_user.empty else ""
    garbage_info = detect_garbage(first_content)

    final_assistant = messages_df[messages_df['role'] != 'user'].tail(1)
    final_answer_length = len(str(final_assistant.iloc[0]['content'])) if not final_assistant.empty else 0

    clarification_needed = llm_eval.get('clarification_needed') or heuristic_clarification

    return {
        'chat_id': chat_row.get('id'),
        'chat_provider_id': chat_row.get('chat_provider_id'),
        'user_id': chat_row.get('user_id'),
        'started_at': normalize_timestamp(chat_row.get('created_at')),
        'date_folder': chat_row.get('date_folder'),
        'messages_count': len(messages_df),
        'user_messages': int((messages_df['role'] == 'user').sum()),
        'assistant_messages': int((messages_df['role'] != 'user').sum()),
        'duration_seconds': compute_duration(messages_df),
        'first_user_message': first_content,
        'is_garbage': garbage_info['is_garbage'],
        'garbage_reason': garbage_info['reason'],
        'clarification_needed': clarification_needed,
        'assistant_fulfilled': llm_eval.get('assistant_fulfilled'),
        'quality_label': llm_eval.get('quality'),
        'quality_notes': llm_eval.get('notes'),
        'final_answer_length': final_answer_length,
        'analysis_version': 1,
    }


def compute_duration(messages: pd.DataFrame) -> Optional[float]:
    if 'created_at' not in messages.columns or messages.empty:
        return None
    start = messages['created_at'].min()
    end = messages['created_at'].max()
    if pd.isna(start) or pd.isna(end):
        return None
    try:
        delta = pd.to_datetime(end) - pd.to_datetime(start)
        return delta.total_seconds()
    except Exception:
        return None


def normalize_timestamp(value):
    if value is None or pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).isoformat()
    except Exception:
        return str(value)


def load_chats_by_mode(loader: SupabaseDataLoader, args: argparse.Namespace) -> pd.DataFrame:
    if args.mode == 'date':
        if not args.date:
            raise SystemExit('--date est requis en mode date')
        date_folder = f"date={args.date}"
        chats = loader.load_chats_in_folder(date_folder, subfolder=args.chats_subfolder)
        if args.chat_fraction < 1.0 and not chats.empty:
            chats = chats.sample(frac=args.chat_fraction, random_state=42)
        return chats

    if args.mode == 'sample':
        return load_sample_chats(loader, args.sample_size, args.chats_subfolder)

    if args.mode == 'user':
        if not args.user_id:
            raise SystemExit('--user-id est requis en mode user')
        return load_user_chats(loader, args.user_id, args.chats_subfolder)

    return pd.DataFrame()


def load_sample_chats(loader: SupabaseDataLoader, sample_size: int, subfolder: str) -> pd.DataFrame:
    folders = loader.list_date_folders(subfolder)
    collected = []
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


def serialize_record(record: Dict) -> Dict:
    serialized = {}
    for key, value in record.items():
        print("=====>❤️", key)
        if key == 'date_folder' or key == 'final_answer_length':
            continue
        if isinstance(value, pd.Timestamp):
            serialized[key] = normalize_timestamp(value)
        else:
            serialized[key] = value
    return serialized


def conversation_summary(messages_df: pd.DataFrame) -> pd.DataFrame:
    return messages_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyser les conversations complètes (chat)')
    parser.add_argument('--mode', choices=['date', 'sample', 'user'], default='date')
    parser.add_argument('--date', help='Date au format YYYY-MM-DD (mode date)')
    parser.add_argument('--chat-fraction', type=float, default=1.0)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--user-id', help='Identifiant utilisateur (mode user)')
    parser.add_argument('--model', default='mistral-small:latest')
    parser.add_argument('--engine', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--chats-subfolder', default='chats')
    parser.add_argument('--messages-subfolder', default='messages')
    parser.add_argument('--max-chats', type=int, default=None)
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--use-context', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--store-supabase', action='store_true')
    parser.add_argument('--supabase-table', default='enriched_chats')
    parser.add_argument('--supabase-batch-size', type=int, default=200)
    args = parser.parse_args()

    loader = SupabaseDataLoader()
    backend_engine = create_engine(args.engine, args.model)
    quality_evaluator = ChatQualityEvaluator(backend_engine)

    supabase_writer = None
    if args.store_supabase:
        supabase_writer = SupabaseResultWriter(
            table_name=args.supabase_table,
            batch_size=args.supabase_batch_size,
        )

    chats_df = load_chats_by_mode(loader, args)
    if chats_df.empty:
        print("Aucun chat trouvé pour les paramètres fournis")
        return

    records = []
    supabase_records = []
    total = args.max_chats or len(chats_df)
    iter_df = chats_df.head(total)

    for _, chat_row in tqdm(iter_df.iterrows(), total=len(iter_df), desc='Chats'):
        chat_id = chat_row.get('chat_provider_id') or chat_row.get('id')
        date_folder = chat_row.get('date_folder')
        if not chat_id or not date_folder:
            continue
        messages_df = loader.load_chat_messages(chat_id, date_folder, subfolder=args.messages_subfolder)
        if messages_df.empty:
            continue

        heuristic_clarification = detect_clarification(messages_df)
        record = evaluate_chat_record(chat_row, messages_df, quality_evaluator, heuristic_clarification)
        records.append(record)

        if supabase_writer:
            supabase_records.append(serialize_record(record))
            if len(supabase_records) >= supabase_writer.batch_size:
                supabase_writer.upsert_messages(supabase_records)
                supabase_records = []

    if supabase_writer and supabase_records:
        supabase_writer.upsert_messages(supabase_records)

    if not records:
        print("Aucun chat exploitable")
        return

    enriched_df = pd.DataFrame(records)
    if args.mode == 'date':
        output_dir = Path('data/processed') / args.chats_subfolder / f"date={args.date}"
    elif args.mode == 'user':
        output_dir = Path('data/processed') / args.chats_subfolder / 'users' / (args.user_id or 'unknown')
    else:
        output_dir = Path('data/processed') / args.chats_subfolder / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'enriched_chats.parquet'

    if args.overwrite or not output_path.exists():
        enriched_df.to_parquet(output_path, index=False)
        print(f"Résultats sauvegardés dans {output_path}")
    else:
        print(f"Fichier déjà existant (utiliser --overwrite pour forcer) : {output_path}")


if __name__ == '__main__':
    main()
