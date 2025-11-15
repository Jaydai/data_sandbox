#!/usr/bin/env python3
"""Copy Supabase Storage parquet exports (chats/messages) into SQL tables."""
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from supabase import Client, create_client
from tqdm import tqdm

from src.data_loader import SupabaseDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Importer tous les parquets chats/messages vers Supabase SQL')
    parser.add_argument('--dest-supabase-url', required=True, help='URL du projet Supabase destination')
    parser.add_argument('--dest-supabase-key', required=True, help='SERVICE_ROLE KEY du projet destination')
    parser.add_argument('--entities', choices=['messages', 'chats', 'both'], default='both')
    parser.add_argument('--messages-table', default='messages')
    parser.add_argument('--chats-table', default='chats')
    parser.add_argument('--messages-subfolder', default='messages')
    parser.add_argument('--chats-subfolder', default='chats')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--limit-files', type=int, default=None, help='Limiter le nombre de fichiers par entité')
    return parser.parse_args()


def create_dest_client(url: str, key: str) -> Client:
    return create_client(url, key)


def sanitize_record(record: Dict) -> Dict:
    sanitized = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, (np.bool_, np.bool8)):
            sanitized[key] = bool(value)
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


def upsert_dataframe(df: pd.DataFrame, client: Client, table: str, batch_size: int) -> None:
    if df.empty:
        return
    records = [sanitize_record(r) for r in df.to_dict(orient='records')]
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.table(table).upsert(batch).execute()


def iter_storage_files(loader: SupabaseDataLoader, subfolder: str, limit_files: Optional[int] = None) -> List[str]:
    folders = loader.list_date_folders(subfolder)
    file_paths: List[str] = []
    for folder in folders:
        paths = loader.list_files_in_folder(folder, subfolder=subfolder)
        file_paths.extend(paths)
        if limit_files and len(file_paths) >= limit_files:
            return file_paths[:limit_files]
    if limit_files:
        return file_paths[:limit_files]
    return file_paths


def process_entity(
    loader: SupabaseDataLoader,
    dest_client: Client,
    entity: str,
    subfolder: str,
    dest_table: str,
    batch_size: int,
    limit_files: Optional[int] = None,
) -> None:
    file_paths = iter_storage_files(loader, subfolder, limit_files)
    if not file_paths:
        print(f"Aucun fichier trouvé dans le dossier {subfolder}")
        return
    print(f"➡️  {entity}: {len(file_paths)} fichiers à traiter")
    for file_path in tqdm(file_paths, desc=f'{entity}'):  # file_path like messages/date=YYYY/file.parquet
        df = loader.load_parquet_to_dataframe(file_path)
        if df is None or df.empty:
            continue
        upsert_dataframe(df, dest_client, dest_table, batch_size)


def main() -> None:
    args = parse_args()
    loader = SupabaseDataLoader()  # utilise les creds source (env)
    dest_client = create_dest_client(args.dest_supabase_url, args.dest_supabase_key)

    if args.entities in ('chats', 'both'):
        process_entity(
            loader,
            dest_client,
            'chats',
            args.chats_subfolder,
            args.chats_table,
            args.batch_size,
            args.limit_files,
        )

    if args.entities in ('messages', 'both'):
        process_entity(
            loader,
            dest_client,
            'messages',
            args.messages_subfolder,
            args.messages_table,
            args.batch_size,
            args.limit_files,
        )


if __name__ == '__main__':
    main()
