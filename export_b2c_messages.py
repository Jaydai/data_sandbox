#!/usr/bin/env python3
"""Aggregate all Supabase messages/chats into a single parquet and upload to destination bucket."""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from supabase import Client, create_client
from tqdm import tqdm

from src.data_loader import SupabaseDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Exporter tous les messages/chats vers un parquet unique (b2c_messages.parquet).')
    parser.add_argument('--dest-supabase-url', required=True, help='URL du projet Supabase destination (bucket cible)')
    parser.add_argument('--dest-supabase-key', required=True, help='SERVICE_ROLE KEY du projet destination')
    parser.add_argument('--dest-bucket', required=True, help='Nom du bucket destination (ex: analytics-exports)')
    parser.add_argument('--messages-subfolder', default='messages', help='Sous-dossier Storage pour les messages source')
    parser.add_argument('--output-filename', default='b2c_messages.parquet', help='Nom du fichier parquet à générer')
    parser.add_argument('--limit-files', type=int, default=None, help='Limiter le nombre de fichiers messages à agréger')
    return parser.parse_args()


def create_dest_client(url: str, key: str) -> Client:
    return create_client(url, key)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce').astype('datetime64[ns]')
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(bool)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = df[col].astype(str).replace({'nan': None})
    return df


def collect_messages(loader: SupabaseDataLoader, subfolder: str, limit_files: int = None) -> pd.DataFrame:
    folders = loader.list_date_folders(subfolder)
    dataframes: List[pd.DataFrame] = []
    file_count = 0
    for folder in folders:
        files = loader.list_files_in_folder(folder, subfolder=subfolder)
        for file_path in files:
            df = loader.load_parquet_to_dataframe(file_path)
            if df is None or df.empty:
                continue
            dataframes.append(df)
            file_count += 1
            if limit_files and file_count >= limit_files:
                return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def upload_to_bucket(client: Client, bucket: str, file_path: Path, dest_key: str) -> None:
    with file_path.open('rb') as f:
        client.storage.from_(bucket).upload(path=dest_key, file=f, file_options={'cacheControl': '3600', 'upsert': True})


def main() -> None:
    args = parse_args()
    loader = SupabaseDataLoader()
    dest_client = create_dest_client(args.dest_supabase_url, args.dest_supabase_key)

    combined_df = collect_messages(loader, args.messages_subfolder, args.limit_files)
    if combined_df.empty:
        print('Aucun message trouvé dans le stockage source.')
        return

    combined_df = sanitize_dataframe(combined_df)

    output_path = Path(args.output_filename)
    combined_df.to_parquet(output_path, index=False)
    print(f"Parquet généré: {output_path}")

    upload_to_bucket(dest_client, args.dest_bucket, output_path, output_path.name)
    print(f"Parquet uploadé vers {args.dest_bucket}/{output_path.name}")


if __name__ == '__main__':
    main()
