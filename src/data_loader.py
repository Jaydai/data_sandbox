import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Optional
import tempfile
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseDataLoader:
    """
    Classe pour charger les fichiers Parquet depuis Supabase Storage
    """
    
    def __init__(self):
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET_MESSAGES", "raw-messages-archive")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL et SUPABASE_KEY doivent être définis dans .env")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("✅ Connexion Supabase établie")
    
    def _list_storage(self, path: str, *, limit: int = 100) -> List[dict]:
        """Retourne l'intégralité des items en paginant la Storage API."""
        results = []
        offset = 0

        while True:
            batch = self.client.storage.from_(self.bucket_name).list(
                path,
                {
                    "limit": limit,
                    "offset": offset,
                    "sortBy": {"column": "name", "order": "asc"},
                },
            )
            if not batch:
                break

            results.extend(batch)

            if len(batch) < limit:
                break

            offset += limit

        return results

    def list_date_folders(self, subfolder: str = "messages") -> List[str]:
        """
        Liste tous les dossiers de dates dans le bucket
        
        Args:
            subfolder: 'messages' ou 'chats'
        
        Returns:
            Liste des dossiers (ex: ['date=2022-12-30', 'date=2023-01-15', ...])
        """
        try:
            files = self._list_storage(subfolder)
            date_folders = sorted([
                f['name'] for f in files 
                if f['name'].startswith('date=')
            ])
            
            logger.info(f"📁 {len(date_folders)} dossiers trouvés dans {subfolder}/")
            return date_folders
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du listing des dossiers: {e}")
            return []
    
    def list_files_in_folder(self, folder_path: str, subfolder: str = "messages") -> List[str]:
        """
        Liste les fichiers Parquet dans un dossier spécifique
        
        Args:
            folder_path: Nom du dossier de date (ex: 'date=2022-12-30')
            subfolder: 'messages' ou 'chats'
        
        Returns:
            Liste des chemins complets de fichiers
        """
        try:
            # Construire le chemin complet
            full_path = f"{subfolder}/{folder_path}"
            
            files = self._list_storage(full_path)
            
            parquet_files = [
                f"{full_path}/{f['name']}" 
                for f in files 
                if f['name'].endswith('.parquet')
            ]
            
            logger.info(f"📄 {len(parquet_files)} fichiers Parquet trouvés dans {full_path}")
            return parquet_files
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du listing des fichiers: {e}")
            return []
    
    def download_parquet_file(self, file_path: str) -> Optional[bytes]:
        """
        Télécharge un fichier Parquet depuis Supabase Storage
        
        Args:
            file_path: Chemin complet du fichier (ex: 'messages/date=2022-12-30/messages_20221230.parquet')
        
        Returns:
            Contenu binaire du fichier ou None si erreur
        """
        try:
            logger.info(f"⬇️  Téléchargement de {file_path}...")
            file_data = self.client.storage.from_(self.bucket_name).download(file_path)
            logger.info(f"✅ Fichier téléchargé ({len(file_data)} bytes)")
            return file_data
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du téléchargement de {file_path}: {e}")
            return None
    
    def load_parquet_to_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Charge un fichier Parquet dans un DataFrame pandas
        
        Args:
            file_path: Chemin complet du fichier dans le bucket
        
        Returns:
            DataFrame pandas ou None si erreur
        """
        file_data = self.download_parquet_file(file_path)
        
        if file_data is None:
            return None
        
        try:
            # Créer un fichier temporaire pour stocker le Parquet
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                tmp_file.write(file_data)
                tmp_path = tmp_file.name
            
            # Charger le Parquet dans pandas
            df = pd.read_parquet(tmp_path)
            
            # Nettoyer le fichier temporaire
            os.unlink(tmp_path)
            
            logger.info(f"✅ DataFrame chargé : {len(df)} lignes, {len(df.columns)} colonnes")
            return df
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du Parquet: {e}")
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return None
    
    def load_date_range(
        self, 
        start_date: str, 
        end_date: str, 
        subfolder: str = "messages",
        sample_fraction: float = 1.0
    ) -> pd.DataFrame:
        """
        Charge tous les messages d'une plage de dates
        
        Args:
            start_date: Date de début (format: 'YYYY-MM-DD')
            end_date: Date de fin (format: 'YYYY-MM-DD')
            subfolder: 'messages' ou 'chats'
            sample_fraction: Fraction d'échantillonnage (0.0 à 1.0)
        
        Returns:
            DataFrame concatené de tous les messages
        """
        # Convertir les dates en format "date=YYYY-MM-DD"
        start_folder = f"date={start_date}"
        end_folder = f"date={end_date}"
        
        # Lister tous les dossiers
        all_folders = self.list_date_folders(subfolder)
        
        # Filtrer les dossiers dans la plage
        selected_folders = [
            folder for folder in all_folders
            if start_folder <= folder <= end_folder
        ]
        
        logger.info(f"📅 Chargement de {len(selected_folders)} dossiers de dates")
        
        all_dataframes = []
        
        for folder in selected_folders:
            # Lister les fichiers dans le dossier
            parquet_files = self.list_files_in_folder(folder, subfolder)
            
            for file_path in parquet_files:
                df = self.load_parquet_to_dataframe(file_path)
                
                if df is not None:
                    # Échantillonnage si nécessaire
                    if sample_fraction < 1.0:
                        df = df.sample(frac=sample_fraction, random_state=42)
                    
                    all_dataframes.append(df)
        
        if not all_dataframes:
            logger.warning("⚠️  Aucune donnée chargée")
            return pd.DataFrame()
        
        # Concatener tous les DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"✅ Total : {len(combined_df)} messages chargés")
        
        return combined_df
    
    def get_sample_messages(self, n_samples: int = 1000, subfolder: str = "messages") -> pd.DataFrame:
        """
        Récupère un échantillon aléatoire de messages
        
        Args:
            n_samples: Nombre de messages à échantillonner
            subfolder: 'messages' ou 'chats'
        
        Returns:
            DataFrame avec l'échantillon
        """
        # Lister tous les dossiers
        all_folders = self.list_date_folders(subfolder)
        
        if not all_folders:
            return pd.DataFrame()
        
        # Sélectionner quelques dossiers aléatoirement
        import random
        selected_folders = random.sample(
            all_folders, 
            min(10, len(all_folders))  # Maximum 10 dossiers
        )
        
        all_messages = []
        total_loaded = 0
        
        for folder in selected_folders:
            if total_loaded >= n_samples:
                break
                
            parquet_files = self.list_files_in_folder(folder, subfolder)
            
            for file_path in parquet_files:
                if total_loaded >= n_samples:
                    break
                    
                df = self.load_parquet_to_dataframe(file_path)
                
                if df is not None:
                    all_messages.append(df)
                    total_loaded += len(df)
        
        if not all_messages:
            return pd.DataFrame()
        
        combined = pd.concat(all_messages, ignore_index=True)
        
        # Échantillonner exactement n_samples messages
        if len(combined) > n_samples:
            return combined.sample(n=n_samples, random_state=42)
        else:
            return combined

    def get_user_conversation(
        self,
        user_id: str,
        subfolder: str = "messages",
        sample_fraction: float = 1.0,
        max_files: Optional[int] = None,
    ) -> pd.DataFrame:
        if not user_id:
            return pd.DataFrame()

        folders = self.list_date_folders(subfolder)
        collected = []
        processed = 0

        for folder in folders:
            parquet_files = self.list_files_in_folder(folder, subfolder)
            for file_path in parquet_files:
                if max_files and processed >= max_files:
                    break
                df = self.load_parquet_to_dataframe(file_path)
                processed += 1
                if df is None or df.empty:
                    continue

                if sample_fraction < 1.0:
                    df = df.sample(frac=sample_fraction, random_state=42)

                if 'user_id' not in df.columns:
                    continue

                user_rows = df[df['user_id'] == user_id]
                if user_rows.empty:
                    continue

                if 'chat_provider_id' in df.columns:
                    chats = user_rows['chat_provider_id'].dropna().unique().tolist()
                    if chats:
                        conversation = df[df['chat_provider_id'].isin(chats)]
                    else:
                        conversation = user_rows
                else:
                    conversation = user_rows

                collected.append(conversation)

            if max_files and processed >= max_files:
                break

        if not collected:
            logger.info("⚠️ Aucun message trouvé pour user_id=%s", user_id)
            return pd.DataFrame()

        combined = pd.concat(collected, ignore_index=True)
        if 'created_at' in combined.columns:
            combined = combined.sort_values('created_at')
        return combined


# Fonctions helper pour utilisation rapide
def quick_load_sample(n_samples: int = 100000, subfolder: str = "messages") -> pd.DataFrame:
    """Charge rapidement un échantillon de messages"""
    loader = SupabaseDataLoader()
    return loader.get_sample_messages(n_samples, subfolder)

def quick_load_date(date: str, subfolder: str = "messages") -> pd.DataFrame:
    """Charge tous les messages d'une date spécifique"""
    loader = SupabaseDataLoader()
    return loader.load_date_range(date, date, subfolder)
