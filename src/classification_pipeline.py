import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import json
from tqdm import tqdm
import time

from src.data_loader import SupabaseDataLoader
from src.classifiers_local import LocalMessageClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationPipeline:
    """
    Pipeline pour classifier les messages en batch
    """
    
    def __init__(
        self, 
        model: str = "qwen2.5:14b",
        output_dir: str = "data/processed"
    ):
        """
        Args:
            model: Modèle Ollama à utiliser
            output_dir: Dossier de sortie pour les résultats
        """
        self.loader = SupabaseDataLoader()
        self.classifier = LocalMessageClassifier(model=model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Pipeline initialisé")
    
    def get_context_for_message(self, df: pd.DataFrame, index: int, n_previous: int = 3) -> str:
        """
        Récupère le contexte (messages précédents) pour un message
        
        Args:
            df: DataFrame contenant les messages
            index: Index du message actuel
            n_previous: Nombre de messages précédents à inclure
        
        Returns:
            Contexte sous forme de string
        """
        if 'chat_provider_id' not in df.columns:
            return ""
        
        current_chat = df.loc[index, 'chat_provider_id']
        current_time = df.loc[index, 'created_at']
        
        # Filtrer les messages du même chat avant ce message
        same_chat = df[
            (df['chat_provider_id'] == current_chat) & 
            (df['created_at'] < current_time)
        ].tail(n_previous)
        
        if same_chat.empty:
            return ""
        
        # Construire le contexte
        context_parts = []
        for _, msg in same_chat.iterrows():
            role = msg.get('role', 'user')
            content = str(msg.get('content', ''))[:200]  # Limiter à 200 chars
            context_parts.append(f"[{role}]: {content}")
        
        return " | ".join(context_parts)
    
    def classify_messages_batch(
        self, 
        df: pd.DataFrame,
        use_context: bool = False,
        save_every: int = 100,
        resume_from: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Classifier un batch de messages
        
        Args:
            df: DataFrame avec les messages
            use_context: Utiliser le contexte des messages précédents
            save_every: Sauvegarder tous les N messages
            resume_from: Reprendre depuis un index spécifique
        
        Returns:
            DataFrame avec colonnes ajoutées : is_work, topic, intent, etc.
        """
        # Filtrer pour ne garder que les messages utilisateur
        user_messages = df[df['role'] == 'user'].copy()
        logger.info(f"📊 {len(user_messages)} messages utilisateur à classifier")
        
        # Résultat
        results = []
        
        # Point de reprise
        start_idx = resume_from if resume_from else 0
        
        # Barre de progression
        for idx, row in tqdm(
            user_messages.iloc[start_idx:].iterrows(), 
            total=len(user_messages) - start_idx,
            desc="Classification"
        ):
            try:
                content = str(row['content'])
                
                # Récupérer le contexte si demandé
                context = ""
                if use_context:
                    context = self.get_context_for_message(df, idx)
                
                # Classifier
                classification = self.classifier.classify_complete(content, context)
                
                # Extraire les résultats
                result = {
                    'message_id': row.get('id', idx),
                    'user_id': row.get('user_id', None),
                    'created_at': row.get('created_at', None),
                    'content_length': len(content),
                    
                    # Work classification
                    'is_work': classification['work']['is_work'],
                    'work_confidence': classification['work']['confidence'],
                    'work_reasoning': classification['work']['reasoning'],
                    
                    # Topic classification
                    'topic': classification['topic']['topic'],
                    'sub_topic': classification['topic']['sub_topic'],
                    'topic_confidence': classification['topic']['confidence'],
                    
                    # Intent classification
                    'intent': classification['intent']['intent'],
                    'intent_confidence': classification['intent']['confidence'],
                    'intent_reasoning': classification['intent']['reasoning'],
                }
                
                results.append(result)
                
                # Sauvegarder périodiquement
                if len(results) % save_every == 0:
                    temp_df = pd.DataFrame(results)
                    temp_path = self.output_dir / f"classifications_temp_{len(results)}.parquet"
                    temp_df.to_parquet(temp_path, index=False)
                    logger.info(f"💾 Sauvegarde intermédiaire : {len(results)} messages classifiés")
            
            except Exception as e:
                logger.error(f"❌ Erreur sur message {idx}: {e}")
                continue
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"✅ {len(results_df)} messages classifiés avec succès")
        return results_df
    
    def run_sample_classification(
        self, 
        n_samples: int = 10000,
        use_context: bool = False
    ) -> pd.DataFrame:
        """
        Classifier un échantillon de messages
        
        Args:
            n_samples: Nombre de messages à échantillonner
            use_context: Utiliser le contexte
        
        Returns:
            DataFrame avec résultats
        """
        logger.info(f"📥 Chargement d'un échantillon de {n_samples} messages...")
        
        # Charger l'échantillon
        df = self.loader.get_sample_messages(n_samples=n_samples)
        
        if df.empty:
            logger.error("❌ Aucun message chargé")
            return pd.DataFrame()
        
        logger.info(f"✅ {len(df)} messages chargés")
        
        # Classifier
        start_time = time.time()
        results_df = self.classify_messages_batch(df, use_context=use_context)
        elapsed = time.time() - start_time
        
        # Sauvegarder
        output_path = self.output_dir / f"classified_sample_{n_samples}.parquet"
        results_df.to_parquet(output_path, index=False)
        logger.info(f"💾 Résultats sauvegardés : {output_path}")
        
        # Statistiques
        logger.info(f"\n{'='*60}")
        logger.info("📊 STATISTIQUES DE CLASSIFICATION")
        logger.info(f"{'='*60}")
        logger.info(f"⏱️  Temps total : {elapsed:.1f}s ({elapsed/len(results_df):.2f}s/message)")
        logger.info(f"\n🏢 WORK/NON-WORK :")
        logger.info(results_df['is_work'].value_counts(normalize=True).to_string())
        logger.info(f"\n📋 TOPICS :")
        logger.info(results_df['topic'].value_counts(normalize=True).head(10).to_string())
        logger.info(f"\n🎯 INTENTS :")
        logger.info(results_df['intent'].value_counts(normalize=True).to_string())
        logger.info(f"{'='*60}\n")
        
        return results_df


def quick_classify_sample(n_samples: int = 1000, model: str = "qwen2.5:14b") -> pd.DataFrame:
    """Fonction helper pour classifier rapidement un échantillon"""
    pipeline = ClassificationPipeline(model=model)
    return pipeline.run_sample_classification(n_samples=n_samples)