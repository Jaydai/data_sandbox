import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from api.services.pii_detector import PIIDetector
from api.services.quality_scorer import QualityScorer
from api.services.risk_analyzer import RiskAnalyzer
from src.classification import HFRouterEngine, MessageClassifier, OllamaEngine
from src.data_loader import SupabaseDataLoader
from src.enrichment import (
    DomainKeywordClassifier,
    RiskHeuristicsAnalyzer,
    SentimentAnalyzer,
    AssistantResponseEvaluator,
    PublicFigureChecker,
)
from src.storage import SupabaseResultWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationPipeline:
    """
    Pipeline pour classifier les messages en batch
    """
    
    def __init__(
        self, 
        model: str = "qwen2.5:14b",
        output_dir: str = "data/processed",
        engine: str = "ollama",
        *,
        enable_pii: bool = True,
        enable_quality: bool = True,
        enable_risk: bool = True,
        store_in_supabase: bool = False,
        supabase_writer: Optional[SupabaseResultWriter] = None,
    ):
        """
        Args:
            model: Nom du modèle à utiliser
            output_dir: Dossier de sortie pour les résultats
            engine: "ollama" pour local, "hf" pour Hugging Face Router
        """
        self.loader = SupabaseDataLoader()
        backend_engine = self._create_engine(engine, model)
        self.classifier = MessageClassifier(engine=backend_engine)
        self.public_figure_checker = PublicFigureChecker(backend_engine) if enable_pii else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pii_detector = self._build_pii_detector(enable_pii, self.public_figure_checker)
        self.quality_scorer = QualityScorer() if enable_quality else None
        self.risk_analyzer = RiskAnalyzer() if enable_risk else None

        self.supabase_writer = supabase_writer
        if store_in_supabase and not self.supabase_writer:
            self.supabase_writer = SupabaseResultWriter()

        self.domain_classifier = DomainKeywordClassifier()
        self.risk_heuristics = RiskHeuristicsAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_evaluator = AssistantResponseEvaluator()

        logger.info("✅ Pipeline initialisé")

    def _create_engine(self, engine: str, model: str):
        engine = engine.lower()
        if engine == "ollama":
            return OllamaEngine(model=model)
        elif engine in {"hf", "huggingface"}:
            return HFRouterEngine(model=model)
        else:
            raise ValueError("engine doit être 'ollama' ou 'hf'")

    @staticmethod
    def _build_pii_detector(enable: bool, public_checker=None) -> Optional[PIIDetector]:
        if not enable:
            return None
        try:
            return PIIDetector(public_figure_checker=public_checker)
        except Exception as exc:
            logger.warning("⚠️  PII Detector indisponible: %s", exc)
            return None
    
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
        results: List[Dict[str, Any]] = []
        supabase_records: List[Dict[str, Any]] = []
        
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
                classification = self.classifier.classify(content, context)

                pii_result = self._detect_pii(content)
                quality_result = self._score_quality(content, context)
                risk_result = self._analyze_risk(content, classification, pii_result, quality_result)
                domain_info = self.domain_classifier.classify(content)
                heuristic_risk = self.risk_heuristics.analyze(
                    content,
                    pii_result.get('has_pii') if pii_result else False,
                )
                sentiment = self.sentiment_analyzer.score(content)

                # Extraire les résultats
                result = self._build_flat_result(
                    row,
                    idx,
                    content,
                    context,
                    classification,
                    pii_result,
                    quality_result,
                    risk_result,
                    domain_info,
                    heuristic_risk,
                    sentiment,
                )

                results.append(result)

                if self.supabase_writer:
                    supabase_records.append(
                        self._build_supabase_record(
                            row,
                            classification,
                            pii_result,
                            quality_result,
                            risk_result,
                            domain_info,
                            heuristic_risk,
                            sentiment,
                            context,
                        )
                    )
                    if len(supabase_records) >= self.supabase_writer.batch_size:
                        self.supabase_writer.upsert_messages(supabase_records)
                        supabase_records = []
                
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

        if self.supabase_writer and supabase_records:
            self.supabase_writer.upsert_messages(supabase_records)
        
        logger.info(f"✅ {len(results_df)} messages classifiés avec succès")
        return results_df

    def _detect_pii(self, content: str) -> Optional[Dict]:
        if not self.pii_detector:
            return None
        return self.pii_detector.detect(content)

    def _score_quality(self, content: str, context: str) -> Optional[Dict]:
        if not self.quality_scorer:
            return None
        context_list = context.split(" | ") if context else []
        return self.quality_scorer.score(content, context_list)

    def _analyze_risk(
        self,
        content: str,
        classification: Optional[Dict],
        pii_result: Optional[Dict],
        quality_result: Optional[Dict],
    ) -> Optional[Dict]:
        if not self.risk_analyzer or not pii_result or not quality_result or not classification:
            return None
        return self.risk_analyzer.analyze(
            content=content,
            classification=classification,
            pii_detection=pii_result,
            quality_score=quality_result,
        )

    def _build_flat_result(
        self,
        row: pd.Series,
        idx: int,
        content: str,
        context: str,
        classification: Dict,
        pii_result: Optional[Dict],
        quality_result: Optional[Dict],
        risk_result: Optional[Dict],
        domain_info: Dict,
        heuristic_risk: Dict,
        sentiment: Dict,
    ) -> Dict[str, Any]:
        result = {
            'message_id': row.get('id', idx),
            'user_id': row.get('user_id', None),
            'created_at': row.get('created_at', None),
            'content_length': len(content),
            'context_used': context,
            'classification_model': self.classifier.engine_name,
            'is_work': classification['work']['is_work'],
            'work_confidence': classification['work']['confidence'],
            'work_reasoning': classification['work']['reasoning'],
            'topic': classification['topic']['topic'],
            'sub_topic': classification['topic']['sub_topic'],
            'topic_confidence': classification['topic']['confidence'],
            'intent': classification['intent']['intent'],
            'intent_confidence': classification['intent']['confidence'],
            'intent_reasoning': classification['intent']['reasoning'],
            'domain_label': domain_info.get('domain', 'other'),
            'domain_keywords': json.dumps(domain_info.get('keywords', [])),
            'manual_risk_level': heuristic_risk.get('risk_level'),
            'manual_risk_flags': json.dumps(heuristic_risk.get('flags', [])),
            'sentiment_label': sentiment.get('label'),
            'sentiment_score': sentiment.get('score'),
        }

        if pii_result:
            result.update({
                'pii_has_pii': pii_result.get('has_pii', False),
                'pii_types': json.dumps(pii_result.get('pii_types', [])),
                'pii_entities': json.dumps(pii_result.get('entities', [])),
                'pii_risk_level': pii_result.get('risk_level'),
            })
        else:
            result.update({
                'pii_has_pii': False,
                'pii_types': json.dumps([]),
                'pii_entities': json.dumps([]),
                'pii_risk_level': None,
            })

        if quality_result:
            result.update({
                'quality_overall_score': quality_result.get('overall_score'),
                'quality_clarity_score': quality_result.get('clarity_score'),
                'quality_context_score': quality_result.get('context_score'),
                'quality_precision_score': quality_result.get('precision_score'),
                'quality_has_role': quality_result.get('has_clear_role'),
                'quality_has_context': quality_result.get('has_context'),
                'quality_has_goal': quality_result.get('has_clear_goal'),
                'quality_strengths': json.dumps(quality_result.get('strengths', [])),
                'quality_weaknesses': json.dumps(quality_result.get('weaknesses', [])),
                'quality_suggestions': json.dumps(quality_result.get('suggestions', [])),
            })
        else:
            result.update({
                'quality_overall_score': None,
                'quality_clarity_score': None,
                'quality_context_score': None,
                'quality_precision_score': None,
                'quality_has_role': None,
                'quality_has_context': None,
                'quality_has_goal': None,
                'quality_strengths': json.dumps([]),
                'quality_weaknesses': json.dumps([]),
                'quality_suggestions': json.dumps([]),
            })

        if risk_result:
            result.update({
                'risk_overall': risk_result.get('overall_risk'),
                'risk_data_leak': risk_result.get('data_leak_risk'),
                'risk_compliance': risk_result.get('compliance_risk'),
                'risk_hallucination': risk_result.get('hallucination_risk'),
                'risk_alerts': json.dumps(risk_result.get('alerts', [])),
                'risk_actions': json.dumps(risk_result.get('mitigation_actions', [])),
            })
        else:
            result.update({
                'risk_overall': None,
                'risk_data_leak': None,
                'risk_compliance': None,
                'risk_hallucination': None,
                'risk_alerts': json.dumps([]),
                'risk_actions': json.dumps([]),
            })

        return result

    def _build_supabase_record(
        self,
        row: pd.Series,
        classification: Dict,
        pii_result: Optional[Dict],
        quality_result: Optional[Dict],
        risk_result: Optional[Dict],
        domain_info: Dict,
        heuristic_risk: Dict,
        sentiment: Dict,
        context: str,
    ) -> Dict[str, Any]:
        record = self._normalize_row(row)
        record.update({
            'context_window': context,
            'classification_model': self.classifier.engine_name,
            'work': classification['work'],
            'topic': classification['topic'],
            'intent': classification['intent'],
            'pii_detection': pii_result,
            'quality_score': quality_result,
            'risk_analysis': risk_result,
            'domain_classification': domain_info,
            'manual_risk': heuristic_risk,
            'sentiment': sentiment,
            'processed_at': datetime.utcnow().isoformat(),
        })
        return record

    def _evaluate_response(self, prompt_content: str, response_content: str) -> Dict:
        return self.response_evaluator.evaluate(prompt_content, response_content)

    def _build_response_result(
        self,
        row: pd.Series,
        prompt_row: Optional[pd.Series],
        evaluation: Dict,
        domain_info: Dict,
        heuristic_risk: Dict,
        sentiment: Dict,
        pii_result: Optional[Dict],
        quality_result: Optional[Dict],
    ) -> Dict[str, Any]:
        prompt_id = prompt_row.get('id') if prompt_row is not None else None
        prompt_provider_id = prompt_row.get('message_provider_id') if prompt_row is not None else None
        return {
            'response_message_id': row.get('id'),
            'response_provider_id': row.get('message_provider_id'),
            'prompt_message_id': prompt_id,
            'prompt_provider_id': prompt_provider_id,
            'response_created_at': row.get('created_at'),
            'prompt_created_at': prompt_row.get('created_at') if prompt_row is not None else None,
            'response_role': row.get('role'),
            'prompt_role': prompt_row.get('role') if prompt_row is not None else None,
            'length_ratio': evaluation.get('length_ratio'),
            'response_length': evaluation.get('response_length'),
            'prompt_length': evaluation.get('prompt_length'),
            'contains_code': evaluation.get('contains_code'),
            'bullet_points': evaluation.get('bullet_points'),
            'numbered_steps': evaluation.get('numbered_steps'),
            'actionability_score': evaluation.get('actionability_score'),
            'disclaimer_present': evaluation.get('disclaimer_present'),
            'potential_refusal': evaluation.get('potential_refusal'),
            'keyword_overlap': evaluation.get('keyword_overlap'),
            'domain_label': domain_info.get('domain', 'other'),
            'domain_keywords': json.dumps(domain_info.get('keywords', [])),
            'manual_risk_level': heuristic_risk.get('risk_level'),
            'manual_risk_flags': json.dumps(heuristic_risk.get('flags', [])),
            'sentiment_label': sentiment.get('label'),
            'sentiment_score': sentiment.get('score'),
            'pii_has_pii': pii_result.get('has_pii') if pii_result else False,
            'pii_types': json.dumps(pii_result.get('pii_types', [])) if pii_result else json.dumps([]),
            'pii_entities': json.dumps(pii_result.get('entities', [])) if pii_result else json.dumps([]),
            'quality_overall_score': quality_result.get('overall_score') if quality_result else None,
            'quality_clarity_score': quality_result.get('clarity_score') if quality_result else None,
            'quality_context_score': quality_result.get('context_score') if quality_result else None,
            'quality_precision_score': quality_result.get('precision_score') if quality_result else None,
        }

    def _build_response_supabase_record(
        self,
        row: pd.Series,
        prompt_row: Optional[pd.Series],
        evaluation: Dict,
        domain_info: Dict,
        heuristic_risk: Dict,
        sentiment: Dict,
        pii_result: Optional[Dict],
        quality_result: Optional[Dict],
    ) -> Dict[str, Any]:
        record = self._normalize_row(row)
        prompt_reference = None
        if prompt_row is not None:
            prompt_reference = {
                'id': prompt_row.get('id'),
                'message_provider_id': prompt_row.get('message_provider_id'),
                'created_at': self._normalize_value(prompt_row.get('created_at')),
            }
        record.update({
            'prompt_reference': prompt_reference,
            'assistant_evaluation': evaluation,
            'domain_classification': domain_info,
            'manual_risk': heuristic_risk,
            'sentiment': sentiment,
            'pii_detection': pii_result,
            'quality_score': quality_result,
            'processed_at': datetime.utcnow().isoformat(),
        })
        return record

    @staticmethod
    def _normalize_row(row: pd.Series) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in row.items():
            normalized[key] = ClassificationPipeline._normalize_value(value)
        return normalized

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (list, dict)):
            return value
        return value
    
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

    def run_full_archive_classification(
        self,
        *,
        use_context: bool = False,
        subfolder: str = "messages",
        overwrite: bool = False
    ) -> None:
        """Classifie tous les fichiers Parquet présents dans Supabase."""

        folders = self.loader.list_date_folders(subfolder=subfolder)
        if not folders:
            logger.warning("⚠️ Aucun dossier trouvé dans Supabase (%s)", subfolder)
            return

        logger.info("📦 Début de la classification complète (%s dossiers)", len(folders))
        total_files = 0
        total_messages = 0

        for folder in folders:
            parquet_files = self.loader.list_files_in_folder(folder, subfolder=subfolder)
            if not parquet_files:
                continue

            for file_path in parquet_files:
                total_files += 1
                df = self.loader.load_parquet_to_dataframe(file_path)
                if df is None or df.empty:
                    logger.info("⚠️ Fichier vide ignoré: %s", file_path)
                    continue

                results_df = self.classify_messages_batch(df, use_context=use_context)
                if results_df.empty:
                    logger.info("⚠️ Aucun message utilisateur dans %s", file_path)
                    continue

                total_messages += len(results_df)
                output_dir = self.output_dir / subfolder / folder
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = Path(file_path).name.replace('.parquet', '_classified.parquet')
                output_path = output_dir / filename

                if output_path.exists() and not overwrite:
                    logger.info("⏭️  Fichier déjà classifié (utiliser overwrite=True pour forcer): %s", output_path)
                    continue

                results_df.to_parquet(output_path, index=False)
                logger.info("💾 %s sauvegardé (%s messages)", output_path, len(results_df))

        logger.info(
            "✅ Classification complète terminée (%s fichiers, %s messages)",
            total_files,
            total_messages,
        )

    def analyze_responses_for_date(
        self,
        date: str,
        *,
        subfolder: str = "messages",
        sample_fraction: float = 1.0,
        use_context: bool = False,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Évalue les réponses assistant pour une journée donnée."""

        df = self.loader.load_date_range(date, date, subfolder=subfolder, sample_fraction=sample_fraction)
        if df.empty:
            logger.warning("⚠️ Aucun message pour %s", date)
            return pd.DataFrame()

        results_df = self._analyze_responses_df(df, use_context=use_context)

        output_dir = self.output_dir / subfolder / f"date={date}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"assistant_responses_{date}.parquet"
        if output_path.exists() and not overwrite:
            logger.info("⏭️ Résultats déjà présents (utiliser overwrite=True): %s", output_path)
        else:
            results_df.to_parquet(output_path, index=False)
            logger.info("💾 Résultats réponses sauvegardés : %s", output_path)

        return results_df

    def analyze_user_conversation(
        self,
        user_id: str,
        *,
        subfolder: str = "messages",
        sample_fraction: float = 1.0,
        max_files: Optional[int] = None,
        use_context: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        df = self.loader.get_user_conversation(
            user_id=user_id,
            subfolder=subfolder,
            sample_fraction=sample_fraction,
            max_files=max_files,
        )
        if df.empty:
            logger.warning("⚠️ Aucun message trouvé pour l'utilisateur %s", user_id)
            return {}

        prompts_df = self.classify_messages_batch(df, use_context=use_context)
        responses_df = self._analyze_responses_df(df, use_context=use_context)

        user_dir = self.output_dir / subfolder / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = user_dir / "prompts.parquet"
        response_path = user_dir / "responses.parquet"

        if overwrite or not prompt_path.exists():
            prompts_df.to_parquet(prompt_path, index=False)
            logger.info("💾 Résultats prompts utilisateur : %s", prompt_path)
        else:
            logger.info("⏭️ Résultats prompts déjà présents (overwrite pour forcer)")

        if overwrite or not response_path.exists():
            responses_df.to_parquet(response_path, index=False)
            logger.info("💾 Résultats réponses utilisateur : %s", response_path)
        else:
            logger.info("⏭️ Résultats réponses déjà présents (overwrite pour forcer)")

        return {"prompts": prompts_df, "responses": responses_df}

    def _analyze_responses_df(self, df: pd.DataFrame, use_context: bool) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        if 'created_at' in df.columns:
            df = df.sort_values('created_at')

        prompts = df[df['role'] == 'user']
        prompt_map = {
            row.get('message_provider_id') or row.get('id'): row
            for _, row in prompts.iterrows()
        }

        assistant_messages = df[df['role'] != 'user']
        logger.info("🧠 %s réponses assistant à analyser", len(assistant_messages))

        results: List[Dict[str, Any]] = []
        supabase_records: List[Dict[str, Any]] = []

        for idx, row in assistant_messages.iterrows():
            response_content = str(row.get('content', ''))
            parent_id = row.get('parent_message_provider_id') or row.get('parent_id')
            prompt_row = prompt_map.get(parent_id)
            prompt_content = str(prompt_row.get('content', '')) if prompt_row is not None else ''

            if use_context and not prompt_content:
                context = self.get_context_for_message(df, idx)
                prompt_content = context

            evaluation = self._evaluate_response(prompt_content, response_content)
            domain_info = self.domain_classifier.classify(response_content)
            pii_result = self._detect_pii(response_content)
            heuristic_risk = self.risk_heuristics.analyze(
                response_content,
                has_pii=pii_result.get('has_pii') if pii_result else False,
            )
            sentiment = self.sentiment_analyzer.score(response_content)
            quality_result = self._score_quality(response_content, prompt_content)

            result = self._build_response_result(
                row,
                prompt_row,
                evaluation,
                domain_info,
                heuristic_risk,
                sentiment,
                pii_result,
                quality_result,
            )
            results.append(result)

            if self.supabase_writer:
                supabase_records.append(
                    self._build_response_supabase_record(
                        row,
                        prompt_row,
                        evaluation,
                        domain_info,
                        heuristic_risk,
                        sentiment,
                        pii_result,
                        quality_result,
                    )
                )
                if len(supabase_records) >= self.supabase_writer.batch_size:
                    self.supabase_writer.upsert_messages(supabase_records)
                    supabase_records = []

        results_df = pd.DataFrame(results)
        if self.supabase_writer and supabase_records:
            self.supabase_writer.upsert_messages(supabase_records)
        return results_df


def quick_classify_sample(n_samples: int = 1000, model: str = "qwen2.5:14b") -> pd.DataFrame:
    """Fonction helper pour classifier rapidement un échantillon"""
    pipeline = ClassificationPipeline(model=model)
    return pipeline.run_sample_classification(n_samples=n_samples)
