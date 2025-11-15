#!/usr/bin/env python3
"""
Script pour classifier un échantillon de messages
"""
import argparse

from src.classification_pipeline import ClassificationPipeline
from src.storage import SupabaseResultWriter

def main():
    parser = argparse.ArgumentParser(description='Classifier un échantillon de messages')
    parser.add_argument(
        '--mode',
        choices=['sample', 'archive', 'responses', 'user'],
        default='sample',
        help="'sample' pour un échantillon, 'archive' pour tout Supabase, 'responses' pour analyser les réponses, 'user' pour un utilisateur"
    )
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=1000,
        help='Nombre de messages à classifier (mode sample)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help="Date (YYYY-MM-DD) pour analyser les réponses assistant"
    )
    parser.add_argument(
        '--user-id',
        type=str,
        default=None,
        help='Identifiant utilisateur pour le mode user'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='mistral-small:latest',
        help='Modèle à utiliser (Ollama ou HF)'
    )
    parser.add_argument(
        '--engine',
        choices=['ollama', 'hf'],
        default='ollama',
        help="Backend d'inférence (ollama local ou hf via router)"
    )
    parser.add_argument(
        '--use-context',
        action='store_true',
        help='Utiliser le contexte des messages précédents'
    )
    parser.add_argument(
        '--subfolder',
        type=str,
        default='messages',
        help="Sous-dossier Supabase (messages ou chats) en mode archive"
    )
    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=1.0,
        help='Fraction aléatoire lors du chargement par date (mode responses)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Nombre maximum de fichiers à parcourir (mode user)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Écrase les fichiers déjà classifiés (mode archive)'
    )
    parser.add_argument(
        '--store-supabase',
        action='store_true',
        help='Enregistrer les messages enrichis dans Supabase SQL'
    )
    parser.add_argument(
        '--supabase-table',
        type=str,
        default=None,
        help='Nom de la table destination (sinon SUPABASE_RESULTS_TABLE)'
    )
    parser.add_argument(
        '--supabase-batch-size',
        type=int,
        default=500,
        help='Taille des batchs lors de lécriture Supabase'
    )
    
    args = parser.parse_args()
    
    date_info = args.date if args.date else 'N/A'
    user_info = args.user_id if args.user_id else 'N/A'
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║          CLASSIFICATION DE MESSAGES CHATGPT                    ║
╚════════════════════════════════════════════════════════════════╝

🎛 Mode : {args.mode}
📊 Taille échantillon : {args.n_samples if args.mode == 'sample' else 'tous les fichiers'}
🤖 Modèle : {args.model}
🔗 Contexte : {'✅ Activé' if args.use_context else '❌ Désactivé'}
 📅 Date ciblée : {date_info}
 👤 Utilisateur : {user_info}

""")
    
    # Initialiser le pipeline
    supabase_writer = None
    if args.store_supabase:
        supabase_writer = SupabaseResultWriter(
            table_name=args.supabase_table,
            batch_size=args.supabase_batch_size,
        )
    pipeline = ClassificationPipeline(
        model=args.model,
        engine=args.engine,
        store_in_supabase=args.store_supabase,
        supabase_writer=supabase_writer,
    )

    if args.mode == 'responses':
        if not args.date:
            raise SystemExit("--date est requis pour le mode responses")
        pipeline.analyze_responses_for_date(
            date=args.date,
            subfolder=args.subfolder,
            sample_fraction=args.sample_fraction,
            use_context=args.use_context,
            overwrite=args.overwrite,
        )
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                 ✅ ANALYSE RÉPONSES TERMINÉE                   ║
╚════════════════════════════════════════════════════════════════╝

📅 Date : {args.date}
📁 Résultats : {pipeline.output_dir}/{args.subfolder}/date={args.date}

""")
        return

    if args.mode == 'user':
        if not args.user_id:
            raise SystemExit("--user-id est requis pour le mode user")
        pipeline.analyze_user_conversation(
            user_id=args.user_id,
            subfolder=args.subfolder,
            sample_fraction=args.sample_fraction,
            max_files=args.max_files,
            use_context=args.use_context,
            overwrite=args.overwrite,
        )
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                 ✅ ANALYSE UTILISATEUR TERMINÉE               ║
╚════════════════════════════════════════════════════════════════╝

👤 Utilisateur : {args.user_id}
📁 Résultats : {pipeline.output_dir}/{args.subfolder}/users/{args.user_id}

""")
        return

    if args.mode == 'archive':
        pipeline.run_full_archive_classification(
            use_context=args.use_context,
            subfolder=args.subfolder,
            overwrite=args.overwrite
        )
        output_target = pipeline.output_dir / args.subfolder
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                     ✅ ARCHIVE TRAITÉE                        ║
╚════════════════════════════════════════════════════════════════╝

📁 Résultats sauvegardés dans : {output_target}
👉 Relancez avec --overwrite pour régénérer un fichier existant

""")
        return

    # Mode sample
    results = pipeline.run_sample_classification(
        n_samples=args.n_samples,
        use_context=args.use_context
    )
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                     ✅ CLASSIFICATION TERMINÉE                 ║
╚════════════════════════════════════════════════════════════════╝

📁 Résultats sauvegardés dans : data/processed/
📊 {len(results)} messages classifiés

Prochaines étapes :
  1. Visualiser : python analyze_results.py
  2. Valider manuellement : python validate_sample.py
  3. Classifier plus de messages : python classify_sample.py --mode archive

""")

if __name__ == "__main__":
    main()
