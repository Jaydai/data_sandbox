#!/usr/bin/env python3
"""
Script pour classifier un échantillon de messages
"""
import argparse
from src.classification_pipeline import ClassificationPipeline

def main():
    parser = argparse.ArgumentParser(description='Classifier un échantillon de messages')
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=1000,
        help='Nombre de messages à classifier (défaut: 1000)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='qwen2.5:14b',
        help='Modèle Ollama à utiliser (défaut: qwen2.5:14b)'
    )
    parser.add_argument(
        '--use-context',
        action='store_true',
        help='Utiliser le contexte des messages précédents'
    )
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║          CLASSIFICATION DE MESSAGES CHATGPT                    ║
╚════════════════════════════════════════════════════════════════╝

📊 Échantillon : {args.n_samples} messages
🤖 Modèle : {args.model}
🔗 Contexte : {'✅ Activé' if args.use_context else '❌ Désactivé'}

""")
    
    # Initialiser le pipeline
    pipeline = ClassificationPipeline(model=args.model)
    
    # Classifier
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
  3. Classifier plus de messages : python classify_sample.py --n-samples 10000

""")

if __name__ == "__main__":
    main()