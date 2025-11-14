#!/usr/bin/env python3
"""
Validation manuelle d'un échantillon de classifications
"""
import pandas as pd
from pathlib import Path
import random
from src.data_loader import SupabaseDataLoader

def load_latest_results(results_dir: str = "data/processed") -> pd.DataFrame:
    """Charge le dernier fichier de résultats"""
    results_path = Path(results_dir)
    files = list(results_path.glob("classified_sample_*.parquet"))
    
    if not files:
        raise FileNotFoundError("Aucun fichier de résultats trouvé")
    
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Chargement de : {latest_file}\n")
    
    return pd.read_parquet(latest_file)

def load_original_messages() -> pd.DataFrame:
    """Charge les messages originaux depuis Supabase"""
    print("📥 Chargement des messages originaux depuis Supabase...")
    loader = SupabaseDataLoader()
    
    # Charger un échantillon large pour avoir les messages
    df = loader.get_sample_messages(n_samples=10000, subfolder="messages")
    print(f"✅ {len(df)} messages originaux chargés\n")
    
    return df

def merge_results_with_content(results_df: pd.DataFrame, messages_df: pd.DataFrame) -> pd.DataFrame:
    """Merge les résultats avec le contenu original"""
    
    # Créer une colonne d'ID commune
    if 'id' in messages_df.columns:
        messages_df = messages_df.rename(columns={'id': 'message_id'})
    
    # Merge
    merged = results_df.merge(
        messages_df[['message_id', 'content', 'role']], 
        on='message_id', 
        how='left',
        suffixes=('', '_original')
    )
    
    return merged

def validate_interactive(df: pd.DataFrame, n_samples: int = 20):
    """Validation interactive d'un échantillon"""
    
    print("=" * 80)
    print("🔍 VALIDATION MANUELLE DES CLASSIFICATIONS")
    print("=" * 80)
    print(f"\nVous allez valider {n_samples} messages aléatoires.")
    print("Pour chaque message, indiquez si la classification est correcte.\n")
    
    # Filtrer seulement les messages avec contenu
    df_with_content = df[df['content'].notna()].copy()
    
    if len(df_with_content) == 0:
        print("❌ Aucun message avec contenu disponible")
        return
    
    print(f"✅ {len(df_with_content)} messages avec contenu disponibles\n")
    
    # Échantillonner
    n_samples = min(n_samples, len(df_with_content))
    sample = df_with_content.sample(n=n_samples, random_state=42)
    
    results = []
    
    for i, (idx, row) in enumerate(sample.iterrows(), 1):
        print("\n" + "=" * 80)
        print(f"MESSAGE {i}/{n_samples}")
        print("=" * 80)
        
        # Afficher le message
        content = str(row['content'])
        
        if len(content) > 500:
            print(f"\n📝 CONTENU (tronqué) :")
            print("-" * 80)
            print(f"{content[:500]}...")
            print("-" * 80)
        else:
            print(f"\n📝 CONTENU :")
            print("-" * 80)
            print(content)
            print("-" * 80)
        
        print(f"\n📊 LONGUEUR : {row['content_length']} caractères")
        
        print(f"\n🔍 CLASSIFICATIONS :")
        print(f"   🏢 Work : {'✅ OUI' if row['is_work'] else '❌ NON'}")
        print(f"      Confiance : {row['work_confidence']}")
        print(f"      Raison : {row['work_reasoning']}")
        
        print(f"\n   📋 Topic : {row['topic']}")
        print(f"      Sous-topic : {row['sub_topic']}")
        print(f"      Confiance : {row['topic_confidence']}")
        
        print(f"\n   🎯 Intent : {row['intent']}")
        print(f"      Confiance : {row['intent_confidence']}")
        print(f"      Raison : {row['intent_reasoning']}")
        
        # Validation
        print(f"\n" + "=" * 80)
        print(f"❓ VOTRE ÉVALUATION :")
        print("=" * 80)
        print(f"   1. ✅ Tout est correct")
        print(f"   2. ❌ Work incorrect (devrait être {'Non-Work' if row['is_work'] else 'Work'})")
        print(f"   3. ❌ Topic incorrect")
        print(f"   4. ❌ Intent incorrect")
        print(f"   5. ❌ Plusieurs erreurs")
        print(f"   s. ⏭️  Skip (passer au suivant)")
        print(f"   q. 🚪 Quitter")
        
        choice = input("\n👉 Votre choix : ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Validation interrompue")
            break
        elif choice == 's':
            continue
        
        result = {
            'message_id': row['message_id'],
            'content_preview': content[:100],
            'is_work': row['is_work'],
            'topic': row['topic'],
            'intent': row['intent'],
            'choice': choice,
            'is_correct': choice == '1',
            'error_type': {
                '2': 'work',
                '3': 'topic',
                '4': 'intent',
                '5': 'multiple'
            }.get(choice, 'none')
        }
        results.append(result)
    
    # Statistiques
    if results:
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        accuracy = correct / total * 100
        
        print("\n" + "=" * 80)
        print("📊 RÉSULTATS DE LA VALIDATION")
        print("=" * 80)
        print(f"\n✅ Classifications correctes : {correct}/{total} ({accuracy:.1f}%)")
        print(f"❌ Classifications incorrectes : {total - correct}/{total} ({100-accuracy:.1f}%)")
        
        # Détail des erreurs
        if total - correct > 0:
            print(f"\n📋 DÉTAIL DES ERREURS :")
            error_types = {}
            for r in results:
                if not r['is_correct']:
                    error_type = r['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"   {error_type:<10} : {count} erreur(s)")
        
        # Sauvegarder
        validation_df = pd.DataFrame(results)
        output_path = Path("data/results/validation.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_df.to_csv(output_path, index=False)
        print(f"\n💾 Validation sauvegardée : {output_path}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS :")
        if accuracy >= 90:
            print("   🎉 Excellente qualité ! Tu peux classifier plus de messages.")
        elif accuracy >= 80:
            print("   ✅ Bonne qualité. Tu peux continuer avec prudence.")
        elif accuracy >= 70:
            print("   ⚠️  Qualité moyenne. Considère ajuster les prompts.")
        else:
            print("   ❌ Qualité insuffisante. Il faut améliorer les prompts ou changer de modèle.")
        
        print("\n" + "=" * 80)

def main():
    print("\n" + "=" * 80)
    print("🔍 VALIDATION MANUELLE DES CLASSIFICATIONS")
    print("=" * 80 + "\n")
    
    # Charger les résultats de classification
    results_df = load_latest_results()
    
    # Charger les messages originaux
    messages_df = load_original_messages()
    
    # Merger pour avoir le contenu
    print("🔗 Fusion des données...")
    merged_df = merge_results_with_content(results_df, messages_df)
    
    messages_with_content = merged_df['content'].notna().sum()
    print(f"✅ {messages_with_content}/{len(merged_df)} messages avec contenu trouvés\n")
    
    if messages_with_content == 0:
        print("❌ Aucun contenu trouvé. Vérifie que les message_id correspondent.")
        return
    
    # Validation interactive
    validate_interactive(merged_df, n_samples=20)

if __name__ == "__main__":
    main()