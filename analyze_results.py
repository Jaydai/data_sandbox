#!/usr/bin/env python3
"""
Analyse et visualisation des résultats de classification
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

def load_latest_results(results_dir: str = "data/processed") -> pd.DataFrame:
    """Charge le dernier fichier de résultats"""
    results_path = Path(results_dir)
    
    # Trouver tous les fichiers de classification
    files = list(results_path.glob("classified_sample_*.parquet"))
    
    if not files:
        raise FileNotFoundError("Aucun fichier de résultats trouvé")
    
    # Prendre le plus récent
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Chargement de : {latest_file}")
    
    df = pd.read_parquet(latest_file)
    print(f"✅ {len(df)} messages chargés\n")
    
    return df

def create_visualizations(df: pd.DataFrame, output_dir: str = "data/results"):
    """Crée des visualisations complètes"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Génération des visualisations...\n")
    
    # ============================================================
    # FIGURE 1 : Vue d'ensemble
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📊 ANALYSE DES MESSAGES CHATGPT', fontsize=16, fontweight='bold')
    
    # 1. Work vs Non-Work
    work_counts = df['is_work'].value_counts()
    axes[0, 0].pie(
        work_counts.values, 
        labels=['Non-Work', 'Work'], 
        autopct='%1.1f%%',
        colors=['#3498db', '#e74c3c'],
        startangle=90
    )
    axes[0, 0].set_title('🏢 Work vs Non-Work', fontweight='bold')
    
    # 2. Topics
    topic_counts = df['topic'].value_counts()
    axes[0, 1].barh(topic_counts.index, topic_counts.values, color='#2ecc71')
    axes[0, 1].set_xlabel('Nombre de messages')
    axes[0, 1].set_title('📋 Distribution des Topics', fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # 3. Intents
    intent_counts = df['intent'].value_counts()
    axes[0, 2].bar(intent_counts.index, intent_counts.values, color='#9b59b6')
    axes[0, 2].set_ylabel('Nombre de messages')
    axes[0, 2].set_title('🎯 Distribution des Intentions', fontweight='bold')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Longueur des messages
    axes[1, 0].hist(df['content_length'], bins=50, color='#34495e', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['content_length'].median(), color='red', linestyle='--', 
                       label=f'Médiane: {df["content_length"].median():.0f}')
    axes[1, 0].set_xlabel('Longueur (caractères)')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('📏 Distribution de la Longueur des Messages', fontweight='bold')
    axes[1, 0].legend()
    
    # 5. Topics par Work/Non-Work
    topic_work = pd.crosstab(df['topic'], df['is_work'], normalize='columns') * 100
    topic_work.plot(kind='bar', ax=axes[1, 1], stacked=False)
    axes[1, 1].set_ylabel('Pourcentage (%)')
    axes[1, 1].set_title('📊 Topics par Catégorie (Work/Non-Work)', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(['Non-Work', 'Work'])
    
    # 6. Confiance des classifications
    confidence_data = pd.concat([
        df['work_confidence'].value_counts(normalize=True) * 100,
        df['topic_confidence'].value_counts(normalize=True) * 100,
        df['intent_confidence'].value_counts(normalize=True) * 100
    ], axis=1, keys=['Work', 'Topic', 'Intent'])
    
    confidence_data.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_ylabel('Pourcentage (%)')
    axes[1, 2].set_title('🎯 Confiance des Classifications', fontweight='bold')
    axes[1, 2].tick_params(axis='x', rotation=0)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Sauvegarder
    fig_path = output_path / "overview.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé : {fig_path}")
    
    # ============================================================
    # FIGURE 2 : Analyse approfondie des Topics
    # ============================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('📋 ANALYSE DÉTAILLÉE DES TOPICS', fontsize=16, fontweight='bold')
    
    # Topics work vs non-work (stacked)
    topic_work_stack = pd.crosstab(df['topic'], df['is_work'])
    topic_work_stack.plot(kind='barh', stacked=True, ax=axes2[0, 0], color=['#3498db', '#e74c3c'])
    axes2[0, 0].set_xlabel('Nombre de messages')
    axes2[0, 0].set_title('Topics : Work vs Non-Work (Empilé)', fontweight='bold')
    axes2[0, 0].legend(['Non-Work', 'Work'])
    
    # Intent par Topic
    intent_topic = pd.crosstab(df['topic'], df['intent'], normalize='index') * 100
    intent_topic.plot(kind='bar', ax=axes2[0, 1], stacked=True)
    axes2[0, 1].set_ylabel('Pourcentage (%)')
    axes2[0, 1].set_title('Intentions par Topic', fontweight='bold')
    axes2[0, 1].tick_params(axis='x', rotation=45)
    
    # Longueur moyenne par topic
    length_by_topic = df.groupby('topic')['content_length'].mean().sort_values()
    axes2[1, 0].barh(length_by_topic.index, length_by_topic.values, color='#16a085')
    axes2[1, 0].set_xlabel('Longueur moyenne (caractères)')
    axes2[1, 0].set_title('Longueur Moyenne par Topic', fontweight='bold')
    
    # Top sous-topics
    top_subtopics = df['sub_topic'].value_counts().head(10)
    axes2[1, 1].barh(top_subtopics.index, top_subtopics.values, color='#8e44ad')
    axes2[1, 1].set_xlabel('Nombre de messages')
    axes2[1, 1].set_title('Top 10 Sous-Topics', fontweight='bold')
    axes2[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    fig2_path = output_path / "topics_analysis.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé : {fig2_path}")
    
    plt.show()

def print_statistics(df: pd.DataFrame):
    """Affiche des statistiques détaillées"""
    
    print("=" * 70)
    print("📊 STATISTIQUES DÉTAILLÉES")
    print("=" * 70)
    
    print(f"\n📈 VOLUME")
    print(f"   Total messages : {len(df)}")
    print(f"   Utilisateurs uniques : {df['user_id'].nunique()}")
    
    print(f"\n🏢 WORK / NON-WORK")
    work_dist = df['is_work'].value_counts(normalize=True) * 100
    print(f"   Work : {work_dist.get(True, 0):.1f}%")
    print(f"   Non-Work : {work_dist.get(False, 0):.1f}%")
    
    print(f"\n📋 TOPICS (Top 5)")
    for topic, count in df['topic'].value_counts().head(5).items():
        pct = count / len(df) * 100
        print(f"   {topic:<25} {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n🎯 INTENTS")
    for intent, count in df['intent'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {intent:<15} {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n📏 LONGUEUR DES MESSAGES")
    print(f"   Moyenne : {df['content_length'].mean():.0f} caractères")
    print(f"   Médiane : {df['content_length'].median():.0f} caractères")
    print(f"   Min : {df['content_length'].min():.0f} caractères")
    print(f"   Max : {df['content_length'].max():.0f} caractères")
    
    print(f"\n🎯 CONFIANCE DES CLASSIFICATIONS")
    print(f"\n   Work Confidence :")
    for conf, count in df['work_confidence'].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {conf:<10} {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n   Topic Confidence :")
    for conf, count in df['topic_confidence'].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {conf:<10} {count:>5} ({pct:>5.1f}%)")
    
    print("\n" + "=" * 70)

def export_summary(df: pd.DataFrame, output_dir: str = "data/results"):
    """Exporte un résumé en CSV et JSON"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Statistiques générales
    summary = {
        "total_messages": len(df),
        "unique_users": df['user_id'].nunique(),
        "work_percentage": (df['is_work'].sum() / len(df) * 100),
        "avg_length": df['content_length'].mean(),
        "median_length": df['content_length'].median(),
    }
    
    # Top topics
    summary['top_topics'] = df['topic'].value_counts().head(5).to_dict()
    
    # Intents
    summary['intents'] = df['intent'].value_counts().to_dict()
    
    # Sauvegarder en JSON
    import json
    json_path = output_path / "summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résumé JSON sauvegardé : {json_path}")
    
    # Sauvegarder le DataFrame complet en CSV
    csv_path = output_path / "classified_messages.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV complet sauvegardé : {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyser les résultats de classification')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/processed',
        help='Dossier contenant les résultats'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Dossier de sortie pour les visualisations'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Ne pas générer de visualisations'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("📊 ANALYSE DES RÉSULTATS DE CLASSIFICATION")
    print("=" * 70 + "\n")
    
    # Charger les données
    df = load_latest_results(args.results_dir)
    
    # Afficher les statistiques
    print_statistics(df)
    
    # Exporter le résumé
    export_summary(df, args.output_dir)
    
    # Créer les visualisations
    if not args.no_viz:
        create_visualizations(df, args.output_dir)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()