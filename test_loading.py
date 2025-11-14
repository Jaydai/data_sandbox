from src.data_loader import SupabaseDataLoader
import pandas as pd

def main():
    print("🚀 Test de chargement des données Supabase\n")
    
    # Initialiser le loader
    loader = SupabaseDataLoader()
    
    # 1. Lister les dossiers disponibles dans messages/
    print("=" * 60)
    print("ÉTAPE 1 : Listing des dossiers de dates dans messages/")
    print("=" * 60)
    
    date_folders = loader.list_date_folders(subfolder="messages")
    print(f"\n📁 {len(date_folders)} dossiers trouvés")
    
    if date_folders:
        print(f"Premier dossier : {date_folders[0]}")
        print(f"Dernier dossier : {date_folders[-1]}")
        print(f"\nPremiers 10 dossiers :")
        for folder in date_folders[:10]:
            print(f"  📁 {folder}")
    else:
        print("❌ Aucun dossier trouvé. Vérifie tes credentials Supabase.")
        return
    
    # 2. Vérifier aussi le dossier chats/
    print("\n" + "=" * 60)
    print("VÉRIFICATION : Dossiers dans chats/")
    print("=" * 60)
    
    chat_folders = loader.list_date_folders(subfolder="chats")
    print(f"📁 {len(chat_folders)} dossiers trouvés dans chats/")
    
    # 3. Lister les fichiers dans le premier dossier de messages
    print("\n" + "=" * 60)
    print(f"ÉTAPE 2 : Fichiers dans messages/{date_folders[0]}")
    print("=" * 60)
    
    files = loader.list_files_in_folder(date_folders[0], subfolder="messages")
    for file in files:
        print(f"  📄 {file}")
    
    # 4. Charger le premier fichier
    if files:
        print("\n" + "=" * 60)
        print(f"ÉTAPE 3 : Chargement du premier fichier")
        print("=" * 60)
        
        df = loader.load_parquet_to_dataframe(files[0])
        
        if df is not None:
            print(f"\n✅ Succès ! DataFrame chargé :")
            print(f"   - {len(df)} lignes")
            print(f"   - {len(df.columns)} colonnes")
            print(f"\nColonnes : {list(df.columns)}")
            print(f"\nAperçu des données :")
            print(df.head(3).to_string())
            
            # Statistiques de base
            print("\n" + "=" * 60)
            print("STATISTIQUES")
            print("=" * 60)
            
            if 'role' in df.columns:
                print(f"\nTypes de rôles :")
                print(df['role'].value_counts())
            
            if 'content' in df.columns:
                df['content_length'] = df['content'].astype(str).str.len()
                print(f"\nLongueur des messages :")
                print(f"   Moyenne : {df['content_length'].mean():.0f} caractères")
                print(f"   Médiane : {df['content_length'].median():.0f} caractères")
                print(f"   Max : {df['content_length'].max():.0f} caractères")
            
            if 'user_id' in df.columns:
                print(f"\nUtilisateurs uniques : {df['user_id'].nunique()}")
    
    # 5. Charger un échantillon aléatoire
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Échantillon aléatoire de 500 messages")
    print("=" * 60)
    
    sample = loader.get_sample_messages(n_samples=500, subfolder="messages")
    
    if not sample.empty:
        print(f"\n✅ {len(sample)} messages échantillonnés")
        
        if 'user_id' in sample.columns:
            print(f"\nUtilisateurs uniques : {sample['user_id'].nunique()}")
            print(f"\nTop 5 utilisateurs les plus actifs :")
            print(sample['user_id'].value_counts().head(5))
        
        if 'model' in sample.columns:
            print(f"\nModèles utilisés :")
            print(sample[sample['role'] == 'assistant']['model'].value_counts())
    
    print("\n" + "=" * 60)
    print("✅ Test terminé avec succès !")
    print("=" * 60)

if __name__ == "__main__":
    main()