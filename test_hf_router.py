#!/usr/bin/env python3
from src.classifiers_hf_router import HFRouterClassifier
import json
import time

def test_hf_router():
    print("🧪 Test HF Router Classifier\n")
    print("=" * 80)
    
    # Initialiser avec Gemma 9B
    classifier = HFRouterClassifier(model="gemma-9b")
    
    # Messages de test
    test_messages = [
        {
            "name": "Email Professionnel",
            "content": "Peux-tu rédiger un email professionnel pour mon manager expliquant le retard du projet Q4 ?"
        },
        {
            "name": "Recette Cuisine",
            "content": "Donne-moi une recette de crêpes facile pour le petit-déjeuner."
        },
        {
            "name": "Code Python",
            "content": "Comment faire une boucle for en Python qui affiche les nombres de 1 à 10 ?"
        },
        {
            "name": "Question Factuelle",
            "content": "Quelle est la capitale du Japon et combien d'habitants y vivent ?"
        }
    ]
    
    total_time = 0
    
    for i, test in enumerate(test_messages, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_messages)} : {test['name']}")
        print(f"{'='*80}")
        print(f"\n📝 MESSAGE : {test['content']}")
        
        # Mesurer le temps
        start_time = time.time()
        
        print(f"\n⏳ Classification en cours...")
        result = classifier.classify_complete(test['content'])
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Afficher les résultats
        print(f"\n⏱️  Temps : {elapsed:.2f}s")
        print(f"\n✅ RÉSULTATS :")
        print(f"\n🏢 WORK : {'✅ OUI' if result['work']['is_work'] else '❌ NON'} ({result['work']['confidence']})")
        print(f"   {result['work']['reasoning']}")
        print(f"\n📋 TOPIC : {result['topic']['topic']} ({result['topic']['confidence']})")
        print(f"   {result['topic']['sub_topic']}")
        print(f"\n🎯 INTENT : {result['intent']['intent']} ({result['intent']['confidence']})")
        print(f"   {result['intent']['reasoning']}")
        
        print(f"\n📄 JSON complet :")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*80}")
    print(f"✅ Tests terminés !")
    print(f"⏱️  Temps total : {total_time:.2f}s")
    print(f"⏱️  Temps moyen : {total_time/len(test_messages):.2f}s par message")
    print("=" * 80)

if __name__ == "__main__":
    test_hf_router()