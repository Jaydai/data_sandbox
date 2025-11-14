from src.classifiers_local import LocalMessageClassifier
import json
import time

def test_local_classifier():
    print("🧪 Test des Classifiers Open-Source Locaux\n")
    print("=" * 80)
    
    # Choisir le modèle
    model = "qwen2.5:14b"  # Change si tu veux tester un autre modèle
    print(f"🤖 Modèle : {model}")
    print("=" * 80)
    
    # Initialiser le classifier
    classifier = LocalMessageClassifier(model=model)
    
    # Messages de test
    test_messages = [
        {
            "name": "Email Professionnel",
            "content": "Peux-tu rédiger un email pour mon manager expliquant pourquoi le projet est en retard ?"
        },
        {
            "name": "Recette Cuisine",
            "content": "Donne-moi une recette de pancakes facile pour le petit-déjeuner."
        },
        {
            "name": "Code Python",
            "content": "Comment faire une boucle for en Python qui affiche les nombres de 1 à 10 ?"
        },
        {
            "name": "Question Factuelle",
            "content": "Quelle est la capitale du Japon ?"
        }
    ]
    
    total_time = 0
    
    for i, test in enumerate(test_messages, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(test_messages)} : {test['name']}")
        print(f"{'=' * 80}")
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
    
    print(f"\n{'=' * 80}")
    print(f"✅ Tests terminés !")
    print(f"⏱️  Temps total : {total_time:.2f}s")
    print(f"⏱️  Temps moyen : {total_time/len(test_messages):.2f}s par message")
    print("=" * 80)

if __name__ == "__main__":
    test_local_classifier()