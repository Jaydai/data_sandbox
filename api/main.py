from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time
from typing import List
import logging
from loguru import logger

from api.models.request import MessageClassificationRequest, BulkClassificationRequest
from api.models.response import (
    MessageClassificationResponse,
    WorkClassification,
    TopicClassification,
    IntentClassification,
    PIIDetection,
    QualityScore,
    RiskAnalysis
)
from api.services.classifier import ClassifierService
from api.services.pii_detector import PIIDetector
from api.services.quality_scorer import QualityScorer
from api.services.risk_analyzer import RiskAnalyzer
from src.enrichment import PublicFigureChecker

# Configuration logging
logging.basicConfig(level=logging.INFO)

# Initialisation FastAPI
app = FastAPI(
    title="Jaydai Classification API",
    description="API de classification et analyse de messages IA pour entreprises",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (à restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: ["https://jaydai.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des services
classifier_service = ClassifierService(model="gemma-9b")
public_checker = PublicFigureChecker(classifier_service.engine)
pii_detector = PIIDetector(public_figure_checker=public_checker)
quality_scorer = QualityScorer()
risk_analyzer = RiskAnalyzer()

logger.info("✅ Services initialisés")


@app.on_event("startup")
async def startup_event():
    """Événement au démarrage"""
    logger.info("🚀 Jaydai Classification API démarrée")
    logger.info(f"📝 Documentation: http://localhost:8000/docs")


@app.get("/")
def root():
    """Endpoint racine"""
    return {
        "service": "Jaydai Classification API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "classify": "/classify",
            "bulk_classify": "/classify/bulk",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "classifier": "operational",
            "pii_detector": "operational",
            "quality_scorer": "operational",
            "risk_analyzer": "operational"
        }
    }


@app.post("/classify", response_model=MessageClassificationResponse)
async def classify_message(request: MessageClassificationRequest):
    """
    Classifier un message et retourner l'analyse complète
    
    ## Paramètres
    - **content** (requis): Contenu du message à analyser
    - **user_id** (requis): ID de l'utilisateur
    - **organization_id** (requis): ID de l'organisation
    - **context** (optionnel): Messages précédents pour contexte
    - **detect_pii** (défaut: true): Activer la détection PII
    - **assess_quality** (défaut: true): Activer l'évaluation qualité
    - **analyze_risks** (défaut: true): Activer l'analyse de risques
    
    ## Retourne
    Classification complète avec:
    - Work/Non-Work classification
    - Topic et sous-topic
    - Intent (Asking/Doing/Expressing)
    - Détection PII (optionnel)
    - Score de qualité (optionnel)
    - Analyse de risques (optionnel)
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Classification pour user={request.user_id}, org={request.organization_id}")
        
        # 1. Classification de base (work/topic/intent)
        classification = classifier_service.classify(
            content=request.content,
            context=request.context
        )
        
        work = WorkClassification(**classification["work"])
        topic = TopicClassification(**classification["topic"])
        intent = IntentClassification(**classification["intent"])
        
        # 2. Détection PII (optionnel)
        pii_detection = None
        if request.detect_pii:
            pii_result = pii_detector.detect(request.content)
            if pii_result:
                pii_detection = PIIDetection(
                    has_pii=pii_result.get("has_pii", False),
                    pii_types=pii_result.get("pii_types", []),
                    entities_found=pii_result.get("entities", []),
                    risk_level=pii_result.get("risk_level", "none"),
                    recommendations=pii_result.get("recommendations", [])
                )
        
        # 3. Score qualité (optionnel)
        quality_score = None
        if request.assess_quality:
            quality_result = quality_scorer.score(
                request.content,
                request.context
            )
            if quality_result:
                quality_score = QualityScore(**quality_result)
        
        # 4. Analyse risques (optionnel)
        risk_analysis = None
        if request.analyze_risks and pii_detection and quality_score:
            risk_result = risk_analyzer.analyze(
                content=request.content,
                classification=classification,
                pii_detection=pii_result,
                quality_score=quality_result
            )
            if risk_result:
                risk_analysis = RiskAnalysis(**risk_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = MessageClassificationResponse(
            message_id=request.message_id,
            user_id=request.user_id,
            organization_id=request.organization_id,
            processed_at=datetime.utcnow(),
            work=work,
            topic=topic,
            intent=intent,
            pii_detection=pii_detection,
            quality_score=quality_score,
            risk_analysis=risk_analysis,
            processing_time_ms=processing_time,
            model_used=classifier_service.model_name
        )
        
        logger.info(f"✅ Classification terminée en {processing_time:.0f}ms")
        return response
    
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/bulk")
async def classify_bulk(
    request: BulkClassificationRequest,
    background_tasks: BackgroundTasks
):
    """
    Classifier plusieurs messages en batch
    
    ## Paramètres
    - **messages**: Liste de messages à classifier
    - **organization_id**: ID de l'organisation
    - **async_processing**: Si true, traitement asynchrone
    
    ## Retourne
    - Mode sync: Liste complète des résultats
    - Mode async: Job ID pour suivi
    """
    if request.async_processing:
        # TODO: Implémenter traitement asynchrone avec queue (Celery/Redis)
        background_tasks.add_task(
            process_bulk_classification, 
            request.messages, 
            request.organization_id
        )
        return {
            "status": "processing",
            "message": f"{len(request.messages)} messages en cours de traitement",
            "job_id": f"job_{int(time.time())}"
        }
    else:
        # Traitement synchrone
        results = []
        for msg_request in request.messages:
            try:
                result = await classify_message(msg_request)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Erreur sur message {msg_request.message_id}: {e}")
                continue
        
        return {
            "status": "completed",
            "total": len(request.messages),
            "processed": len(results),
            "results": results
        }


async def process_bulk_classification(messages: List, org_id: str):
    """Traiter un batch en arrière-plan (TODO: implémenter avec queue)"""
    logger.info(f"📦 Traitement batch pour org={org_id}, {len(messages)} messages")
    # TODO: Implémenter avec Celery + Redis
    pass


@app.get("/stats/organization/{org_id}")
async def get_organization_stats(org_id: str, days: int = 30):
    """
    Statistiques d'utilisation pour une organisation
    
    ## Paramètres
    - **org_id**: ID de l'organisation
    - **days**: Nombre de jours (défaut: 30)
    
    ## Retourne
    Statistiques agrégées:
    - Volume de messages
    - Répartition work/non-work
    - Topics principaux
    - Scores de qualité moyens
    - Alertes de risques
    """
    # TODO: Implémenter avec base de données
    return {
        "organization_id": org_id,
        "period_days": days,
        "total_messages": 0,
        "work_percentage": 0.0,
        "top_topics": [],
        "avg_quality_score": 0.0,
        "risk_alerts": []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
