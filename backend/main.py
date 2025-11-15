#!/usr/bin/env python3
"""
FastAPI Educational RAG Chatbot - Main Application
Refactored for better maintainability and security
"""
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from config import Config
from models import HealthResponse, ErrorResponse
from middleware import (
    setup_cors,
    setup_rate_limiting,
    log_requests,
    SecurityHeadersMiddleware,
    limiter
)
from db_service import DatabaseService

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Educational RAG API",
    description="AI-powered educational chatbot with course-specific context",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state
db_service: DatabaseService = None
rag_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global db_service, rag_system

    try:
        logger.info("=" * 50)
        logger.info("Starting Educational RAG API v2.0.0")
        logger.info("=" * 50)

        # Initialize database service
        logger.info(f"Initializing database at: {Config.DATABASE_PATH}")
        db_service = DatabaseService(Config.get_database_path())

        # Note: RAG system initialization moved to separate module
        # This is imported here to avoid circular dependencies
        from rag_service import EducationalRAGService
        logger.info("Initializing RAG system...")
        rag_system = EducationalRAGService(db_service)

        logger.info("✅ Application started successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_service

    logger.info("Shutting down application...")

    if db_service:
        db_service.close()

    logger.info("✅ Application shutdown complete")


# Setup middleware
setup_cors(app, Config.ALLOWED_ORIGINS)
setup_rate_limiting(app)
app.middleware("http")(log_requests)
app.add_middleware(SecurityHeadersMiddleware)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
@limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
async def health_check():
    """Health check endpoint"""
    try:
        total_docs = 0
        courses_loaded = 0

        if rag_system:
            courses_loaded = len(rag_system.course_indexes)
            for course_docs in rag_system.course_documents.values():
                total_docs += len(course_docs)

        return HealthResponse(
            status="healthy",
            model_loaded=rag_system is not None,
            courses_loaded=courses_loaded,
            total_documents=total_docs,
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Health check failed",
                detail=str(e)
            ).dict()
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Educational RAG Chatbot API v2.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "courses": "/courses/{course_id}/load",
            "health": "/health"
        }
    }


# Import and include routes
# This is done after app creation to avoid circular imports
try:
    from routes import setup_routes
    setup_routes(app, rag_system, db_service, limiter)
    logger.info("✅ Routes configured successfully")
except ImportError as e:
    logger.warning(f"Routes not loaded: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    )
