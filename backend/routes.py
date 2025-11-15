"""
API Routes for Educational RAG System
"""
import logging
import json
from typing import Optional
from fastapi import HTTPException, status, Request
from fastapi.responses import StreamingResponse
from datetime import datetime

from models import QuestionRequest, ChatResponse
from utils import calculate_confidence, format_timestamp
from config import Config

logger = logging.getLogger(__name__)


def setup_routes(app, rag_system, db_service, limiter):
    """
    Setup all API routes

    Args:
        app: FastAPI application
        rag_system: RAG service instance
        db_service: Database service instance
        limiter: Rate limiter instance
    """

    @app.get("/courses/{course_id}/load")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def load_course(request: Request, course_id: int):
        """Load course materials into RAG system"""
        try:
            logger.info(f"Loading course {course_id}")

            if not rag_system:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG system not initialized"
                )

            success = rag_system.load_course_materials(course_id)

            if success:
                doc_count = len(rag_system.course_documents.get(course_id, []))
                return {
                    "success": True,
                    "course_id": course_id,
                    "documents_loaded": doc_count,
                    "message": f"Course {course_id} loaded successfully"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Failed to load course {course_id}"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading course {course_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @app.post("/chat")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def chat(request: Request, question_req: QuestionRequest):
        """Non-streaming chat endpoint"""
        try:
            if not rag_system:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG system not initialized"
                )

            # Search for relevant content
            search_results = rag_system.search(
                question_req.question,
                question_req.course_id,
                k=5
            )

            if not search_results:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No relevant content found for this question"
                )

            # Generate response (collect streamed response)
            response_text = ""
            async for chunk in rag_system.generate_response_stream(
                question_req.question,
                search_results,
                question_req.course_id,
                question_req.user_profile
            ):
                response_text += chunk

            # Calculate metrics
            scores = [r['score'] for r in search_results]
            confidence = calculate_confidence(scores)

            # Get course name
            course_name = None
            if question_req.course_id:
                course_info = db_service.get_course_info(question_req.course_id)
                if course_info:
                    course_name = course_info.get('name')

            return ChatResponse(
                response=response_text,
                sources_count=len(search_results),
                confidence=confidence,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                top_score=max(scores) if scores else 0.0,
                course_name=course_name,
                timestamp=format_timestamp()
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @app.post("/chat/stream")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def chat_stream(request: Request, question_req: QuestionRequest):
        """Streaming chat endpoint using Server-Sent Events"""
        try:
            if not rag_system:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG system not initialized"
                )

            # Search for relevant content
            search_results = rag_system.search(
                question_req.question,
                question_req.course_id,
                k=5
            )

            if not search_results:
                # Send error message
                async def error_stream():
                    yield f"data: {json.dumps({'type': 'error', 'data': 'No relevant content found'})}\n\n"

                return StreamingResponse(
                    error_stream(),
                    media_type="text/event-stream"
                )

            # Stream response
            async def generate():
                try:
                    # Send metadata first
                    scores = [r['score'] for r in search_results]
                    confidence = calculate_confidence(scores)

                    course_name = None
                    if question_req.course_id:
                        course_info = db_service.get_course_info(question_req.course_id)
                        if course_info:
                            course_name = course_info.get('name')

                    metadata = {
                        'sources_count': len(search_results),
                        'confidence': confidence,
                        'avg_score': sum(scores) / len(scores) if scores else 0.0,
                        'top_score': max(scores) if scores else 0.0,
                        'course_name': course_name,
                        'timestamp': format_timestamp()
                    }

                    yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

                    # Stream content
                    async for chunk in rag_system.generate_response_stream(
                        question_req.question,
                        search_results,
                        question_req.course_id,
                        question_req.user_profile
                    ):
                        yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"

                    # Send done signal
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                except Exception as e:
                    logger.error(f"Stream generation error: {e}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    logger.info("✅ Routes configured successfully")
