#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Enhanced RAG Chatbot with Streaming Support
Modified to support real-time streaming responses
"""

import os
import re
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from typing import Optional, Dict, List, AsyncGenerator
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import logging
import asyncio
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    stream: bool = True  # Add streaming option

class ChatResponse(BaseModel):
    response: str
    sources_count: int
    confidence: str
    avg_score: float
    top_score: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    documents_count: int
    index_status: str

# Enhanced RAG class with streaming support
class EnhancedTawjihRAG:
    def __init__(self, deepseek_api_key):
        """Initialize enhanced RAG system"""
        self.api_key = deepseek_api_key
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_document(self, file_path):
        """Load and process document"""
        logger.info("Loading document...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Enhanced text cleaning
        text = self.clean_text(text)
        
        # Split into chunks
        chunks = self.split_text(text, chunk_size=700, overlap=100)
        self.documents = chunks
        
        logger.info(f"Created {len(chunks)} document chunks")
        self.build_index()
        
    def clean_text(self, text):
        """Enhanced text cleaning"""
        # Remove non-French characters
        text = re.sub(r'[ⵏⴳⵜⵜⵓⵃⵙⵓ\u2D30-\u2D7F\u0600-\u06FF]+', '', text)
        
        # Clean up spacing and formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.\-_]{3,}', ' ', text)
        text = re.sub(r'=+', ' ', text)
        
        return text.strip()
        
    def split_text(self, text, chunk_size=700, overlap=100):
        """Improved text splitting"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = [chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?')]
                best_cut = max([pos for pos in sentence_endings if pos > start + chunk_size // 2] + [-1])
                
                if best_cut > -1:
                    chunk = text[start:start + best_cut + 1]
                    end = start + best_cut + 1
                else:
                    # Fallback to newline
                    last_newline = chunk.rfind('\n')
                    if last_newline > start + chunk_size // 2:
                        chunk = text[start:start + last_newline]
                        end = start + last_newline
            
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks
    
    def build_index(self):
        """Build FAISS vector index"""
        logger.info("Building vector index...")
        
        # Generate embeddings
        self.embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info("Index built successfully!")
    
    def search(self, query, k=5):
        """Search for relevant documents with better scoring"""
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                # Add keyword matching bonus
                keyword_bonus = self.calculate_keyword_bonus(query, self.documents[idx])
                adjusted_score = float(score) + keyword_bonus
                
                results.append({
                    'text': self.documents[idx],
                    'score': adjusted_score,
                    'original_score': float(score),
                    'keyword_bonus': keyword_bonus
                })
        
        # Re-sort by adjusted score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def calculate_keyword_bonus(self, query, text):
        """Calculate keyword matching bonus"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Important keywords get higher weight
        important_keywords = {
            'médecine', 'ingénieur', 'encg', 'ensa', 'université', 'institut',
            'condition', 'accès', 'inscription', 'formation', 'école'
        }
        
        bonus = 0
        for word in query_words:
            if word in text_words:
                if word in important_keywords:
                    bonus += 0.1  # Higher bonus for important words
                else:
                    bonus += 0.05
        
        return bonus
    
    async def generate_response_stream(self, query, search_results) -> AsyncGenerator[str, None]:
        """Generate streaming response using DeepSeek API"""
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            context_parts.append(f"Source {i} (Score: {result['score']:.2f}):\n{result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Enhanced system prompt
        system_prompt = """Tu es un conseiller d'orientation spécialisé dans le système éducatif marocain. 

Règles importantes:
- Réponds UNIQUEMENT en français
- Si l'utilisateur dit juste "salut", "bonjour", "hi" ou une salutation simple, réponds par une salutation amicale et demande-lui comment tu peux l'aider avec son orientation
- Base-toi uniquement sur les informations fournies pour répondre aux vraies questions d'orientation
- Structure ta réponse de manière claire et organisée
- Mentionne les conditions d'accès, procédures et contacts quand disponibles
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois encourageant et pratique dans tes conseils
- N'invente pas d'informations qui ne sont pas dans les sources
- ne donne pas les liens des sitwebs
"""
        
        # Enhanced user prompt
        user_prompt = f"""Question de l'étudiant: {query}

Informations trouvées dans le guide d'orientation:
{context}

Instructions spéciales:
- Si la question est juste une salutation (salut, bonjour, hi, hello), réponds par une salutation et demande comment tu peux aider avec l'orientation scolaire
- Sinon, réponds de manière structurée en aidant l'étudiant avec des informations précises basées uniquement sur le contexte fourni"""
        
        # Use httpx for async streaming
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1200,
            "stream": True  # Enable streaming
        }
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    'POST',
                    'https://api.deepseek.com/v1/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=30.0
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            json_str = line[6:]  # Remove 'data: ' prefix
                            
                            if json_str == '[DONE]':
                                break
                                
                            try:
                                chunk = json.loads(json_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                                
            except httpx.HTTPStatusError as e:
                logger.error(f"API Error: {e.response.status_code} - {e.response.text}")
                yield self.fallback_response(search_results)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield self.fallback_response(search_results)
    
    def generate_response(self, query, search_results):
        """Non-streaming response generation (fallback)"""
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            context_parts.append(f"Source {i} (Score: {result['score']:.2f}):\n{result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # System and user prompts (same as streaming version)
        system_prompt = """Tu es un conseiller d'orientation spécialisé dans le système éducatif marocain. 

Règles importantes:
- Réponds UNIQUEMENT en français
- Si l'utilisateur dit juste "salut", "bonjour", "hi" ou une salutation simple, réponds par une salutation amicale et demande-lui comment tu peux l'aider avec son orientation
- Base-toi uniquement sur les informations fournies pour répondre aux vraies questions d'orientation
- Structure ta réponse de manière claire et organisée
- Mentionne les conditions d'accès, procédures et contacts quand disponibles
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois encourageant et pratique dans tes conseils
- N'invente pas d'informations qui ne sont pas dans les sources"""
        
        user_prompt = f"""Question de l'étudiant: {query}

Informations trouvées dans le guide d'orientation:
{context}

Instructions spéciales:
- Si la question est juste une salutation (salut, bonjour, hi, hello), réponds par une salutation et demande comment tu peux aider avec l'orientation scolaire
- Sinon, réponds de manière structurée en aidant l'étudiant avec des informations précises basées uniquement sur le contexte fourni"""
        
        # Call DeepSeek API
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1200
            }
            
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return self.fallback_response(search_results)
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return self.fallback_response(search_results)
    
    def fallback_response(self, search_results):
        """Enhanced fallback response"""
        if not search_results:
            return "❌ Désolé, je n'ai pas trouvé d'informations pertinentes pour votre question. Essayez de reformuler avec des mots-clés différents."
        
        response = "📚 **Informations trouvées dans le guide d'orientation:**\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            text = result['text']
            # Truncate intelligently
            if len(text) > 250:
                # Try to cut at sentence end
                cut_pos = text.rfind('.', 0, 250)
                if cut_pos > 100:
                    text = text[:cut_pos + 1]
                else:
                    text = text[:250] + "..."
            
            response += f"**{i}.** {text}\n\n"
        
        response += "💡 **Conseil:** Pour une information plus précise, contactez directement l'institution qui vous intéresse."
        return response
    
    async def chat_stream(self, question):
        """Streaming chat function"""
        # Search for relevant documents
        search_results = self.search(question, k=10)
        
        # Calculate confidence based on scores
        avg_score = np.mean([r['score'] for r in search_results]) if search_results else 0
        confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
        
        # Metadata to send before streaming
        metadata = {
            'sources_count': len(search_results),
            'confidence': confidence,
            'avg_score': float(avg_score),
            'top_score': float(search_results[0]['score']) if search_results else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # First, yield the metadata as a special message
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
        
        # Then stream the response
        async for chunk in self.generate_response_stream(question, search_results):
            yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
        
        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    def chat(self, question):
        """Non-streaming chat function"""
        # Search for relevant documents
        search_results = self.search(question, k=6)
        
        # Generate response
        response = self.generate_response(question, search_results)
        
        # Calculate confidence based on scores
        avg_score = np.mean([r['score'] for r in search_results]) if search_results else 0
        confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
        
        return {
            'response': response,
            'sources_count': len(search_results),
            'confidence': confidence,
            'avg_score': float(avg_score),
            'top_score': float(search_results[0]['score']) if search_results else 0
        }
    
    def save_conversation(self, question, response, filename="conversation_log.json"):
        """Save conversation for later analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response.get('response', ''),
            'confidence': response.get('confidence', ''),
            'sources_count': response.get('sources_count', 0)
        }
        
        # Load existing log or create new one
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except FileNotFoundError:
            conversations = []
        
        conversations.append(log_entry)
        
        # Save updated log
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)

# Initialize FastAPI app
app = FastAPI(
    title="Tawjih RAG Chatbot API",
    description="API for Moroccan Higher Education Orientation Chatbot with Streaming Support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot: Optional[EnhancedTawjihRAG] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup"""
    global chatbot
    
    # Configuration
    API_KEY = "sk-c6d95fab838d450eac64cb68c223d7ef"
    DOCUMENT_PATH = r"D:\DIRASSA\3_eme_annee\S6\DL\PFM\RAG\PDF\french_output.txt"
    
    try:
        # Check if document exists
        if not os.path.exists(DOCUMENT_PATH):
            logger.error(f"Document {DOCUMENT_PATH} not found!")
            raise FileNotFoundError(f"Document {DOCUMENT_PATH} not found!")
        
        # Initialize chatbot
        logger.info("Initializing Enhanced Tawjih RAG Chatbot...")
        chatbot = EnhancedTawjihRAG(API_KEY)
        
        # Load document
        chatbot.load_document(DOCUMENT_PATH)
        logger.info("Chatbot initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Tawjih RAG Chatbot API",
        "status": "active",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global chatbot
    
    return HealthResponse(
        status="healthy" if chatbot else "unhealthy",
        model_loaded=chatbot is not None,
        documents_count=len(chatbot.documents) if chatbot else 0,
        index_status="ready" if chatbot and chatbot.index else "not_ready"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    """
    Non-streaming chat endpoint
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        ChatResponse with the complete answer and metadata
    """
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        # Get response from chatbot
        result = chatbot.chat(request.question)
        
        # Save conversation
        chatbot.save_conversation(request.question, result)
        
        # Return response
        return ChatResponse(
            response=result['response'],
            sources_count=result['sources_count'],
            confidence=result['confidence'],
            avg_score=result['avg_score'],
            top_score=result['top_score'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/chat/stream")
async def chat_stream(request: QuestionRequest):
    """
    Streaming chat endpoint - returns Server-Sent Events
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        StreamingResponse with SSE data
    """
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    async def generate():
        try:
            async for chunk in chatbot.chat_stream(request.question):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )

@app.get("/examples", response_model=Dict[str, List[str]])
async def get_examples():
    """Get example questions"""
    return {
        "examples": [
            "Quelles sont les conditions pour étudier la médecine ?",
            "Comment s'inscrire à l'ENCG de Tanger ?",
            "Instituts de technologie avec bac Sciences Math",
            "Frais d'inscription université Hassan II",
            "Salut, je cherche une orientation",
            "Bonjour, j'ai besoin d'aide"
        ]
    }

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "main:app",  # Change "main" to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )