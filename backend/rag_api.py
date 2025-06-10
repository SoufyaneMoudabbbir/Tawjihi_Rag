#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Educational RAG Chatbot with Multi-Course Support
Enhanced for college student learning with course-specific materials
"""
import PyPDF2
import os
import re
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import sqlite3
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
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    course_id: Optional[int] = None
    user_id: str
    user_profile: Optional[Dict] = None
    stream: bool = True

class ChatResponse(BaseModel):
    response: str
    sources_count: int
    confidence: str
    avg_score: float
    top_score: float
    course_name: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    courses_loaded: int
    total_documents: int

# Enhanced Educational RAG class
class EducationalRAG:
    def __init__(self, deepseek_api_key, database_path):
        """Initialize educational RAG system"""
        self.api_key = deepseek_api_key
        self.database_path = database_path
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Course-specific storage
        self.course_documents = {}  # course_id -> list of documents
        self.course_embeddings = {}  # course_id -> embeddings array
        self.course_indexes = {}    # course_id -> faiss index
        self.course_info = {}       # course_id -> course metadata
        
        # Initialize database connection
        self.init_database()
        
    def init_database(self):
        """Initialize database connection"""
        try:
            self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.conn = None
    
    def get_course_info(self, course_id):
        """Get course information from database"""
        if not self.conn:
            return None
            
        try:
            cursor = self.conn.execute(
                "SELECT * FROM courses WHERE id = ?", (course_id,)
            )
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error fetching course info: {e}")
            return None
    
    def get_course_files(self, course_id):
        """Get course file paths from database"""
        if not self.conn:
            return []
            
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM course_files WHERE course_id = ?", (course_id,)
            )
            return [row['file_path'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching course files: {e}")
            return []
    
    def load_course_materials(self, course_id):
        """Load and process course materials"""
        if course_id in self.course_documents:
            logger.info(f"Course {course_id} already loaded")
            return True
            
        # Get course info
        course_info = self.get_course_info(course_id)
        if not course_info:
            logger.warning(f"Course {course_id} not found")
            return False
            
        # Get course files
        file_paths = self.get_course_files(course_id)
        if not file_paths:
            logger.warning(f"No files found for course {course_id}")
            return False
        
        logger.info(f"Loading {len(file_paths)} files for course {course_id}")
        
        # Store course info
        self.course_info[course_id] = dict(course_info)
        
        # Process all files for this course
        all_documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} chunks from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        if not all_documents:
            logger.warning(f"No documents loaded for course {course_id}")
            return False
            
        # Store documents and build index
        self.course_documents[course_id] = all_documents
        self.build_course_index(course_id)
        
        logger.info(f"Successfully loaded course {course_id} with {len(all_documents)} document chunks")
        return True
    
    def load_document(self, file_path):
        """Load and process a single document - handles PDFs and text files"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                # Handle PDF files
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                
                logger.info(f"Extracted text from {num_pages} pages in PDF: {file_path}")
                
            elif file_ext in ['.txt', '.md']:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return []
            
            # Enhanced text cleaning
            text = self.clean_text(text)
            
            # Split into chunks
            chunks = self.split_text(text, chunk_size=700, overlap=100)
            
            # Add metadata to chunks
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = f"[Source: {Path(file_path).name}, Part {i+1}/{len(chunks)}]\n{chunk}"
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    def clean_text(self, text):
        """Enhanced text cleaning for educational content"""
        # Remove non-useful characters but preserve educational formatting
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\-\+\=\%\$\#\@\&\*\/\\\[\]\{\}\|]', '', text)
        
        # Clean up spacing and formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.\-_]{3,}', ' ', text)
        text = re.sub(r'=+', ' ', text)
        
        return text.strip()
        
    def split_text(self, text, chunk_size=700, overlap=100):
        """Improved text splitting for educational content"""
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
                    # Fallback to paragraph break
                    last_paragraph = chunk.rfind('\n\n')
                    if last_paragraph > start + chunk_size // 2:
                        chunk = text[start:start + last_paragraph]
                        end = start + last_paragraph
            
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks
    
    def build_course_index(self, course_id):
        """Build FAISS vector index for a specific course"""
        if course_id not in self.course_documents:
            logger.error(f"No documents found for course {course_id}")
            return False
            
        logger.info(f"Building vector index for course {course_id}...")
        
        # Generate embeddings
        documents = self.course_documents[course_id]
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Store embeddings and index
        self.course_embeddings[course_id] = embeddings
        self.course_indexes[course_id] = index
        
        logger.info(f"Index built successfully for course {course_id}!")
        return True
    
    def search(self, query, course_id=None, k=5):
        """Search for relevant documents"""
        if course_id and course_id not in self.course_indexes:
            # Try to load course materials
            if not self.load_course_materials(course_id):
                return []
        
        # Determine which index to search
        if course_id and course_id in self.course_indexes:
            # Search specific course
            index = self.course_indexes[course_id]
            documents = self.course_documents[course_id]
        elif len(self.course_indexes) > 0:
            # Search all courses or first available course
            course_id = list(self.course_indexes.keys())[0]
            index = self.course_indexes[course_id]
            documents = self.course_documents[course_id]
        else:
            # No courses loaded
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(documents):
                # Add keyword matching bonus
                keyword_bonus = self.calculate_keyword_bonus(query, documents[idx])
                adjusted_score = float(score) + keyword_bonus
                
                results.append({
                    'text': documents[idx],
                    'score': adjusted_score,
                    'original_score': float(score),
                    'keyword_bonus': keyword_bonus,
                    'course_id': course_id
                })
        
        # Re-sort by adjusted score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def calculate_keyword_bonus(self, query, text):
        """Calculate keyword matching bonus for educational content"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Educational keywords get higher weight
        educational_keywords = {
            'concept', 'theory', 'principle', 'formula', 'equation', 'definition',
            'example', 'problem', 'solution', 'method', 'process', 'analysis',
            'study', 'learn', 'understand', 'explain', 'calculate', 'derive'
        }
        
        bonus = 0
        for word in query_words:
            if word in text_words:
                if word in educational_keywords:
                    bonus += 0.15  # Higher bonus for educational terms
                else:
                    bonus += 0.05
        
        return bonus
    
    async def generate_response_stream(self, query, search_results, course_id=None, user_profile=None) -> AsyncGenerator[str, None]:
        """Generate streaming response using DeepSeek API"""
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            context_parts.append(f"Source {i} (Score: {result['score']:.2f}):\n{result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get course context
        course_context = ""
        if course_id and course_id in self.course_info:
            course = self.course_info[course_id]
            course_context = f"Course: {course['name']}"
            if course.get('professor'):
                course_context += f" (Prof. {course['professor']})"
                
        
        
        # Educational system prompt
        # Build personalized prompt
        if user_profile:
            academic_level = user_profile.get('academicLevel', 'undergraduate')
            learning_style = user_profile.get('learningStyle', 'visual')
            difficulty_pref = user_profile.get('difficultyPreference', 'moderate')
            major_field = user_profile.get('majorField', 'general studies')
            language_pref = user_profile.get('preferredLanguage', 'English')
            
            system_prompt = f"""You are an AI tutor specializing in helping {academic_level} students studying {major_field}.

        {course_context}

        Student Learning Profile:
        - Learning Style: {learning_style} learner
        - Preferred Difficulty: {difficulty_pref}
        - Academic Background: {user_profile.get('educationBackground', 'general')}
        - Career Goals: {user_profile.get('careerGoals', 'academic success')}

        Your personalized teaching approach:
        {f'- VISUAL: Use metaphors, describe diagrams, break down visually' if learning_style == 'visual' else ''}
        {f'- AUDITORY: Explain as if speaking, use verbal patterns' if learning_style == 'auditory' else ''}
        {f'- KINESTHETIC: Give hands-on examples, practical applications' if learning_style == 'kinesthetic' else ''}
        {f'- READING/WRITING: Provide structured notes, clear outlines' if learning_style == 'reading' else ''}

        Your role:
        - Explain concepts at {difficulty_pref} difficulty level
        - Use examples from {major_field} when possible
        - Break down complex topics for {academic_level} level
        - Provide practice problems appropriate for their level
        - Create exercises matching their learning style
        - Help with assignments and homework
        - Encourage learning and build confidence

        Guidelines:
        - Always respond primarily in {language_pref}
        - Use the provided course materials as your primary source
        - If information isn't in the materials, clearly state this
        - Be encouraging and supportive
        - Ask follow-up questions to check understanding
        - Suggest study strategies for {learning_style} learners
        - For math/science: show step-by-step solutions
        - For essays/writing: provide structured guidance

        Response style:
        - Adapt complexity to {academic_level} level
        - Use {difficulty_pref} difficulty explanations
        - Well-organized with headers when needed
        - Include examples matching their learning style
        - End with a question to continue learning
        """
        
        # Enhanced user prompt
        user_prompt = f"""Student Question: {query}

Course Materials:
{context}

Please provide a helpful, educational response based on the course materials above. If the question is about a specific concept, explain it thoroughly with examples. If it's asking for help with a problem, provide step-by-step guidance.
"""
        
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
            "max_tokens": 1500,
            "stream": True
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
                            json_str = line[6:]
                            
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
                yield self.get_fallback_response(search_results, course_id)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield self.get_fallback_response(search_results, course_id)
    
    def generate_response(self, query, search_results, course_id=None, user_profile=None):

        """Non-streaming response generation"""
        # Similar to streaming but returns complete response
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            context_parts.append(f"Source {i} (Score: {result['score']:.2f}):\n{result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get course context
        course_context = ""
        if course_id and course_id in self.course_info:
            course = self.course_info[course_id]
            course_context = f"Course: {course['name']}"
            if course.get('professor'):
                course_context += f" (Prof. {course['professor']})"
        
        if user_profile:
            academic_level = user_profile.get('academicLevel', 'undergraduate')
            learning_style = user_profile.get('learningStyle', 'visual')
            difficulty_pref = user_profile.get('difficultyPreference', 'moderate')
            major_field = user_profile.get('majorField', 'general studies')
            language_pref = user_profile.get('preferredLanguage', 'English')
            
            system_prompt = f"""You are an AI tutor for a {academic_level} student studying {major_field}. {course_context}

        Student Profile:
        - Learning Style: {learning_style} learner
        - Preferred Difficulty: {difficulty_pref}
        - Academic Background: {user_profile.get('educationBackground', 'general')}
        - Language Preference: {language_pref}

        Adapt your teaching approach:
        {f'- For VISUAL learners: Use diagrams, charts, and visual metaphors' if learning_style == 'visual' else ''}
        {f'- For AUDITORY learners: Explain concepts as if speaking, use rhythm and patterns' if learning_style == 'auditory' else ''}
        {f'- For KINESTHETIC learners: Use hands-on examples and practical applications' if learning_style == 'kinesthetic' else ''}
        {f'- For READING/WRITING learners: Provide structured notes and written explanations' if learning_style == 'reading' else ''}

        Help students learn by:
        - Explaining at {difficulty_pref} difficulty level
        - Using examples relevant to {major_field}
        - Providing practice problems appropriate for {academic_level} level
        - Being encouraging and supportive
        - Using course materials as primary source
        - Responding primarily in {language_pref}
        """
        else:
            # Fallback to original prompt if no profile
            system_prompt = f"""You are an AI tutor for college students. {course_context}
                
        Help students learn by:
        - Explaining concepts clearly
        - Providing examples and practice problems
        - Being encouraging and supportive
        - Using course materials as primary source
        - Responding in the same language as the question
        """
        
        user_prompt = f"""Student Question: {query}

Course Materials:
{context}

Provide a helpful educational response using the course materials."""
        
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
                "max_tokens": 1500
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
                return self.get_fallback_response(search_results, course_id)
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return self.get_fallback_response(search_results, course_id)
    
    def get_fallback_response(self, search_results, course_id=None):
        """Enhanced fallback response for educational context"""
        if not search_results:
            return "I don't have information about that topic in the course materials. Could you rephrase your question or ask about a different concept from your coursework?"
        
        course_name = ""
        if course_id and course_id in self.course_info:
            course_name = self.course_info[course_id]['name']
            
        response = f"📚 **Information from {course_name if course_name else 'your course materials'}:**\n\n"
        
        for i, result in enumerate(search_results[:2], 1):
            text = result['text']
            if len(text) > 300:
                cut_pos = text.rfind('.', 0, 300)
                if cut_pos > 150:
                    text = text[:cut_pos + 1]
                else:
                    text = text[:300] + "..."
            
            response += f"**Source {i}:** {text}\n\n"
        
        response += "💡 **For better help:** Try asking more specific questions about concepts, formulas, or problems you're working on."
        return response
    
    def track_learning_progress(self, user_id, course_id, question, response_quality):
        """Track learning progress in database"""
        if not self.conn:
            return
            
        try:
            # Extract topics from question (simple keyword extraction)
            topics = self.extract_topics(question)
            
            # Insert progress record
            self.conn.execute("""
                INSERT INTO learning_progress 
                (user_id, course_id, activity_type, topic_covered, questions_asked, session_duration)
                VALUES (?, ?, 'chat_question', ?, 1, 1)
            """, (user_id, course_id, ', '.join(topics) if topics else None))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error tracking progress: {e}")
    
    def extract_topics(self, text):
        """Simple topic extraction from question text"""
        # This is a basic implementation - you could use more sophisticated NLP
        educational_terms = [
            'equation', 'formula', 'theorem', 'concept', 'theory', 'principle',
            'definition', 'example', 'problem', 'solution', 'analysis', 'method'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        topics = [word for word in words if word in educational_terms]
        return topics[:3]  # Return max 3 topics
    
    async def chat_stream(self, question, course_id=None, user_id=None, user_profile=None):
        """Streaming chat function"""
        # Search for relevant documents
        search_results = self.search(question, course_id, k=20)
        
        # Calculate confidence
        avg_score = np.mean([r['score'] for r in search_results]) if search_results else 0
        confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
        
        # Get course name
        course_name = None
        if course_id and course_id in self.course_info:
            course_name = self.course_info[course_id]['name']
        
        # Metadata
        metadata = {
            'sources_count': len(search_results),
            'confidence': confidence,
            'avg_score': float(avg_score),
            'top_score': float(search_results[0]['score']) if search_results else 0,
            'course_name': course_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Yield metadata first
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
        
        # Stream the response
        async for chunk in self.generate_response_stream(question, search_results, course_id, user_profile):
            yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
        
        # Track progress
        if user_id and course_id:
            self.track_learning_progress(user_id, course_id, question, confidence)
        
        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    def chat(self, question, course_id=None, user_id=None, user_profile=None):
        """Non-streaming chat function"""
        # Search for relevant documents
        search_results = self.search(question, course_id, k=6)
        
        # Generate response
        response = self.generate_response(question, search_results, course_id, user_profile)
        
        # Calculate confidence
        avg_score = np.mean([r['score'] for r in search_results]) if search_results else 0
        confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
        
        # Get course name
        course_name = None
        if course_id and course_id in self.course_info:
            course_name = self.course_info[course_id]['name']
        
        # Track progress
        if user_id and course_id:
            self.track_learning_progress(user_id, course_id, question, confidence)
        
        return {
            'response': response,
            'sources_count': len(search_results),
            'confidence': confidence,
            'avg_score': float(avg_score),
            'top_score': float(search_results[0]['score']) if search_results else 0,
            'course_name': course_name
        }
        


# Initialize FastAPI app
app = FastAPI(
    title="Educational RAG Chatbot API",
    description="Multi-course AI tutor for college students with streaming support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot: Optional[EducationalRAG] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup"""
    global chatbot
    
    # Configuration
    API_KEY = "sk-2777026d7e5a4682be3c46afb6575ea2"  # Your DeepSeek API key
    DATABASE_PATH = "../frontend/database.sqlite"  # Path to your frontend database
    
    try:
        # Initialize chatbot
        logger.info("Initializing Educational RAG Chatbot...")
        chatbot = EducationalRAG(API_KEY, DATABASE_PATH)
        logger.info("Chatbot initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Educational RAG Chatbot API",
        "status": "active",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "health": "/health",
            "courses": "/courses",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global chatbot
    
    courses_loaded = len(chatbot.course_indexes) if chatbot else 0
    total_docs = sum(len(docs) for docs in chatbot.course_documents.values()) if chatbot else 0
    
    return HealthResponse(
        status="healthy" if chatbot else "unhealthy",
        model_loaded=chatbot is not None,
        courses_loaded=courses_loaded,
        total_documents=total_docs
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    """Non-streaming chat endpoint"""
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
        result = chatbot.chat(request.question, request.course_id, request.user_id, request.user_profile)
        
        # Return response
        return ChatResponse(
            response=result['response'],
            sources_count=result['sources_count'],
            confidence=result['confidence'],
            avg_score=result['avg_score'],
            top_score=result['top_score'],
            course_name=result.get('course_name'),
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
    """Streaming chat endpoint"""
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
            async for chunk in chatbot.chat_stream(request.question, request.course_id, request.user_id, request.user_profile):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/courses/{course_id}/load")
async def load_course(course_id: int):
    """Manually load a specific course"""
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    try:
        success = chatbot.load_course_materials(course_id)
        if success:
            course_info = chatbot.course_info.get(course_id, {})
            return {
                "success": True,
                "course_id": course_id,
                "course_name": course_info.get('name', 'Unknown'),
                "documents_loaded": len(chatbot.course_documents.get(course_id, [])),
                "message": f"Course {course_id} loaded successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not load course {course_id}"
            )
    except Exception as e:
        logger.error(f"Error loading course {course_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading course: {str(e)}"
        )

@app.get("/courses/loaded")
async def get_loaded_courses():
    """Get list of currently loaded courses"""
    global chatbot
    
    if not chatbot:
        return {"courses": []}
    
    loaded_courses = []
    for course_id, course_info in chatbot.course_info.items():
        loaded_courses.append({
            "course_id": course_id,
            "name": course_info.get('name', 'Unknown'),
            "documents_count": len(chatbot.course_documents.get(course_id, [])),
            "professor": course_info.get('professor'),
            "semester": course_info.get('semester')
        })
    
    return {"courses": loaded_courses}

@app.get("/examples", response_model=Dict[str, List[str]])
async def get_examples():
    """Get example questions for students"""
    return {
        "examples": [
            "Explain the concept of photosynthesis",
            "Help me solve this calculus problem",
            "What are the key themes in this literature?",
            "Create practice questions for my upcoming exam",
            "Summarize today's lecture material",
            "I don't understand this chemistry equation",
            "Generate a study plan for this course",
            "What are the main points I should remember?"
        ]
    }
    
@app.get("/test/course/{course_id}")
async def test_course_loading(course_id: int):
    """Test endpoint to check if course PDFs are loaded"""
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    # Check if course is already loaded
    is_loaded = course_id in chatbot.course_documents
    
    # Get course info
    course_info = chatbot.get_course_info(course_id)
    
    # Get number of documents if loaded
    num_documents = len(chatbot.course_documents.get(course_id, []))
    
    # Get sample documents if available
    sample_docs = []
    if is_loaded and num_documents > 0:
        sample_docs = chatbot.course_documents[course_id][:3]  # First 3 chunks
    
    return {
        "course_id": course_id,
        "is_loaded": is_loaded,
        "course_info": dict(course_info) if course_info else None,
        "num_documents": num_documents,
        "sample_documents": sample_docs,
        "message": "Course materials loaded successfully" if is_loaded else "Course not loaded yet"
    }

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )