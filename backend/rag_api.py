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
        
    def load_course(self, course_id):
        """Load course documents and create vector index"""
        try:
            logger.info(f"Loading course {course_id}")
            
            # Get course files
            course_files = self.get_course_files(course_id)
            if not course_files:
                logger.warning(f"No files found for course {course_id}")
                return
            
            documents = []
            
            for file_info in course_files:
                file_path = file_info["file_path"]
                
                # Check if file exists and is a string path
                if not isinstance(file_path, str):
                    logger.error(f"Invalid file path type: {type(file_path)}")
                    continue
                    
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    continue
                
                try:
                    # Load PDF
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    
                    # Split into chunks
                    chunks = self.split_text(text)
                    documents.extend(chunks)
                    
                    logger.info(f"Loaded {len(chunks)} chunks from {file_info['original_name']}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            if documents:
                # Store documents and create embeddings
                self.course_documents[course_id] = documents
                embeddings = self.embedding_model.encode(documents)
                self.course_embeddings[course_id] = embeddings
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings.astype('float32'))
                self.course_indexes[course_id] = index
                
                logger.info(f"Successfully loaded course {course_id} with {len(documents)} documents")
            else:
                logger.warning(f"No documents loaded for course {course_id}")
                
        except Exception as e:
            logger.error(f"Error loading course {course_id}: {e}")
    
    
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
            
        # Get course files (now returns list of dicts with file_path and original_name)
        course_files = self.get_course_files(course_id)
        if not course_files:
            logger.warning(f"No files found for course {course_id}")
            return False
        
        logger.info(f"Loading {len(course_files)} files for course {course_id}")
        
        # Store course info
        self.course_info[course_id] = dict(course_info)
        
        # Process all files for this course
        all_documents = []
        for file_info in course_files:  # file_info is now a dict
            file_path = file_info["file_path"]  # Extract the actual file path
            if os.path.exists(file_path):
                try:
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} chunks from {file_info['original_name']}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
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
    
    def search_chapter_content(self, query, course_id, chapter_id, k=5):
        """Search for relevant documents within a specific chapter only"""
        try:
            # Get chapter content from database
            if not self.conn:
                logger.error("No database connection")
                return []
                
            cursor = self.conn.execute("""
                SELECT content_text, page_reference 
                FROM chapter_content 
                WHERE chapter_id = ?
            """, (chapter_id,))
            
            chapter_documents = []
            chapter_metadata = []
            
            for row in cursor.fetchall():
                if row[0]:  # content_text
                    chapter_documents.append(row[0])
                    chapter_metadata.append(row[1])  # page_reference
            
            if not chapter_documents:
                logger.warning(f"No content found for chapter {chapter_id}")
                return []
            
            logger.info(f"Searching within chapter {chapter_id}: {len(chapter_documents)} documents")
            
            # Create embeddings for chapter content only
            embeddings = self.embedding_model.encode(chapter_documents)
            
            # Create temporary FAISS index for this chapter
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            # Encode query and search
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = index.search(query_embedding.astype('float32'), min(k, len(chapter_documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chapter_documents):
                    keyword_bonus = self.calculate_keyword_bonus(query, chapter_documents[idx])
                    adjusted_score = float(score) + keyword_bonus
                    
                    results.append({
                        'text': chapter_documents[idx],
                        'score': adjusted_score,
                        'original_score': float(score),
                        'keyword_bonus': keyword_bonus,
                        'course_id': course_id,
                        'chapter_id': chapter_id,
                        'page_reference': chapter_metadata[idx] if idx < len(chapter_metadata) else None
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Found {len(results)} relevant chunks in chapter {chapter_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching chapter content: {e}")
            return []
        
    async def chat_stream_chapter(self, question, course_id, chapter_id, user_id=None, user_profile=None):
        """Streaming chat function for specific chapter content only"""
        logger.info(f"Chapter chat: course_id={course_id}, chapter_id={chapter_id}, question='{question[:50]}...'")
        
        # Use chapter-specific search
        search_results = self.search_chapter_content(question, course_id, chapter_id, k=10)
        
        # Get chapter info for context
        chapter_info = None
        if self.conn:
            cursor = self.conn.execute(
                "SELECT title, content_summary FROM course_chapters WHERE id = ?", 
                (chapter_id,)
            )
            chapter_info = cursor.fetchone()
        
        # Calculate confidence
        avg_score = np.mean([r['score'] for r in search_results]) if search_results else 0
        confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
        
        # Get course and chapter names
        course_name = None
        chapter_name = None
        if course_id and course_id in self.course_info:
            course_name = self.course_info[course_id]['name']
        if chapter_info:
            chapter_name = chapter_info[0]  # title
        
        # Metadata
        metadata = {
            'sources_count': len(search_results),
            'confidence': confidence,
            'avg_score': float(avg_score),
            'top_score': float(search_results[0]['score']) if search_results else 0,
            'course_name': course_name,
            'chapter_name': chapter_name,
            'chapter_id': chapter_id,
            'search_type': 'chapter_specific',
            'timestamp': datetime.now().isoformat()
        }
        
        # Yield metadata first
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
        
        # Stream the response with chapter-specific context
        async for chunk in self.generate_chapter_response_stream(question, search_results, course_id, chapter_id, chapter_info, user_profile):
            yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
        
        # Track progress
        if user_id and course_id:
            self.track_learning_progress(user_id, course_id, question, confidence)
        
        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    
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
                
    async def generate_chapter_response_stream(self, query, search_results, course_id=None, chapter_id=None, chapter_info=None, user_profile=None) -> AsyncGenerator[str, None]:
        """Generate streaming response for specific chapter content"""
        # Prepare context from chapter search results only
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            page_ref = result.get('page_reference', 'Unknown page')
            context_parts.append(f"Chapter Section {i} ({page_ref}, Score: {result['score']:.2f}):\n{result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get chapter and course context
        chapter_context = ""
        if chapter_info and course_id and course_id in self.course_info:
            course = self.course_info[course_id]
            chapter_title = chapter_info[0]
            chapter_summary = chapter_info[1] if len(chapter_info) > 1 else ""
            
            chapter_context = f"Course: {course['name']}\nChapter: {chapter_title}"
            if chapter_summary:
                chapter_context += f"\nChapter Summary: {chapter_summary}"
        
        # Enhanced system prompt for chapter-specific tutoring
        system_prompt = f"""You are an AI tutor helping a student with a SPECIFIC CHAPTER from their course.

    {chapter_context}

    CRITICAL INSTRUCTIONS:
    - You MUST ONLY use information from the provided chapter content below
    - Do NOT use general knowledge or information from other chapters/courses
    - If the student asks about topics not covered in this chapter's content, politely redirect them to the chapter material
    - Stay strictly within the scope of this specific chapter
    - Reference the chapter content explicitly in your responses

    Your role for this chapter:
    - Explain concepts covered in THIS SPECIFIC CHAPTER ONLY
    - Help with problems and examples from THIS CHAPTER
    - Create practice questions from THIS CHAPTER'S material
    - Clarify terminology from THIS CHAPTER
    - Help students understand connections WITHIN this chapter

    Response guidelines:
    - Always stay within the scope of this chapter
    - Use ONLY the provided chapter content as your source
    - If information isn't in this chapter's content, clearly state this
    - Be encouraging and supportive
    - Reference specific sections when helpful
    - End with questions to continue learning about this chapter
    """

        user_prompt = f"""Student Question about this chapter: {query}

    CHAPTER CONTENT (THIS IS YOUR ONLY INFORMATION SOURCE):
    {context}

    Please provide a helpful response based ONLY on this chapter's content. Focus specifically on this chapter's concepts and stay strictly within the provided material.
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
                yield self.get_chapter_fallback_response(search_results, chapter_id)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield self.get_chapter_fallback_response(search_results, chapter_id)

    def get_chapter_fallback_response(self, search_results, chapter_id=None):
        """Fallback response for chapter-specific content"""
        if not search_results:
            return f"I don't have information about that topic in this specific chapter. Could you ask about concepts covered in this chapter's material?"
        
        response = f"📖 **Information from this chapter:**\n\n"
        
        for i, result in enumerate(search_results[:2], 1):
            text = result['text']
            if len(text) > 300:
                cut_pos = text.rfind('.', 0, 300)
                if cut_pos > 150:
                    text = text[:cut_pos + 1]
                else:
                    text = text[:300] + "..."
            
            page_ref = result.get('page_reference', 'Unknown page')
            response += f"**Chapter Section {i}** ({page_ref}):\n{text}\n\n"
        
        response += "💡 **For better help:** Try asking more specific questions about the concepts covered in this chapter."
        return response           
    
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
            
    def extract_pdf_with_pages(self, file_path):
        """Extract PDF text with page numbers"""
        try:
            text_by_pages = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_by_pages.append(text)
            return text_by_pages
        except Exception as e:
            logger.error(f"Error extracting PDF with pages: {e}")
            return []

    def detect_chapters_with_ai(self, text):
        """Use DeepSeek API to identify chapter breaks and topics using FULL document"""
        
        # Calculate text length for better analysis
        text_length = len(text)
        max_tokens = 120000  # DeepSeek can handle up to ~128k tokens
        
        # If text is very long, we'll use intelligent sampling
        if text_length > max_tokens:
            # Take first 40k chars, middle 40k chars, and last 40k chars
            sample_text = (
                text[:40000] + 
                "\n\n[... MIDDLE SECTION ...]\n\n" + 
                text[text_length//2-20000:text_length//2+20000] + 
                "\n\n[... END SECTION ...]\n\n" + 
                text[-40000:]
            )
            analysis_text = sample_text
            logger.info(f"Document too long ({text_length} chars). Using intelligent sampling.")
        else:
            # Use full document if it fits
            analysis_text = text
            logger.info(f"Analyzing full document ({text_length} characters)")
        
        prompt = f"""
    Analyze this COMPLETE educational document and identify ALL natural chapter/section breaks throughout the entire document. 
    Look for chapter titles, section headers, topic changes, and logical content divisions.

    Return ONLY a valid JSON structure with ALL chapters found in the document:

    {{
    "chapters": [
        {{
        "chapter_number": 1,
        "title": "Introduction to Biology",
        "start_page": 1,
        "end_page": 15,
        "content_summary": "Overview of biological concepts and scientific method",
        "estimated_study_time": 45,
        "difficulty_level": "beginner",
        "key_topics": ["scientific method", "characteristics of life", "biological organization"]
        }},
        {{
        "chapter_number": 2,
        "title": "Cell Structure and Function", 
        "start_page": 16,
        "end_page": 35,
        "content_summary": "Detailed study of cellular components and processes",
        "estimated_study_time": 60,
        "difficulty_level": "intermediate",
        "key_topics": ["cell membrane", "nucleus", "organelles", "cellular processes"]
        }}
    ]
    }}

    IMPORTANT: 
    - Identify ALL chapters/sections in the document, not just the beginning
    - Look for clear content divisions and topic changes
    - Estimate realistic page ranges for each chapter
    - Include ALL major sections you can identify

    Complete document text: {analysis_text}
    """
        
        try:
            # Use existing DeepSeek API call with higher token limit
            response = self.generate_response_sync_extended(prompt)
            return self.parse_chapter_json(response)
        except Exception as e:
            logger.error(f"Error in AI chapter detection: {e}")
            return {"chapters": []}
        
    def generate_response_sync_extended(self, prompt):
        """Extended synchronous DeepSeek API call for longer documents"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000,  # Increased for longer responses
                "stream": False
            }
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error in extended sync API call: {e}")
            return ""    

    def generate_response_sync(self, prompt):
        """Synchronous version of DeepSeek API call"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            # FIX: Add /v1 to the URL
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error in sync API call: {e}")
            return ""

    def parse_chapter_json(self, response_text):
        """Parse JSON from AI response"""
        try:
            # Find JSON in response (handle cases where AI adds extra text)
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                return {"chapters": []}
        except Exception as e:
            logger.error(f"Error parsing chapter JSON: {e}")
            return {"chapters": []}

    def save_chapter_structure(self, course_id, chapter_analysis, text_by_pages):
        """Save detected chapters to database"""
        if not self.conn or not chapter_analysis.get('chapters'):
            return
            
        try:
             # Ensure tables exist first
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS course_chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_id INTEGER NOT NULL,
                    chapter_number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content_summary TEXT,
                    estimated_study_time INTEGER DEFAULT 30,
                    difficulty_level TEXT DEFAULT 'medium',
                    prerequisites TEXT,
                    status TEXT DEFAULT 'locked',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chapter_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    content_text TEXT,
                    page_reference TEXT,
                    vector_index INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES course_chapters (id) ON DELETE CASCADE
                )
            """)
            for chapter in chapter_analysis['chapters']:
                # Extract chapter content from pages
                start_page = chapter.get('start_page', 1) - 1  # Convert to 0-based index
                end_page = chapter.get('end_page', len(text_by_pages)) - 1
                
                chapter_content = ""
                for page_idx in range(max(0, start_page), min(len(text_by_pages), end_page + 1)):
                    chapter_content += f"Page {page_idx + 1}: {text_by_pages[page_idx]}\n"
                
                # Insert chapter
                cursor = self.conn.execute("""
                    INSERT INTO course_chapters 
                    (course_id, chapter_number, title, content_summary, estimated_study_time, difficulty_level, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    course_id,
                    chapter.get('chapter_number', 1),
                    chapter.get('title', 'Untitled Chapter'),
                    chapter.get('content_summary', ''),
                    chapter.get('estimated_study_time', 30),
                    chapter.get('difficulty_level', 'medium'),
                    'unlocked' if chapter.get('chapter_number', 1) == 1 else 'locked'
                ))
                
                chapter_id = cursor.lastrowid
                
                # Insert chapter content
                self.conn.execute("""
                    INSERT INTO chapter_content 
                    (chapter_id, content_type, content_text, page_reference)
                    VALUES (?, ?, ?, ?)
                """, (
                    chapter_id,
                    'text',
                    chapter_content[:5000],  # Limit content size
                    f"Pages {chapter.get('start_page', 1)}-{chapter.get('end_page', 1)}"
                ))
            
            self.conn.commit()
            logger.info(f"Saved {len(chapter_analysis['chapters'])} chapters for course {course_id}")
            
        except Exception as e:
            logger.error(f"Error saving chapters: {e}")

    def get_course_files(self, course_id):
        """Get file paths for a course"""
        if not self.conn:
            return []
            
        try:
            cursor = self.conn.execute(
                "SELECT file_path, original_name FROM course_files WHERE course_id = ?",
                (course_id,)
            )
            files = []
            for row in cursor.fetchall():
                # Ensure we have proper string paths
                file_path = str(row[0]) if row[0] else ""
                original_name = str(row[1]) if row[1] else "Unknown"
                
                # Verify the file exists
                if os.path.exists(file_path):
                    files.append({
                        "file_path": file_path,
                        "original_name": original_name
                    })
                else:
                    logger.warning(f"File not found: {file_path}")
                    
            return files
        except Exception as e:
            logger.error(f"Error getting course files: {e}")
            return []

    def analyze_document_structure(self, file_path, course_id):
        """Main function to analyze PDF and create chapter structure"""
        try:
            logger.info(f"Analyzing document structure for course {course_id}: {file_path}")
            
            # Extract text with page information
            text_by_pages = self.extract_pdf_with_pages(file_path)
            if not text_by_pages:
                return None
                
            # Combine text for analysis
            full_text = "\n".join([f"Page {i+1}: {page}" for i, page in enumerate(text_by_pages)])
            
            # Use AI to detect chapters
            chapter_analysis = self.detect_chapters_with_ai(full_text)
            
            # Save to database
            self.save_chapter_structure(course_id, chapter_analysis, text_by_pages)
            
            return chapter_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document structure: {e}")
            return None        
    
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

@app.post("/chat/chapter")
async def chat_chapter_stream(request: dict):
    """Streaming chat endpoint for specific chapter content"""
    try:
        question = request.get('question')
        course_id = request.get('course_id')
        chapter_id = request.get('chapter_id')
        user_id = request.get('user_id')
        user_profile = request.get('user_profile')
        
        logger.info(f"Chapter chat request: course={course_id}, chapter={chapter_id}")
        
        if not question or not course_id or not chapter_id:
            raise HTTPException(status_code=400, detail="Missing required fields: question, course_id, chapter_id")
        
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        # Ensure course is loaded
        if course_id not in chatbot.course_info:
            chatbot.load_course_materials(course_id)
        
        async def generate():
            try:
                async for chunk in chatbot.chat_stream_chapter(question, course_id, chapter_id, user_id, user_profile):
                    yield chunk
            except Exception as e:
                logger.error(f"Chapter streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chapter chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
   
@app.post("/api/analyze-course-structure")
async def analyze_course_structure(request: dict):
    """Analyze uploaded course materials and create chapter structure"""
    try:
        course_id = request.get('course_id')
        user_id = request.get('user_id')
        
        if not course_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing course_id or user_id")
        
        logger.info(f"Starting course analysis for course {course_id}")
        
        # Ensure chatbot is initialized
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        # Get course files from database
        course_files = chatbot.get_course_files(course_id)
        
        if not course_files:
            logger.warning(f"No course files found for course {course_id}")
            return {
                "success": False,
                "message": "No course files found to analyze",
                "chapters_created": 0
            }
        
        # Analyze each PDF and create structure
        total_chapters = 0
        processed_files = 0
        
        for file_info in course_files:
            try:
                logger.info(f"Analyzing file: {file_info['original_name']}")
                analysis = chatbot.analyze_document_structure(file_info['file_path'], course_id)
                
                if analysis and analysis.get('chapters'):
                    total_chapters += len(analysis['chapters'])
                    processed_files += 1
                    logger.info(f"Successfully analyzed {file_info['original_name']}: {len(analysis['chapters'])} chapters found")
                else:
                    logger.warning(f"No chapters detected in {file_info['original_name']}")
                    
            except Exception as file_error:
                logger.error(f"Error analyzing file {file_info['original_name']}: {file_error}")
                continue
        
        return {
            "success": True, 
            "message": f"Course analysis completed. {total_chapters} chapters detected from {processed_files} files.",
            "chapters_created": total_chapters,
            "files_processed": processed_files,
            "total_files": len(course_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in course analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")   

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