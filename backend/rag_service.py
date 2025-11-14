"""
Educational RAG Service
Core RAG functionality with course-specific context isolation
"""
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, AsyncGenerator
import PyPDF2
from pathlib import Path
import json
import httpx

from config import Config
from utils import clean_text, split_text, calculate_confidence, format_timestamp
from db_service import DatabaseService

logger = logging.getLogger(__name__)


class EducationalRAGService:
    """RAG service for educational content with course isolation"""

    def __init__(self, db_service: DatabaseService):
        self.db = db_service
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Course-specific storage (in-memory for now, can be moved to Redis)
        self.course_documents: Dict[int, List[str]] = {}
        self.course_embeddings: Dict[int, np.ndarray] = {}
        self.course_indexes: Dict[int, faiss.Index] = {}
        self.course_info: Dict[int, Dict] = {}

        logger.info(f"RAG Service initialized with model: {Config.EMBEDDING_MODEL}")

    def load_course_materials(self, course_id: int) -> bool:
        """
        Load and process course materials into vector index

        Args:
            course_id: Course ID to load

        Returns:
            True if successful, False otherwise
        """
        try:
            if course_id in self.course_indexes:
                logger.info(f"Course {course_id} already loaded")
                return True

            # Get course info
            course_info = self.db.get_course_info(course_id)
            if not course_info:
                logger.warning(f"Course {course_id} not found")
                return False

            # Get course files
            course_files = self.db.get_course_files(course_id)
            if not course_files:
                logger.warning(f"No files found for course {course_id}")
                return False

            logger.info(f"Loading {len(course_files)} files for course {course_id}")

            # Store course info
            self.course_info[course_id] = course_info

            # Process all files
            all_documents = []
            for file_info in course_files:
                file_path = file_info["file_path"]
                if Path(file_path).exists():
                    try:
                        documents = self._load_document(file_path)
                        all_documents.extend(documents)
                        logger.info(
                            f"Loaded {len(documents)} chunks from {file_info['original_name']}"
                        )
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                else:
                    logger.warning(f"File not found: {file_path}")

            if not all_documents:
                logger.warning(f"No documents loaded for course {course_id}")
                return False

            # Store documents and build index
            self.course_documents[course_id] = all_documents
            self._build_course_index(course_id)

            logger.info(
                f"✅ Course {course_id} loaded with {len(all_documents)} document chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading course {course_id}: {e}", exc_info=True)
            return False

    def _load_document(self, file_path: str) -> List[str]:
        """Load and process a single PDF document"""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n\n"

                # Clean and split text
                text = clean_text(text)
                chunks = split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)

                # Add metadata
                enhanced_chunks = []
                for i, chunk in enumerate(chunks):
                    enhanced_chunk = (
                        f"[Source: {Path(file_path).name}, "
                        f"Part {i+1}/{len(chunks)}]\n{chunk}"
                    )
                    enhanced_chunks.append(enhanced_chunk)

                return enhanced_chunks
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return []

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []

    def _build_course_index(self, course_id: int) -> bool:
        """Build FAISS vector index for a course"""
        try:
            if course_id not in self.course_documents:
                logger.error(f"No documents found for course {course_id}")
                return False

            logger.info(f"Building vector index for course {course_id}...")

            # Generate embeddings
            documents = self.course_documents[course_id]
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Create FAISS index with cosine similarity
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))

            # Store
            self.course_embeddings[course_id] = embeddings
            self.course_indexes[course_id] = index

            logger.info(f"✅ Index built for course {course_id}")
            return True

        except Exception as e:
            logger.error(f"Error building index for course {course_id}: {e}")
            return False

    def search(
        self,
        query: str,
        course_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant documents

        Args:
            query: Search query
            course_id: Course ID to search in
            k: Number of results to return

        Returns:
            List of search results with text and scores
        """
        if not course_id:
            logger.warning("No course_id provided for search")
            return []

        # Ensure course is loaded
        if course_id not in self.course_indexes:
            logger.info(f"Loading course {course_id}...")
            if not self.load_course_materials(course_id):
                logger.error(f"Failed to load course {course_id}")
                return []

        try:
            # Get index and documents
            index = self.course_indexes[course_id]
            documents = self.course_documents[course_id]

            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = index.search(query_embedding.astype('float32'), k)

            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(documents):
                    results.append({
                        'text': documents[idx],
                        'score': float(score),
                        'course_id': course_id,
                        'index': int(idx)
                    })

            logger.info(f"Found {len(results)} results for course {course_id}")
            return results

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return []

    async def generate_response_stream(
        self,
        query: str,
        search_results: List[Dict],
        course_id: Optional[int] = None,
        user_profile: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using DeepSeek API"""
        # Prepare context
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            context_parts.append(
                f"Source {i} (Score: {result['score']:.2f}):\n{result['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Build system prompt
        system_prompt = """You are an AI tutor helping college students learn from their course materials.

Your role:
- Explain concepts clearly and thoroughly
- Provide step-by-step guidance for problems
- Use examples to illustrate points
- Encourage learning and build confidence
- Stay focused on the course materials provided

Guidelines:
- Use ONLY the provided course materials
- If information isn't in the materials, clearly state this
- Be encouraging and supportive
- End with a question to check understanding
"""

        user_prompt = f"""Student Question: {query}

Course Materials:
{context}

Please provide a helpful, educational response based on the course materials above."""

        # Call DeepSeek API
        headers = {
            'Authorization': f'Bearer {Config.DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": Config.DEEPSEEK_MODEL,
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
                    Config.DEEPSEEK_API_URL,
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

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                # Fallback response
                yield "I apologize, but I'm having trouble generating a response. "
                yield "Here's what I found in the course materials:\n\n"
                for result in search_results[:2]:
                    yield f"- {result['text'][:200]}...\n\n"
