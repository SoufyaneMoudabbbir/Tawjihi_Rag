# Tawjihi RAG - Educational Platform

An AI-powered educational platform combining RAG (Retrieval-Augmented Generation) technology with course management for college students in Morocco.

## 🚀 Features

- **Course Management**: Upload and manage PDF course materials
- **AI-Powered Chat**: RAG-based Q&A system with course-specific context
- **Student Profiling**: Dynamic questionnaire for personalized learning
- **Chapter Analysis**: Automatic detection and structuring of course chapters
- **Multi-Language Support**: Arabic, English, and French

## 🏗️ Architecture

### Tech Stack

**Backend**
- FastAPI with async support
- Sentence-Transformers for embeddings
- FAISS for vector similarity search
- PyPDF2 for PDF processing
- SQLite3 for database
- DeepSeek API for LLM responses

**Frontend**
- Next.js 15 with App Router
- React 19
- Tailwind CSS + shadcn/ui
- Clerk for authentication
- TypeScript/JavaScript

## 📦 Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd Tawjihi_Rag
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Backend Setup**
```bash
cd backend
python -m venv rag_env_310
source rag_env_310/bin/activate  # On Windows: rag_env_310\Scripts\activate
pip install -r requirements.txt
python main.py
```

4. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📚 Documentation

See [CLAUDE.md](./CLAUDE.md) for detailed development guide and architecture documentation.

## 🔒 Security

- All API keys are managed through environment variables
- Input validation on all endpoints
- Rate limiting implemented
- SQL injection protection via parameterized queries
- CORS configuration for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- DeepSeek AI for LLM API
- Sentence-Transformers for embeddings
- FAISS for vector search
- Clerk for authentication
