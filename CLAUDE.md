# CLAUDE.md - Development Guide for AI Assistants

## Project Overview

**Tawjihi RAG** is an educational platform combining RAG (Retrieval-Augmented Generation) technology with course management for college students in Morocco. The system helps students upload course materials, interact with AI-powered chat sessions, and receive personalized educational guidance.

### Core Features
- **Course Management**: Upload and manage PDF course materials
- **AI-Powered Chat**: RAG-based Q&A system using DeepSeek API with course-specific context
- **Student Profiling**: Dynamic questionnaire system for personalized learning
- **Chapter Analysis**: Automatic detection and structuring of course chapters
- **Multi-Language Support**: Arabic, English, French support

## Architecture

### Tech Stack

**Backend (Python/FastAPI)**
- FastAPI web framework with async support
- Sentence-Transformers for embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- FAISS for vector similarity search
- PyPDF2 for PDF processing
- SQLite3 for database
- DeepSeek API for LLM responses
- Uvicorn as ASGI server

**Frontend (Next.js 15)**
- React 19 with Next.js App Router
- TypeScript and JavaScript (mixed codebase)
- Tailwind CSS for styling
- shadcn/ui component library
- Clerk for authentication
- SQLite3 for local database
- React Hook Form + Zod for validation

### System Flow
```
User → Next.js Frontend → API Routes → SQLite DB
                        ↓
                 Backend FastAPI → DeepSeek API
                        ↓
                 FAISS Vector Search
```

## Directory Structure

```
Tawjihi_Rag/
├── backend/
│   ├── rag_api.py           # Main FastAPI application (2433 lines)
│   └── requirements.txt      # Python dependencies
│
├── frontend/
│   ├── app/                  # Next.js App Router
│   │   ├── layout.js         # Root layout with Clerk provider
│   │   ├── page.js           # Landing page
│   │   ├── api/              # API route handlers
│   │   │   ├── chats/        # Chat session management
│   │   │   ├── courses/      # Course CRUD operations
│   │   │   ├── database/     # Migration endpoints
│   │   │   ├── form-config/  # Questionnaire config
│   │   │   ├── progress/     # User progress tracking
│   │   │   └── responses/    # Form responses
│   │   ├── chat/[sessionId]/ # Chat interface
│   │   ├── courses/          # Course dashboard
│   │   ├── dashboard/        # Main dashboard
│   │   ├── form-builder/     # Form creation UI
│   │   └── questionnaire/    # Student profiling
│   │
│   ├── components/
│   │   ├── ui/               # shadcn/ui components
│   │   ├── forms/            # Form components
│   │   ├── ChatMessage.js    # Chat message renderer
│   │   └── DynamicFormRenderer.js
│   │
│   ├── lib/
│   │   ├── db.js             # SQLite connection helper
│   │   ├── chatApi.js        # Backend API client
│   │   ├── fallbackService.js # Offline fallback
│   │   ├── formConfig.json   # Student questionnaire schema
│   │   └── utils.ts          # Utility functions
│   │
│   ├── hooks/                # React hooks
│   ├── styles/               # Global CSS
│   ├── public/               # Static assets
│   ├── uploads/              # User-uploaded files (gitignored)
│   └── database.sqlite       # SQLite database file
│
├── package.json              # Root dependencies (react-markdown)
└── .gitignore
```

## Database Schema

### SQLite Tables

**courses**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT
user_id         TEXT NOT NULL
name            TEXT NOT NULL
description     TEXT
professor       TEXT
semester        TEXT
status          TEXT DEFAULT 'active'
file_count      INTEGER DEFAULT 0
chat_count      INTEGER DEFAULT 0
progress        INTEGER DEFAULT 0
created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
last_accessed   DATETIME
```

**course_files**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT
course_id       INTEGER NOT NULL (FK -> courses.id)
filename        TEXT NOT NULL
original_name   TEXT NOT NULL
file_path       TEXT NOT NULL
file_size       INTEGER
upload_date     DATETIME DEFAULT CURRENT_TIMESTAMP
processed       BOOLEAN DEFAULT FALSE
```

**chat_sessions**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT
user_id         TEXT NOT NULL
course_id       INTEGER (FK -> courses.id)
title           TEXT NOT NULL
created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
```

**chat_messages**
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT
session_id      INTEGER NOT NULL (FK -> chat_sessions.id)
role            TEXT NOT NULL ('user' | 'assistant')
content         TEXT NOT NULL
timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
metadata        TEXT (JSON)
```

**Indexes**
- `idx_courses_user_id` on courses(user_id)
- `idx_course_files_course_id` on course_files(course_id)
- `idx_chat_sessions_user_id` on chat_sessions(user_id)
- `idx_chat_sessions_course_id` on chat_sessions(course_id)

## Key Development Patterns

### 1. API Routes Structure

**Location**: `frontend/app/api/*/route.js`

All API routes follow Next.js 15 App Router conventions:

```javascript
import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function GET(request) {
  const { searchParams } = new URL(request.url)
  const db = await openDb()
  // ... logic
  return NextResponse.json({ data })
}

export async function POST(request) {
  const data = await request.json()
  // or: const formData = await request.formData()
  const db = await openDb()
  // ... logic
  return NextResponse.json({ success: true })
}
```

### 2. Database Access Pattern

**Always use the centralized db helper**:

```javascript
import { openDb } from "@/lib/db"

const db = await openDb()
const result = await db.run("INSERT INTO...", [params])
const rows = await db.all("SELECT * FROM...", [params])
```

**Database connection**: Singleton pattern in `lib/db.js` prevents multiple connections.

### 3. Backend RAG System

**Key Components** (in `backend/rag_api.py`):

- `EducationalRAG` class: Main RAG orchestrator
- Course-specific indexes: `course_indexes[course_id]` with FAISS
- Multi-lingual embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
- Chunk size: 700 characters with 100 character overlap
- DeepSeek API integration for response generation

**Document Processing Flow**:
1. PDF uploaded via frontend
2. Stored in `uploads/{userId}/{courseId}/`
3. Backend `/courses/{courseId}/load` endpoint triggered
4. PyPDF2 extracts text
5. Text chunked with metadata
6. Embeddings created with SentenceTransformer
7. FAISS index built for course
8. Ready for similarity search

### 4. File Upload Pattern

**Frontend uploads go to**: `uploads/{userId}/{courseId}/{timestamp}_{filename}`

```javascript
const formData = new FormData()
formData.append('userId', userId)
formData.append('name', courseName)
formData.append('files', pdfFile)

const response = await fetch('/api/courses', {
  method: 'POST',
  body: formData
})
```

**Backend processing triggered automatically** after course creation.

### 5. Chat Implementation

**Streaming Response Pattern**:

Frontend uses Server-Sent Events (SSE) for streaming:

```javascript
await chatApi.sendMessageStream(
  question,
  (chunk) => { /* handle content chunk */ },
  (metadata) => { /* handle metadata */ },
  (error) => { /* handle error */ },
  () => { /* handle completion */ }
)
```

Backend streams with FastAPI:

```python
async def stream_response():
    yield f"data: {json.dumps({'type': 'metadata', 'data': {...}})}\n\n"
    for chunk in response:
        yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

return StreamingResponse(stream_response(), media_type="text/event-stream")
```

### 6. Authentication

**Clerk Integration**:
- Root layout wraps app in `<ClerkProvider>`
- User IDs from Clerk stored as `user_id` (string) in database
- Check `userId` in API routes for auth

### 7. Component Patterns

**shadcn/ui components**: Located in `components/ui/`
- Pre-configured with Tailwind CSS
- Use Radix UI primitives
- Import with `@/components/ui/button`

**Form handling**:
- React Hook Form for validation
- Zod schemas for type safety (where TypeScript is used)
- Dynamic forms driven by `lib/formConfig.json`

### 8. Styling Conventions

**Tailwind CSS**:
- Custom color scheme defined in `tailwind.config.js`
- CSS variables for theming (HSL format)
- Responsive design with mobile-first approach
- Use `className` for all styling

**Common patterns**:
```javascript
className="bg-gradient-to-br from-blue-50 to-indigo-100"
className="rounded-2xl shadow-lg p-6"
className="text-sm text-gray-600"
```

## Development Workflows

### Starting Development

**Backend**:
```bash
cd backend
python -m venv rag_env_310
source rag_env_310/bin/activate  # or rag_env_310\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn rag_api:app --reload --port 8000
```

**Frontend**:
```bash
cd frontend
npm install  # or pnpm install
npm run dev  # Starts on port 3000
```

**Environment Variables Required**:
- Backend: `DEEPSEEK_API_KEY` for LLM
- Frontend: Clerk authentication keys

### Adding New Features

#### Adding a New API Route

1. Create file at `frontend/app/api/your-route/route.js`
2. Export GET/POST/PUT/DELETE functions
3. Use `openDb()` for database access
4. Return `NextResponse.json()`

#### Adding a New Database Table

1. Add migration in `frontend/app/api/database/migrate/route.js`
2. Create table with `IF NOT EXISTS`
3. Add indexes for foreign keys
4. Update this documentation

#### Modifying the RAG System

1. Edit `backend/rag_api.py`
2. Modify `EducationalRAG` class methods
3. Adjust chunk_size/overlap in `split_text()` if needed
4. Test with diverse PDF content

#### Adding UI Components

1. Use shadcn CLI to add components: `npx shadcn@latest add [component]`
2. Or manually create in `components/ui/`
3. Follow Tailwind CSS conventions
4. Export from component file

### Testing Strategy

**Current State**: No automated tests present

**Recommended approach**:
- Manual testing via browser dev tools
- Test API routes with Postman/curl
- Verify database changes with SQLite browser
- Test file uploads with various PDF formats

## Important Conventions

### Code Style

**JavaScript/TypeScript**:
- Mixed .js and .ts files (migration in progress)
- Use async/await for all async operations
- Prefer arrow functions for components
- Use destructuring for props

**Python**:
- Follow PEP 8 conventions
- Type hints where appropriate
- Comprehensive error logging
- Docstrings for all major functions

### Error Handling

**Frontend**:
```javascript
try {
  const response = await fetch(url)
  if (!response.ok) throw new Error()
  return await response.json()
} catch (error) {
  console.error("Error:", error)
  return NextResponse.json({ error: "Message" }, { status: 500 })
}
```

**Backend**:
```python
try:
    # operation
except Exception as e:
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Naming Conventions

- **Components**: PascalCase (`ChatMessage.js`)
- **Routes**: kebab-case (`/api/form-config`)
- **Database tables**: snake_case (`chat_sessions`)
- **Variables**: camelCase in JS/TS, snake_case in Python
- **API endpoints**: RESTful conventions

### File Organization

- **One component per file**
- **Co-locate related files** (route handlers with their dynamic segments)
- **Separate UI from logic** (components vs lib)
- **Keep API routes thin** (delegate to services if complex)

## Common Pitfalls & Solutions

### 1. SQLite Database Locking
**Issue**: "Database is locked" errors
**Solution**: Use single connection via `openDb()`, avoid concurrent writes

### 2. PDF Processing Failures
**Issue**: Some PDFs fail to extract text
**Solution**: Check file encoding, validate PDF structure, add error logging

### 3. FAISS Index Memory
**Issue**: Large courses consume significant memory
**Solution**: Limit chunk size, implement lazy loading, consider disk-based indexes

### 4. CORS Issues
**Issue**: Frontend can't reach backend
**Solution**: CORS middleware configured in FastAPI, verify ports match

### 5. File Path Issues
**Issue**: Upload paths break on Windows
**Solution**: Always use `path.join()`, never string concatenation

### 6. Build Errors
**Config**: Build errors for TypeScript/ESLint are **intentionally ignored**
```javascript
// next.config.js
eslint: { ignoreDuringBuilds: true },
typescript: { ignoreBuildErrors: true }
```

## Security Considerations

### Current Implementation
- ✅ Clerk authentication for user management
- ✅ User-scoped data access (userId checks)
- ✅ File upload validation (PDF only)
- ✅ SQL injection prevention (parameterized queries)

### Areas to Enhance
- ⚠️ Add file size limits on uploads
- ⚠️ Sanitize user inputs in chat
- ⚠️ Implement rate limiting
- ⚠️ Add API key rotation
- ⚠️ Secure environment variable handling

## API Endpoints Reference

### Frontend API Routes (Next.js)

**Courses**:
- `GET /api/courses?userId={id}` - List user courses
- `POST /api/courses` - Create course + upload files
- `GET /api/courses/[courseId]` - Get course details
- `DELETE /api/courses/[courseId]` - Delete course
- `POST /api/courses/[courseId]/analyze` - AI chapter analysis

**Chats**:
- `GET /api/chats?userId={id}` - List chat sessions
- `POST /api/chats` - Create chat session
- `GET /api/chats/[sessionId]` - Get session details
- `POST /api/chats/[sessionId]/messages` - Send message
- `GET /api/chats/[sessionId]/messages` - Get message history

**Forms**:
- `GET /api/form-config` - Get questionnaire config
- `POST /api/responses` - Submit questionnaire response
- `GET /api/progress?userId={id}` - Get user progress

### Backend API Routes (FastAPI)

**Health**:
- `GET /health` - System health check

**Chat**:
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - Streaming chat (SSE)

**Courses**:
- `GET /courses/{course_id}/load` - Process course materials
- Additional endpoints in rag_api.py

## Environment Setup Checklist

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] SQLite3 available
- [ ] Backend virtual environment created
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] DeepSeek API key configured
- [ ] Clerk authentication configured
- [ ] Database initialized
- [ ] Upload directories created

## Git Workflow

**Branch naming**: `claude/claude-md-{session-id}`

**Commit messages**: Descriptive, action-oriented
```
Add chapter analysis feature
Fix PDF upload path handling
Update database schema for quizzes
```

**Push command**: Always use
```bash
git push -u origin <branch-name>
```

## Quick Reference Commands

**Database inspection**:
```bash
sqlite3 frontend/database.sqlite
.tables
.schema courses
SELECT * FROM courses LIMIT 5;
```

**Find files by pattern**:
```bash
find . -name "*.js" -path "*/api/*"
```

**Check backend health**:
```bash
curl http://localhost:8000/health
```

**Test file upload**:
```bash
curl -X POST http://localhost:3000/api/courses \
  -F "userId=test" \
  -F "name=Test Course" \
  -F "files=@course.pdf"
```

## Additional Notes

### Deployment Considerations
- Frontend: Vercel-ready (Next.js)
- Backend: Containerize FastAPI with Docker
- Database: Migrate to PostgreSQL for production
- File storage: Consider S3 for uploads

### Performance Optimization
- Implement Redis caching for embeddings
- Add pagination for course/chat lists
- Lazy load course indexes
- Optimize FAISS index parameters

### Future Enhancements
- Real-time collaboration features
- Mobile app (React Native)
- Advanced analytics dashboard
- Multi-tenant support
- Exam scheduling system

---

**Last Updated**: 2025-11-14
**Version**: 1.0.0
**Maintainer**: AI Assistant / Development Team
