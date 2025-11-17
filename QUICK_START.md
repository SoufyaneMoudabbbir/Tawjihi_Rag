# Quick Start Guide - Refactored Tawjihi RAG

## 🚀 Get Started in 5 Minutes

This guide will get you running with the improved, more secure codebase.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (optional, recommended)

## Option 1: Run with Docker (Recommended)

### Step 1: Set Up Environment

```bash
# Copy environment files
cp .env.example .env
cp frontend/.env.example frontend/.env.local

# Edit .env and add your keys:
# - DEEPSEEK_API_KEY=your_key_here
# - NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_key
# - CLERK_SECRET_KEY=your_key
```

### Step 2: Build and Run

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Step 3: Access

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/health

### Step 4: Stop

```bash
docker-compose down

# Or keep data and just stop
docker-compose stop
```

## Option 2: Run Locally (Development)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv rag_env_310
source rag_env_310/bin/activate  # On Windows: rag_env_310\Scripts\activate

# Install NEW dependencies
pip install -r requirements-new.txt

# Set environment variables
export DEEPSEEK_API_KEY=your_key_here
export DATABASE_PATH=./database.sqlite

# Run the server (using ORIGINAL file for now)
uvicorn rag_api:app --reload --port 8000

# Or to test new modules:
python -c "from app.core.config import settings; print('Config loaded:', settings.APP_NAME)"
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
echo "NEXT_PUBLIC_FRONTEND_URL=http://localhost:3000" >> .env.local

# Add Clerk keys
echo "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_key" >> .env.local
echo "CLERK_SECRET_KEY=your_secret" >> .env.local

# Run development server
npm run dev
```

## 📁 What's New? Key Files to Know

### Backend (New Modular Structure)

```
backend/app/
├── core/
│   ├── config.py          # 🔧 All settings here
│   ├── logging.py         # 📝 Structured logging
│   └── exceptions.py      # ❌ Error handling
├── models/
│   ├── requests.py        # ✅ Input validation
│   └── responses.py       # 📤 Output models
├── database/
│   └── connection.py      # 🗄️ Thread-safe DB
├── services/
│   ├── embedding_service.py    # 🔢 Embeddings
│   └── deepseek_client.py      # 🤖 LLM client
└── utils/
    ├── file_validator.py  # 🔒 Security
    └── text_processing.py # ✂️ Text utils
```

### Frontend (Improved)

```
frontend/
├── lib/
│   └── config.ts          # 🔧 Centralized config (NEW)
├── next.config.js         # ✅ Security headers added
└── .env.example           # 📝 Environment template (NEW)
```

### Root Files

```
Tawjihi_Rag/
├── docker-compose.yml     # 🐳 Docker setup (NEW)
├── Dockerfile.backend     # 🐳 Backend container (NEW)
├── Dockerfile.frontend    # 🐳 Frontend container (NEW)
├── IMPROVEMENTS_SUMMARY.md # 📚 All changes (NEW)
├── REFACTORING_PROGRESS.md # 📊 Status tracker (NEW)
└── QUICK_START.md         # 🚀 This file (NEW)
```

## 🔍 Testing the Improvements

### 1. Test Configuration Management

```bash
cd backend
python -c "
from app.core.config import settings
print(f'✅ App: {settings.APP_NAME}')
print(f'✅ Environment: {settings.ENVIRONMENT}')
print(f'✅ Max file size: {settings.MAX_FILE_SIZE_MB}MB')
print(f'✅ API key set: {len(settings.DEEPSEEK_API_KEY) > 0}')
"
```

### 2. Test File Validation

```bash
python -c "
from app.utils.file_validator import FileValidator

# This should pass
pdf_data = b'%PDF-1.4\n%some pdf content'
print('✅ Valid PDF:', FileValidator.validate_pdf_signature(pdf_data))

# This should fail
try:
    FileValidator.validate_pdf_signature(b'Not a PDF')
    print('❌ Should have failed')
except Exception as e:
    print('✅ Invalid PDF rejected:', str(e))
"
```

### 3. Test Logging

```bash
python -c "
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger('test')
logger.info('Test log message', extra={'test_id': 123})
print('✅ Check logs/ directory for structured JSON logs')
"
```

### 4. Test Frontend Config

```bash
cd frontend
node -e "
const config = require('./lib/config.ts');
console.log('✅ API URL:', config.default.api.baseUrl);
console.log('✅ Max file size:', config.default.upload.maxFileSize);
"
```

## 🔧 Environment Variables Reference

### Backend (.env)

```bash
# Required
DEEPSEEK_API_KEY=sk-...

# Optional (with defaults)
DATABASE_PATH=./database.sqlite
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

### Frontend (.env.local)

```bash
# Required
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...

# Optional
NEXT_PUBLIC_FRONTEND_URL=http://localhost:3000
NODE_ENV=development
```

## 🐛 Troubleshooting

### "Module not found" errors (Backend)

```bash
cd backend
pip install -r requirements-new.txt  # Make sure you use the NEW file
```

### "Cannot find module '@/lib/config'" (Frontend)

```bash
cd frontend
npm install
# Make sure you created lib/config.ts
```

### Database locked errors

The new async database manager fixes this, but if using old code:
```bash
# Stop all processes using the database
lsof database.sqlite  # On macOS/Linux
# Kill processes, then restart
```

### Docker build fails

```bash
# Clear cache and rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Port already in use

```bash
# Backend (8000)
lsof -ti:8000 | xargs kill -9

# Frontend (3000)
lsof -ti:3000 | xargs kill -9
```

## 📊 What's Working vs. What's Next

### ✅ Working Now
- All new utility modules (config, logging, exceptions)
- File validation with security checks
- Database connection manager (async, thread-safe)
- Embedding service
- DeepSeek client
- Docker deployment
- Frontend configuration
- Security headers

### ⏳ In Progress (Can Still Use Old Code)
- RAG service (original `rag_api.py` still works)
- API routes (original endpoints still work)
- Rate limiting (infrastructure ready)
- Database migrations (Alembic ready to configure)

### 📋 Coming Soon
- Complete RAG service migration
- Comprehensive tests
- PostgreSQL migration
- Redis caching
- CI/CD pipeline

## 🎯 Your First Task

Try this to see the improvements:

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your keys

# 2. Test new backend modules
cd backend
source rag_env_310/bin/activate
pip install -r requirements-new.txt
python -c "from app.core.config import settings; print('✅ Config works!')"

# 3. Run original backend (it still works!)
uvicorn rag_api:app --reload

# 4. In another terminal, test health endpoint
curl http://localhost:8000/health

# 5. See the new structured logs
cat logs/app.log
```

## 📚 Learn More

- **IMPROVEMENTS_SUMMARY.md** - Detailed list of all fixes
- **REFACTORING_PROGRESS.md** - Migration status and timeline
- **Backend code** - Check `backend/app/` for new modules
- **Inline comments** - All new code is heavily documented

## 🆘 Need Help?

1. **Check logs**: `backend/logs/app.log` (JSON formatted)
2. **Check health**: `curl http://localhost:8000/health`
3. **Check docs**: Each module has docstrings
4. **Check issues**: GitHub issues (if applicable)

## 🎉 Success Criteria

You'll know it's working when:

- ✅ `http://localhost:8000/health` returns JSON
- ✅ Frontend loads at `http://localhost:3000`
- ✅ You can create a course and upload PDFs
- ✅ Chat works and gets responses
- ✅ Logs appear in `backend/logs/app.log`
- ✅ No console errors in browser

## 🚀 Next Steps

Once you have it running:

1. **Review the improvements**: Read `IMPROVEMENTS_SUMMARY.md`
2. **Plan the migration**: Check `REFACTORING_PROGRESS.md`
3. **Start using new modules**: Begin with utils and services
4. **Add tests**: Use the new infrastructure
5. **Deploy with Docker**: Use `docker-compose.yml`

---

**Need Help?** All new code has extensive comments and docstrings!

**Questions?** Check the documentation files in the root directory.

**Ready to contribute?** Start with `REFACTORING_PROGRESS.md` to see what's next!
