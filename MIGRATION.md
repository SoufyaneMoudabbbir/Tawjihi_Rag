# Migration Guide: v1.0 → v2.0

This guide helps you migrate from the monolithic v1.0 to the modular v2.0 architecture.

## ⚠️ Breaking Changes

### 1. Backend Architecture Completely Refactored

**Old (v1.0):**
```python
# Single file: backend/rag_api.py (2433 lines)
from rag_api import app
```

**New (v2.0):**
```python
# Modular structure:
from main import app            # Application entry point
from routes import setup_routes # API routes
from models import *            # Pydantic models
from rag_service import *       # RAG logic
from config import Config       # Configuration
```

### 2. Environment Variables Required

**Action Required:**
1. Copy `.env.example` to `.env`
2. Fill in your API keys:
   ```bash
   DEEPSEEK_API_KEY=your_actual_key_here
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_key
   CLERK_SECRET_KEY=your_clerk_secret
   ```

3. **Remove hardcoded keys** from code

### 3. Database Connection Changes

**Old:**
```python
conn = sqlite3.connect(db_path, check_same_thread=False)
```

**New:**
```python
from db_service import DatabaseService
db = DatabaseService(db_path)  # Thread-safe with connection pooling
```

### 4. API Endpoint Changes

Most endpoints remain the same, but error responses are now standardized:

**Old:**
```json
{"error": "Some error"}
```

**New:**
```json
{
  "error": "Error message",
  "detail": "Detailed explanation",
  "code": "ERROR_CODE"
}
```

## 🚀 Migration Steps

### Step 1: Backup Your Data

```bash
# Backup database
cp frontend/database.sqlite frontend/database.sqlite.backup

# Backup uploads
cp -r frontend/uploads frontend/uploads.backup
```

### Step 2: Update Backend

```bash
cd backend

# Create new virtual environment
python -m venv rag_env_310
source rag_env_310/bin/activate

# Install updated dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env  # or your preferred editor
```

**Required variables:**
- `DEEPSEEK_API_KEY` - Your DeepSeek API key
- `DATABASE_PATH` - Path to SQLite database
- `BACKEND_HOST` - Usually 0.0.0.0
- `BACKEND_PORT` - Usually 8000

### Step 4: Run Database Migration

The new version uses the same database schema, but with improved indexes:

```bash
# No migration needed - compatible with v1.0 database
# But recommended to add new indexes:
sqlite3 frontend/database.sqlite << 'EOF'
CREATE INDEX IF NOT EXISTS idx_courses_user_id ON courses(user_id);
CREATE INDEX IF NOT EXISTS idx_course_files_course_id ON course_files(course_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_course_id ON chat_sessions(course_id);
EOF
```

### Step 5: Update Frontend

```bash
cd frontend

# Install new dependencies
npm install

# Build to check for errors
npm run build
```

### Step 6: Test Migration

```bash
# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

**Verify:**
- [ ] Backend starts without errors
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] Frontend loads
- [ ] Existing courses still visible
- [ ] Can upload new files
- [ ] Chat works with existing courses

## 🔧 Code Changes Needed

### If You Modified Backend Code

#### Example: Custom RAG Logic

**Old:**
```python
# In rag_api.py
class EducationalRAG:
    def my_custom_method(self):
        pass
```

**New:**
Create a new file or modify `rag_service.py`:
```python
# In rag_service.py or custom_rag.py
from rag_service import EducationalRAGService

class CustomRAGService(EducationalRAGService):
    def my_custom_method(self):
        pass
```

#### Example: Custom API Route

**Old:**
```python
# In rag_api.py
@app.get("/my-custom-route")
def my_route():
    pass
```

**New:**
Add to `routes.py`:
```python
def setup_routes(app, rag_system, db_service, limiter):
    # ... existing routes ...

    @app.get("/my-custom-route")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def my_route(request: Request):
        # Your logic here
        pass
```

### If You Modified Frontend Code

No major changes needed. The API interface remains mostly compatible.

**One change:** UI components should NOT be in API route files:
```javascript
// ❌ Old (v1.0): In /app/api/courses/route.js
const UIComponent = () => <div>...</div>

// ✅ New (v2.0): In /components/UIComponent.js
export default function UIComponent() {
  return <div>...</div>
}
```

## 📊 What's Improved

### Security ✅
- ✅ No hardcoded API keys
- ✅ Environment variable management
- ✅ Input validation on all endpoints
- ✅ Rate limiting
- ✅ SQL injection protection
- ✅ Security headers

### Architecture ✅
- ✅ Modular backend (7 files vs 1 massive file)
- ✅ Separation of concerns
- ✅ Thread-safe database connections
- ✅ Proper error handling
- ✅ Logging throughout

### DevOps ✅
- ✅ Docker support
- ✅ Docker Compose for easy deployment
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Pre-commit hooks
- ✅ Testing framework

### Testing ✅
- ✅ Pytest setup with coverage
- ✅ Unit tests for models and utils
- ✅ Test coverage reporting
- ✅ CI/CD integration

### Documentation ✅
- ✅ Comprehensive CLAUDE.md
- ✅ API documentation
- ✅ Security policy
- ✅ Contributing guidelines
- ✅ This migration guide

## 🐛 Troubleshooting

### Backend won't start

**Error:** `Configuration validation failed: DEEPSEEK_API_KEY is required`

**Fix:**
```bash
# Add to .env file
echo "DEEPSEEK_API_KEY=your_key_here" >> .env
```

### Import errors

**Error:** `ModuleNotFoundError: No module named 'config'`

**Fix:**
```bash
# Ensure you're in the backend directory
cd backend
pip install -r requirements.txt
```

### Database errors

**Error:** `Database is locked`

**Fix:**
The new version uses thread-safe connections. This shouldn't happen.
If it does, check that only one backend instance is running.

### Frontend API errors

**Error:** `Failed to fetch`

**Fix:**
Check that backend is running and URL is correct:
```bash
# In .env or .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🔄 Rollback Procedure

If migration fails, rollback:

```bash
# Stop services
# Restore database
cp frontend/database.sqlite.backup frontend/database.sqlite

# Restore uploads
rm -rf frontend/uploads
cp -r frontend/uploads.backup frontend/uploads

# Checkout v1.0
git checkout v1.0  # or appropriate tag/branch
```

## 📞 Support

If you encounter issues:

1. Check GitHub Issues
2. Review CLAUDE.md for architecture details
3. Open a new issue with:
   - Error messages
   - Steps to reproduce
   - Environment details
   - Migration step where it failed

## ✨ Next Steps

After successful migration:

1. **Test thoroughly** with your data
2. **Update documentation** if you have custom modifications
3. **Configure CI/CD** for your repository
4. **Set up monitoring** (recommended)
5. **Plan PostgreSQL migration** for production

---

**Migration completed?** Welcome to v2.0! 🎉
