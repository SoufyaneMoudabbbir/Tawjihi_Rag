# Contributing to Tawjihi RAG

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating an issue
3. **Include**:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Environment details (OS, browser, versions)

### Suggesting Features

1. **Check existing feature requests** first
2. **Describe the problem** you're trying to solve
3. **Propose a solution** with examples
4. **Consider alternatives** you've evaluated

### Pull Requests

#### Before You Start

1. **Fork the repository**
2. **Create a branch** from `develop`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Check open issues** to see if someone else is working on it

#### Development Setup

1. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Tawjihi_Rag.git
   cd Tawjihi_Rag
   ```

2. **Install dependencies**
   ```bash
   # Backend
   cd backend
   python -m venv rag_env_310
   source rag_env_310/bin/activate
   pip install -r requirements.txt

   # Frontend
   cd ../frontend
   npm install
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

#### Making Changes

1. **Write code** following our style guide
2. **Add tests** for new features
3. **Update documentation** as needed
4. **Run tests** to ensure nothing breaks
   ```bash
   # Backend
   cd backend
   pytest tests/ -v

   # Frontend
   cd frontend
   npm test
   ```

5. **Run linting**
   ```bash
   # Backend
   flake8 backend/

   # Frontend
   cd frontend
   npm run lint
   ```

#### Commit Guidelines

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semi-colons, etc)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(chat): add streaming response support

- Implement SSE streaming
- Update frontend to handle chunks
- Add progress indicators

Closes #123
```

```
fix(auth): resolve session timeout issue

Sessions were expiring too quickly due to incorrect
token refresh logic.

Fixes #456
```

#### Submitting Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub
   - Use the PR template
   - Link related issues
   - Describe changes clearly
   - Add screenshots for UI changes

3. **Wait for review**
   - Address feedback promptly
   - Keep PR scope focused
   - Be patient with reviewers

#### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts
- [ ] PR description is clear
- [ ] Related issues linked
- [ ] Screenshots added (for UI changes)

## Development Guidelines

### Code Style

**Python (Backend)**
- Follow PEP 8
- Use type hints
- Maximum line length: 127
- Use docstrings for functions/classes
- Run `black` for formatting

**JavaScript/TypeScript (Frontend)**
- Use ES6+ features
- Prefer functional components
- Use TypeScript for new files
- Follow React best practices
- Run `eslint` for linting

### Testing

**Backend**
- Write unit tests for new functions
- Test edge cases
- Aim for >70% coverage
- Use pytest fixtures

**Frontend**
- Test components in isolation
- Test user interactions
- Test error states
- Use React Testing Library

### Documentation

- Update CLAUDE.md for architecture changes
- Add JSDoc/docstrings for public APIs
- Update README.md for setup changes
- Add inline comments for complex logic

## Project Structure

```
Tawjihi_Rag/
├── backend/          # FastAPI backend
│   ├── main.py       # Application entry
│   ├── routes.py     # API routes
│   ├── models.py     # Pydantic models
│   └── tests/        # Backend tests
├── frontend/         # Next.js frontend
│   ├── app/          # App router
│   ├── components/   # React components
│   └── lib/          # Utilities
└── docs/             # Documentation
```

## Getting Help

- **Discord**: [Join our Discord](#)
- **GitHub Discussions**: Ask questions
- **Documentation**: Check CLAUDE.md
- **Email**: contact@example.com

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README

Thank you for contributing to Tawjihi RAG! 🎓
