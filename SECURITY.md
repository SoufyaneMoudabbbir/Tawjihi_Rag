# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :x:                |

## Reporting a Vulnerability

We take the security of Tawjihi RAG seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Disclose Publicly

Please do not open a public issue for security vulnerabilities.

### 2. Report Privately

Send an email to [security@example.com] with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity (Critical: 7 days, High: 14 days, Medium: 30 days)

## Security Best Practices

### For Developers

1. **Environment Variables**
   - Never commit API keys or secrets
   - Use `.env.example` as a template
   - Rotate API keys regularly

2. **Dependencies**
   - Keep dependencies up to date
   - Run `npm audit` and `pip check` regularly
   - Review security advisories

3. **Code Review**
   - All PRs require review
   - Security-sensitive changes need additional scrutiny
   - Use pre-commit hooks

### For Deployment

1. **API Keys**
   - Use environment variables only
   - Implement key rotation
   - Monitor API usage

2. **Database**
   - Use PostgreSQL in production (not SQLite)
   - Enable SSL connections
   - Regular backups

3. **Network**
   - Use HTTPS only
   - Configure CORS properly
   - Implement rate limiting

4. **Access Control**
   - Implement least privilege principle
   - Use Clerk for authentication
   - Validate user permissions

## Known Security Considerations

### Database
- SQLite is for development only
- Migrate to PostgreSQL for production
- Enable connection pooling

### File Uploads
- PDF files only
- Size limits enforced (50MB default)
- Virus scanning recommended for production

### API Security
- Rate limiting: 60 requests/minute default
- Input validation on all endpoints
- Sanitization of user inputs

## Security Checklist

- [ ] All API keys in environment variables
- [ ] HTTPS enabled in production
- [ ] CORS configured properly
- [ ] Rate limiting active
- [ ] Input validation implemented
- [ ] SQL injection protection verified
- [ ] XSS protection enabled
- [ ] File upload validation active
- [ ] Error messages don't leak sensitive info
- [ ] Logging configured (no sensitive data logged)
- [ ] Dependencies updated
- [ ] Security headers configured

## Contact

For security concerns, contact: [security@example.com]
