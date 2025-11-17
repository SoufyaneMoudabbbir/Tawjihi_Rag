/**
 * Frontend Configuration
 * Centralized configuration management for the frontend
 */

export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    timeout: 30000,
  },

  frontend: {
    baseUrl: process.env.NEXT_PUBLIC_FRONTEND_URL || 'http://localhost:3000',
  },

  upload: {
    maxFileSize: 50 * 1024 * 1024, // 50MB
    allowedTypes: ['application/pdf'],
    allowedExtensions: ['.pdf'],
  },

  features: {
    autoAnalysis: true,
    offlineMode: false,
    debugMode: process.env.NODE_ENV === 'development',
  },

  pagination: {
    defaultPageSize: 10,
    maxPageSize: 100,
  },
} as const

// Validate required environment variables at build time
export function validateConfig() {
  const required = ['NEXT_PUBLIC_API_URL']
  const missing = required.filter(key => !process.env[key])

  if (missing.length > 0 && process.env.NODE_ENV === 'production') {
    console.warn(`⚠️  Missing environment variables: ${missing.join(', ')}`)
    console.warn('Using default values. Set these in .env.local for production.')
  }
}

// Run validation
if (typeof window === 'undefined') {
  // Server-side only
  validateConfig()
}

export default config
