#!/usr/bin/env node

/**
 * Database Initialization Test Script
 * Tests that all required tables and indexes are created automatically
 */

import { openDb } from './lib/db.js'

const EXPECTED_TABLES = [
  'courses',
  'course_files',
  'chat_sessions',
  'chat_messages',
  'user_responses',
  'course_chapters',
  'chapter_content',
  'chapter_quizzes'
]

const EXPECTED_INDEXES = [
  'idx_courses_user_id',
  'idx_course_files_course_id',
  'idx_chat_sessions_user_id',
  'idx_chat_sessions_course_id',
  'idx_chat_sessions_type',
  'idx_chat_messages_session_id',
  'idx_user_responses_user_id',
  'idx_course_chapters_course_id',
  'idx_chapter_content_chapter_id',
  'idx_chapter_quizzes_chapter_id'
]

async function testDatabaseInitialization() {
  console.log('🧪 Testing database initialization...\n')

  try {
    // Open database (should trigger initialization)
    const db = await openDb()
    console.log('✅ Database connection established\n')

    // Check all tables exist
    console.log('📋 Checking tables...')
    const tables = await db.all(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    const tableNames = tables.map(t => t.name)

    let allTablesExist = true
    for (const expectedTable of EXPECTED_TABLES) {
      if (tableNames.includes(expectedTable)) {
        console.log(`  ✅ ${expectedTable}`)
      } else {
        console.log(`  ❌ ${expectedTable} - MISSING!`)
        allTablesExist = false
      }
    }

    // Check all indexes exist
    console.log('\n📑 Checking indexes...')
    const indexes = await db.all(
      "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name"
    )
    const indexNames = indexes.map(i => i.name)

    let allIndexesExist = true
    for (const expectedIndex of EXPECTED_INDEXES) {
      if (indexNames.includes(expectedIndex)) {
        console.log(`  ✅ ${expectedIndex}`)
      } else {
        console.log(`  ❌ ${expectedIndex} - MISSING!`)
        allIndexesExist = false
      }
    }

    // Test critical columns
    console.log('\n🔍 Checking critical columns...')

    // Check chat_sessions has session_type column
    const chatSessionsColumns = await db.all("PRAGMA table_info(chat_sessions)")
    const hasSessionType = chatSessionsColumns.some(col => col.name === 'session_type')
    if (hasSessionType) {
      console.log('  ✅ chat_sessions.session_type column exists')
    } else {
      console.log('  ❌ chat_sessions.session_type column MISSING!')
      allTablesExist = false
    }

    // Check chat_messages table structure
    const chatMessagesColumns = await db.all("PRAGMA table_info(chat_messages)")
    const hasTypeColumn = chatMessagesColumns.some(col => col.name === 'type')
    if (hasTypeColumn) {
      console.log('  ✅ chat_messages.type column exists')
    } else {
      console.log('  ❌ chat_messages.type column MISSING!')
      allTablesExist = false
    }

    // Final result
    console.log('\n' + '='.repeat(50))
    if (allTablesExist && allIndexesExist) {
      console.log('🎉 ALL TESTS PASSED! Database is properly initialized.')
      process.exit(0)
    } else {
      console.log('❌ TESTS FAILED! Some tables or indexes are missing.')
      process.exit(1)
    }

  } catch (error) {
    console.error('❌ Error during testing:', error)
    process.exit(1)
  }
}

testDatabaseInitialization()
