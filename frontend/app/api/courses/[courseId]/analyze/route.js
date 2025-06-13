import { NextResponse } from "next/server"

export async function POST(request, { params }) {
  try {
    // Fix: Await params in Next.js 15
    const { courseId } = await params
    const { userId } = await request.json()
    
    console.log(`Starting analysis for course ${courseId}`)
    
    // Call backend analysis
    const response = await fetch(`http://localhost:8000/api/analyze-course-structure`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ 
        course_id: parseInt(courseId), 
        user_id: userId 
      })
    })
    
    if (!response.ok) {
      const errorData = await response.text()
      console.error('Backend analysis failed:', errorData)
      throw new Error(`Analysis failed: ${response.status}`)
    }
    
    const result = await response.json()
    console.log('Analysis result:', result)
    
    return NextResponse.json(result)
  } catch (error) {
    console.error('Analysis error:', error)
    return NextResponse.json({ 
      error: error.message,
      success: false 
    }, { status: 500 })
  }
}