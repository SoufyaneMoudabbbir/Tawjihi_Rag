import { Inter } from "next/font/google"
import { ClerkProvider } from "@clerk/nextjs"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "EduPath Morocco - Find Your Perfect Educational Journey",
  description:
    "Discover the best educational opportunities in Morocco through AI-powered guidance. Get personalized recommendations based on your background, interests, and career goals.",
    generator: 'v0.dev'
}

export default function RootLayout({ children }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className={inter.className}>
          <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">{children}</div>
        </body>
      </html>
    </ClerkProvider>
  )
}
