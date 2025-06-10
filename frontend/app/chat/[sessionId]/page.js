"use client";

import { useUser } from "@clerk/nextjs";
import dynamic from "next/dynamic";
import { useState, useEffect, useRef } from "react";
import { useRouter, useParams } from "next/navigation";
import {
  Send,
  Bot,
  User,
  Loader,
  BookOpen,
  FileText,
  Menu,
  X,
  Home,
  Plus,
  ChevronRight,
  Target,
  Brain,
} from "lucide-react";
import Link from "next/link";

// Dynamically import AnimatePresence to avoid SSR issues
const AnimatePresence = dynamic(
  () => import("framer-motion").then((mod) => mod.AnimatePresence),
  { ssr: false }
);
const MotionDiv = dynamic(() => import("framer-motion").then((mod) => mod.motion.div), {
  ssr: false,
});
const MotionAside = dynamic(() => import("framer-motion").then((mod) => mod.motion.aside), {
  ssr: false,
});

export default function ChatPage() {
  const { user, isLoaded } = useUser();
  const router = useRouter();
  const params = useParams();
  const sessionId = params.sessionId;
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [courseInfo, setCourseInfo] = useState(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [userProfile, setUserProfile] = useState(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [chatSessions, setChatSessions] = useState([]);

  useEffect(() => {
    if (isLoaded && user && sessionId) {
      loadChatSession();
      loadChatHistory();
    }
  }, [isLoaded, user, sessionId]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      scrollToBottom();
    }
  }, [messages]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      adjustTextareaHeight();
    }
  }, [inputMessage]);

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`/api/chats?userId=${user.id}`);
      if (response.ok) {
        const data = await response.json();
        setChatSessions(data.sessions || []);
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  const loadChatSession = async () => {
    setIsLoading(true);
    try {
      // Load session details
      const sessionResponse = await fetch(`/api/chats/${sessionId}`);
      if (!sessionResponse.ok) {
        router.push("/dashboard");
        return;
      }

      const sessionData = await sessionResponse.json();

      // Load messages
      const messagesResponse = await fetch(`/api/chats/${sessionId}/messages`);
      if (messagesResponse.ok) {
        const messagesData = await messagesResponse.json();
        setMessages(messagesData.messages || []);
      }

      // Load course info if available
      const courseId = sessionData?.session?.course_id || sessionData?.course?.id;
      if (courseId) {
        const courseResponse = await fetch(`/api/courses?userId=${user.id}`);
        if (courseResponse.ok) {
          const coursesData = await courseResponse.json();
          const course = coursesData.courses?.find((c) => c.id === courseId);
          if (course) {
            setCourseInfo(course);
            generateSuggestedQuestions(course);
          } else {
            generateGeneralSuggestedQuestions();
          }
        }
      } else {
        setCourseInfo(null);
        generateGeneralSuggestedQuestions();
      }

      // Load user profile
      const profileResponse = await fetch(`/api/responses?userId=${user.id}`);
      if (profileResponse.ok) {
        const profileData = await profileResponse.json();
        setUserProfile(profileData.responses || {});
        console.log("User profile loaded:", profileData.responses);
      }
    } catch (error) {
      console.error("Error loading chat session:", error);
      router.push("/dashboard");
    } finally {
      setIsLoading(false);
    }
  };

  const generateSuggestedQuestions = (course) => {
    const questions = [
      `Explain the key concepts in ${course.name}`,
      `Create a study plan for ${course.name}`,
      `What are common exam questions for this course?`,
      `Generate practice problems for ${course.name}`,
    ];
    setSuggestedQuestions(questions);
  };

  const generateGeneralSuggestedQuestions = () => {
    const questions = [
      "Help me create a study schedule",
      "Explain a concept I'm struggling with",
      "What are effective study techniques?",
      "How can I improve my note-taking?",
    ];
    setSuggestedQuestions(questions);
  };

  const handleSendMessage = async (messageText = null) => {
    const message = messageText || inputMessage.trim();
    if (!message || isTyping || isStreaming) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: message,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsTyping(true);
    setIsStreaming(true);

    try {
      // Save user message to database
      await fetch(`/api/chats/${sessionId}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
      });

      // Call the educational RAG API
      const response = await fetch("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: message,
          course_id: courseInfo?.id || null,
          user_id: user.id,
          user_profile: userProfile,
          stream: true,
        }),
      });

      if (response.ok) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let botResponse = "";
        const responseId = Date.now() + 1;

        setMessages((prev) => [
          ...prev,
          { id: responseId, type: "bot", content: "", timestamp: new Date() },
        ]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === "content") {
                  botResponse += data.data;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === responseId ? { ...msg, content: botResponse } : msg
                    )
                  );
                } else if (data.type === "done") {
                  await fetch(`/api/chats/${sessionId}/messages`, {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                      message: {
                        id: responseId,
                        type: "bot",
                        content: botResponse,
                        timestamp: new Date(),
                      },
                    }),
                  });
                }
              } catch (parseError) {
                console.error("Error parsing SSE data:", parseError);
              }
            }
          }
        }
      } else {
        const fallbackResponse = {
          id: Date.now() + 1,
          type: "bot",
          content: courseInfo
            ? `I'm here to help you learn ${courseInfo.name}. Could you please rephrase your question or ask about a specific topic from your course materials?`
            : "I'm here to help with your studies. Could you please provide more details about what you'd like to learn or ask about?",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, fallbackResponse]);
        await fetch(`/api/chats/${sessionId}/messages`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: fallbackResponse }),
        });
      }
    } catch (error) {
      console.error("Error in handleSendMessage:", error);
      const errorResponse = {
        id: Date.now() + 1,
        type: "bot",
        content: "I'm having trouble connecting right now. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const createNewChat = async () => {
    try {
      const response = await fetch("/api/chats", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId: user.id,
          title: `New Learning Session - ${new Date().toLocaleDateString()}`,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        router.push(`/chat/${data.sessionId}`);
      }
    } catch (error) {
      console.error("Error creating chat:", error);
    }
  };

  if (!isLoaded || isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <Loader className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading your learning session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-gray-50">
      <AnimatePresence>
        {showSidebar && (
          <>
            <MotionDiv
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowSidebar(false)}
              className="fixed inset-0 bg-black/20 z-40 lg:hidden"
            />
            <MotionAside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: "spring", damping: 25 }}
              className="fixed lg:relative w-72 h-full bg-white shadow-lg z-50 flex flex-col"
            >
              <div className="p-4 border-b">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">Learning Sessions</h2>
                  <button
                    onClick={() => setShowSidebar(false)}
                    className="lg:hidden p-1 hover:bg-gray-100 rounded-lg"
                  >
                    <X className="h-5 w-5 text-gray-500" />
                  </button>
                </div>
                <button
                  onClick={createNewChat}
                  className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Plus className="h-4 w-4" />
                  <span>New Chat</span>
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {chatSessions.map((session) => (
                  <Link
                    key={session.id}
                    href={`/chat/${session.id}`}
                    className={`block p-3 rounded-lg hover:bg-gray-100 transition-colors ${
                      session.id === parseInt(sessionId) ? "bg-blue-50 border-l-4 border-blue-600" : ""
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <Brain className="h-4 w-4 text-gray-400 mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">{session.title}</p>
                        <p className="text-xs text-gray-500">
                          {session.course_name || "General Learning"}
                        </p>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
              <div className="p-4 border-t space-y-2">
                <Link
                  href="/dashboard"
                  className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <Home className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-700">Dashboard</span>
                </Link>
                <Link
                  href="/courses"
                  className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <BookOpen className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-700">My Courses</span>
                </Link>
              </div>
            </MotionAside>
          </>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col h-screen">
        <header className="bg-white border-b px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowSidebar(!showSidebar)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <Menu className="h-5 w-5 text-gray-600" />
              </button>
              <div>
                <h1 className="text-lg font-semibold text-gray-900">
                  {courseInfo ? courseInfo.name : "Learning Assistant"}
                </h1>
                {courseInfo && (
                  <p className="text-sm text-gray-500 flex items-center space-x-2">
                    <FileText className="h-3 w-3" />
                    <span>{courseInfo.file_count || 0} materials</span>
                    <span className="text-gray-300">•</span>
                    <Target className="h-3 w-3" />
                    <span>{courseInfo.progress || 0}% complete</span>
                  </p>
                )}
              </div>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 && !isTyping && (
              <div className="py-8 px-4">
                <div className="text-center mb-8">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="h-10 w-10 text-white" />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    {courseInfo ? `Let's learn ${courseInfo.name}!` : "Ready to learn!"}
                  </h2>
                  <p className="text-gray-600 max-w-md mx-auto">
                    {courseInfo
                      ? "I have access to your course materials and can help explain concepts, create practice problems, and answer questions."
                      : "Ask me anything about your studies. I'm here to help you learn and understand."}
                  </p>
                </div>
                {suggestedQuestions.length > 0 && (
                  <div className="space-y-3 max-w-2xl mx-auto">
                    <p className="text-sm text-gray-500 text-center mb-4">Try asking:</p>
                    {suggestedQuestions.map((question, index) => (
                      <button
                        key={index}
                        onClick={() => handleSendMessage(question)}
                        className="w-full text-left p-4 bg-white rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all group"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700 group-hover:text-gray-900">
                            {question}
                          </span>
                          <ChevronRight className="h-4 w-4 text-gray-400 group-hover:text-blue-600 transition-colors" />
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            <div className="px-4 py-6 space-y-6">
              {messages.map((message) => (
                <MotionDiv
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`flex items-start space-x-3 max-w-3xl ${
                      message.type === "user" ? "flex-row-reverse space-x-reverse" : ""
                    }`}
                  >
                    <div
                      className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                        message.type === "user" ? "bg-blue-600" : "bg-gray-200"
                      }`}
                    >
                      {message.type === "user" ? (
                        <User className="h-5 w-5 text-white" />
                      ) : (
                        <Bot className="h-5 w-5 text-gray-600" />
                      )}
                    </div>
                    <div className={`flex-1 ${message.type === "user" ? "text-right" : ""}`}>
                      <div
                        className={`inline-block px-4 py-2 rounded-2xl ${
                          message.type === "user"
                            ? "bg-blue-600 text-white"
                            : "bg-white border border-gray-200 text-gray-800"
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </MotionDiv>
              ))}
              {isTyping && (
                <MotionDiv
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-start space-x-3"
                >
                  <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                    <Bot className="h-5 w-5 text-gray-600" />
                  </div>
                  <div className="bg-white border border-gray-200 rounded-2xl px-4 py-2">
                    <div className="flex space-x-1">
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      ></div>
                    </div>
                  </div>
                </MotionDiv>
              )}
            </div>
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="border-t bg-white">
          <div className="max-w-4xl mx-auto p-4">
            <div className="flex items-end space-x-4">
              <div className="flex-1 relative">
                <textarea
                  ref={textareaRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={courseInfo ? `Ask about ${courseInfo.name}...` : "Ask a question..."}
                  className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="1"
                  disabled={isStreaming}
                />
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!inputMessage.trim() || isStreaming}
                  className={`absolute right-2 bottom-2 p-2 rounded-lg transition-colors ${
                    inputMessage.trim() && !isStreaming
                      ? "bg-blue-600 text-white hover:bg-blue-700"
                      : "bg-gray-100 text-gray-400"
                  }`}
                >
                  <Send className="h-4 w-4" />
                </button>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              AI can make mistakes. Verify important information with your course materials.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}