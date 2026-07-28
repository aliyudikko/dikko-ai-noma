// app/page.tsx

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Send, 
  Menu, 
  X, 
  Plus, 
  Copy, 
  RefreshCw, 
  ThumbsUp, 
  ThumbsDown,
  Search,
  Settings,
  Info,
  User,
  Bot,
  Database,
  Sun,
  Moon,
  Loader2,
  Sparkles,
  BookOpen,
  Check,
  AlertCircle,
  ChevronLeft,
  PanelLeft
} from 'lucide-react';
import { apiService } from '@/lib/api';
import { Message, Conversation } from '@/types/chat';

// ============ TYPES ============

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  isGenerating: boolean;
  error: string | null;
  ragActive: boolean;
}

// ============ COMPONENTS ============

// ---------- Welcome Screen ----------
function WelcomeScreen({ onPromptSelect }: { onPromptSelect: (prompt: string) => void }) {
  const examplePrompts = [
    'Yaya ake noman masara?',
    'Yaushe ne lokacin dashen shinkafa?',
    'Ta yaya zan magance kwari a gonar tumatir?',
    'Wane taki ya dace da masara?',
    'Yaya zan kula da shanu a lokacin rani?',
    'Menene mafi kyawun hanyar sarrafa ciyawa?',
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-3xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-16 h-16 bg-green-600 rounded-2xl flex items-center justify-center shadow-lg">
          <Bot className="w-10 h-10 text-white" />
        </div>
      </div>
      
      <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
        Assalamu alaikum!
      </h1>
      <p className="text-2xl font-semibold text-green-700 dark:text-green-400 mb-3">
        Ni Dikko AI Noma ne.
      </p>
      <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
        Zan iya taimaka maka da bayanai da shawarwari kan noma.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
        {examplePrompts.map((prompt, index) => (
          <button
            key={index}
            onClick={() => onPromptSelect(prompt)}
            className="text-left p-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border border-gray-200 dark:border-gray-700 rounded-xl hover:border-green-500 dark:hover:border-green-500 hover:bg-white dark:hover:bg-gray-800 transition-all hover:shadow-md text-gray-700 dark:text-gray-300 text-sm"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------- Chat Message ----------
function ChatMessage({ 
  message, 
  onCopy, 
  onRegenerate, 
  onLike, 
  onDislike 
}: { 
  message: Message; 
  onCopy: (content: string) => void;
  onRegenerate: () => void;
  onLike: () => void;
  onDislike: () => void;
}) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState(false);
  const [disliked, setDisliked] = useState(false);

  const handleCopy = () => {
    onCopy(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleLike = () => {
    setLiked(!liked);
    setDisliked(false);
    onLike();
  };

  const handleDislike = () => {
    setDisliked(!disliked);
    setLiked(false);
    onDislike();
  };

  // Render markdown-like content (simplified for Hausa text)
  const renderContent = (content: string) => {
    const paragraphs = content.split(/\n\s*\n/);
    
    return paragraphs.map((paragraph, idx) => {
      if (paragraph.trim().match(/^[\d]+\.\s/)) {
        const items = paragraph.split(/\n/).filter(line => line.trim());
        return (
          <ol key={idx} className="list-decimal list-inside space-y-1 my-2">
            {items.map((item, i) => (
              <li key={i} className="pl-2">{item.replace(/^[\d]+\.\s/, '')}</li>
            ))}
          </ol>
        );
      }
      
      if (paragraph.trim().match(/^[•\-]\s/)) {
        const items = paragraph.split(/\n/).filter(line => line.trim());
        return (
          <ul key={idx} className="list-disc list-inside space-y-1 my-2">
            {items.map((item, i) => (
              <li key={i} className="pl-2">{item.replace(/^[•\-]\s/, '')}</li>
            ))}
          </ul>
        );
      }
      
      return <p key={idx} className="mb-3 last:mb-0 leading-relaxed">{paragraph}</p>;
    });
  };

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}>
      {!isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-green-600 flex items-center justify-center shadow-sm">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}
      
      <div className={`max-w-[85%] ${isUser ? 'order-1' : ''}`}>
        <div className={`
          rounded-2xl px-5 py-4
          ${isUser 
            ? 'bg-green-600 text-white shadow-sm' 
            : 'bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm border border-gray-200/50 dark:border-gray-700/50 text-gray-800 dark:text-gray-200 shadow-sm'
          }
        `}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose dark:prose-invert prose-sm max-w-none">
              {renderContent(message.content)}
              
              {message.usedRag && (
                <div className="mt-3 flex items-center gap-2 text-xs text-green-600 dark:text-green-400 bg-green-50/80 dark:bg-green-900/30 backdrop-blur-sm px-3 py-1.5 rounded-full w-fit">
                  <Database className="w-3.5 h-3.5" />
                  <span>An samo bayanai daga tushen ilimi</span>
                </div>
              )}
            </div>
          )}
        </div>
        
        {!isUser && (
          <div className="flex items-center gap-1 mt-2 px-1">
            <button 
              onClick={handleCopy}
              className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors"
              title="Kwafi"
            >
              {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
            <button 
              onClick={onRegenerate}
              className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors"
              title="Sake samarwa"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button 
              onClick={handleLike}
              className={`p-1.5 rounded-md transition-colors ${liked ? 'text-green-500' : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100/50 dark:hover:bg-gray-800/50'}`}
              title="Na ji daɗi"
            >
              <ThumbsUp className="w-4 h-4" />
            </button>
            <button 
              onClick={handleDislike}
              className={`p-1.5 rounded-md transition-colors ${disliked ? 'text-red-500' : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100/50 dark:hover:bg-gray-800/50'}`}
              title="Ban ji daɗi ba"
            >
              <ThumbsDown className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
      
      {isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-gray-200/80 dark:bg-gray-700/80 backdrop-blur-sm flex items-center justify-center">
          <User className="w-5 h-5 text-gray-600 dark:text-gray-300" />
        </div>
      )}
    </div>
  );
}

// ---------- Chat Input ----------
function ChatInput({ 
  onSend, 
  onStop,
  isLoading, 
  isGenerating,
  disabled 
}: { 
  onSend: (message: string) => void;
  onStop: () => void;
  isLoading: boolean;
  isGenerating: boolean;
  disabled?: boolean;
}) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }, [input]);

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <div className="flex items-end gap-2 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border border-gray-200/50 dark:border-gray-700/50 rounded-2xl p-2 shadow-[0_4px_24px_rgba(0,0,0,0.06)] dark:shadow-[0_4px_24px_rgba(0,0,0,0.3)] transition-all focus-within:border-green-500/50 focus-within:ring-2 focus-within:ring-green-500/20">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Tambaya kan noma..."
          className="flex-1 resize-none bg-transparent px-3 py-2 text-gray-700 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 outline-none min-h-[44px] max-h-[200px] text-base"
          rows={1}
          disabled={disabled || isLoading}
        />
        
        {isGenerating ? (
          <button
            type="button"
            onClick={onStop}
            className="flex-shrink-0 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-colors flex items-center gap-2 shadow-sm"
          >
            <X className="w-4 h-4" />
            <span className="text-sm font-medium hidden sm:inline">Tsaya</span>
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim() || isLoading || disabled}
            className={`flex-shrink-0 px-4 py-2 rounded-xl transition-all flex items-center gap-2 ${
              input.trim() && !isLoading && !disabled
                ? 'bg-green-600 hover:bg-green-700 text-white shadow-sm hover:shadow-md'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span className="text-sm font-medium hidden sm:inline">Aika</span>
              </>
            )}
          </button>
        )}
      </div>
    </form>
  );
}

// ---------- Sidebar ----------
function Sidebar({ 
  isOpen, 
  onClose,
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onToggleDarkMode,
  isDark,
}: { 
  isOpen: boolean;
  onClose: () => void;
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onToggleDarkMode: () => void;
  isDark: boolean;
}) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredConversations = conversations.filter(conv =>
    conv.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      <aside className={`
        fixed top-0 left-0 z-50 h-full w-[280px] bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl border-r border-gray-200/50 dark:border-gray-800/50
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static lg:z-0
        flex flex-col shadow-2xl dark:shadow-2xl dark:shadow-black/30
      `}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200/50 dark:border-gray-800/50">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-green-600 flex items-center justify-center shadow-sm">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="font-semibold text-gray-800 dark:text-white text-sm">Dikko AI Noma</div>
              <div className="text-[10px] text-gray-500 dark:text-gray-400">AI for Farmers</div>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="lg:hidden p-1.5 rounded-md hover:bg-gray-100/50 dark:hover:bg-gray-800/50 text-gray-500 dark:text-gray-400 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Conversation */}
        <div className="p-3">
          <button
            onClick={onNewConversation}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-green-600 hover:bg-green-700 text-white rounded-xl transition-colors text-sm font-medium shadow-sm hover:shadow-md"
          >
            <Plus className="w-4 h-4" />
            Sabuwar Hira
          </button>
        </div>

        {/* Search */}
        <div className="px-3 pb-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Nemo tattaunawa..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-gray-100/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-xl text-sm text-gray-700 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 outline-none border border-transparent focus:border-green-500/50 focus:bg-white dark:focus:bg-gray-800 transition-all"
            />
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-3 pb-3">
          <div className="space-y-1">
            {filteredConversations.length === 0 ? (
              <div className="text-center text-gray-400 dark:text-gray-500 text-sm py-8">
                {searchQuery ? 'Ba a sami tattaunawa ba' : 'Babu tattaunawa a yanzu'}
              </div>
            ) : (
              filteredConversations.map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => onSelectConversation(conv.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl transition-all text-sm ${
                    currentConversationId === conv.id
                      ? 'bg-green-50/80 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                      : 'hover:bg-gray-100/50 dark:hover:bg-gray-800/50 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <div className="truncate font-medium">{conv.title}</div>
                  <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                    {conv.messages.length} saƙonni
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200/50 dark:border-gray-800/50 p-3 space-y-1">
          <button 
            onClick={() => window.location.href = '/about'}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors text-sm text-gray-600 dark:text-gray-400"
          >
            <Info className="w-4 h-4" />
            Game da Dikko AI Noma
          </button>
          <button 
            onClick={onToggleDarkMode}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors text-sm text-gray-600 dark:text-gray-400"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            {isDark ? 'Hasken rana' : 'Yanayin dare'}
          </button>
          <div className="px-3 pt-2 text-[10px] text-gray-400 dark:text-gray-500 border-t border-gray-200/50 dark:border-gray-800/50 mt-1">
            Developer: Yahya Aliyu Dikko
          </div>
        </div>
      </aside>
    </>
  );
}

// ============ MAIN PAGE ============

export default function Home() {
  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ragActive, setRagActive] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [apiAvailable, setApiAvailable] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentMessageId = useRef(0);

  // Load dark mode preference
  useEffect(() => {
    const saved = localStorage.getItem('dikko-dark-mode');
    if (saved === 'true' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      setIsDark(true);
      document.documentElement.classList.add('dark');
    }
  }, []);

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDark = !isDark;
    setIsDark(newDark);
    localStorage.setItem('dikko-dark-mode', String(newDark));
    document.documentElement.classList.toggle('dark');
  };

  // Check API health
  useEffect(() => {
    apiService.healthCheck().then(available => {
      setApiAvailable(available);
    });
  }, []);

  // Auto scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Generate unique ID
  const generateId = () => {
    currentMessageId.current += 1;
    return `msg-${Date.now()}-${currentMessageId.current}`;
  };

  // Create new conversation
  const createNewConversation = useCallback(() => {
    const id = `conv-${Date.now()}`;
    const newConv: Conversation = {
      id,
      title: 'Sabuwar tattaunawa',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setConversations(prev => [newConv, ...prev]);
    setCurrentConversationId(id);
    setMessages([]);
    setError(null);
    setSidebarOpen(false);
  }, []);

  // Load conversation
  const loadConversation = useCallback((id: string) => {
    const conv = conversations.find(c => c.id === id);
    if (conv) {
      setCurrentConversationId(id);
      setMessages(conv.messages);
      setError(null);
      setSidebarOpen(false);
    }
  }, [conversations]);

  // Send message
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setIsGenerating(true);
    setError(null);

    if (messages.length === 0) {
      const title = content.trim().slice(0, 50) + (content.trim().length > 50 ? '...' : '');
      setConversations(prev => prev.map(conv => 
        conv.id === currentConversationId ? { ...conv, title } : conv
      ));
    }

    try {
      const response = await apiService.sendMessage({
        message: content.trim(),
        conversation_id: currentConversationId || undefined,
      });

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        sources: response.sources,
        usedRag: response.used_rag,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setRagActive(response.used_rag);

      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [...conv.messages, userMessage, assistantMessage],
            updatedAt: new Date(),
          };
        }
        return conv;
      }));

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An samu matsala';
      setError(errorMsg);
      
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: 'Yi haƙuri, an samu matsala wajen haɗawa da sabar. Da fatan za a sake gwadawa.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsGenerating(false);
    }
  }, [messages, currentConversationId]);

  // Stop generation
  const stopGeneration = useCallback(() => {
    setIsGenerating(false);
    setIsLoading(false);
  }, []);

  // Copy message
  const copyMessage = useCallback((content: string) => {
    navigator.clipboard.writeText(content).catch(() => {
      alert('Da fatan za a kwafi saƙon da hannu.');
    });
  }, []);

  // Regenerate
  const regenerateLast = useCallback(() => {
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMessage) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'assistant') {
        setMessages(prev => prev.slice(0, -1));
      }
      sendMessage(lastUserMessage.content);
    }
  }, [messages, sendMessage]);

  // Like/Dislike handlers
  const handleLike = useCallback(() => {
    console.log('Liked');
  }, []);

  const handleDislike = useCallback(() => {
    console.log('Disliked');
  }, []);

  // Initialize with welcome message
  useEffect(() => {
    if (conversations.length === 0 && messages.length === 0) {
      createNewConversation();
    }
  }, []);

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-50/50 via-white to-gray-50/50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950 overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={loadConversation}
        onNewConversation={createNewConversation}
        onToggleDarkMode={toggleDarkMode}
        isDark={isDark}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Background decorative elements */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-20 right-20 w-96 h-96 bg-green-500/5 rounded-full blur-3xl"></div>
          <div className="absolute bottom-20 left-20 w-80 h-80 bg-emerald-500/5 rounded-full blur-3xl"></div>
        </div>

        {/* Header */}
        <header className="relative flex items-center justify-between px-4 py-3 border-b border-gray-200/50 dark:border-gray-800/50 bg-white/60 dark:bg-gray-900/60 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100/50 dark:hover:bg-gray-800/50 text-gray-500 dark:text-gray-400 transition-colors"
            >
              <Menu className="w-5 h-5" />
            </button>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="hidden lg:flex p-2 rounded-lg hover:bg-gray-100/50 dark:hover:bg-gray-800/50 text-gray-500 dark:text-gray-400 transition-colors"
              title={sidebarOpen ? 'Rufe sidebar' : 'Bude sidebar'}
            >
              <PanelLeft className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-green-600 flex items-center justify-center shadow-sm">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-semibold text-gray-800 dark:text-white text-sm">Dikko AI Noma</span>
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">Shirye</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {!apiAvailable && (
              <span className="text-xs text-amber-500 flex items-center gap-1 bg-amber-50/80 dark:bg-amber-900/20 px-2 py-1 rounded-full backdrop-blur-sm">
                <AlertCircle className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Mock mode</span>
              </span>
            )}
            <button
              onClick={() => window.location.href = '/about'}
              className="p-2 rounded-lg hover:bg-gray-100/50 dark:hover:bg-gray-800/50 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              title="About"
            >
              <Info className="w-4 h-4" />
            </button>
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg hover:bg-gray-100/50 dark:hover:bg-gray-800/50 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              title={isDark ? 'Hasken rana' : 'Yanayin dare'}
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* Chat Messages - with transparent background */}
        <div className="flex-1 overflow-y-auto px-4 py-6 relative">
          <div className="max-w-4xl mx-auto space-y-6">
            {!hasMessages ? (
              <WelcomeScreen onPromptSelect={sendMessage} />
            ) : (
              <>
                {messages.map((msg, index) => (
                  <ChatMessage
                    key={msg.id}
                    message={msg}
                    onCopy={copyMessage}
                    onRegenerate={index === messages.length - 1 && msg.role === 'assistant' ? regenerateLast : () => {}}
                    onLike={handleLike}
                    onDislike={handleDislike}
                  />
                ))}
                
                {isLoading && (
                  <div className="flex items-center gap-3 text-gray-500 dark:text-gray-400">
                    <div className="w-8 h-8 rounded-xl bg-green-600 flex items-center justify-center shadow-sm">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-sm">Dikko AI Noma yana tunani</span>
                      <span className="inline-flex gap-1">
                        <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                        <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                        <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></span>
                      </span>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </div>

        {/* Chat Input - Transparent like ChatGPT/You.com */}
        <div className="relative border-t border-gray-200/50 dark:border-gray-800/50 bg-transparent px-4 py-4">
          <div className="max-w-4xl mx-auto">
            {error && (
              <div className="mb-2 text-sm text-red-500 flex items-center gap-2 bg-red-50/80 dark:bg-red-900/20 backdrop-blur-sm px-3 py-2 rounded-xl">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
            <ChatInput
              onSend={sendMessage}
              onStop={stopGeneration}
              isLoading={isLoading}
              isGenerating={isGenerating}
              disabled={!apiAvailable && !apiService['isMockMode']}
            />
            <div className="text-center text-[10px] text-gray-400 dark:text-gray-500 mt-3">
              Dikko AI Noma • AI don manoman Hausa • © 2026 Yahya Aliyu Dikko
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}