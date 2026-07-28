
'use client';

import { ArrowLeft, Bot, Database, Code, Cpu, BookOpen, Sparkles } from 'lucide-react';
import Link from 'next/link';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <Link 
          href="/" 
          className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-white transition-colors mb-6"
        >
          <ArrowLeft className="w-4 h-4" />
          Koma zuwa tattaunawa
        </Link>

        <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 rounded-2xl bg-green-600 flex items-center justify-center">
              <Bot className="w-10 h-10 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Dikko AI Noma</h1>
              <p className="text-gray-500 dark:text-gray-400">AI don manoman Hausa</p>
            </div>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed">
              Dikko AI Noma wani mataimaki ne na AI da aka tsara don taimakawa manoma masu magana da harshen Hausa su sami bayanai da shawarwari kan noma cikin sauƙi.
            </p>

            <hr className="my-6 border-gray-200 dark:border-gray-800" />

            <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Game da Mai Ƙirƙira</h2>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Yahya Aliyu Dikko</strong> - Mai ƙirƙira da haɓaka Dikko AI Noma.
            </p>

            <hr className="my-6 border-gray-200 dark:border-gray-800" />

            <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Fasahar da Aka Yi Amfani Da Ita</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
              <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 flex items-start gap-3">
                <Cpu className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-800 dark:text-white">Custom Hausa Transformer</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Model ɗin AI da aka horar da harshen Hausa</div>
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 flex items-start gap-3">
                <Database className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-800 dark:text-white">RAG (Retrieval-Augmented Generation)</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Samun bayanai daga tushen ilimi</div>
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 flex items-start gap-3">
                <Code className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-800 dark:text-white">PyTorch & SentencePiece</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Framework ɗin horar da model da tokenizer</div>
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 flex items-start gap-3">
                <Sparkles className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-800 dark:text-white">Next.js & FastAPI</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Frontend da backend na zamani</div>
                </div>
              </div>
            </div>

            <hr className="my-6 border-gray-200 dark:border-gray-800" />

            <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Manufa</h2>
            <p className="text-gray-700 dark:text-gray-300">
              Dikko AI Noma an ƙirƙira shi ne domin taimakawa manoma Hausawa su sami ilimin noma mai amfani ta hanyar amfani da fasahar AI, don inganta aikin noma da rayuwar al'umma.
            </p>

            <hr className="my-6 border-gray-200 dark:border-gray-800" />

            <div className="text-center text-sm text-gray-400 dark:text-gray-500">
              © 2026 Dikko AI Noma • Yahya Aliyu Dikko
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}