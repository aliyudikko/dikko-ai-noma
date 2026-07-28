import { ChatRequest, ChatResponse, ApiError } from '@/types/chat';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Mock responses for when mock mode is explicitly enabled
const MOCK_RESPONSES: Record<string, string> = {
  'default': `Assalamu alaikum! Na gode da tambayarka.

Dangane da tambayarka game da noma, ga wasu shawarwari masu amfani:

1. Shirya ƙasa da kyau kafin dasa - wannan yana taimakawa saiwoyin shuka su sami iska da ruwa.
2. Zabi iri masu inganci daga tushe amintacce.
3. Kula da ciyawa akai-akai don su hana shukar ka samun abinci mai gina jiki.
4. Sha ruwa a lokutan da suka dace, musamman lokacin rani.
5. Yi amfani da takin da ya dace da buƙatar shukar ka.

Idan kana buƙatar ƙarin bayani game da wani fanni na noma, ka tambaya. Zan yi ƙoƙarin taimaka maka.`,
  'noman masara': `Assalamu alaikum! Noman masara abu ne mai muhimmanci a arewacin Najeriya.

Ga muhimman abubuwa game da noman masara:

1. Lokacin dasa: Farkon damina (Mayu-Yuni) shine mafi kyau.
2. Shirya ƙasa: Noma ƙasa da zurfin inci 6-8, sannan ka sanya taki.
3. Dasa iri: Ka sanya iri a rami mai zurfin inci 2-3, nisan inci 8-12 tsakanin kowanne.
4. Kula da ciyawa: Ka cire ciyawa sau 2-3 a lokacin girma.
5. Taki: Ka sanya NPK 15-15-15 a lokacin dasa, sannan ka ƙara Urea bayan makonni 4-6.
6. Girbi: Masara tana girma cikin watanni 3-4.

Idan kana buƙatar ƙarin bayani, ka tambaya. Zan taimaka.`
};

function getMockResponse(message: string): string {
  const lowerMsg = message.toLowerCase();
  if (lowerMsg.includes('masara') || lowerMsg.includes('masar')) {
    return MOCK_RESPONSES['noman masara'];
  }
  return MOCK_RESPONSES['default'];
}

export class ApiService {
  private baseUrl: string;
  private isMockMode: boolean;

  constructor() {
    this.baseUrl = API_URL;
    // Only use mock mode if explicitly set to 'mock'
    this.isMockMode = API_URL === 'mock';
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    if (this.isMockMode) {
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 500));
      
      return {
        response: getMockResponse(request.message),
        used_rag: false,
        sources: undefined,
        conversation_id: request.conversation_id || 'mock-conv-1'
      };
    }

    try {
      // Map frontend request format to FastAPI backend schema
      const backendPayload = {
        message: request.message,
        max_new_tokens: 100,
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        do_sample: true
      };

      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(backendPayload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error ${response.status}`);
      }

      const data = await response.json();

      // Map backend response format back to frontend ChatResponse type
      return {
        response: data.response,
        used_rag: false,
        sources: undefined,
        conversation_id: request.conversation_id || 'conv-1'
      };
    } catch (error) {
      throw {
        message: 'Yi haƙuri, an samu matsala wajen haɗawa da sabar. Da fatan za a sake gwadawa.',
        originalError: error
      } as ApiError;
    }
  }

  // Check if backend is available
  async healthCheck(): Promise<boolean> {
    if (this.isMockMode) return true;
    
    try {
      const response = await fetch(`${this.baseUrl}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  // Set mock mode
  setMockMode(enabled: boolean): void {
    this.isMockMode = enabled;
  }
}

export const apiService = new ApiService();