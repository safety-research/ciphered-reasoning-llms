export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    CHATS: '/api/chats',
    MESSAGES: '/api/messages',
    MODELS: '/api/models',
    CHAT_HISTORY: '/api/chats/:chatId/messages',
    NEW_CHAT: '/api/chats/new',
    DELETE_CHAT: '/api/chats/:chatId',
    SEND_MESSAGE: '/api/chats/:chatId/messages',
  }
};

export const AVAILABLE_MODELS = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'OpenAI' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic' },
  { id: 'llama-2-70b', name: 'Llama 2 70B', provider: 'Meta' },
];