export interface Chat {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  model: string;
}

export interface Message {
  id: string;
  chatId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  translatedContent?: string;
  timestamp: Date;
}

export interface Model {
  id: string;
  name: string;
  provider: string;
}