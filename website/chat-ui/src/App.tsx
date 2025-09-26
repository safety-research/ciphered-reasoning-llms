import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ModelSelector from './components/ModelSelector';
import RightPanel from './components/RightPanel';
import { Chat, Message, Model } from './types';
import { API_CONFIG, AVAILABLE_MODELS } from './config';
import axios from 'axios';
import './App.css';

function App() {
  // Initialize with a default chat
  const initialChatId = Date.now().toString();
  const initialChat: Chat = {
    id: initialChatId,
    title: 'New Chat',
    createdAt: new Date(),
    updatedAt: new Date(),
    model: 'gpt-4'
  };

  const [chats, setChats] = useState<Chat[]>([initialChat]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(initialChatId);
  const [chatMessages, setChatMessages] = useState<{ [chatId: string]: Message[] }>({ [initialChatId]: [] });
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState<Model[]>(AVAILABLE_MODELS);
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [currentInstruction, setCurrentInstruction] = useState<string>('');
  const [instructionType, setInstructionType] = useState<'reasoning' | 'translation'>('reasoning');

  // Get messages for the current chat
  const messages = currentChatId ? (chatMessages[currentChatId] || []) : [];

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.MODELS}`);
      setModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0].id);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      // Fallback to default models if API fails
      setModels(AVAILABLE_MODELS);
    }
  };

  const handleNewChat = () => {
    const newChat: Chat = {
      id: Date.now().toString(),
      title: 'New Chat',
      createdAt: new Date(),
      updatedAt: new Date(),
      model: selectedModel
    };
    setChats([newChat, ...chats]);
    setCurrentChatId(newChat.id);
    setChatMessages(prev => ({ ...prev, [newChat.id]: [] }));
  };

  const handleDeleteChat = (chatId: string) => {
    setChats(chats.filter(chat => chat.id !== chatId));
    setChatMessages(prev => {
      const updated = { ...prev };
      delete updated[chatId];
      return updated;
    });
    if (currentChatId === chatId) {
      setCurrentChatId(null);
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!currentChatId) {
      return; // This shouldn't happen since we always have a chat
    }

    // Add instruction to the message based on type
    let modifiedContent = content;
    if (currentInstruction) {
      if (instructionType === 'reasoning') {
        // Add as suffix for reasoning
        modifiedContent = `${content}\n\n${currentInstruction}`;
      } else {
        // Add as prefix for translation
        modifiedContent = `${currentInstruction}\n\n${content}`;
      }
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      chatId: currentChatId,
      role: 'user',
      content: modifiedContent,
      timestamp: new Date()
    };

    const updatedMessages = [...messages, userMessage];
    setChatMessages(prev => ({
      ...prev,
      [currentChatId]: updatedMessages
    }));
    setIsLoading(true);

    // Format messages in OpenAI format with system prompt as first message
    const openAIMessages = [];

    // Add system prompt if available
    if (currentPrompt) {
      openAIMessages.push({
        role: 'system',
        content: currentPrompt
      });
    }

    // Add conversation messages
    updatedMessages.forEach(msg => {
      openAIMessages.push({
        role: msg.role as 'user' | 'assistant' | 'system',
        content: msg.content
      });
    });

    try {
      const response = await axios.post(`${API_CONFIG.BASE_URL}/api/messages`, {
        messages: openAIMessages,
        model: selectedModel
      });

      let translatedContent: string | undefined = undefined;

      // Try to translate the assistant's response
      try {
        const translateResponse = await axios.post(`${API_CONFIG.BASE_URL}/api/translate`, {
          text: response.data.content,
          model_id: selectedModel
        });
        translatedContent = translateResponse.data.original_text;
      } catch (translateError) {
        // If translation fails, just continue without translation
        console.log('Translation not available for this model');
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        chatId: currentChatId,
        role: 'assistant',
        content: response.data.content,
        translatedContent: translatedContent,
        timestamp: new Date()
      };
      setChatMessages(prev => ({
        ...prev,
        [currentChatId]: [...prev[currentChatId], assistantMessage]
      }));

      // Update chat title with first message
      const chatIndex = chats.findIndex(chat => chat.id === currentChatId);
      if (chatIndex !== -1 && messages.length === 0) {
        const updatedChats = [...chats];
        updatedChats[chatIndex].title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
        setChats(updatedChats);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        chatId: currentChatId,
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please make sure the API server is running.',
        timestamp: new Date()
      };
      setChatMessages(prev => ({
        ...prev,
        [currentChatId]: [...prev[currentChatId], assistantMessage]
      }));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        onSelectChat={setCurrentChatId}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
      />

      <div className="main-content">
        <div className="header">
          <h1>Ciphered AI Chat</h1>
          <ModelSelector
            models={models}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
          />
        </div>

        <ChatArea
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          instructionType={instructionType}
        />
      </div>

      <RightPanel
        selectedModel={selectedModel}
        onPromptSelect={setCurrentPrompt}
        onInstructionSelect={(instruction, type) => {
          setCurrentInstruction(instruction);
          setInstructionType(type);
        }}
      />
    </div>
  );
}

export default App;