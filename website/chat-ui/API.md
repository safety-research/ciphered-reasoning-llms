# Chat UI API Documentation

This document describes the simplified API endpoints for the Chat UI application. The backend is stateless and does not persist any data between page refreshes.

## Base Configuration

Set the API base URL using the environment variable:
```
REACT_APP_API_URL=http://localhost:8000
```

## API Endpoints

### 1. Get Available Models
**GET** `/api/models`

Retrieves a list of available AI models.

**Response:**
```json
[
  {
    "id": "gpt-4",
    "name": "GPT-4",
    "provider": "OpenAI"
  },
  {
    "id": "gpt-3.5-turbo",
    "name": "GPT-3.5 Turbo",
    "provider": "OpenAI"
  },
  {
    "id": "claude-3-opus",
    "name": "Claude 3 Opus",
    "provider": "Anthropic"
  }
]
```

---

### 2. Send Messages
**POST** `/api/messages`

Sends the full conversation history in OpenAI format and receives an AI response.

**Request Body (OpenAI Format):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, can you help me with React?"
    },
    {
      "role": "assistant",
      "content": "Of course! I'd be happy to help you with React."
    },
    {
      "role": "user",
      "content": "What are React hooks?"
    }
  ],
  "model": "gpt-4"
}
```

**Response:**
```json
{
  "content": "React hooks are functions that allow you to use state and other React features...",
  "role": "assistant",
  "model": "gpt-4"
}
```

**Note:**
- The `messages` array contains the complete conversation history in OpenAI's chat completion format
- Each message has `role` ("user", "assistant", or "system") and `content` (text)
- The server is stateless - no conversation history is stored

---

### 3. Health Check
**GET** `/health`

Health check endpoint to verify the server is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00.000000"
}
```

---

## Important Notes

1. **No Persistence**: The application does not persist any data. All chats and messages are lost on page refresh.
2. **Stateless**: The backend is completely stateless - it only processes the messages sent in each request.
3. **Session-Only**: All data exists only in the browser's memory during the current session.

## Implementation Guide

To connect to real LLM APIs, replace the `generate_placeholder_response()` function in the backend with actual API calls:

```python
# Example with OpenAI
import openai

def get_real_response(messages, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
```

## CORS Configuration

The backend allows CORS requests from:
- `http://localhost:3000`
- `http://localhost:3001`