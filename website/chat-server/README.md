# Chat Server - Flask Backend

A Flask server that provides API endpoints for the Chat UI application. Currently uses placeholder responses for demonstration.

## Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

4. **Run the server:**
   ```bash
   python app.py
   ```

The server will start on `http://localhost:8000`

## API Endpoints

All endpoints are documented in the Chat UI's `API.md` file. The server implements:

- `GET /api/models` - Get available AI models
- `GET /api/chats` - Get all chats
- `POST /api/chats/new` - Create new chat
- `DELETE /api/chats/:chatId` - Delete a chat
- `GET /api/chats/:chatId/messages` - Get chat messages
- `POST /api/chats/:chatId/messages` - Send message (OpenAI format)
- `GET /health` - Health check

## Features

- ✅ Full OpenAI-format message handling
- ✅ In-memory chat storage
- ✅ CORS support for frontend
- ✅ Contextual placeholder responses
- ✅ Multiple model support

## Placeholder Responses

The server generates contextual placeholder responses based on keywords in user messages. In production, replace the `generate_placeholder_response()` function with actual LLM API calls.

## Next Steps

To connect to real LLM APIs:

1. Add API keys to `.env`:
   ```
   OPENAI_API_KEY=your-key
   ANTHROPIC_API_KEY=your-key
   ```

2. Install LLM libraries:
   ```bash
   pip install openai anthropic
   ```

3. Replace placeholder logic in `generate_placeholder_response()` with actual API calls