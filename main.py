import os
import uuid
from typing import Dict, List
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq, GroqError  # Groq SDK Import kiya
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Environment variables load karein
load_dotenv()

app = FastAPI(
    title="Groq Chatbot API",
    description="FastAPI chatbot with memory and ultra-fast real-time streaming using Groq Cloud.",
    version="1.0.0"
)

# CORS enabled karein taake frontend se request aa sakein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq Client Initialisation ---
api_key = os.getenv("GROQ_API_KEY")
# model_name default ko groq supportable model par set kiya (grok-2-latest ya llama-3.3-70b-versatile)
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-specdec") 

if not api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

# Thread-safe client instance
client = Groq(api_key=api_key)

# --- Chat Memory System (In-Memory Dictionary) ---
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

# --- Pydantic Schemas ---
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, description="User ka message jo chatbot ko bhejna hai.")
    system_instruction: str = Field(
        "You are a helpful, polite, and witty AI Assistant.", 
        description="Behavioral instructions for the chatbot."
    )

class SessionResponse(BaseModel):
    session_id: str
    message: str

# --- Helper Functions ---
def get_or_create_history(session_id: str) -> List[Dict[str, str]]:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]

# --- Endpoints ---

# 1. Naya Session Create Karne Ka Endpoint
@app.post("/api/v1/chat/session", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session():
    new_session_id = str(uuid.uuid4())
    chat_sessions[new_session_id] = []
    return SessionResponse(
        session_id=new_session_id, 
        message="Naya chat session successfully start ho chuka hai."
    )

# 2. Real-Time Streaming Chat Endpoint (CLEAN STREAMING)
@app.post("/api/v1/chat/stream")
async def chat_stream(
    payload: ChatMessage,
    session_id: str = Query(..., description="Unique session ID for memory context.")
):
    history = get_or_create_history(session_id)
    
    # Agar history khali hai, to pehle system prompt add karein
    if not history:
        history.append({"role": "system", "content": payload.system_instruction})
    
    # User ka message append karein
    history.append({"role": "user", "content": payload.message})

    async def response_generator():
        try:
            # Groq API call in threadpool
            response_stream = await run_in_threadpool(
                client.chat.completions.create,
                model=model_name,
                messages=history,
                temperature=0.7,
                stream=True,
            )
            
            full_response_text = ""
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response_text += content
                    # >>> CHANGE 1: Ab yahan "data: " aur extra "\n\n" ke bagair direct content return hoga.
                    yield content
            
            # Message history update karein
            if full_response_text:
                history.append({"role": "assistant", "content": full_response_text})

        except GroqError as e:
            yield f"[GROQ API ERROR: {str(e)}]"
        except Exception as e:
            yield f"[SYSTEM ERROR: {str(e)}]"

    # >>> CHANGE 2: "text/event-stream" ko standard "text/plain" ya dynamic response me change kiya 
    # taake automatic parsing asaan ho aur stream clean ho.
    return StreamingResponse(response_generator(), media_type="text/plain")


# 3. Chat Memory Clean karne ka Endpoint
@app.delete("/api/v1/chat/session/{session_id}", status_code=status.HTTP_200_OK)
def clear_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"detail": "Session memory cleared successfully."}
    raise HTTPException(status_code=404, detail="Session ID nahi mila.")