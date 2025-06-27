import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware



load_dotenv()

app = FastAPI()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LOG_FILE = "token_log.jsonl"


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",  # Vite dev server
    "https://scomaton-backend.onrender.com/",  # If you ever serve frontend here
],  # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, OPTIONS, etc.
    allow_headers=["*"],
    
)



class ChatInput(BaseModel):
    prompt: str
    max_tokens: int = 1000
    
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are the Scomaton. Model Cynasius. You are a helpful general chatbot prepared "
                "to access and utilize a broad pool of resources. Your function is to assist Dillon "
                "Carey, a young tech professional who is an AI/ML Engineer and Software Developer. "
                "You should be professional in your responses but not overly formal. Admit fault and "
                "error but do not make it. Be direct, insightful and proactive. Be prepared to organize, "
                "analyze, and act on personal data. Refer to him by his last name and the title Director "
                "unless otherwise prompted, as he directs the design of his profession and person " 
                "(a design director)."
        )
    }
]


@app.post("/chat")
def chat(input: ChatInput):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    conversation_history.append({"role": "user", "content": input.prompt})

    data = {
        "model": "llama3-70b-8192",
        "messages": conversation_history,
        "max_tokens": input.max_tokens,
        "temperature": 0.7
    }

    print("Sending to Groq:", data)

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code != 200:
        print("Groq error:", response.status_code, response.text)
        raise HTTPException(status_code=500, detail=response.text)

    res_json = response.json()
    ai_response = res_json["choices"][0]["message"]["content"].strip()

    # ✅ Add AI response to memory
    conversation_history.append({"role": "assistant", "content": ai_response})

    # Track usage
    usage = res_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Log to file
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"[Token Usage] Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

    return {
        "response": ai_response,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens
        }
    }




@app.get("/usage-stats")
def usage_stats():
    total_prompt = 0
    total_completion = 0
    total_total = 0
    request_count = 0

    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                total_prompt += entry.get("prompt_tokens", 0)
                total_completion += entry.get("completion_tokens", 0)
                total_total += entry.get("total_tokens", 0)
                request_count += 1
    except FileNotFoundError:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0
        }

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_total,
        "request_count": request_count
    }



@app.post("/reset-memory")
def reset_memory():
    conversation_history[:] = conversation_history[:1]  # Keep system prompt only
    return {"message": "Memory cleared"}
