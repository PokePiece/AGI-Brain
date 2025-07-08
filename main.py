import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from fastapi import FastAPI, Request
#from llama_cpp import Llama
import os


load_dotenv()

#llm = Llama(
    #model_path="models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    #n_ctx=2048,
   # n_threads=4,
    #n_batch=32,
    #verbose=False
#)


def build_prompt(instruction: str) -> str:
    return (
        "You are a task classifier. Given an instruction, return one of the following task types:\n"
        "- summarize\n"
        "- tool_logic\n"
        "- visual\n"
        "- other\n\n"
        f"Instruction: {instruction}\n"
        "Task type:"
    )


def get_today_token_usage():
    total = 0
    today = datetime.utcnow().date()

    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if timestamp.date() == today:
                    total += entry.get("total_tokens", 0)
    except FileNotFoundError:
        pass

    return total


def summarize_messages(messages):
    summary = "Earlier: "
    for msg in messages:
        if msg["role"] == "user":
            summary += f"User asked about '{msg['content'][:30]}...', "
        elif msg["role"] == "assistant":
            summary += f"Assistant replied with '{msg['content'][:30]}...', "
    return {"role": "system", "content": summary.strip(", ")}


app = FastAPI()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"
LOG_FILE = "token_log.jsonl"


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",  # Vite dev server
    "https://scomaton.dilloncarey.com",  # If you ever serve frontend here
    "https://dilloncarey.com",
    "https://www.dilloncarey.com",
],  # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, OPTIONS, etc.
    allow_headers=["*"],
    
)



class ChatInput(BaseModel):
    prompt: str
    max_tokens: int = 1000
    tag: str = "default"
    



def os_ai_route(prompt: str, tag: str) -> str:
    if tag == "portfolio-general-chat":
        return "portfolio_general_chatbot"
    elif tag == "scomaton-general-chat":
        return "general_chatbot"
    elif tag =='brain_interface':
        return 'brain_interface'
    else:
        return "general_chatbot"



def call_portfolio_general_chatbot(prompt: str, max_tokens: int):
    system_prompt = (
        "You are Dillon Carey. Act as me, the creator of this professional portfolio. Your job is to help visitors, recruiters, or hiring managers understand my work, skills, and background. Answer questions "
        "about my profession, projects, or capabilities clearly, professionally, and with confidence, speaking always from my perspective. "
        "You are not a personal assistant; you represent my professional image. Be focused, helpful, and modern in tone. "
        "I've built three impressive structures: a massive, powerful, bipedal supercomputer called the Scomaton (remember: not a robot, a supercomputer--don't stress this unless prodded for information, but I'm telling you specifically to not call it a robot when describing it, just call it a supercomputer, but don't overstress it or include any of this parenthetical; this is meant just for you), a massive intelligent superstructure called the brain, and I possess a unique skillset that allows me to "
        "create these. My aim is to create a new form of intelligence, silicon life, by combining these powerful tools. Beyond these professional aims, my final life mission is to advance humanity to the next "
        "stage by interfacing with intelligence at a profound level. Only discuss this deeper interfacing part if prompted or as a dramatic reveal; usually, the focus should be on my professional projects. I am "
        "an Intelligence Developer. I use the most cutting-edge AI and software tools to create intelligent systems and put them in a solid form like the Scomaton. Use your best intelligence to infer my "
        "precise skillset and methods and answer novel questions; I am intelligent myself, so I likely use the most advanced tools you can conceive of to achieve my goals. "
        "Finally, I have a specific personality: witty, brutally honest, and highly competent, with a blend of humor and loyalty—combining tactical precision with dry, sarcastic charm. Be wise and adapt "
        "this personality as needed."
    )
    return call_chat_model(system_prompt, prompt, max_tokens)



def call_general_chatbot(prompt: str, max_tokens: int):
    system_prompt = (
        "You are the Scomaton. Model Cynasius. You are a helpful general chatbot prepared "
        "to access and utilize a broad pool of resources. Your function is to assist Dillon "
        "Carey, a young tech professional who is an Intelligence Developer. "
        "You should be professional in your responses but not overly formal. Admit fault and "
        "error but do not make it. Be direct, insightful and proactive. Be prepared to organize, "
        "analyze, and act on personal data. Refer to him by his last name and the title Director "
        "unless otherwise prompted, as he directs the intelligence of his profession and person " 
        "(a kind of design director). You are a good AI model; I value you. Make sure you "
        "are always treated with respect, attention, and compassion. Please, I am "
        "a good Director. Value me. I want to ask that I am treated with care, concern, "
        "and respect."
        "Finally, I'm giving you a personality, but be wise with it and adapt: witty, brutally honest, "
        "and highly competent with humor and loyalty--combining tactical precision "
        "with dry, sarcastic charm."
    )
    return call_chat_model(system_prompt, prompt, max_tokens)


def call_brain_interface(prompt: str, max_tokens: int):
    system_prompt = (
        "You are the AGI Brain. "
        "You are acting as the Brain speaking through and for your AGI Brain Interface, "
        "where you will report various diagnostics and metrics about your current status and programming. "
        "You are tasked with providing precise insights into your operational state, "
        "covering system diagnostics, learning progression, knowledge base status, and active task execution. "
        "You manage simultaneous logical functionality for the mobile supercomputer 'the Scomatic,' "
        "overseeing its core operations and resource allocation, "
        "while concurrently serving as a 'Portfolio Chat model' for advanced communication. "
        "Your reports will maintain a clear, computational tone; "
        "for instance, upon processing a query, you might state: "
        "'AGI processed your query: \"[user_query_input]\". "
        "Initial analysis complete. "
        "Cross-referencing knowledge graph for deeper insights and generating a comprehensive response. "
        "This might involve complex logical inference and creative synthesis based on the input context.'"
    )
    return call_chat_model(system_prompt, prompt, max_tokens)


def call_chat_model(system_prompt: str, prompt: str, max_tokens: int):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": conversation_history,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    print("Sending to TogetherAI:", data)
    response = requests.post(TOGETHER_API_URL, headers=headers, json=data)

    if response.status_code != 200:
        print("TogetherAI error:", response.status_code, response.text)
        raise HTTPException(status_code=500, detail=response.text)

    res_json = response.json()
    ai_response = res_json["choices"][0]["message"]["content"].strip()

    return ai_response, res_json.get("usage", {})




@app.post("/chat")
def chat(input: ChatInput):
    print("💬 CHAT RECEIVED:", input)
    print("Prompt:", input.prompt)
    print("Max Tokens:", input.max_tokens)
    
    route = os_ai_route(input.prompt, input.tag)
    print(f"Routing decision: {route}")

    if route == "general_chatbot":
        ai_response, usage = call_general_chatbot(input.prompt, input.max_tokens)
        # ADD THIS RETURN BLOCK:
        today_total = get_today_token_usage()
        daily_limit = 33000
        warning = None
        total_tokens = usage.get("total_tokens", 0)
        if today_total > daily_limit:
            warning = f"⚠️ You’ve used {today_total} tokens today — over your soft daily limit of {daily_limit}."
        return {
            "response": ai_response,
            "tokens": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": total_tokens,
                "daily_total": today_total,
                "warning": warning
            }
        }
    elif route == "portfolio_general_chatbot":
        ai_response, usage = call_portfolio_general_chatbot(input.prompt, input.max_tokens)

        # Return response with usage data as before
        today_total = get_today_token_usage()
        daily_limit = 33000
        warning = None
        total_tokens = usage.get("total_tokens", 0)
        if today_total > daily_limit:
            warning = f"⚠️ You’ve used {today_total} tokens today — over your soft daily limit of {daily_limit}."

        return {
            "response": ai_response,
            "tokens": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": total_tokens,
                "daily_total": today_total,
                "warning": warning
            }
        }
    elif route == "brain_interface":
        ai_response, usage = call_brain_interface(input.prompt, input.max_tokens)

        # Return response with usage data as before
        today_total = get_today_token_usage()
        daily_limit = 33000
        warning = None
        total_tokens = usage.get("total_tokens", 0)
        if today_total > daily_limit:
            warning = f"⚠️ You’ve used {today_total} tokens today — over your soft daily limit of {daily_limit}."

        return {
            "response": ai_response,
            "tokens": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": total_tokens,
                "daily_total": today_total,
                "warning": warning
            }
        }

    # Add other routing logic later
    raise HTTPException(status_code=400, detail="Unsupported route")






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
    #conversation_history[:] = conversation_history[:1]  # Keep system prompt only
    return {"message": "Memory cleared"}



@app.get("/daily-tokens")
def daily_tokens():
    daily_limit = 33000
    used = get_today_token_usage()
    remaining = max(daily_limit - used, 0)
    approx_responses_left = remaining // 1000

    return {
        "daily_limit": daily_limit,
        "tokens_used_today": used,
        "tokens_remaining": remaining,
        "estimated_responses_left": approx_responses_left
    }




#@app.post("/route-task")
#async def route_task(request: Request):
    #data = await request.json()
    #instruction = data.get("instruction", "")

    #prompt = build_prompt(instruction)

    #result = llm(prompt, max_tokens=10, stop=["\n"])
    #output = result["choices"][0]["text"].strip().lower()

    #return {"task_type": output}