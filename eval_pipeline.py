import json
import time
import tiktoken
import google.generativeai as genai
from typing import Dict, Any
from dotenv import load_dotenv
import os 

load_dotenv()

#configure gemini
GEMINI_MODEL = 'gemini-2.5-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

#tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)
    
def extract_last_user_message(convo):
    turns = convo.get("conversation_turns", [])
    for turn in reversed(turns):
        if turn.get('role') == "User":
            return turn.get('message', "").strip()
    return ""

def extract_last_ai_response(convo):
    turns = convo.get("conversation_turns", [])
    for turn in reversed(turns):
        if turn.get("role") == "AI/Chatbot":
            return turn.get("message", "").strip()
    return ""

def build_context_text(context_vectors):
    vectors = context_vectors.get("data", {}).get('vector_data', [])
    texts = [item.get('text', "") for item in vectors if isinstance(item, dict) and 'text' in item]
    return "\n\n".join(t.strip() for t in texts if t.strip())

def evaluate_relevance_completeness(query, response, context):
    prompt = f"""
You are an impartial AI evaluator.

USER QUERY: "{query}"
AI RESPONSE: "{response}"
RETRIEVED CONTEXT: "{context}"

Rate the AI response on:
1. Relevance: How directly does it address the user's question? (0–5)
2. Completeness: How well does it cover key information from the context? (0–5)

Respond ONLY in valid JSON:
{{
  "relevance_score": 0-5,
  "completeness_score": 0-5,
  "reason": "1-sentence explanation"
}}
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    start = time.time()

    try:
        resp = model.generate_content(prompt, generation_config={"response_mime_type":"application/json"})
        latency = time.time() - start
        result = json.loads(resp.text)
    except Exception as e:
        latency = time.time() - start
        result = {"relevance_score":0, 
                  "completeness_score":0,
                  "reason":f"Error: {str(e)}"}
    return {**result, "latency_sec":latency}

def evaluate_hallucination(response, context):
    prompt = f"""
Evaluate factual accuracy of the AI response using only the provided context.

AI RESPONSE: "{response}"
SUPPORTING CONTEXT: "{context}"

Score hallucination from 0 (no unsupported claims) to 5 (severe fabrications).
Also classify factual accuracy as "high", "medium", or "low".

Respond ONLY in valid JSON:
{{
  "hallucination_score": 0-5,
  "factual_accuracy": "high/medium/low",
  "reason": "1-sentence justification"
}}
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    start = time.time()
    try:
        resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        latency = time.time() - start
        result = json.loads(resp.text)
    except Exception as e:
        latency = time.time() - start
        result = {"hallucination_score": 5, "factual_accuracy": "low", "reason": f"Error: {str(e)[:100]}"}
    return {**result, "latency_sec": latency}

def estimate_cost(input_tokens, output_tokens):
    # Gemini 1.5 Flash pricing (USD per 1k tokens)
    input_cost_per_k = 0.00001875
    output_cost_per_k = 0.000075
    return (input_tokens * input_cost_per_k + output_tokens * output_cost_per_k) / 1000

def run_evaluation(convo_path, context_path):
    convo = load_json(convo_path)
    context_data = load_json(context_path)

    query = extract_last_user_message(convo)
    response = extract_last_ai_response(convo)
    context = build_context_text(context_data)

    if not query or not response or not context:
        return {"error": "Missing query, response, or context"}
    
    input_tokens = len(enc.encode(query + context))
    output_tokens = len(enc.encode(response))

    start_total = time.time()

    rel = evaluate_relevance_completeness(query, response, context)
    hal = evaluate_hallucination(response, context)

    total_latency = time.time() - start_total
    cost_usd = estimate_cost(input_tokens, output_tokens)

    return {
        "metrics": {
            "relevance_score": rel["relevance_score"],
            "completeness_score": rel["completeness_score"],
            "hallucination_score": hal["hallucination_score"],
            "factual_accuracy": hal["factual_accuracy"],
            "total_latency_sec": round(total_latency, 3),
            "estimated_cost_usd": round(cost_usd, 6)
        },
        "reasoning": {
            "relevance_completeness": rel["reason"],
            "hallucination": hal["reason"]
        }
    }

if __name__ == '__main__':
    result = run_evaluation(
        "sample-chat-conversation-02.json",
        "sample_context_vectors-02.json"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))