#backend/ollama_client.py -> Local LLM client

import os
import requests
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:latest")

def generate(messages, max_tokens: int = 512) -> str:
    
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": messages[0]["content"],
        "options": {"num_predict": max_tokens}
    }

    try:
        with requests.post(url, json=payload, stream=True) as resp:
            resp.raise_for_status()
            full_reply = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        full_reply.append(data["message"]["content"])
                    if data.get("done"):
                        break
                except Exception:
                    continue
            return "".join(full_reply)
    except Exception as e:
        return f"[Error contacting Ollama: {e}]"


if __name__ == "__main__":
    out = generate([{ "role": "user", "content": "Say hello in JSON" }])
    print(out)
