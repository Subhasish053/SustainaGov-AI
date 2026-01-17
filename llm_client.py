import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"


def call_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 2048,
            "num_predict": 512
        }
    }

    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300  # ⬅️ 10 minutes (safe for first run)
    )

    if response.status_code != 200:
        raise Exception(
            f"Ollama Error {response.status_code}: {response.text}"
        )

    return response.json()["response"].strip()

