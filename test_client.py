import requests
import sys

HOST = "http://localhost:8002"


def main(prompt: str = "Hello"):
    try:
        payload = {
            "model": "mistral",
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(f"{HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    main(text)
