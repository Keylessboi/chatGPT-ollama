# Local AI Assistant Backend

This repository provides a minimal self-hosted AI assistant that exposes an OpenAI-compatible API using [Ollama](https://github.com/jmorganca/ollama), [LangChain](https://python.langchain.com/), and [ChromaDB](https://www.trychroma.com/). The service runs completely locally and includes persistent memory and basic multimodal support.

## Features

- **OpenAI-Compatible** `/v1/chat/completions` endpoint usable by apps like the Enchanted iOS client, web interfaces, or the CLI.
- **Persistent Memory** using LangChain's `VectorStoreRetrieverMemory` backed by ChromaDB. Conversations are stored on disk and recalled for future prompts.
- **Ollama Models** for language generation. You can switch models by providing the model name in each request. Works with the latest models like `llama3` or `phi3` via `ollama pull`.
- **Multimodal** example endpoint `/v1/vision` using the `llava` model to handle image inputs (requires the model to be installed in Ollama).
- **Ollama API Proxy** exposing `/api/generate` and `/api/chat` so tools can
  communicate using the standard Ollama protocol.

## Requirements

- Python 3.11+
- [Ollama](https://github.com/jmorganca/ollama) installed and running locally
- Enough disk space to store ChromaDB data (defaults to `./chroma_db`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `fastapi`, `uvicorn`, `langchain`, and `chromadb` plus
`langchain-community` and the new split packages `langchain-ollama` and
`langchain-chroma` along with `python-multipart` for uploads.
`ollama` is included to proxy the official API.

## Usage

The server listens on **port 8001** by default. Use `--host 0.0.0.0` to allow connections from other devices on your network. Make sure the Ollama service is running (`ollama serve`) and pull the newest models like `llama3` or `phi3` before starting the server.

1. Ensure Ollama is running and your desired models are pulled, e.g.:

```bash
ollama pull mistral
ollama pull llava
```

2. Start the server:

```bash
    python server.py --host 0.0.0.0 --port 8001
```

The server also proxies the real Ollama API. You can send requests that match
`/api/generate` and `/api/chat` exactly as documented in
[Ollama's API docs](https://github.com/ollama/ollama/blob/main/docs/api.md).
Use the `OLLAMA_URL` environment variable if your Ollama instance is running on
another host or port (default `http://localhost:11434`).

3. Send API requests compatible with OpenAI's format. Example `curl`:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral", "messages": [{"role": "user", "content": "Hello"}]}'
```

The server returns a response in the same structure as `openai.ChatCompletion.create()`.
To connect from another machine, replace `localhost` with your server's IP address (e.g. http://192.168.1.10:8001). Ensure port 8001 is open in any firewall.

For vision requests:

```bash
curl -F file=@image.png -F prompt="describe" http://localhost:8001/v1/vision
```

### Quick test script

You can run `test_client.py` to verify the API without using `curl`.
Set the `SERVER_URL` environment variable if the server is on another host:

```bash
# local server
python test_client.py "Hello"

# or specify a remote server
SERVER_URL=http://192.168.1.10:8001 python test_client.py "Hello"
```

### Enchanted iOS setup

If you use the Enchanted mobile client, open the **Settings** screen and enable
**Custom Server**. Set the base URL to `http://<your-server-ip>:8001`. The app
will automatically append `/v1` to this URL when sending requests. Make sure the
server is reachable from your phone (use `--host 0.0.0.0` when starting the
server).

## Persistence

Conversation history is stored in a local ChromaDB directory (`./chroma_db` by default). Delete this folder to reset memory.
Set `CHROMA_DB` to change where memory is stored.

## Notes

- This project is a small example. Expand it to add authentication, file/document support, or more advanced memory and retrieval strategies.
- When running behind a VPN or local network, ensure only trusted devices can reach the server.

