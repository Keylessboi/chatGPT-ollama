# Local AI Assistant Backend

This repository provides a minimal self-hosted AI assistant that exposes an OpenAI-compatible API using [Ollama](https://github.com/jmorganca/ollama), [LangChain](https://python.langchain.com/), and [ChromaDB](https://www.trychroma.com/). The service runs completely locally and includes persistent memory and basic multimodal support.

## Features

- **OpenAI-Compatible** `/v1/chat/completions` endpoint usable by apps like the Enchanted iOS client, web interfaces, or the CLI.
- **Persistent Memory** using LangChain's `VectorStoreRetrieverMemory` backed by ChromaDB. Conversations are stored on disk and recalled for future prompts.
- **Ollama Models** for language generation. You can switch models by providing the model name in each request.
- **Multimodal** example endpoint `/v1/vision` using the `llava` model to handle image inputs (requires the model to be installed in Ollama).

## Requirements

- Python 3.11+
- [Ollama](https://github.com/jmorganca/ollama) installed and running locally
- Enough disk space to store ChromaDB data (defaults to `./chroma_db`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists common packages including `fastapi`, `uvicorn`, `langchain`, and `chromadb`. plus `langchain-community` and `python-multipart` for uploads.

## Usage

1. Ensure Ollama is running and your desired models are pulled, e.g.:

```bash
ollama pull mistral
ollama pull llava
```

2. Start the server:

```bash
python server.py --host 0.0.0.0 --port 8000
```

3. Send API requests compatible with OpenAI's format. Example `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral", "messages": [{"role": "user", "content": "Hello"}]}'
```

The server returns a response in the same structure as `openai.ChatCompletion.create()`.

For vision requests:

```bash
curl -F file=@image.png -F prompt="describe" http://localhost:8000/v1/vision
```

## Persistence

Conversation history is stored in a local ChromaDB directory (`./chroma_db` by default). Delete this folder to reset memory.

## Notes

- This project is a small example. Expand it to add authentication, file/document support, or more advanced memory and retrieval strategies.
- When running behind a VPN or local network, ensure only trusted devices can reach the server.

