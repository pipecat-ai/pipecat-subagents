# Voice Code Assistant (Local)

Talk to your codebase hands-free. Ask questions about code, project structure, or file contents and get spoken answers based on actual files. The Claude Agent SDK worker navigates the filesystem using Read, Write, Bash, Glob, and Grep tools.

## Architecture

```
CodeAssistant (transport + BusBridge)
  └── Voice Agent (LLM, bridged)
        └── @tool ask_code(question)
              └── Code Worker (Claude Agent SDK)
```

- **CodeAssistant** (`code_assistant.py`): Owns the transport (STT, TTS) and bridges frames to the bus.
- **VoiceAgent** (`voice_agent.py`): Bridged LLM agent that talks to the user. Has an `ask_code` tool that dispatches questions to the code worker using `request_task()`. The agent stays conversational while the worker runs in the background.
- **CodeWorker** (`code_worker.py`): Runs a Claude Agent SDK session with filesystem tools to explore the project and answer questions.

## Setup

```bash
uv sync
uv pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero,examples]"
pip install claude-agent-sdk
```

Copy `env.example` to `.env` and add your API keys:

```bash
cp examples/env.example .env
```

## Running

```bash
# Default: explores the current directory
uv run examples/local/code-assistant/code_assistant.py

# Specify a project path
PROJECT_PATH=/path/to/your/project uv run examples/local/code-assistant/code_assistant.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

To use Daily transport:

```bash
uv run examples/local/code-assistant/code_assistant.py --transport daily
```

## Environment variables

| Variable            | Required by                    |
|---------------------|--------------------------------|
| `ANTHROPIC_API_KEY` | Code Worker (Claude Agent SDK) |
| `OPENAI_API_KEY`    | Voice Agent (LLM)              |
| `DEEPGRAM_API_KEY`  | STT                            |
| `CARTESIA_API_KEY`  | TTS                            |
| `DAILY_API_KEY`     | Only with `--transport daily`  |
| `PROJECT_PATH`      | Optional, defaults to cwd      |

## Example questions

- "What does the main module do?"
- "Find all TODO comments in the project"
- "How is error handling implemented?"
- "What dependencies does this project use?"
- "Explain the test structure"
