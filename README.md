# um-agent-coder

`um-agent-coder` is a simple AI coding agent that can be extended to support different LLM providers.

## Project Structure

```
.
├── config
│   └── config.yaml.example
├── requirements.txt
├── src
│   └── um_agent_coder
│       ├── __init__.py
│       ├── __main__.py
│       ├── agent
│       │   ├── __init__.py
│       │   └── agent.py
│       ├── config.py
│       ├── llm
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── providers
│       │       ├── __init__.py
│       │       └── openai.py
│       └── main.py
└── tests
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/UM-GPT/um-agent-coder.git
    cd um-agent-coder
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Copy the sample configuration file:
    ```bash
    cp config/config.yaml.example config/config.yaml
    ```

2.  Open `config/config.yaml` and add your OpenAI API key:
    ```yaml
    llm:
      provider: openai
      openai:
        api_key: "YOUR_OPENAI_API_KEY"
        model: "gpt-3.5-turbo"
    ```

    Alternatively, you can set the `OPENAI_API_KEY` environment variable.

## Usage

To run the agent, use the following command:

```bash
python -m src.um_agent_coder "YOUR_PROMPT"
```

For example:

```bash
python -m src.um_agent_coder "Write a python function that calculates the factorial of a number."
```

## LLM Provider Pricing

This section provides a summary of the pricing for the different LLM providers. Please note that this information may not be up-to-date. Always check the official pricing pages for the latest information.

### OpenAI

*   **GPT-4o (2024-08-06):** $2.50 per 1M input tokens, $1.25 for cached input, and $10.00 per 1M output tokens.
*   **GPT-4o Mini:** $0.15 per 1M input tokens, $0.075 for cached input, and $0.60 per 1M output tokens.
*   **GPT-4.5 Preview:** $75.00 per 1M input tokens and $150.00 per 1M output tokens.
*   **GPT-3.5 Turbo:** $0.50 per 1M input tokens and $1.50 per 1M output tokens.

### Anthropic

**Latest Models**

*   **Claude Opus 4.1:**
    *   Input: $15 / MTok
    *   Output: $75 / MTok
*   **Claude Sonnet 4:**
    *   Input: $3 / MTok
    *   Output: $15 / MTok
*   **Claude Haiku 3.5:**
    *   Input: $0.80 / MTok
    *   Output: $4 / MTok

**Legacy Models**

*   **Claude Opus 4:**
    *   Input: $15 / MTok
    *   Output: $75 / MTok
*   **Claude Opus 3:**
    *   Input: $15 / MTok
    *   Output: $75 / MTok
*   **Claude Sonnet 3.7:**
    *   Input: $3 / MTok
    *   Output: $15 / MTok
*   **Claude Haiku 3:**
    *   Input: $0.25 / MTok
    *   Output: $1.25 / MTok

### Google

**Gemini 2.5 Pro**

*   **Input:**
    *   $1.25 per 1M tokens (prompts <= 200k tokens)
    *   $2.50 per 1M tokens (prompts > 200k tokens)
*   **Output:**
    *   $10.00 per 1M tokens (prompts <= 200k tokens)
    *   $15.00 per 1M tokens (prompts > 200k)

**Gemini 2.5 Flash**

*   **Input:**
    *   $0.30 per 1M tokens (text / image / video)
    *   $1.00 per 1M tokens (audio)
*   **Output:** $2.50 per 1M tokens

**Gemini 2.5 Flash-Lite**

*   **Input:**
    *   $0.10 per 1M tokens (text / image / video)
    *   $0.30 per 1M tokens (audio)
*   **Output:** $0.40 per 1M tokens

**Gemini 1.5 Pro**

*   **Input:**
    *   $1.25 per 1M tokens (prompts <= 128k tokens)
    *   $2.50 per 1M tokens (prompts > 128k tokens)
*   **Output:**
    *   $5.00 per 1M tokens (prompts <= 128k tokens)
    *   $10.00 per 1M tokens (prompts > 128k tokens)

**Gemini 1.5 Flash**

*   **Input:**
    *   $0.075 per 1M tokens (prompts <= 128k tokens)
    *   $0.15 per 1M tokens (prompts > 128k tokens)
*   **Output:**
    *   $0.30 per 1M tokens (prompts <= 128k tokens)
    *   $0.60 per 1M tokens (prompts > 128k tokens)
