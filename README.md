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
