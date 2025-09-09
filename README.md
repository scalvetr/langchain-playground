# LangChain Playground

## Local setup

Install Ollama and Pull llama3 model

```shell
brew install ollama
ollama serve &
ollama pull llama3
ollama list
ollama run llama3
```

## Build and run

Define a config file named `config.ini`

```editorconfig
[HuggingFace]
API_KEY=xxx

[OPENAI]
API_KEY=xxx
```

Install the dependencies
```shell
pipenv install
```

Run the program
```shell
pipenv run python main.py --llm ollama --question "Explain what AI is?"
pipenv run python main.py --llm local --question "Explain what AI is?"
pipenv run python main.py --llm openai --question "Rome"

```