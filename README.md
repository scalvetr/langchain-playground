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
pipenv run python main.py --example ollama --input "Explain what AI is?"
pipenv run python main.py --example local --input "Explain what AI is?"
pipenv run python main.py --example openai --input "Rome"
pipenv run python main.py --example agent_sqrroot --input "101"
pipenv run python main.py --example rag_character_splitter_openai --input "Explain what AI is?"
pipenv run python main.py --example rag_semantic_splitter_openai --input "Explain what AI is?"
pipenv run python main.py --example rag_token_splitter_openai --input "Explain what AI is?"

```