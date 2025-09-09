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

Install the dependencies

```shell
pipenv install
```
Run the program
```shell
pipenv run python main.py

```