import argparse
import configparser

from agent_sqrroot import AgentSqrRoot
from example import Example
from local_llm import LocalLLM
from ollama_llm import OllamaLLM
from openai_llm import OpenAILLM
from rag_with_openai_llm import RAGWithOpenAILLM


class ExampleFactory:
    @staticmethod
    def get_example(example: str, config: configparser.ConfigParser) -> Example:
        if example == "ollama":
            return OllamaLLM(config)
        elif example == "local":
            return LocalLLM(config)
        elif example == "openai":
            return OpenAILLM(config)
        elif example == "agent_sqrroot":
            return AgentSqrRoot(config)
        elif example == "rag_with_openai_llm":
            return RAGWithOpenAILLM(config)
        else:
            raise ValueError(f"Unknown example: {example}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=[
        "ollama",
        "local",
        "openai",
        "agent_sqrroot",
        "rag_with_openai_llm"
    ],
                        required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')
    example = ExampleFactory.get_example(args.example, config)
    response = example.run(args.input)
    print(f"{args.example.capitalize()} example:", response)
