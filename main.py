import argparse
import configparser
from example import Example
from ollama_llm import OllamaLLM
from local_llm import LocalLLM
from openai_llm import OpenAILLM
from agent_sqrroot import AgentSqrRoot

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
        else:
            raise ValueError(f"Unknown example: {example}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=["ollama", "local", "openai", "agent_sqrroot"], required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')
    example = ExampleFactory.get_example(args.example, config)
    response = example.run(args.input)
    print(f"{args.example.capitalize()} example:", response)
