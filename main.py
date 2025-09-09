import argparse
import configparser
from llm_base import LLMBase
from ollama_llm import OllamaLLM
from local_llm import LocalLLM
from openai_llm import OpenAILLM

class LLMFactory:
    @staticmethod
    def get_llm(llm_type: str, config: configparser.ConfigParser) -> LLMBase:
        if llm_type == "ollama":
            return OllamaLLM(config)
        elif llm_type == "local":
            return LocalLLM(config)
        elif llm_type == "openai":
            return OpenAILLM(config)
        else:
            raise ValueError(f"Unknown llm type: {llm_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["ollama", "local", "openai"], required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')
    llm = LLMFactory.get_llm(args.llm, config)
    response = llm.run(args.prompt)
    print(f"{args.llm.capitalize()} model:", response)
