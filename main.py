import argparse
from llm_base import LLMBase
from ollama_llm import OllamaLLM
from local_llm import LocalLLM
from openapi_llm import OpenAPILLM

class LLMFactory:
    @staticmethod
    def get_llm(llm_type: str) -> LLMBase:
        if llm_type == "ollama":
            return OllamaLLM()
        elif llm_type == "local":
            return LocalLLM()
        elif llm_type == "openapi":
            return OpenAPILLM()
        else:
            raise ValueError(f"Unknown llm type: {llm_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["ollama", "local", "openapi"], required=True)
    args = parser.parse_args()

    question = "Explain event-driven architecture in simple terms."
    llm = LLMFactory.get_llm(args.llm)
    response = llm.run(question)
    print(f"{args.llm.capitalize()} model:", response)
