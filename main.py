import argparse
import configparser

from agent_sqrroot import AgentSqrRoot
from example import Example
from local_llm import LocalLLM
from ollama_llm import OllamaLLM
from openai_llm import OpenAILLM
from rag_character_splitter_openai import RAGCharacterSplitterOpenAI
from rag_token_splitter_openai import RAGTokenSplitterOpenAI
from rag_semantic_splitter_openai import RAGSemanticSplitterOpenAI
from rag_qa_langsmitht_openai import RAGQALangsmithOpenAI
from rag_qa_ragas_openai import RAGQARagasOpenAI
from rag_qa_semantic_splitter_ragas_openai import RAGQASemanticSplitterRagasOpenAI
from rag_vector_db_openai import RAGVectorDBOpenAI

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
        elif example == "rag_character_splitter_openai":
            return RAGCharacterSplitterOpenAI(config)
        elif example == "rag_token_splitter_openai":
            return RAGTokenSplitterOpenAI(config)
        elif example == "rag_semantic_splitter_openai":
            return RAGSemanticSplitterOpenAI(config)
        elif example == "rag_bm25_retrival_openai":
            return RAGSemanticSplitterOpenAI(config)
        elif example == "rag_qa_langsmitht_openai":
            return RAGQALangsmithOpenAI(config)
        elif example == "rag_qa_ragas_openai":
            return RAGQARagasOpenAI(config)
        elif example == "rag_qa_semantic_splitter_ragas_openai":
            return RAGQASemanticSplitterRagasOpenAI(config)
        elif example == "rag_vector_db_openai":
            return RAGVectorDBOpenAI(config)
        else:
            raise ValueError(f"Unknown example: {example}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=[
        "ollama",
        "local",
        "openai",
        "agent_sqrroot",
        "rag_character_splitter_openai",
        "rag_token_splitter_openai",
        "rag_semantic_splitter_openai",
        "rag_bm25_retrival_openai",
        "rag_qa_langsmitht_openai",
        "rag_qa_ragas_openai",
        "rag_qa_semantic_splitter_ragas_openai",
        "rag_vector_db_openai"
    ],
                        required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')
    example = ExampleFactory.get_example(args.example, config)
    response = example.run(args.input)
    print(f"{args.example.capitalize()} example:", response)
