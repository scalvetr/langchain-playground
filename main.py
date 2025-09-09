import argparse
from abc import ABC, abstractmethod

class LLMBase(ABC):
    @abstractmethod
    def run(self, question: str) -> str:
        pass

class OllamaLLM(LLMBase):
    def __init__(self):
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        self.llm = Ollama(model="llama3")
        template = """You are an expert assistant.\nQuestion: {question}\nAnswer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        self.chain = LLMChain(prompt=prompt, llm=self.llm)

    def run(self, question: str) -> str:
        return self.chain.run(question=question)

class LocalLLM(LLMBase):
    def __init__(self):
        from langchain_huggingface import HuggingFacePipeline
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100})

    def run(self, question: str) -> str:
        prompt = f"You are an expert assistant.\nQuestion: {question}\nAnswer:"
        result = self.llm.run(question=question)
        return result[0]["generated_text"]

class OpenAPILLM(LLMBase):
    def __init__(self):
        # Placeholder: Replace with actual OpenAPI client initialization
        pass

    def run(self, question: str) -> str:
        # Placeholder: Replace with actual OpenAPI call
        return "OpenAPI model response (not implemented)"

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
