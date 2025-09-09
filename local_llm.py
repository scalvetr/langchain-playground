import configparser

from llm_base import LLMBase

class LocalLLM(LLMBase):
    def __init__(self, config: configparser.ConfigParser):
        from huggingface_hub import login
        login(token=config["HUGGINGFACE"]["API_KEY"], add_to_git_credential=False)
        from langchain_huggingface import HuggingFacePipeline
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-3.2-1B",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100})

    def run(self, question: str) -> str:
        prompt = f"You are an expert assistant.\nQuestion: {question}\nAnswer:"
        result = self.llm.run(question=question)
        return result[0]["generated_text"]

