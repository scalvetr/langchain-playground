import configparser

from example import Example

class OllamaLLM(Example):
    def __init__(self, config: configparser.ConfigParser):
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        self.llm = Ollama(model="llama3")
        template = """You are an expert assistant.
Question: {question}
Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        self.chain = LLMChain(prompt=prompt, llm=self.llm)

    def run(self, input: str) -> str:
        return self.chain.run(question=input)
