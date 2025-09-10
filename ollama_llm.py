import configparser

from example import Example

class OllamaLLM(Example):
    def __init__(self, config: configparser.ConfigParser):
        from langchain_ollama.llms import OllamaLLM
        from langchain.prompts import PromptTemplate
        self.llm = OllamaLLM(model="llama3")
        template = """You are an expert assistant.
Question: {question}
Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        self.chain = prompt | self.llm

    def run(self, input: str) -> str:
        return self.chain.invoke(input=input)
