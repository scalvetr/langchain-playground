from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLaMA 3 model running on Ollama
llm = Ollama(model="llama3")

# Define a simple prompt template
template = """You are an expert assistant.
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain
chain = LLMChain(prompt=prompt, llm=llm)

if __name__ == "__main__":
    question = "Explain event-driven architecture in simple terms."
    response = chain.run(question=question)
    print("LLaMA 3 (via LangChain):", response)
