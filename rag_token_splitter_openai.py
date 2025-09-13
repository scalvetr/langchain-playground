import configparser
import os

import tiktoken
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

from example import Example


class RAGTokenSplitterOpenAI(Example):
    def __init__(self, config: configparser.ConfigParser):
        loader = UnstructuredHTMLLoader("sample_document.html")
        data = loader.load()

        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        splitter = TokenTextSplitter(
            encoding_name=encoding.name,
            chunk_size=10,
            chunk_overlap=2)

        docs = splitter.split_documents(data)

        embedding_function = OpenAIEmbeddings(
            api_key=config["OPENAI"]["API_KEY"],
            model="text-embedding-3-small")

        vectorstore = Chroma.from_documents(
            docs,
            embedding=embedding_function,
            persist_directory=os.getcwd()
        )
        # configure the vector store as a retriever
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            seach_kwargs={"k", 2}
        )

        message = f"""
        Answer the following question using the context provided:
        
        Context:
        {retriever}
        
        Question:
        {input}
        
        Answer:
        """

        self.prompt_template = ChatPromptTemplate.from_messages([("human", message)])

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        self.rag_chain = (
                {"retriever": self.retriever, "input": RunnablePassthrough()}
                | self.prompt_template
                | llm
                | StrOutputParser()
        )

    def run(self, input: str) -> str:
        return prompt_template.invoke({"input", input})
