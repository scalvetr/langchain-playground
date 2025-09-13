import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker

from example import Example


class RAGSemanticSplitterOpenAI(Example):
    def __init__(self, config: configparser.ConfigParser):
        # test BM25 retriever from_texts
        chunks = [
            "Python was created by Guido van Rossum and released in 1991.",
            "Python is popular language for machine learning (ML).",
            "The PyTorch Libreary is popular Python library for AI and ML."
        ]
        retriever = BM25Retriever.from_texts(
            chunks,
            k=3)
        results = retriever.invoke("When was Python created?")
        print("Most Relevant Document")
        print(results[0].page_content)

        # test BM25 retriever from_documents
        loader = UnstructuredHTMLLoader("sample_document.html")
        data = loader.load()

        embeddings = OpenAIEmbeddings(
            api_key=config["OPENAI"]["API_KEY"],
            model="text-embedding-3-small")

        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.8)

        chunks = splitter.split_documents(data)

        retriever = BM25Retriever.from_documents(
            documents=chunks,
            k=5)

        message = f"""
        Answer the following question using the context provided:
        
        Context:
        {retriever}
        
        Question:
        {input}
        
        Answer:
        """

        prompt = ChatPromptTemplate.from_messages([("human", message)])

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        self.rag_chain = (
                {"retriever": retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

    def run(self, input: str) -> str:
        return self.rag_chain.invoke({"input", input})
