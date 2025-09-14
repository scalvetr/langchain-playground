import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langsmith import LangChainStringEvaluator
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import context_precision, faithfulness

from example import Example


class RAGQASemanticSplitterRagasOpenAI(Example):
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        loader = UnstructuredHTMLLoader("sample_document.html")
        data = loader.load()

        embeddings = OpenAIEmbeddings(
            api_key=config["OPENAI"]["API_KEY"],
            model="text-embedding-3-small")

        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.8)

        docs = splitter.split_documents(data)

        vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
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

        prompt_template = ChatPromptTemplate.from_messages([("human", message)])

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        self.rag_chain = (
                {"retriever": self.retriever, "input": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
        )

    def run(self, input: str) -> str:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=self.config["OPENAI"]["API_KEY"])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.config["OPENAI"]["API_KEY"])
        # Faithfulness
        faithfulness_chain = EvaluatorChain(
            metric=faithfulness,
            llm=llm,
            embeddings=embeddings
        )
        eval_result = faithfulness_chain({
            "question": input,
            "answer": self.rag_chain.invoke(input),
            "contexts": self.retriever.invoke(input)
        })
        print(eval_result)
        return eval_result['faithfulness']
