import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langsmith.evaluation import LangChainStringEvaluator
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import context_precision, faithfulness

from example import Example


class RAGQARagasOpenAI(Example):
    def __init__(self, config: configparser.ConfigParser):
        self.config = config

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
            "question": "How does the RAG model improve question answering with LLms?",
            "answer": "The RAG model improves question answering by combining the retrieval of documents...",
            "contexts": [
                "The RAG model integrates document retrieval with LLMs by first retrieving relevant passages...",
                "By incorporating retrieval mechanisms, RAG leverages external knowledge sources, allowing the.."
            ]
        })
        print(eval_result)

        # Context precision
        context_precision_chain = EvaluatorChain(
            metric=context_precision,
            llm=llm,
            embeddings=embeddings
        )
        eval_result = context_precision_chain({
            "question": "How does the RAG model improve question answering with LLms?",
            "context_truth": "The RAG model improves question answering by combining the retrieval of documents...",
            "contexts": [
                "The RAG model integrates document retrieval with LLMs by first retrieving relevant passages...",
                "By incorporating retrieval mechanisms, RAG leverages external knowledge sources, allowing the.."
            ]

        })
        print(f"Context Precision: {eval_result['context_precision']}")
        return eval_result['context_precision']
