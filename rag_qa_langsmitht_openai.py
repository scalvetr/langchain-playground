import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langsmith.evaluation import LangChainStringEvaluator

from example import Example


class RAGQALangsmithOpenAI(Example):
    def __init__(self, config: configparser.ConfigParser):
        self.config = config

    def run(self, input: str) -> str:
        query = "What are the main components of RAG architecture?"
        predicted_answer = "Training and encoding"
        ref_answer = "Retrival and generation"

        promt_template = """You are an expert professor specialized in grading students' answers to
            You are grading the following question:{query}
            Here is the real answer:{answer}
            You are grading the following predicted answer:{result}
            Respond with CORRECT and INCORRECT:
            Grade:"""

        prompt = PromptTemplate(input_variables=["query", "answer", "result"], template=promt_template)

        eval_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=self.config["OPENAI"]["API_KEY"])

        qa_evaluator = LangChainStringEvaluator(
            "qa",
            config={
                "llm": eval_llm,
                "prompt": prompt
            }
        )

        score = qa_evaluator.evalutor.evaluate_strings(
            prediction=predicted_answer,
            reference=ref_answer,
            input=query
        )
        print(f"Score: {score}")
        return score
