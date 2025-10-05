import configparser

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter

from example import Example


class RAGVectorDBOpenAI(Example):
    def __init__(self, config: configparser.ConfigParser):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=config["OPENAI"]["API_KEY"])
        llm_transformer = LLMGraphTransformer(llm=llm)

        raw_documents = WikipediaLoader(query="large language model").load()
        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
        documents = text_splitter.split_documents(raw_documents[:3])

        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        print(graph_documents)

        print(documents[0])

        graph = Neo4jGraph(url=config["NEO4J"]["URL"], username=config["NEO4J"]["USERNAME"],
                           password=config["NEO4J"]["PASSWORD"])

        graph.add_graph_documents(
            graph_documents,
            include_source=True,
            baseEntityLabel=True
        )
        graph.refresh_schema()

        print(graph.get_schema)
        results = graph.query("""
        MATCH (gpt4:Model {id: "Gpt-4"})-[:DEVELOPED_BY]->(org:Organization)
        RETURN org""")
        print(results)

        examples = [
            {
                "question": "How many notable large laguage models are mentioned in the article?",
                "query": "MATCH (m:Concept {id: 'Large Language Model'}) RETURN COUNT(DISTINCT m)"
            },
            {
                "question": "Which companies or organizations have developed the large language model mentioned?",
                "query": "MATCH (Organizations)-[:DEVELOPED_BY]->(m:Concept {id: 'Large Language Model'}) RETURN COUNT(DISTINCT o.id)"
            },
            {
                "question": "What is the largest model size mentioned in the article, in terms of number of parameters?",
                "query": "MATCH (m:Concept {id: 'Large Language Model'}) RETURN MAX(m.parameters) AS largest_model"
            },
        ]
        example_prompt = PromptTemplate.from_template("User input: {question}\nCypher query: {query}")
        cypher_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="""You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
            
            Here is the schema information
            {schema}.
            
            Below are a number of examples of questions and their corresponding Cypher queries.
            """,
            suffix="User input: {question}\nCypher query: ",
            input_variables=["question"]
        )
        self.chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=cypher_prompt,
            verbose=True,
            exclude_types=["Concept"],
            validate_cypher=True)

    def run(self, input: str) -> str:
        result = self.chain.invoke({"query": input})
        return result['result']
