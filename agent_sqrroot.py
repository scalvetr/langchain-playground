import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering.stuff_prompt import messages
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools

from example import Example

class AgentSqrRoot(Example):
    def __init__(self, config: configparser.ConfigParser):
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        tools = load_tools(["llm-math"], llm=llm)
        self.agent = create_react_agent(llm, tools)


    def run(self, input: str) -> str:
        messages = self.agent.invoke({"messages": [("human", f"What is the square root of {input}?")]})
        return messages[-1]
