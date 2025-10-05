import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub

from example import Example

class AgentSqrRoot(Example):
    def __init__(self, config: configparser.ConfigParser):
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        tools = load_tools(["llm-math"], llm=llm)
        # Create a prompt with the required variables
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(llm, tools, prompt)


    def run(self, input: str) -> str:
        messages = self.agent.invoke({"messages": [("human", f"What is the square root of {input}?")]})
        return messages['messages'][-1].content
