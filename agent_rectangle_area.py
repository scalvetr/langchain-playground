import configparser

from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import tool

from example import Example


class AgentRectangleArea(Example):
    def __init__(self, config: configparser.ConfigParser):
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        tools = [self.rectangle_area]
        # Create a prompt with the required variables
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(llm, tools, prompt)

    @tool
    def rectangle_area(input: str):
        """Calculates the area of a rectangle
        given the lengths of sides 1 and b"""

        sides = input.split(',')
        a = float(sides[0].strip())
        b = float(sides[1].strip())
        return a * b

    def run(self, input: str) -> str:
        messages = self.agent.invoke({"messages": [("human", input)]})
        return messages['messages'][-1].content
