import configparser

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from example import Example

class Agent(Example):
    def __init__(self, config: configparser.ConfigParser):
        destination_prompt = PromptTemplate(
            input_variables=["destination"],
            template="I'm planning a trip to {destination}. Can you suggest some activities to do there?"
        )
        activities_prompt = PromptTemplate(
            input_variables=["activities"],
            template="I only have one day, so can you create an itinerary from our top three activities: {activities}"
        )
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=config["OPENAI"]["API_KEY"])
        self.seq_chain = ({"activities": destination_prompt | llm | StrOutputParser() } |
                          activities_prompt | llm | StrOutputParser())

    def run(self, input: str) -> str:
        return self.seq_chain.invoke({"destination", input})
