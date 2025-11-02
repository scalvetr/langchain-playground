import configparser

# Modules for structuring text
from typing import Annotated
from typing_extensions import TypedDict

# LangGraph modules for defining graphs
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Module for setting up OpenAI
from langchain_openai import ChatOpenAI



config = configparser.ConfigParser()
config.read('config.ini')

# Define the llm
llm = ChatOpenAI(model="gpt-40-mini", api_key=config["OPENAI"]["API_KEY"])

# Define the State
class State(TypedDict):

    # Define messages with metadata
    messages: Annotated[list, add_messages]

# Initialize StateGraph
graph_builder = StateGraph(State)

# define chatbot function to respond with the model
def chatbot(state: State):
    return {"messages":
            [llm.invoke(state["messages"])]}

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the start and end of the conversation flow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph to prepare for execution
graph = graph_builder.compile()

# Define a function to execute the chatbot based on user input
def stream_graph_updates(user_input: str):
    # Start streaming events from the graph with the user's input
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Agent:", value["messages"])
