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
            [llm_with_tools.invoke(state["messages"])]}

# Modules for building a Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Initialize Wikipedia API wrapper to fetch top 1 result
api_wrapper = WikipediaAPIWrapper(top_k_results=1)

# Create a Wikipedia query tool using the API wrapper
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

# Bind the wikipedia tool to the language model
llm_with_tools = llm.bind_tools(tools)

# Modules for adding tool conditions and nodes
from langgraph.prebuilt import ToolNode, tools_condition

# Add chatboot node to the graph

graph_builder.add_node("chatbot", chatbot)

# Create a ToolNode to handle tool calls and add it to the graph
tool_node = ToolNode(tools=[wikipedia_tool])
graph_builder.add_node("tools", tool_node)

# Set up a condition to direct from chatbot to tool or to end node
graph_builder.add_conditional_edges("chatbot", tools_condition)

# Connect tools back to the chatbot and add START and END nodes
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# Import the modules for saving memory
from langgraph.checkpoint.memory import MemorySaver

# Modify the graph with memory checkpointing
memory = MemorySaver()

# Compile the graph passing in memory as the checkpoint
graph = graph_builder.compile(checkpointer=memory)

# Import modules for chatbot diagram
from IPython.display import display, Image
# Try generationg displaying the graph diagram
display(Image(graph.get_graph().draw_mermaid_png()))

# Define a function to execute the chatbot based on user input
def stream_memory_responses(user_input: str):
    config = {"configurable": {"thread_id": "single_session_memmory"}}

    # Start streaming events from the graph with the user's input
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            if "message" in value and value["messages"]:
                print("Agent:", value["messages"])

stream_memory_responses("What is the Colosseum?")
stream_memory_responses("Who build it?")

