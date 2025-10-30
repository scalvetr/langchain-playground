import configparser

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool


@tool
def rectangle_area(input: str):
    """Calculates the area of a rectangle
    given the lengths of sides 1 and b"""

    sides = input.split(',')
    a = float(sides[0].strip())
    b = float(sides[1].strip())
    return a * b


def create_model():
    config = configparser.ConfigParser()
    config.read('config.ini')

    from huggingface_hub import login
    login(token=config["HuggingFace"]["API_KEY"], add_to_git_credential=False)
    from langchain_huggingface import HuggingFacePipeline

    return HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
    )


if __name__ == "__main__":
    # Example usage when running the module directly.
    model = create_model()
    tools = [rectangle_area]
    prompt = hub.pull("hwchase17/react")
    app = create_react_agent(model, tools, prompt)

    query = "Compute the area for a rectangle with length 5 and width 3."
    # The agent will use the registered `rectangle_area` tool when appropriate.
    messages = app.invoke({
        "messages": [("human", query)]
    })
    print({
        "user_input": query,
        "agent_output": messages["messages"][-1].content
    })
    from langchain_core.messages import AIMessage, HumanMessage

    message_history = messages["messages"]
    new_query = "What about one with sides 4 and 3?"

    # Invoke the app with the full message history
    messages = app.invoke({
        "messages": message_history + [("human", new_query)]
    })

    filtered_messages = [msg for msg in messages["messages"] if
                         isinstance(msg, (HumanMessage, AIMessage) and msg.content.strip())]

    format({
        "user_input": new_query,
        "agent_output": [f"{msg.__class__.__name__}: {msg.content}: {msg.content}" for msg in filtered_messages]
    })
    # agent_executor = AgentExecutor(agent=app, tools=tools, handle_parsing_errors=True)

    # agent_executor.invoke({"input": "hi"})

    # Use with chat history
    # from langchain_core.messages import AIMessage, HumanMessage

    # agent_executor.invoke(
    #    {
    #        "input": "what's my name?",
    # Notice that chat_history is a string
    # since this prompt is aimed at LLMs, not chat models
    #        "chat_history": "Human: My name is Bob\nAI: Hello Bob!",
    #    }
    # )
