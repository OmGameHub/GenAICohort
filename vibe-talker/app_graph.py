import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import SystemMessage

load_dotenv()

@tool()
def run_command(command: str):
    """Takes a command line prompt and executes it on the user's machine"""
    result = os.system(command=command)
    return result


llm = init_chat_model(
    model_provider="openai",
    model="gpt-4.1",
)
llm_with_tools = llm.bind_tools([run_command])
tools = [run_command]

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    # message = llm.invoke(state.get("messages", []))

    SYSTEM_PROMPT = SystemMessage("""
    You are an AI assistant who takes an input from user and bases on available tools you choose the correct tool and execute the commands.

    You can even execute commands and help user with the output of the command.

    Always make sure to keep your generated codes and file in example/ folder.
    """)

    message = llm_with_tools.invoke([SYSTEM_PROMPT] + state.get("messages", []))

    assert len(message.tool_calls) <= 1
    return { "messages": [message] }

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)