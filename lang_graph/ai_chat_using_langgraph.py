from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class DetectCallResponse(BaseModel):
    is_coding_question: bool

class CodingQuestionResponse(BaseModel):
    answer: str

class NonCodingQuestionResponse(BaseModel):
    answer: str

# Initialize OpenAI API
client = wrap_openai(OpenAI())

class State(TypedDict):
    user_message: str
    assistant_message: str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state["user_message"]

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related to coding or not.
    Return the response in specified JSON boolean only.
    """

    # OpenAI Call
    result = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=DetectCallResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
    )

    state["is_coding_question"] = result.choices[0].message.parsed.is_coding_question
    return state

def route_edge(state: State) -> Literal["solve_coding_question", "solve_non_coding_question"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_non_coding_question"

def solve_coding_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to resolve the user query based on coding problem he is facing.
    Return the response in specified JSON format.
    """

    # OpenAI Call (Coding Question gpt-4.1)
    result = client.beta.chat.completions.parse(
        model="gpt-4.1",
        response_format=CodingQuestionResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
    )

    state["assistant_message"] = result.choices[0].message.parsed.answer
    return state

def solve_non_coding_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to chat with the user.
    Return the response in specified JSON format.
    """

    # OpenAI Call (Non-Coding Question
    result = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=NonCodingQuestionResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
    )

    state["assistant_message"] = result.choices[0].message.parsed.answer
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_non_coding_question", solve_non_coding_question)
graph_builder.add_node("route_edge", route_edge)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)

graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_non_coding_question", END)

graph = graph_builder.compile()

def call_graph():
    state = {
        "user_message": "What is the color of sky?",
        "assistant_message": "",
        "is_coding_question": False
    }

    result = graph.invoke(state)
    print("Final Result:", result)

if __name__ == "__main__":
    call_graph()