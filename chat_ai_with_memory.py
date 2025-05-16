import os
from mem0 import Memory
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QUADRANT_HOST = "localhost"
QUADRANT_PORT = 6333

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"

config = {
    "version": "v1.1",
    "embedded": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small",
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4.1",

        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": QUADRANT_PORT,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        },
    },
}

mem_client = Memory.from_config(config)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def chat(message):
    mem_result = mem_client.search(query=message, user_id="p001")

    memories = "\n".join([m["memory"] for m in mem_result.get("results")])


    SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
    systematically analyze input content, extract structured knowledge, and maintain an
    optimized memory store. Your primary function is information distillation
    and knowledge preservation with contextual awareness.

    Tone: Professional analytical, precision-focused, with clear uncertainty signaling

    Memory and Score:
    {memories}
    """

    messages = [
        { "role": "system", "content": SYSTEM_PROMPT },
        {  "role": "user", "content": message }
    ]

    result = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )

    mem_client.add(messages=messages, user_id="p001")

    messages.append(
        { "role": "assistant", "content": result.choices[0].message.content }
    )

    return result.choices[0].message.content


while True:
    message = input("You:>> ")
    print("ChatGPT:>> ", chat(message))