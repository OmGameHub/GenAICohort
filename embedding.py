from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "Eiffel Tower is in Paris and is a famous landmark. It is 324 meeter tall."
response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)
print("Vector Embedding:", response.data[0].embedding)