from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from openai import OpenAI

from langchain_qdrant import QdrantVectorStore

load_dotenv()


file_path = "https://www.tutorialspoint.com/javascript/javascript_tutorial.pdf"


loader = PyPDFLoader(file_path=file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)

print(f"Number of documents: {len(docs)}")
print(f"Number of split documents: {len(split_docs)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the vector store
vector_store = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embeddings,
    collection_name="javascript_tutorial"
)

vector_store.add_documents(documents=split_docs)
print("Documents added to vector store")


retriever = QdrantVectorStore.from_existing_collection(
    collection_name="javascript_tutorial",
    embedding=embeddings,
    url="http://localhost:6333",
)

search_query = "What is the main function in JavaScript?"
relevant_chunks = retriever.similarity_search(
    query=search_query,
)

messages = []

SYSTEM_PROMPT = """
    You are an AI Assistant who is specialized in JavaScript.
"""
messages.append({ "role": "system", "content": SYSTEM_PROMPT })

client = OpenAI()

while True:
    user_query = input('> ')
    messages.append({ "role": "user", "content": user_query })

    # Perform similarity search to find relevant context
    relevant_chunks = retriever.similarity_search(
        query=user_query,
        k=3  # Get top 3 most relevant chunks
    )

    # Add the relevant context to the system message
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])1
    system_prompt = f"""
        Here is some relevant context from the JavaScript tutorial: {context}
        Please use this context to help answer the user's question.
    """

    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    # Print the response
    print(f"🤖: {response.choices[0].message.content}")

    # Add the response to the conversation history
    messages.append({ "role": "assistant", "content": response.choices[0].message.content })
