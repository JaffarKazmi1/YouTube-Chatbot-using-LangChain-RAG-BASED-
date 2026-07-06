from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

import os

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# -------------------------------
# Configuration
# -------------------------------

VIDEO_ID = "Gfr50f6ZBvo"

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable not found.\n"
        "Linux/macOS:\n"
        'export GROQ_API_KEY="your_groq_api_key"\n'
        "Windows CMD:\n"
        'set GROQ_API_KEY=your_groq_api_key'
    )


# -------------------------------
# Step 1 : Fetch Transcript
# -------------------------------

try:
    yt = YouTubeTranscriptApi()

    transcript_list = yt.fetch(
        VIDEO_ID,
        languages=["en"]
    )

    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("This video has no captions.")
    exit()

print("\nTranscript Loaded Successfully!\n")


# -------------------------------
# Step 2 : Split into Chunks
# -------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.create_documents([transcript])

print(f"Number of chunks: {len(chunks)}")


# -------------------------------
# Step 3 : Create Embeddings
# -------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# -------------------------------
# Step 4 : Load Groq LLM
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=api_key,
)


# -------------------------------
# Step 5 : Prompt
# -------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
"""
)



# -------------------------------
# Step 7 : Build RAG Chain
# -------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

main_chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)


# -------------------------------
# Step 8 : Ask Questions
# -------------------------------

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer = main_chain.invoke(query)

    print("\nAnswer:\n")
    print(answer)