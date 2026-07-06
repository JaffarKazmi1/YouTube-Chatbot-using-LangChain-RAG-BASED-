# YouTube-Chatbot-using-LangChain-RAG-BASED

Retrieval-Augmented Generation (RAG) chatbot built with LangChain that extracts transcripts from YouTube videos, processes the text into vector embeddings, and enables users to have conversations about the video's content using OpenAI's language models.

---

## Project Description
This project provides a scalable solution for analyzing lengthy videos without the need to read through massive raw transcripts or repeatedly pay heavy API token costs. 

By implementing a structured RAG pipeline, the system extracts the text from a given YouTube video ID rather than URL, shards it into contextually overlapping chunks, and registers them into a high-performance local vector database (**FAISS**). When a user submits a query, the application performs a semantic similarity search to pull only the most relevant sections of the transcript, passing them as highly focused context to **GROQ**. This architecture ensures rapid, precise, and cost-efficient responses.

---

##  Tech Stack

* **Orchestration Framework:** LangChain 
* **Data Extraction:** YouTube Transcript API
* **Vector Store:** FAISS (Facebook AI Similarity Search - CPU version)
* **Embeddings Model:** OpenAI `text-embedding-3-small`
* **LLM Engine:** GROQ 
* **Language:** Python 3.10+

---

## Step-by-Step Project Flow

The system processes data and answers questions through a structured, multi-phase pipeline:

[YouTube Video ID]
│
▼

DATA EXTRACTION  ──> YouTubeTranscriptApi fetches raw timestamped captions.
│
▼

TEXT SPLITTING   ──> RecursiveCharacterTextSplitter fragments text into 1000-char chunks.
│
▼

EMBEDDING & VDB  ──> OpenAI text-embedding-3-small vectors are indexed in FAISS.
│
▼

RETRIEVAL        ──> Similarity search fetches top (k=4) matched chunks.
│
▼

PROMPT INTERFACE ──> Chunks are formatted into text context alongside user query.
│
▼

INFERENCE (LLM)  ──> ChatOpenAI analyzes context and streams back the final answer.