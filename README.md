# YouTube-Chatbot-using-LangChain-RAG-BASED

Retrieval-Augmented Generation (RAG) chatbot built with LangChain that extracts transcripts from YouTube videos, processes the text into vector embeddings, and enables users to have conversations about the video's content using OpenAI's language models.

**Live Demo:** https://yt-transcript-rag-chatbot.streamlit.app/

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


```text
                    [YouTube Video ID]
                            │
                            ▼

1. DATA EXTRACTION
   ───────────────────────────────────────────────────────────
   YouTubeTranscriptApi fetches raw timestamped captions.
                            │
                            ▼

2. TEXT SPLITTING
   ───────────────────────────────────────────────────────────
   RecursiveCharacterTextSplitter fragments text into
   1000-character chunks.
                            │
                            ▼

3. EMBEDDING & VECTOR DATABASE
   ───────────────────────────────────────────────────────────
   OpenAI text-embedding-3-small generates embeddings,
   which are indexed in FAISS.
                            │
                            ▼

4. RETRIEVAL
   ───────────────────────────────────────────────────────────
   Similarity search retrieves the top k=4 most relevant
   text chunks.
                            │
                            ▼

5. PROMPT INTERFACE
   ───────────────────────────────────────────────────────────
   Retrieved chunks are formatted into context and combined
   with the user's query.
                            │
                            ▼

6. INFERENCE (LLM)
   ───────────────────────────────────────────────────────────
   ChatOpenAI analyzes the provided context and streams
   the final response back to the user.
```


1. **Data Ingestion:** The script takes a raw YouTube text payload from subtitles based on the specified video ID.
2. **Document Processing:** The long continuous string is broken down into structured semantic units using a recursive character splitter. An overlap window ensures context is not broken across block boundaries.
3. **Vector Vectorization:** Text slices are translated into mathematical vectors capturing high-dimensional meaning and cached locally inside a FAISS search tree.
4. **Context-Aware Retrieval:** The runtime parses user questions, queries the vector space for mathematical alignment, and surfaces the top 4 matched nodes.
5. **LCEL Pipeline Generation:** A parallelized chain formats data payloads into string contexts, passing structural prompts dynamically to the inference parser.


---

### 1. Install System Dependencies Globally
use the following command to install the libraries globally rather than in a virtual environment :


pip install --break-system-packages youtube-transcript-api langchain langchain-community langchain-openai faiss-cpu

## Limitations

1. **The website may occasionally show an error** due to one of the following reasons:
   - **IP gets temporarily rate-limited** because too many transcript requests are made within a short period.
   - **The application is running on a different network/IP** that YouTube does not trust, causing transcript retrieval to fail.

2. **Language Support**
   - The application is currently configured for **English transcripts**.
   - To use the website with another language, you must manually change the **language option** in the source code.

3. **Chat Session Persistence**
   - Chat history is **not stored**.
   - Refreshing the page clears the conversation since no persistent storage is implemented.
   - As a result, previously discussed context is lost after a refresh.