import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# Page Layout & Configuration
# -------------------------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Custom Styling (UI only)
# -------------------------------
st.markdown(
    """
    <style>
        /* Overall app background */
        .stApp {
            background: linear-gradient(180deg, #0f1116 0%, #14171f 100%);
        }

        /* Main title block */
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            background: linear-gradient(90deg, #ff4b4b, #ff8a5c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            color: #9aa0ac;
            font-size: 0.95rem;
            margin-bottom: 1.4rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #12141a;
            border-right: 1px solid #262a33;
        }
        section[data-testid="stSidebar"] .stButton button {
            width: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #ff4b4b, #ff8a5c);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.6rem 0;
            transition: transform 0.15s ease, opacity 0.15s ease;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            transform: translateY(-1px);
            opacity: 0.92;
        }

        /* Status pill */
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .status-ready {
            background: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.35);
        }
        .status-idle {
            background: rgba(255, 255, 255, 0.06);
            color: #9aa0ac;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        /* Chat container card */
        .chat-card {
            background: #161922;
            border: 1px solid #262a33;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.4rem 0.2rem;
        }

        /* Info / empty state box */
        .empty-state {
            border: 1px dashed #333947;
            border-radius: 16px;
            padding: 2.2rem 1.5rem;
            text-align: center;
            color: #9aa0ac;
            background: #12141a;
        }
        .empty-state h3 {
            color: #e6e6e6;
            margin-bottom: 0.3rem;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Header
# -------------------------------
header_col1, header_col2 = st.columns([0.08, 0.92])
with header_col1:
    st.markdown("<div style='font-size:2.6rem;'>🎥</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown("<div class='hero-title'>YouTube Transcript RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Paste a video ID, build a knowledge base from its transcript, "
        "and chat with the content directly.</div>",
        unsafe_allow_html=True,
    )

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("🔑 GROQ_API_KEY environment variable not found. Please set it in your .env file.")
    st.stop()

# Helper function to extract text chunks from a retriever
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------
# Sidebar: Video Processing Panel
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Video Settings")

    if "retriever" in st.session_state:
        st.markdown("<span class='status-pill status-ready'>● Knowledge base ready</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill status-idle'>○ No video processed yet</span>", unsafe_allow_html=True)

    st.caption("Paste the ID from the video URL (the part after `v=`).")
    video_id = st.text_input("YouTube Video ID", value="Gfr50f6ZBvo", placeholder="e.g. Gfr50f6ZBvo")
    process_button = st.button("🚀 Process & Build Knowledge Base")

    st.divider()
    with st.expander("ℹ️ How this works"):
        st.markdown(
            "1. Fetches the video's transcript\n"
            "2. Splits it into overlapping chunks\n"
            "3. Embeds & indexes chunks with FAISS\n"
            "4. Retrieves relevant chunks per question\n"
            "5. Answers using only that retrieved context"
        )

# -------------------------------
# Core RAG Pipeline Setup
# -------------------------------
# Cache the vector store construction so it doesn't rebuild on every chat interaction
@st.cache_resource(show_spinner="Fetching transcript and indexing text...")
def initialize_rag(v_id):
    try:
        yt = YouTubeTranscriptApi()
        transcript_list = yt.fetch(v_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return None, "This video has no captions or transcript available."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return retriever, f"Successfully processed video! Found {len(chunks)} text chunks."

# Handle RAG initialization state
if process_button:
    # If user clicks the button, wipe the old chat history and build a new index
    st.session_state.messages = []
    retriever, message = initialize_rag(video_id)
    if retriever is None:
        st.sidebar.error(f"⚠️ {message}")
    else:
        st.sidebar.success(f"✅ {message}")
        st.session_state.retriever = retriever

# --- FIXED SECTION: ONLY RENDER CHAT IF RETRIEVER IS IN SESSION STATE ---
if "retriever" in st.session_state:
    st.markdown("### 💬 Chat with the Video")

    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-card'>", unsafe_allow_html=True)

        # Initialize message log history in streamlit session state if empty
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if not st.session_state.messages:
            st.caption("Ask anything about the video below — answers are grounded strictly in its transcript. 👇")

        # Display prior chat messages
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        st.markdown("</div>", unsafe_allow_html=True)

    # Setup RAG chain logic safely using current cached retriever
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant.\nAnswer ONLY using the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )

    parallel_chain = RunnableParallel({
        "context": st.session_state.retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    # Capture new user questions
    if user_query := st.chat_input("Ask a question about this video..."):
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Run RAG Chain inference and display response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                answer = main_chain.invoke(user_query)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.markdown(
        """
        <div class='empty-state'>
            <h3> Get started</h3>
            Enter a YouTube Video ID in the sidebar and click
            <b>“Process &amp; Build Knowledge Base”</b> to initialize the chat interface.
        </div>
        """,
        unsafe_allow_html=True,
    )