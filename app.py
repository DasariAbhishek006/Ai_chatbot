import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from streamlit_lottie import st_lottie
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
# -------------------------
# CONFIG
# -------------------------
BASE_DB_DIR = "chroma_dbs"
UPLOAD_DIR = "uploaded_docs"
os.makedirs(BASE_DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="🤖 AI CHATBOT", layout="wide")

# -------------------------
# CUSTOM STYLING
# -------------------------
st.markdown("""
    <style>
        .main { 
            background-color: #F3F4F6; 
            color: #111827;
            font-family: 'Inter', sans-serif;
        }
        .sidebar .sidebar-content { 
            background-color: #FFFFFF; 
            border-left: 2px solid #E5E7EB;
            color: #111827; 
        }
        h1, h2, h3, h4 { 
            color: #2563EB; 
            font-weight: 600;
        }
        .user-bubble {
            background-color: #2563EB;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 65%;
            word-wrap: break-word;
            margin-left: auto;
            margin-bottom: 14px;
            box-shadow: 0 2px 6px rgba(37,99,235,0.25);
        }
        .bot-bubble {
            background-color: #FFFFFF;
            color: #111827;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 65%;
            word-wrap: break-word;
            margin-right: auto;
            margin-bottom: 14px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }
        .source-box {
            background: #F9FAFB;
            padding: 10px;
            margin-top: 6px;
            border-radius: 8px;
            font-size: 13px;
            color: #374151;
            border: 1px dashed #D1D5DB;
        }
    </style>
""", unsafe_allow_html=True)
# -------------------------
# HEADER
# -------------------------
st.markdown("<h1 style='text-align:center;'>🤖 AI CHATBOT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Talk to your documents.</p>", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
from streamlit_lottie import st_lottie
import requests
import streamlit as st

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Example Lottie animation (robot)
lottie_robot = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# -----------------------
# Sidebar Toggle State
# -----------------------
import requests
import streamlit as st

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Example Lottie animation (robot)
lottie_robot = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# -----------------------
# Sidebar Toggle State
# -----------------------
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False

# Floating hamburger button
st.markdown("""
    <style>
    .glow-button {
        display: inline-block;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
        border: none;
        border-radius: 12px;
        cursor: pointer;
        text-align: center;
        box-shadow: 0px 0px 10px rgba(59, 130, 246, 0.7);
        transition: 0.3s ease-in-out;
    }
    .glow-button:hover {
        box-shadow: 0px 0px 20px rgba(6, 182, 212, 0.9);
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Session state for sidebar toggle ---
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False

# --- Single CTA button that toggles sidebar ---
if st.button("📂 Upload & Chat", key="toggle_sidebar"):
    st.session_state.show_sidebar = not st.session_state.show_sidebar

# --- Layout with toggle ---
main_col, right_col = st.columns([3, 1])

# -----------------------
# Main Content
# -----------------------

# Fallback so variable always exists
uploaded_file = None  

# -----------------------
# Right Floating Sidebar
# -----------------------
if st.session_state.show_sidebar:
    st.markdown("""
    <style>
    .right-sidebar {
        position: fixed;
        top: 60px;      /* distance from top */
        right: 20px;    /* distance from right */
        width: 300px;   /* sidebar width */
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)

    # Robot animation
    st_lottie(lottie_robot, height=100, key="robot")

    st.markdown("### 🌀 Workspace")

    with st.expander("🖇️ Attach File", expanded=True):
        uploaded_file = st.file_uploader("Choose a document (PDF, TXT, or MD)", type=["pdf", "txt", "md"])

    with st.expander("🗝️ Intelligence Controls", expanded=True):
        llm_model = st.selectbox("Choose LLM", ["mistral", "llama2"])
        num_ctx = st.slider("Context size", 128, 2048, 512, step=128)
        max_tokens = st.slider("Max tokens per response", 64, 1024, 256, step=64)
        k_value = st.slider("Chunks retrieved (k)", 1, 5, 2)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.success("✅ Chat cleared!")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Use uploaded_file safely
# -----------------------
if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

# -------------------------
# MAIN AREA
# -------------------------
if uploaded_file:
    # Save file
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ File uploaded: {uploaded_file.name}")

    # DB path
    file_db = os.path.splitext(uploaded_file.name)[0] + "_db"
    db_path = os.path.join(BASE_DB_DIR, file_db)
    st.session_state.db_path = db_path

    # Load docs + embeddings
    start_time = time.perf_counter()
    if not os.path.exists(db_path):
        with st.spinner("🔄 Creating new vector DB..."):
            loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            st.info(f"📄 Number of chunks: {len(split_docs)}")

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=db_path)
            st.success("✅ Embeddings created and cached!")
    else:
        with st.spinner("⚡ Loading cached embeddings..."):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
            st.success("✅ Cached embeddings loaded!")

    st.info(f"⏱️ DB load & ingestion time: {time.perf_counter() - start_time:.2f}s")

    # Show DB size
    if os.path.exists(db_path):
        db_size = sum(
            os.path.getsize(os.path.join(db_path, f))
            for f in os.listdir(db_path)
            if os.path.isfile(os.path.join(db_path, f))
        )
        st.sidebar.info(f"💾 DB size: {db_size / 1024:.2f} KB")

    # Setup retriever + LLM
   # Setup retriever + LLM
    retriever = vectordb.as_retriever(search_kwargs={"k": k_value})

    # ✅ Use the new Ollama class instead of OllamaLLM
    llm = Ollama(
        model=llm_model,
        num_ctx=num_ctx,
        num_predict=max_tokens
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


    # -------------------------
    # Chat
    # -------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    def submit():
        query = st.session_state.input_text
        if query:
            with st.spinner("🤔 Thinking..."):
                start = time.perf_counter()
                result = qa.invoke(query)
                latency = time.perf_counter() - start
                answer = result["result"]
                sources = result.get("source_documents", [])
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": sources,
                    "latency": latency
                })
            st.session_state.input_text = ""

    # Chat Display
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='user-bubble'>{chat['question']}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='bot-bubble'>{chat['answer']}<br><span style='font-size:12px; color:gray;'>⏱ {chat.get('latency', 0):.2f}s</span></div>",
                unsafe_allow_html=True
            )
            if chat["sources"]:
                with st.expander("📚 Sources"):
                    for doc in chat["sources"]:
                        st.markdown(f"<div class='source-box'>{doc.page_content[:800]}{'...' if len(doc.page_content)>800 else ''}</div>", unsafe_allow_html=True)

    # Input
    st.text_input("💬 Ask something about the document:", key="input_text", on_change=submit)