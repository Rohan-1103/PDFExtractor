import streamlit as st
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ---------------- APP CONFIG ---------------- #

st.set_page_config(page_title="PDF Insight Extractor", layout="wide")
st.title("PDF Insight Extractor")

st.sidebar.header("Upload Document")

# ---------------- LOAD PDF ---------------- #

@st.cache_data
def load_pdf(pdf_file):

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    text = re.sub(r"\s+", " ", text)

    return text


# ---------------- TEXT CHUNKING ---------------- #

def chunk_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


# ---------------- MODELS ---------------- #

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():

    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        device=-1
    )

    return generator


# ---------------- VECTOR INDEX ---------------- #

def create_faiss_index(chunks, model):

    embeddings = model.encode(chunks, convert_to_numpy=True)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ---------------- SEMANTIC SEARCH ---------------- #

def semantic_search(query, model, index, chunks, k=5):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0]]


# ---------------- SUMMARY ---------------- #

def summarize_document(text, llm):

    prompt = f"""
Summarize the following document in 5 bullet points.

Document:
{text[:2000]}

Summary:
"""

    result = llm(prompt, max_new_tokens=150)

    return result[0]["generated_text"]


# ---------------- QUESTION ANSWERING ---------------- #

def answer_question(question, context, llm):

    prompt = f"""
Answer the question using the context below.

Context:
{context[:1500]}

Question:
{question}

Answer:
"""

    result = llm(prompt, max_new_tokens=150)

    return result[0]["generated_text"]


# ---------------- STREAMLIT UI ---------------- #

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    embedding_model = load_embedding_model()
    llm = load_llm()

    if "data" not in st.session_state or st.session_state.get("file") != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            text = load_pdf(uploaded_file)

            chunks = chunk_text(text)

            index = create_faiss_index(chunks, embedding_model)

            summary = summarize_document(text, llm)

            st.session_state.data = {
                "chunks": chunks,
                "index": index,
                "summary": summary
            }

            st.session_state.file = uploaded_file.name

    data = st.session_state.data

    # -------- SUMMARY -------- #

    st.subheader("Document Summary")

    st.write(data["summary"])

    # -------- Q&A -------- #

    st.subheader("Ask Questions About the Document")

    question = st.text_input("Enter your question")

    if question:

        context_chunks = semantic_search(
            question,
            embedding_model,
            data["index"],
            data["chunks"]
        )

        context = " ".join(context_chunks)

        answer = answer_question(question, context, llm)

        st.write("### Answer")

        st.write(answer)

        with st.expander("Context used"):
            st.write(context)

else:

    st.info("Upload a PDF to start analysis.")
