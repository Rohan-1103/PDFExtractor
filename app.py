import streamlit as st
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

st.set_page_config(page_title="PDF Insight Extractor", layout="wide")
st.title("PDF Insight Extractor")

# ---------------- PDF PROCESSING ---------------- #

@st.cache_data
def load_and_clean_pdf(pdf_file):

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    text = re.sub(r'\s+', ' ', text).strip()

    return text


@st.cache_data
def chunk_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks


# ---------------- MODELS ---------------- #

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base",
        device=-1
    )


# ---------------- VECTOR INDEX ---------------- #

def generate_embeddings(text_chunks, model):

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ---------------- SEARCH ---------------- #

def semantic_search(query, model, index, chunks, k=5):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0]]


# ---------------- LLM TASKS ---------------- #

def generate_summary(text, llm):

    prompt = f"""
Summarize the following document:

{text[:2000]}
"""

    result = llm(prompt, max_new_tokens=120)

    return result[0]["generated_text"]


def answer_question(question, context, llm):

    prompt = f"""
Answer the question using the context below.

Context:
{context[:1500]}

Question:
{question}

Answer:
"""

    result = llm(prompt, max_new_tokens=120)

    return result[0]["generated_text"]


# ---------------- STREAMLIT UI ---------------- #

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    embedding_model = load_embedding_model()
    llm = load_llm()

    if "data" not in st.session_state:

        with st.spinner("Processing PDF..."):

            text = load_and_clean_pdf(uploaded_file)

            chunks = chunk_text(text)

            index = generate_embeddings(chunks, embedding_model)

            summary = generate_summary(text, llm)

            st.session_state.data = {
                "chunks": chunks,
                "index": index,
                "summary": summary
            }

    data = st.session_state.data

    st.subheader("Document Summary")

    st.write(data["summary"])

    st.subheader("Ask Questions")

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
