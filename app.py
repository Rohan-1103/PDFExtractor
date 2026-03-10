import streamlit as st
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# --- App Configuration --- #
st.set_page_config(page_title="PDF Insight Extractor", layout="wide")
st.title("PDF Insight Extractor")

st.sidebar.title("Controls")

# --- PDF Processing --- #
@st.cache_data
def load_and_clean_pdf(pdf_file):

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    return cleaned_text


@st.cache_data
def chunk_text(text, chunk_size=500):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# --- Load Models --- #
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_summarization_models():

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    return tokenizer, model


@st.cache_resource
def load_generation_pipeline():

    return pipeline(
        task="text-generation",
        model="google/flan-t5-base",
        device=-1
    )


# --- Core Functions --- #
def generate_embeddings_and_index(text_chunks, model):

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return embeddings, index


def generate_summary(text, tokenizer, model):

    inputs = tokenizer(
        [text],
        max_length=1024,
        return_tensors="pt",
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=150,
        min_length=30,
        early_stopping=True,
        forced_bos_token_id=0
    )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


def semantic_search(query, embedding_model, faiss_index, text_chunks, k=5):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = faiss_index.search(query_embedding, k)

    return [text_chunks[i] for i in indices[0]]


def answer_question(question, context, generator):

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    result = generator(prompt, max_new_tokens=120)

    answer = result[0]["generated_text"].split("Answer:")[-1].strip()

    return answer


# --- Streamlit UI --- #

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success("PDF uploaded successfully!")

    embedding_model = load_embedding_model()
    sum_tokenizer, sum_model = load_summarization_models()
    generator = load_generation_pipeline()

    if "processed_data" not in st.session_state or \
       st.session_state.get("uploaded_file_name") != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            raw_text = load_and_clean_pdf(uploaded_file)

            text_chunks = chunk_text(raw_text)

            st.write(f"Extracted {len(text_chunks)} chunks")

            embeddings, faiss_index = generate_embeddings_and_index(
                text_chunks,
                embedding_model
            )

            st.write(f"FAISS index created with {faiss_index.ntotal} vectors")

            summary = generate_summary(
                " ".join(text_chunks[:3]),
                sum_tokenizer,
                sum_model
            )

            st.session_state.processed_data = {
                "text_chunks": text_chunks,
                "faiss_index": faiss_index,
                "summary": summary
            }

            st.session_state.uploaded_file_name = uploaded_file.name

    processed_data = st.session_state.processed_data

    # --- Summary --- #
    st.subheader("Document Summary")
    st.write(processed_data["summary"])

    # --- Question Answering --- #
    st.subheader("Ask Questions")

    question = st.text_input("Ask something about the document")

    if question:

        with st.spinner("Searching and generating answer..."):

            retrieved_chunks = semantic_search(
                question,
                embedding_model,
                processed_data["faiss_index"],
                processed_data["text_chunks"]
            )

            context = " ".join(retrieved_chunks)

            answer = answer_question(
                question,
                context,
                generator
            )

            st.write("### Answer")
            st.write(answer)

            with st.expander("Show retrieved context"):
                st.write(context)

else:
    st.info("Upload a PDF from the sidebar to begin.")
