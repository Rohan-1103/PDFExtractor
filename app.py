
import streamlit as st
import fitz # PyMuPDF for PDF reading
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import torch

# --- Configuration --- #
st.set_page_config(page_title="PDF Insight Extractor", layout="wide")
st.title("PDF Insight Extractor")

st.sidebar.title("Controls")

# --- PDF Processing Functions --- #
@st.cache_data
def load_and_clean_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    cleaned_text = re.sub(r'\s+', ' ', text).strip() # Corrected regex here
    return cleaned_text

@st.cache_data
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# --- Model Loading (Cached) --- #
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarization_models():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Core Logic Functions --- #
def generate_embeddings_and_index(text_chunks, model):
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embeddings, index

def generate_summary(text, tokenizer, model, max_length=150, min_length=30):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, min_length=min_length, early_stopping=True, forced_bos_token_id=0)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

def semantic_search(query, embedding_model, faiss_index, text_chunks, k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k)
    return [text_chunks[i] for i in indices[0]]

def answer_question(question, context, qa_pipeline):
    result = qa_pipeline(question=question, context=context)
    return result['answer'], result['score']

# --- Streamlit UI --- #

uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    # Load models
    embedding_model = load_embedding_model()
    sum_tokenizer, sum_model = load_summarization_models()
    qa_pipe = load_qa_pipeline()

    # Use session state to store processed data and avoid re-processing on every interaction
    if "processed_data" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            raw_text = load_and_clean_pdf(uploaded_file)
            text_chunks = chunk_text(raw_text)
            st.write(f"Extracted {len(text_chunks)} text chunks from the PDF.")

            embeddings, faiss_index = generate_embeddings_and_index(text_chunks, embedding_model)
            st.write(f"Created FAISS index with {faiss_index.ntotal} vectors.")

            summary_text = generate_summary(text_chunks[0], sum_tokenizer, sum_model)

            st.session_state.processed_data = {
                "raw_text": raw_text,
                "text_chunks": text_chunks,
                "embeddings": embeddings,
                "faiss_index": faiss_index,
                "summary_text": summary_text
            }
            st.session_state.uploaded_file_name = uploaded_file.name

    processed_data = st.session_state.processed_data

    st.subheader("Document Summary")
    st.write(processed_data["summary_text"])

    st.subheader("Question Answering")
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching for relevant context and answering..."):
            # Semantic search for context
            retrieved_chunks = semantic_search(
                question, 
                embedding_model, 
                processed_data["faiss_index"], 
                processed_data["text_chunks"],
                k=5 # Retrieve top 5 most relevant chunks
            )
            context = " ".join(retrieved_chunks)

            # Answer the question
            answer, confidence = answer_question(question, context, qa_pipe)
            
            st.write(f"**Answer:** {answer}")
            st.write(f"**Confidence:** {confidence:.2f}")
            
            with st.expander("Show retrieved context"): # Optional: show context
                st.write(context)

else:
    st.info("Please upload a PDF file using the sidebar to get started.")
