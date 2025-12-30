import os
import gradio as gr
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. CONFIGURATION ---
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Initialize Azure Client
try:
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )
except Exception as e:
    client = None
    print(f"Error init Azure: {e}")

# Load Models (Cached globally)
print("⏳ Loading AI Models... (This happens only once)")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Models Loaded.")

# --- 2. CORE LOGIC ---

def custom_chunker(text, chunk_size=300, overlap=50):
    """Sliding window chunking to preserve context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = 384

    def build_index(self, text_chunks):
        self.chunks = text_chunks
        embeddings = embed_model.encode(text_chunks)
        faiss.normalize_L2(embeddings) # Normalize for Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        return len(text_chunks)

    def retrieve(self, query, k=10):
        if self.index is None:
            return []
        query_vec = embed_model.encode([query])
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)
        return [self.chunks[i] for i in indices[0] if i != -1]

# Global Store (Simple for local demo)
store = VectorStore()

# --- 3. GRADIO INTERFACE FUNCTIONS ---

def ingest_data(text):
    """Handles the 'Build Knowledge Base' button click."""
    if not text or not text.strip():
        return "⚠️ Warning: No text provided. Please paste content first."
    
    try:
        chunks = custom_chunker(text)
        count = store.build_index(chunks)
        return f"✅ Success: Knowledge Base built with {count} chunks. Ready for queries!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_answer(query):
    """Handles the 'Generate Answer' button click."""
    if not query:
        return "⚠️ Please enter a question.", ""
    
    if store.index is None:
        return "⚠️ System is empty. Please ingest text in step 1 first.", ""

    # 1. Retrieve
    retrieved = store.retrieve(query, k=10)
    if not retrieved:
        return "❌ No relevant info found in documents.", ""

    # 2. Rerank
    pairs = [[query, doc] for doc in retrieved]
    scores = rerank_model.predict(pairs)
    # Sort (doc, score) by score descending
    ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:3]]

    # 3. Generate
    context_str = "\n\n---\n\n".join(top_docs)
    system_msg = "You are a helpful enterprise assistant. Answer based strictly on the context below."
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ OpenAI Error: {str(e)}"

    # Format sources for the UI
    sources_display = ""
    for i, doc in enumerate(top_docs):
        sources_display += f"**Source {i+1}:** {doc[:200]}...\n\n"

    return answer, sources_display

# --- 4. PROFESSIONAL UI LAYOUT ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Enterprise RAG") as demo:
    gr.Markdown("# 🏢 Enterprise RAG System")
    gr.Markdown("Build a custom Knowledge Base from text and ask questions using **Faiss + Reranking + GPT-4o**.")

    with gr.Row():
        # --- LEFT COLUMN: Ingestion ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Knowledge Ingestion")
            txt_source = gr.Textbox(
                lines=10, 
                placeholder="Paste your articles, emails, or reports here...", 
                label="Raw Text Data"
            )
            btn_build = gr.Button("⚡ Build Knowledge Base", variant="primary")
            
            # Status Box (Fixed: Uses Textbox instead of Label)
            txt_status = gr.Textbox(
                label="System Status", 
                value="Waiting for data...", 
                interactive=False
            )

        # --- RIGHT COLUMN: Q&A ---
        with gr.Column(scale=2):
            gr.Markdown("### 2. Q & A Interface")
            txt_query = gr.Textbox(
                label="Enter your question", 
                placeholder="e.g., What is the summary of the document?"
            )
            btn_ask = gr.Button("🔍 Generate Answer", variant="secondary")
            
            gr.Markdown("#### 🤖 AI Response")
            out_answer = gr.Markdown(value="_Answer will appear here..._")
            
            with gr.Accordion("📄 Retrieved Context (Transparency)", open=False):
                out_sources = gr.Markdown()

    # --- WIRING ---
    btn_build.click(
        fn=ingest_data, 
        inputs=[txt_source], 
        outputs=[txt_status]
    )
    
    btn_ask.click(
        fn=get_answer, 
        inputs=[txt_query], 
        outputs=[out_answer, out_sources]
    )

if __name__ == "__main__":
    demo.launch()