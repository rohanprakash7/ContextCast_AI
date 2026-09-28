# ContextCast AI: Conversational RAG Platform for YouTube

An end-to-end Retrieval-Augmented Generation (RAG) system that enables intelligent, context-aware question answering over YouTube video content using LangChain, FAISS/Pinecone, and OpenAI.

---

## Architecture

![System Architecture](./artifacts/artifacts/contextcast-ai-architecture.drawio.png)

---

---

## Overview

ContextCast AI extracts YouTube video transcripts, processes them into semantic chunks, stores them in a vector database, and uses retrieval-augmented generation to deliver fast, accurate, and context-grounded responses to user queries.

---

## System Architecture

### Three-Layer Design

**Client Layer**
- Chrome Extension (JavaScript) for user interaction
- FastAPI backend for API handling

**RAG Layer**
- Indexing Pipeline: Transcript extraction → Chunking → Embedding → Vector storage
- Retrieval Pipeline: Query embedding → Similarity search → Prompt engineering → Response generation

**Infrastructure Layer**
- Docker for containerized deployment
- Local + cloud-ready architecture
- Evaluation pipeline for performance tracking

---

## Technology Stack

| Layer | Component | Technology |
|------|----------|-----------|
| Backend API | Web Framework | FastAPI |
| LLM Orchestration | Framework | LangChain |
| LLM & Embeddings | Model APIs | OpenAI (GPT-4o-mini, text-embedding-3-small) |
| Vector Search (Local) | Similarity Engine | FAISS |
| Vector Database (Cloud) | Managed Vector DB | Pinecone |
| Data Ingestion | Transcript Extraction | youtube-transcript-api |
| Text Processing | Chunking | LangChain TextSplitter |
| Retrieval | Similarity Search | Cosine Similarity, Top-K Retrieval |
| Evaluation | RAG Metrics | RAGAS (Faithfulness, Precision, Recall) |
| Frontend | UI Interface | Chrome Extension (JavaScript) |
| Deployment | Containerization | Docker |
| API Testing | Testing | Postman / Curl |
| Programming | Language | Python |

---

## RAG Pipeline

### Indexing Pipeline

**Data Ingestion**
- Extracts transcripts via YouTube Transcript API
- Converts into structured LangChain documents

**Chunking Strategy**
- RecursiveCharacterTextSplitter
- Chunk size: ~1000 characters
- Overlap: ~200 characters
- Preserves semantic continuity

**Embedding + Storage**
- Generates vector embeddings (1536-dim)
- Stores in FAISS (local) or Pinecone (cloud)
- Uses cosine similarity for retrieval

---

### Retrieval Pipeline

**Query Flow**
- Convert query → embedding vector
- Retrieve top-k similar chunks (k=4–6)
- Pass context + query to LLM
- Generate grounded response

**Optimization Techniques**
- Top-k tuning (k=6 optimal)
- Prompt engineering for factual responses
- Semantic similarity ranking
- Context filtering

---

## Evaluation (RAGAS)

### Metrics Used
- Faithfulness (Hallucination control)
- Context Precision
- Context Recall

### Results

| Configuration | Faithfulness | Precision | Recall | Score |
|--------------|-------------|----------|--------|------|
| simple_k6 | 1.0 | 0.87 | 0.83 | **90.8%** |

 Achieved **zero hallucination (Faithfulness = 1.0)**  
⚡ Response latency: **~2.5–4 seconds**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|------------|
| GET | `/health` | Health check |
| POST | `/index` | Index YouTube video |
| GET | `/index/status/{video_id}` | Check indexing status |
| POST | `/query` | Query video content |

---

## Example Usage

### Index Video
```json
POST /index
{
  "video_id": "VIDEO_ID"
}