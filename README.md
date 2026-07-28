# Dikko AI Noma

**Dikko AI Noma** is a Hausa-language agricultural AI assistant designed to provide practical knowledge and guidance on **Noma (crop farming and agriculture)**.

The system combines a custom lightweight **decoder-only Transformer language model** with **Retrieval-Augmented Generation (RAG)** to improve the accuracy and relevance of Hausa agricultural responses.

## Features

* Hausa-focused agricultural AI assistant
* Custom decoder-only Transformer language model
* Custom SentencePiece tokenizer
* Fine-tuned on Hausa agricultural instruction-response data
* Retrieval-Augmented Generation using agricultural knowledge
* ChromaDB vector database
* LangChain-based RAG pipeline
* FastAPI backend
* Next.js frontend
* Persistent local vector database
* GPU/CPU-compatible inference
* Support for Hausa characters such as `Ƙ`, `ƙ`, `Ɗ`, and `ɗ`

## System Architecture

```text
                         Dikko AI Noma
                              │
                              ▼
                         Next.js UI
                              │
                              ▼
                       FastAPI Backend
                              │
                              ▼
                         User Query
                              │
                              ▼
                     LangChain RAG System
                              │
                              ▼
                       ChromaDB Search
                              │
                              ▼
                    data/rag/noma.txt
                              │
                              ▼
                  Relevant Agricultural Context
                              │
                              ▼
                 Custom Dikko AI Noma Model
                              │
                              ▼
                      Hausa AI Response
                              │
                              ▼
                         Next.js UI
```

## Technology Stack

### AI / Machine Learning

* Python
* PyTorch
* Custom Decoder-only Transformer
* SentencePiece
* Hugging Face
* LangChain
* ChromaDB
* Hugging Face Sentence Transformers

### Backend

* FastAPI
* Uvicorn

### Frontend

* Next.js
* React
* TypeScript

### Data

* Hausa agricultural text
* Hausa instruction-response fine-tuning data
* `noma.txt` knowledge base
* ChromaDB vector index

## Model

Dikko AI Noma uses a lightweight custom decoder-only Transformer designed for experimentation and deployment with limited computational resources.

Current model configuration:

```text
Vocabulary Size: 8,000
Hidden Size: 256
Transformer Layers: 2
Attention Heads: 4
Context Length: Configurable
Tokenizer: SentencePiece
Language: Hausa
Domain: Noma / Agriculture
```

The model is trained using causal language modeling and fine-tuned using Hausa agricultural instruction-response examples.

## Retrieval-Augmented Generation

The RAG system provides the model with relevant agricultural information from the project's knowledge base before generating an answer.

The knowledge source is:

```text
data/rag/noma.txt
```

The RAG pipeline:

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
ChromaDB Similarity Search
      │
      ▼
Relevant Hausa Agricultural Documents
      │
      ▼
Context + User Question
      │
      ▼
Dikko AI Noma Transformer
      │
      ▼
Final Hausa Answer
```

RAG is used to provide the model with additional factual agricultural context and reduce unsupported responses.

## Project Structure

```text
dikko-ai-noma/
│
├── data/
│   ├── datasets/
│   │   ├── train.bin
│   │   └── val.bin
│   │
│   └── rag/
│       └── noma.txt
│
├── scripts/
│   ├── build_rag_index.py
│   └── ...
│
├── src/
│   ├── api/
│   ├── model/
│   ├── tokenizer/
│   │   └── hausa_tokenizer.model
│   ├── rag/
│   └── ...
│
├── tests/
│
├── ui/
│   ├── app/
│   ├── components/
│   └── ...
│
├── checkpoints/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The Hausa agricultural fine-tuning dataset contains instruction-response examples focused on agriculture and crop farming.

The dataset is available on Hugging Face:

[Dikko AI Noma Hausa Dataset](https://huggingface.co/datasets/AliyuDikko/dikko-ai-noma-da-kiwo-hausa-dataset/?utm_source=chatgpt.com)

> The current dataset repository name contains `da-kiwo`, but the project's intended AI domain is **Noma (crop farming/agriculture)**.

## Training

The model was developed and trained using Google Colab with GPU acceleration.

The training process includes:

1. Hausa dataset preparation
2. Data cleaning and deduplication
3. Train/validation splitting
4. SentencePiece tokenization
5. Token ID generation
6. Custom Transformer pretraining
7. Agricultural fine-tuning
8. Validation loss monitoring
9. Model checkpoint saving
10. Inference testing
11. RAG integration

Training experiments and notebooks are available through the project resources.

## Running the Backend

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

## Running the Frontend

Navigate to the UI directory:

```bash
cd ui
```

Install dependencies:

```bash
npm install
```

Start the Next.js development server:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:3000
```

## API

The primary chat endpoint is:

```text
POST /api/chat
```

Example request:

```json
{
  "message": "Yaya ake noman masara?",
  "use_rag": true,
  "top_k": 3
}
```

Example response:

```json
{
  "response": "Ga wasu shawarwari kan yadda ake noman masara...",
  "model": "dikko-ai-noma",
  "rag_used": true
}
```

The frontend displays the final Hausa response while the RAG context remains an internal component of the generation pipeline.

## RAG Index

Build the ChromaDB index from the Hausa agricultural knowledge base:

```bash
python scripts/build_rag_index.py
```

The index is stored locally and reused by the API to avoid rebuilding embeddings on every request.

## Scope

Dikko AI Noma is specifically designed for:

* Noma
* Amfanin gona
* Noman masara
* Noman shinkafa
* Noman tumatir
* Noman kayan lambu
* Shirya ƙasa
* Ban ruwa
* Takin zamani
* Kula da amfanin gona
* Kwari da cututtukan amfanin gona
* Sauran batutuwan da suka shafi noma

The primary focus of the system is **Hausa agricultural knowledge and crop farming**.

## Project Goals

The project aims to demonstrate the development of a domain-specific Hausa AI assistant by combining:

* Custom Transformer architecture
* Hausa language modeling
* Domain-specific fine-tuning
* Retrieval-Augmented Generation
* Vector databases
* FastAPI AI serving
* Modern web application development

The goal is to provide Hausa-speaking users with a practical AI assistant capable of answering agricultural questions using both its learned language capabilities and retrieved agricultural knowledge.

## Developer

**Developed by Aliyu Dikko**

**Project:** Dikko AI Noma

**Domain:** Hausa AI for Agriculture

## Links

**GitHub Repository**

[Dikko AI Noma GitHub Repository](https://github.com/aliyudikko/dikko-ai-noma?utm_source=chatgpt.com)

**Hugging Face Dataset**

[Dikko AI Noma Hausa Dataset](https://huggingface.co/datasets/AliyuDikko/dikko-ai-noma-da-kiwo-hausa-dataset/?utm_source=chatgpt.com)

**Google Colab Training Notebook**

[Dikko AI Noma Training Notebook](https://colab.research.google.com/drive/1V9iCFzBzs0I82aGiRyuhqyMZzaNRz4_M?utm_source=chatgpt.com#scrollTo=PNQPtVrPUF8M)
