# Dikko AI Noma

**Dikko AI Noma** is a Hausa-language agricultural AI assistant designed to make practical agricultural knowledge more accessible to Hausa-speaking farmers, students, and communities.

The project is built around a compact **decoder-only Transformer language model developed with PyTorch**, trained specifically for Hausa agricultural applications. It combines a locally trained language model with **Retrieval-Augmented Generation (RAG)** to provide more useful and knowledge-grounded responses.

---

## Project Overview

Dikko AI Noma focuses on helping Hausa-speaking users understand agricultural topics through natural Hausa conversations.

The system is designed to support:

* Agricultural question answering
* Agricultural advice
* Instruction following
* Agricultural explanations
* Summarization
* Classification
* Hausa agricultural conversations
* Retrieval of relevant agricultural knowledge through RAG

The project demonstrates the development of a complete AI pipeline, from data preparation and tokenizer training to Transformer training, inference, retrieval, and application integration.

---

## Key Features

* Hausa-first agricultural AI assistant
* Custom SentencePiece tokenizer
* Custom decoder-only Transformer language model
* PyTorch-based training pipeline
* Causal language modeling
* Rotary Positional Embeddings (RoPE)
* RMSNorm
* SwiGLU MLP
* Custom training and validation pipeline
* RAG-based agricultural knowledge retrieval
* FastAPI inference backend
* Next.js web interface

---

## Model Architecture

Dikko AI Noma uses a compact decoder-only Transformer designed for experimentation and training with limited computational resources.

| Component           | Configuration            |
| ------------------- | ------------------------ |
| Architecture        | Decoder-only Transformer |
| Framework           | PyTorch                  |
| Transformer Layers  | 2                        |
| Hidden Size         | 256                      |
| Attention Heads     | 4                        |
| Vocabulary Size     | 8,000                    |
| Context Length      | 128 tokens               |
| Positional Encoding | RoPE                     |
| Normalization       | RMSNorm                  |
| Activation          | SwiGLU                   |
| Training Objective  | Causal Language Modeling |
| Tokenizer           | SentencePiece            |

The model is trained using **next-token prediction**. Given a sequence of tokens, the model learns to predict the next token at each position.

---

## Agricultural Domain

The model is focused on Hausa agricultural knowledge, including:

* Maize farming
* Rice farming
* Millet farming
* Sorghum farming
* Cowpea farming
* Groundnut farming
* Tomato farming
* Onion farming
* Pepper farming
* Cassava farming
* Soil preparation
* Seed selection
* Planting
* Irrigation
* Fertilizer application
* Soil management
* Weed management
* Crop pests and diseases
* Harvesting
* Crop storage
* Agricultural practices

---

## Dataset

The project uses a Hausa agricultural dataset prepared specifically for training and evaluating the system.

The data processing pipeline includes:

1. Data collection
2. Text cleaning
3. Removal of unnecessary content
4. Duplicate removal
5. Quality filtering
6. Dataset preparation
7. Train/validation splitting
8. SentencePiece tokenizer training
9. Tokenization
10. Binary dataset generation

The resulting dataset is used to train and validate the Transformer language model.

---

## System Architecture

```text
Hausa Agricultural Data
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Duplicate Removal
        │
        ▼
Quality Filtering
        │
        ▼
SentencePiece Tokenizer
        │
        ▼
Tokenized Dataset
        │
        ├───────────────┐
        ▼               ▼
   Training Data    Validation Data
        │
        ▼
Decoder-only Transformer
        │
        ▼
PyTorch Training
        │
        ▼
Trained Hausa Model
        │
        ▼
Inference
        │
        ▼
RAG Knowledge Retrieval
        │
        ▼
FastAPI Backend
        │
        ▼
Next.js Web Interface
```

---

## Project Structure

```text
dikko-ai-noma/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── datasets/
│   └── tokenizer/
│
├── src/
│   ├── preprocessing/
│   ├── tokenizer/
│   ├── model/
│   ├── training/
│   ├── inference/
│   └── rag/
│
├── scripts/
├── tests/
├── checkpoints/
│
├── ui/
│
├── requirements.txt
├── preprocess.py
└── README.md
```

---

## Technologies

* **Python**
* **PyTorch**
* **SentencePiece**
* **NumPy**
* **FastAPI**
* **RAG**
* **Next.js**
* **React**
* **TypeScript**
* **Tailwind CSS**
* **Git**
* **GitHub**

---

## Development Pipeline

The project follows a complete machine learning development workflow:

```text
Data
  ↓
Preprocessing
  ↓
Deduplication
  ↓
Tokenizer Training
  ↓
Dataset Tokenization
  ↓
Model Construction
  ↓
PyTorch Training
  ↓
Validation
  ↓
Inference
  ↓
RAG Integration
  ↓
API Integration
  ↓
Web Application
```

---

## Purpose

The purpose of Dikko AI Noma is to explore how a compact language model can be developed for a **local language and a local problem domain**.

Rather than relying entirely on general-purpose models, the project demonstrates the process of building a specialized Hausa agricultural AI system, including the creation of a domain-focused dataset, training a custom Transformer model, and integrating the model with retrieval-based agricultural knowledge.

The long-term goal is to contribute to the development of useful **Hausa language AI technology** and improve access to agricultural information for Hausa-speaking communities.

---

## Developer and Creator

**Dikko AI Noma** was created and developed by:

### Yahya Aliyu Dikko

Yahya Aliyu Dikko is the **developer and creator** of Dikko AI Noma and is responsible for the design and implementation of the project, including:

* Hausa agricultural data preparation
* Data preprocessing and deduplication
* SentencePiece tokenizer development
* Decoder-only Transformer architecture
* PyTorch model implementation
* Training pipeline
* Model inference
* RAG integration
* FastAPI backend integration
* Next.js application interface

---

## Project Identity

| Field                   | Information              |
| ----------------------- | ------------------------ |
| Project Name            | Dikko AI Noma            |
| Developer               | Yahya Aliyu Dikko        |
| Creator                 | Yahya Aliyu Dikko        |
| Primary Language        | Hausa                    |
| Domain                  | Agriculture              |
| Model Architecture      | Decoder-only Transformer |
| Deep Learning Framework | PyTorch                  |
| Tokenizer               | SentencePiece            |
| Knowledge System        | RAG                      |
| Backend                 | FastAPI                  |
| Frontend                | Next.js                  |

---

## Author

**Yahya Aliyu Dikko**

**Dikko AI Noma** — A Hausa-language AI assistant for agriculture.

> Built to explore local-language AI, Transformer architecture, and practical AI solutions for local agricultural problems.
