#  Dikko AI Noma

### Hausa Agricultural AI Assistant for Local Farming Communities

Dikko AI Noma is a Hausa-language agricultural AI assistant designed to help farmers and agricultural communities access useful farming knowledge in a language they understand.

The project addresses a real local problem: many Hausa-speaking farmers may have limited access to agricultural information in their preferred language. Dikko AI Noma provides a simple AI-powered interface where users can ask agricultural questions in Hausa and receive practical responses about crop production and farming.

The application combines a custom lightweight Transformer language model built with PyTorch, a Hausa tokenizer, agricultural knowledge retrieval (RAG), and a FastAPI inference system.

---

## 🎯 The Local Problem

Agriculture is an important part of life and economic activity in Northern Nigeria.

However, farmers often face challenges such as:

- Limited access to agricultural information in Hausa.
- Difficulty finding reliable information about crop production.
- Lack of easily accessible information about pests and crop diseases.
- Limited access to agricultural extension services.
- Language barriers when using digital agricultural resources.

Many existing AI tools and agricultural resources are primarily designed for English-speaking users.

Dikko AI Noma aims to make agricultural knowledge more accessible by providing an AI assistant that communicates directly in Hausa.

---

## 💡 The Solution

Dikko AI Noma allows users to ask questions about agriculture in Hausa.

For example:

```text
User:
Yaushe ne mafi kyawun lokacin dasa masara a Sokoto?

Dikko AI Noma:
Mafi kyawun lokacin dasa masara a yankunan arewacin Najeriya
yawanci yana da alaƙa da farkon damina. Ya kamata manomi ya
duba lokacin da ruwan sama ya fara sauka akai-akai...
````

The system is designed to provide simple, practical, and understandable agricultural information.

---

## 🌾 What Dikko AI Noma Can Help With

The assistant focuses on crop agriculture, including:

* 🌽 Noman masara
* 🌾 Noman shinkafa
* 🌱 Noman gero
* 🌾 Noman dawa
* 🥜 Noman gyada
* 🫘 Noman wake
* 🍅 Noman tumatir
* 🧅 Noman albasa
* 🌶️ Noman barkono
* 🌿 Noman rogo
* Shirya ƙasa
* Zaɓin iri
* Lokacin dasa amfanin gona
* Ban ruwa
* Taki
* Sarrafa ciyawa
* Kwari
* Cututtukan amfanin gona
* Girbi
* Adana amfanin gona

The application focuses specifically on **noma (crop agriculture)** rather than general-purpose AI.

---

# 🧠 AI Technology

The project meaningfully uses **PyTorch** to implement and train a custom decoder-only Transformer language model.

The model is not simply an API wrapper around an external AI service.

The development pipeline includes:

```text
Hausa Agricultural Data
        │
        ▼
Data Cleaning
        │
        ▼
Deduplication
        │
        ▼
Hausa Tokenizer
        │
        ▼
Tokenization
        │
        ▼
PyTorch Transformer
        │
        ▼
Model Training
        │
        ▼
Instruction Fine-Tuning
        │
        ▼
Model Inference
        │
        ▼
RAG Knowledge Retrieval
        │
        ▼
Hausa Agricultural Answer
```

---

# 🧠 Custom Transformer Model

Dikko AI Noma uses a lightweight decoder-only Transformer implemented with PyTorch.

### Model Configuration

| Component              | Configuration            |
| ---------------------- | ------------------------ |
| Architecture           | Decoder-only Transformer |
| Language               | Hausa                    |
| Domain                 | Agriculture              |
| Transformer Layers     | 3                        |
| Hidden Size            | 256                      |
| Attention Heads        | 4                        |
| Feed-Forward Dimension | 1024                     |
| Vocabulary Size        | 8,000                    |
| Normalization          | RMSNorm                  |
| Positional Encoding    | RoPE                     |
| Activation             | SwiGLU                   |
| Attention              | Causal Self-Attention    |
| Framework              | PyTorch                  |

The model is intentionally lightweight so that it can be trained and demonstrated using limited computational resources such as Google Colab.

---

# 🔤 Hausa Tokenizer

A custom SentencePiece tokenizer was trained for the Hausa agricultural domain.

The tokenizer uses an 8,000-token vocabulary and supports important Hausa characters such as:

```text
Ƙ  ƙ
Ɗ  ɗ
Ɓ  ɓ
```

Special tokens include:

```text
<|begin_of_sample|>
<|end_of_sample|>
<|instruction|>
<|response|>
```

The tokenizer was tested with:

* Casual Hausa text
* Agricultural Hausa text
* Hausa special characters
* Instruction-response samples

---

# 📚 Hausa Agricultural Dataset

A specialized Hausa agricultural dataset was created and prepared for training.

The dataset pipeline includes:

1. Data collection
2. Data cleaning
3. Duplicate removal
4. Empty-line removal
5. Dataset validation
6. Train/validation splitting
7. Tokenization
8. Binary dataset generation

The dataset is designed specifically around agricultural knowledge relevant to Hausa-speaking communities.

---

# 🔍 Retrieval-Augmented Generation (RAG)

The system integrates Retrieval-Augmented Generation to improve the usefulness and factual grounding of responses.

When a user asks a question, the system can retrieve relevant agricultural information from a knowledge base and provide it as context to the language model.

```text
User Question
      │
      ▼
Retrieve Relevant Information
      │
      ▼
Agricultural Knowledge Base
      │
      ▼
Relevant Context
      │
      ▼
Hausa Transformer
      │
      ▼
Hausa Agricultural Response
```

This approach allows the system to combine the language-generation capabilities of the Transformer with external agricultural knowledge.

---

# ⚙️ FastAPI Backend

The AI model is integrated into a FastAPI backend.

The backend provides an interface for sending Hausa agricultural questions to the AI system and receiving generated responses.

```text
User
 │
 ▼
Application Interface
 │
 ▼
FastAPI
 │
 ├───────────────┐
 ▼               ▼
RAG          Transformer
 │               │
 └───────┬───────┘
         ▼
   Hausa Response
```

The backend architecture makes it possible to connect Dikko AI Noma to web or mobile applications.

---

# 🚀 How the Application Works

### Step 1 — User asks a question

The user enters an agricultural question in Hausa.

### Step 2 — Question processing

The system processes the user's question.

### Step 3 — Knowledge retrieval

The RAG system searches the agricultural knowledge base for relevant information.

### Step 4 — AI generation

The retrieved information and user question are provided to the AI system.

### Step 5 — Hausa response

Dikko AI Noma generates a clear agricultural response in Hausa.

---

# 🏆 Why This Project Matters

Dikko AI Noma is designed around a specific local need rather than being a general-purpose chatbot.

The project focuses on:

* A real local problem.
* A local language.
* Local agricultural needs.
* Accessible AI technology.
* Practical agricultural information.

The goal is to demonstrate how AI can be adapted to serve communities that are often underserved by mainstream AI systems.

---

# 💻 Technology Stack

### AI / Machine Learning

* Python
* PyTorch
* SentencePiece
* NumPy

### AI Architecture

* Decoder-only Transformer
* Causal Self-Attention
* RMSNorm
* RoPE
* SwiGLU
* Causal Language Modeling

### Knowledge System

* Retrieval-Augmented Generation (RAG)
* Vector Database

### Backend

* FastAPI
* REST API

### Development

* Git
* GitHub
* Hugging Face
* Google Colab

---

# 📁 Project Structure

```text
dikko-ai-noma/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tokenizer/
│   └── datasets/
│
├── src/
│   ├── preprocessing/
│   ├── tokenizer/
│   ├── model/
│   ├── training/
│   ├── inference/
│   └── rag/
│
├── tests/
│
├── checkpoints/
│
├── scripts/
│
├── requirements.txt
│
└── README.md
```

---

# 🧪 Example Use Cases

### Crop Production

```text
Yaya zan shirya ƙasa kafin dasa masara?
```

### Irrigation

```text
Ta yaya zan kula da gonar tumatir lokacin rani?
```

### Crop Diseases

```text
Me zan yi idan ganyen wake ya fara yin rawaya?
```

### Agricultural Education

```text
Ka bayyana min yadda ake noman shinkafa cikin sauƙi.
```

The system responds in Hausa and provides agricultural information relevant to the user's question.

---

# 🌍 Local Impact

Dikko AI Noma aims to contribute to local communities by making agricultural information easier to access.

Potential users include:

* Small-scale farmers
* Young farmers
* Agricultural students
* Hausa-speaking communities
* Agricultural educators
* Community organizations

By combining Hausa language technology with agricultural knowledge, the project demonstrates how AI can be applied to a practical local challenge.

---

# 🏗️ Built for the Local Impact Hackathon

This project was developed as part of the **Local Impact Hackathon** challenge.

The hackathon challenge encourages participants to:

* Identify a real local problem.
* Build a working AI application.
* Use technology to address a meaningful community need.

Dikko AI Noma addresses these requirements by focusing on agricultural information accessibility for Hausa-speaking communities.

The project also meaningfully uses **PyTorch** to implement and train a custom Transformer-based language model.

---

# 👨‍💻 Creator

**Yahya Aliyu Dikko**

**Project:** Dikko AI Noma

**Focus:** Hausa NLP and Agricultural AI

**Language:** Hausa

**Domain:** Agriculture

---

# ⚠️ Disclaimer

Dikko AI Noma is an experimental AI-based agricultural information system.

The information generated by the system should not replace professional agricultural advice.

For serious crop diseases, pesticide applications, or other high-risk agricultural decisions, users should consult qualified agricultural extension officers and follow official product instructions.

---

