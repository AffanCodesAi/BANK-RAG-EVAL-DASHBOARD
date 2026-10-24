# BANK-RAG-EVAL-DASHBOARD

A Retrieval-Augmented Generation (RAG) application with a web dashboard for querying a knowledge base, evaluating answer quality, and inspecting retrieval performance.

## Overview

BANK-RAG-EVAL-DASHBOARD combines semantic search, retrieval-based question answering, and evaluation tools in one project. It is designed to explore how effectively a RAG system retrieves relevant information and generates grounded answers.

## Features

- **RAG-based question answering** — retrieve relevant context to answer user questions.
- **Semantic search** — find relevant knowledge using vector similarity.
- **Evaluation tools** — test predefined questions and inspect answer quality.
- **Interactive dashboard** — access the application through a web interface.
- **Retrieval inspection** — examine retrieved results to help debug and improve search.
- **Modular architecture** — separate ingestion, retrieval, evaluation, API, and frontend components.

## Tech Stack

- **Python** — backend and RAG pipeline
- **FastAPI** — API layer
- **ChromaDB** — vector database
- **Sentence Transformers** — text embeddings
- **React** — frontend dashboard
- **Vite** — frontend development and build tooling

## Project Structure

```text
BANK-RAG-EVAL-DASHBOARD/
├── main.py
├── requirements.txt
├── src/
│   ├── ingest.py
│   ├── search.py
│   ├── evaluate.py
│   └── dashboard.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AffanCodesAi/BANK-RAG-EVAL-DASHBOARD.git
cd BANK-RAG-EVAL-DASHBOARD
```

### 2. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure the knowledge base

Configure any required environment variables and provide an authorized knowledge source. Keep API keys, credentials, private documents, and other sensitive information out of Git.

### 5. Run the application

Start the backend using the project's configured entry point:

```bash
python main.py
```

In a separate terminal, run the frontend development server:

```bash
cd frontend
npm run dev
```

Use the local URLs displayed in the terminal to access the application. Check the project's configuration for any required environment variables or additional startup steps.

## Evaluation

The project includes evaluation and dashboard modules for testing RAG behavior. Evaluation can help investigate:

- Relevance of retrieved context
- Answer quality and grounding
- Retrieval failures
- Opportunities to improve chunking, embeddings, and search configuration

## Future Improvements

- Add automated retrieval and answer-quality metrics.
- Compare embedding models and retrieval strategies.
- Track evaluation results across different configurations.
- Improve observability and error analysis.
- Add regression tests for retrieval and answer generation.

## Security and Data Handling

This repository should be used with authorized data sources only. Never commit API keys, credentials, customer records, confidential documents, or proprietary datasets.

## Disclaimer

This is an AI/RAG engineering portfolio project. Outputs should be independently verified before being used for operational or financial decisions.

## Demo Screenshots

<img width="1900" height="933" alt="image" src="https://github.com/user-attachments/assets/7ea5b4e0-0795-4504-a675-9b0bdd5b5f62" />

<img width="1900" height="933" alt="image" src="https://github.com/user-attachments/assets/5c678068-34c8-4ac0-bc2d-05e887f3a5eb" />

<img width="1900" height="933" alt="image" src="https://github.com/user-attachments/assets/8de249cb-9d33-467c-86cb-28fb45536418" />




