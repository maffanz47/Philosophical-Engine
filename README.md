# Philosophical Text Engine

A hybrid machine learning pipeline for classifying and analyzing philosophical text passages through supervised (SVM, Neural Networks) and unsupervised (K-Means, PCA) methods.

## 🛠 Tech Stack

- **Core**: Python 3.x
- **Web Framework**: FastAPI, Uvicorn
- **Machine Learning**: Scikit-learn (SVM, K-Means, PCA), PyTorch (Hierarchical MLP)
- **NLP**: NLTK (Cleaning, Lemmatization, Stopwords)
- **Containerization**: Docker, Docker Compose
- **Utilities**: NumPy, SciPy, Joblib, tqdm, python-dotenv

## 📁 Project Structure

- `api.py`: FastAPI backend exposing the `/predict` endpoint for real-time text analysis.
- `train.py`: Primary entry point to ingest data, train the model pipeline, and save artifacts.
- `engine_core.py`: Central orchestration class that manages model lifecycle and inference logic.
- `models_supervised.py`: PyTorch architectures for the Hierarchical MLP and SVM training modules.
- `models_unsupervised.py`: Logic for thematic clustering (K-Means) and dimensionality reduction (PCA).
- `ingestion.py`: Handles text ingestion and cleaning from philosophical corpora.
- `taxonomy.py`: Defines the classification hierarchy (Branches vs. Schools).
- `engine_artifacts/`: (Auto-generated) Stores serialized model weights and vectorizers.

## ⚙️ How it Works

1.  **Vectorization**: Philosophical text is converted into high-dimensional vectors using a TF-IDF pipeline (10,000 features).
2.  **Hybrid Modeling**:
    - **SVM**: Provides a deterministic baseline for high-level branch classification.
    - **Hierarchical MLP**: A multi-tier Neural Network that predicts both the philosophical branch (e.g., Ethics) and the specific school (e.g., Stoicism).
    - **K-Means Clustering**: Identifies latent thematic groups across the text corpus.
3.  **Spatial Analysis**: Uses **PCA** (Principal Component Analysis) to project passages into 2D space, allowing for visual analysis of philosophical "distance."
4.  **Inference**: The `PhilosophyEngine` class aggregates outputs from all models to provide a comprehensive analysis of any given passage.

## 🚀 Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/maffanz47/Philosophical-Engine.git
    cd Philosophical-Engine
    ```

2.  **Install dependencies**:
    ```bash
    python -m venv .venv
    # Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Train models & generate artifacts**:
    ```bash
    python train.py
    ```

4.  **Launch the API server**:
    ```bash
    uvicorn api:app --reload
    ```

5.  **Run with Docker**:
    ```bash
    docker-compose up --build
    ```

## ✨ Features

- **Multi-Tier Classification**: Hierarchical prediction of both broad philosophical branches and specific schools of thought.
- **Hybrid Inference Pipeline**: Combines deterministic, probabilistic, and unsupervised ML perspectives.
- **2D Cluster Mapping**: Visualizes philosophical passages in 2D space using PCA coordinates.
- **Dual Performance Modes**: Support for "Fast" (25% data model) and "Slow" (full accuracy model) inference.
- **Interactive CLI**: Built-in console utility (`engine_core.py`) for manual testing and verification.
- **Production-Ready API**: Fully documented FastAPI backend with Docker support for easy deployment.
