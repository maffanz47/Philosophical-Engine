# Master Instruction: Philosophical Text Engine (AI221 Project)

## 1. Project Overview & Scope
**Project Title:** The Philosophical Text Engine  
**Selected Domain:** Entertainment & Media (Literary Intelligence).  
**Objective:** Build a production-grade ML system that ingests philosophical literature, analyzes its "intellectual DNA," predicts readability, and serves insights via a containerized API[cite: 1].

---

## 2. Integrated Machine Learning Tasks
The system must implement and execute the following tasks within a single, unified workflow[cite: 1]:
*   **Multi-Class Classification (The "Thinker"):** A PyTorch Artificial Neural Network (ANN) that categorizes text into 5–7 philosophical schools (e.g., Stoicism, Existentialism, Islamic Philosophy)[cite: 1].
*   **Regression (The "Complexity Scorer"):** A model to predict a continuous "Reading Complexity Score" for text snippets[cite: 1].
*   **Recommendation (The "Librarian"):** A K-Nearest Neighbors (KNN) engine using Cosine Similarity to find thematically similar readings[cite: 1].
*   **Unsupervised Clustering:** A K-Means implementation to discover latent thematic groupings within the corpus[cite: 1].
*   **Dimensionality Reduction:** Use PCA or t-SNE to generate 2D visualizations of the high-dimensional embedding space for reporting[cite: 1].

---

## 3. Experimentation & Model Comparison
The project must include multiple ML experiments to compare a baseline version against an improved version[cite: 1]:

### Tier A: Baseline Model ("Fast")
*   **Feature Engineering:** Standard TF-IDF vectorization.
*   **Model:** A simple Logistic Regression or a shallow 1-layer ANN.

### Tier B: Improved Model ("Pro")
*   **Feature Engineering:** Dense Word Embeddings using **DistilBERT** (Transformers library).
*   **Model:** A deep **PyTorch** ANN with multiple hidden layers, Dropout, and ReLU activations.

---

## 4. Technical Specifications

### Data Ingestion & Prefect Orchestration[cite: 1]
Build a **Prefect 3.0** pipeline consisting of the following tasks:
1.  **Ingestion:** Automated fetching of .txt files from the Project Gutenberg API with retry logic[cite: 1].
2.  **Feature Engineering:** Dedicated task for **Tokenization** and **Lemmatization** (using NLTK or spaCy) and text segmentation into 300–500 word chunks[cite: 1].
3.  **Training & Evaluation:** Concurrent training of Baseline and Pro models, logging F1-Score (Classification) and RMSE (Regression)[cite: 1].
4.  **Versioning:** A mechanism to save models with timestamped version numbers (e.g., `ANN_v1.0_Pro.pt`)[cite: 1].
5.  **Notifications:** A success/failure notification sent to a **Discord Webhook**[cite: 1].

### FastAPI Deployment[cite: 1]
*   **Endpoint:** `/predict`
*   **Input Handling:** Support for both raw JSON text and **.txt file uploads** using `UploadFile`[cite: 1].
*   **Output:** Return a JSON object with: `predicted_school`, `confidence_score`, `complexity_index`, and `top_3_recommendations`.

### Automated Testing & MLOps[cite: 1]
*   **Deepchecks:** Integrate automated checks for Data Integrity (nulls/duplicates), Train-Test Leakage, and Distribution Drift[cite: 1].
*   **Pytest:** Unit tests for preprocessing functions and API endpoint responsiveness[cite: 1].

---

## 5. Infrastructure & CI/CD
*   **Containerization:** A `Dockerfile` for the FastAPI service and a `docker-compose.yml` to orchestrate the API and Prefect worker services[cite: 1].
*   **GitHub Actions:** A `.github/workflows/ci.yml` file to automate testing, data validation, and Docker image builds on every push[cite: 1].

---

## 6. Strict Development Constraints
*   **No Arrow Functions:** Strictly use traditional `function` declarations for any JavaScript or script-based logic.
*   **Frameworks:** Exclusively use **PyTorch** for the neural network architectures[cite: 1].
*   **Mathematical Context:** Use the following Cosine Similarity formula for documentation:
    $$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
*   **Modular Codebase:** Maintain a clear structure:
    *   `src/data_loader.py`
    *   `src/preprocessor.py`
    *   `src/models.py`
    *   `workflows/main_flow.py`
    *   `tests/deepchecks_suite.py`

---

**AI Agent Action:** Begin by initializing the repository structure. Start with `src/data_loader.py` and `src/preprocessor.py`. Notify me once the core ingestion and lemmatization tasks are functional.