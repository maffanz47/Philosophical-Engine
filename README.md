# The Philosophical Engine

*"From ancient text to machine insight — classifying, recommending, and forecasting human philosophy."*

An end-to-end MLOps academic project exploring the domain of classic philosophical literature sourced from Project Gutenberg.

---

## 🏛 Architecture Overview

```text
+-------------------+      +-------------------+      +------------------+
|                   |      |                   |      |                  |
| Project Gutenberg +----->+ Gutenberg Scraper +----->+ Data & Embeddings|
|                   |      | (spaCy, BeautifulSoup)   | (CSV, NPY)       |
+-------------------+      +---------+---------+      +--------+---------+
                                     |                         |
                                     v                         v
                           +---------+---------+      +--------+---------+
                           | Prefect Pipeline  |      | 7 ML Modules     |
                           | Orchestration     +----->+ (Train/Eval)     |
                           +---------+---------+      +--------+---------+
                                     |                         |
                                     v                         v
                           +---------+---------+      +--------+---------+
                           | MLflow Server     |<-----+ Models & Metrics |
                           | (Tracking & Reg)  |      | (Saved Artifacts)|
                           +-------------------+      +--------+---------+
                                                               |
                                     +-------------------------+
                                     v
                           +---------+---------+      +------------------+
                           | FastAPI Backend   |      | Interactive      |
                           | (REST Endpoints)  +----->+ Frontend         |
                           +-------------------+      | (Dashboard)      |
                                                      +------------------+
```

## 🚀 Quick Start

Ensure you have Docker and Docker Compose installed.

1. **Clone the repository** and navigate to the project directory.
2. **Start all services** using Docker Compose:
   ```bash
   cd docker
   docker-compose up --build -d
   ```
3. **Access the Application**:
   - **Interactive Frontend**: http://localhost:8000/dashboard/
   - **FastAPI Docs (Swagger)**: http://localhost:8000/docs
   - **Prefect Dashboard**: http://localhost:4200
   - **MLflow UI**: http://localhost:5000

## 📡 API Endpoints

| Method | Path | Description | Example cURL |
|--------|------|-------------|--------------|
| `GET` | `/health` | API Healthcheck | `curl http://localhost:8000/health` |
| `GET` | `/models/info` | Loaded Models Meta | `curl http://localhost:8000/models/info` |
| `POST` | `/api/v1/classify/` | Classify School | `curl -X POST -H "Content-Type: application/json" -d '{"text": "I think therefore I am"}' http://localhost:8000/api/v1/classify/` |
| `POST` | `/api/v1/regression/` | Predict Influence | `curl -X POST -H "Content-Type: application/json" -d '{"full_text": "Sample"}' http://localhost:8000/api/v1/regression/` |
| `GET` | `/api/v1/recommend/` | Hybrid Recommendation | `curl "http://localhost:8000/api/v1/recommend/?query=123&n=5"` |
| `GET` | `/api/v1/embeddings/map` | Get UMAP 2D Map | `curl http://localhost:8000/api/v1/embeddings/map` |
| `GET` | `/api/v1/timeseries/sentiment` | Sentiment Forecast | `curl http://localhost:8000/api/v1/timeseries/sentiment` |
| `GET` | `/api/v1/clusters/` | Thought Clusters | `curl http://localhost:8000/api/v1/clusters/` |
| `GET` | `/api/v1/associations/` | Concept Rules | `curl "http://localhost:8000/api/v1/associations/?concept=truth"` |
| `POST` | `/api/v1/pipeline/run` | Run Full Pipeline | `curl -X POST http://localhost:8000/api/v1/pipeline/run` |

## ⚙️ Running the Prefect Pipeline

To trigger the end-to-end pipeline (Scraping -> Preprocessing -> Model Training -> Eval):
1. **From the Frontend**: Go to the **Pipeline Runner** tab and click **Run Pipeline**.
2. **From the CLI** (if running locally without docker):
   ```bash
   python -m pipeline.philosophical_pipeline
   ```
3. Monitor progress at http://localhost:4200.

## 📊 Viewing MLflow Experiments

All training runs are tracked via MLflow.
- Open http://localhost:5000 in your browser.
- You can view metrics (F1, Accuracy, R2, RMSE) and artifacts (Confusion Matrices, Scatter Plots).
- An interactive comparison notebook is provided in `notebooks/experiment_comparison.ipynb`.

## 🧪 Running Tests & DeepChecks

1. **Unit Tests**:
   ```bash
   pytest tests/unit/ -v
   ```
2. **DeepChecks (Data & Model Integrity)**:
   ```bash
   pytest tests/ml_tests/test_deepchecks.py -v
   ```
   *Note: DeepChecks generates HTML reports saved in `reports/deepchecks/`.*

## 🔄 CI/CD Pipeline

We use GitHub Actions with three distinct workflows:
1. **CI (`ci.yml`)**: Runs on `pull_request` to `main`. Executes `ruff` linting, `black` formatting, `pytest`, and DeepChecks, then validates the Docker build.
2. **Train (`train.yml`)**: Runs on `push` to `main` or `workflow_dispatch`. Executes the Prefect pipeline to train models and uploads the resulting models as a GitHub Release artifact.
3. **Deploy (`deploy.yml`)**: Triggered upon completion of the Train workflow. Builds the Docker image and pushes it to GitHub Container Registry (GHCR).

## ⚠️ Known Limitations & Future Work
- **Scraping Time**: Scraping 500 books takes time. A cached synthetic dataset might be beneficial for rapid CI testing.
- **DistilBERT Hardware Constraint**: Fine-tuning BERT locally without a GPU is extremely slow. We've truncated sequence lengths for demonstration.
- **Frontend Enhancements**: Further integrate WebSocket connections to stream real-time logs from the Prefect pipeline directly to the UI.
