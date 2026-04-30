import os
import requests
import logging
from prefect import flow, task
from prefect.logging import get_run_logger

# Import modules
from scraper.gutenberg_scraper import scrape_gutenberg
from ml.dimensionality.embedding_reducer import generate_and_reduce_embeddings
from ml.classification.school_classifier import train_models as train_classifier
from ml.regression.influence_predictor import train_models as train_regressor
from ml.timeseries.trend_analyzer import analyze_trends
from ml.clustering.thought_clusterer import cluster_thoughts
from ml.association.concept_miner import mine_concepts

# --- Hooks ---
def send_discord_notification(flow, flow_run, state):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
        
    status = "SUCCESS" if state.is_completed() else "FAILED"
    color = 65280 if status == "SUCCESS" else 16711680 # Green vs Red
    
    payload = {
        "embeds": [{
            "title": f"Philosophical Engine Pipeline: {status}",
            "description": f"Flow '{flow.name}' finished with state: {state.name}",
            "color": color
        }]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        # Don't fail the flow if discord notification fails
        pass

def on_completion_hook(flow, flow_run, state):
    send_discord_notification(flow, flow_run, state)

def on_failure_hook(flow, flow_run, state):
    send_discord_notification(flow, flow_run, state)

# --- Tasks ---
@task(retries=3, retry_delay_seconds=30)
def run_scraper(max_books: int):
    logger = get_run_logger()
    logger.info("Task: Running web scraper")
    scrape_gutenberg(max_books=max_books)

@task(retries=3, retry_delay_seconds=30)
def run_preprocessing():
    logger = get_run_logger()
    logger.info("Task: Generating embeddings and reducing dimensions")
    generate_and_reduce_embeddings()

@task(retries=3, retry_delay_seconds=30)
def train_classification():
    logger = get_run_logger()
    logger.info("Task: Training classification models")
    train_classifier()

@task(retries=3, retry_delay_seconds=30)
def train_regression():
    logger = get_run_logger()
    logger.info("Task: Training regression models")
    train_regressor()

@task(retries=3, retry_delay_seconds=30)
def analyze_time_series():
    logger = get_run_logger()
    logger.info("Task: Analyzing time series trends")
    analyze_trends()

@task(retries=3, retry_delay_seconds=30)
def run_clustering():
    logger = get_run_logger()
    logger.info("Task: Clustering thoughts")
    cluster_thoughts()

@task(retries=3, retry_delay_seconds=30)
def run_association_mining():
    logger = get_run_logger()
    logger.info("Task: Mining concept associations")
    mine_concepts()

@task(retries=3, retry_delay_seconds=30)
def evaluate_models():
    # Placeholder for DeepChecks evaluation call which is handled in tests/
    logger = get_run_logger()
    logger.info("Task: Evaluating models (DeepChecks skipped in raw pipeline, runs in CI)")
    pass

@task(retries=3, retry_delay_seconds=30)
def save_artifacts():
    logger = get_run_logger()
    logger.info("Task: Saving artifacts and finalizing versions")
    # Joblib saving is already handled inside the training functions.
    pass


# --- Sub-Flows ---
@flow(name="1_Scrape_Gutenberg", on_failure=[on_failure_hook], on_completion=[on_completion_hook])
def scrape_gutenberg_flow(max_books: int = 500):
    run_scraper(max_books)

@flow(name="2_Preprocess_Data", on_failure=[on_failure_hook], on_completion=[on_completion_hook])
def preprocess_flow():
    run_preprocessing()

@flow(name="3_Train_ML_Models", on_failure=[on_failure_hook], on_completion=[on_completion_hook])
def train_all_models_flow():
    # Since they use CPU/Memory, running sequentially is safer, but can be run concurrently with Dask/Ray
    train_classification()
    train_regression()
    analyze_time_series()
    run_clustering()
    run_association_mining()

@flow(name="4_Evaluate_Models", on_failure=[on_failure_hook], on_completion=[on_completion_hook])
def evaluate_flow():
    evaluate_models()

@flow(name="5_Save_Artifacts", on_failure=[on_failure_hook], on_completion=[on_completion_hook])
def save_artifacts_flow():
    save_artifacts()

# --- Master Flow ---
@flow(name="Philosophical_Engine_Master_Pipeline", on_failure=[on_failure_hook], on_completion=[on_completion_hook])
def philosophical_pipeline(max_books: int = 500):
    logger = get_run_logger()
    logger.info("Starting Master Pipeline...")
    
    scrape_gutenberg_flow(max_books)
    preprocess_flow()
    train_all_models_flow()
    evaluate_flow()
    save_artifacts_flow()
    
    logger.info("Master Pipeline Complete!")

if __name__ == "__main__":
    # Can be run manually for testing
    philosophical_pipeline(max_books=10) # 10 for quick testing
