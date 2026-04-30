"""
train.py — Model Training Script

Run this script ONCE to scrape the Gutenberg texts, train all models
(SVM, MLP, K-Means, PCA), and save the artifacts to disk.
"""

from engine_core import PhilosophyEngine

if __name__ == "__main__":
    print("Initializing Philosophy Engine Training Pipeline...")
    engine = PhilosophyEngine()
    
    # Run the full pipeline
    engine.build(nn_epochs=50, batch_size=64)
    
    # Persist the models to disk
    engine.save(path="engine_artifacts")
