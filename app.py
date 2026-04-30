"""
app.py — Interactive Inference Console

Run this script to instantly load the pre-trained models from disk
and start the interactive console loop. You do NOT need to wait for
scraping or training to run this.
"""

from engine_core import PhilosophyEngine, manual_test

if __name__ == "__main__":
    engine = PhilosophyEngine()
    
    try:
        engine.load(path="engine_artifacts")
        manual_test(engine)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run `python train.py` first to generate the models.")
