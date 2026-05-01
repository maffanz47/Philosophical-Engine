from .argument_classifier import argument_classifier
from .sentiment_analyzer import sentiment_analyzer
from .complexity_scorer import complexity_scorer
from .philosopher_finder import philosopher_finder
from .fallacy_detector import fallacy_detector
from .topic_clusterer import topic_clusterer
from .concept_visualizer import concept_visualizer
from .neural_extractor import neural_extractor
from .socratic_agent import socratic_agent

def load_all_models():
    """Load all trained ML models into memory on startup."""
    argument_classifier.load_models()
    sentiment_analyzer.load_models()
    complexity_scorer.load_models()
    philosopher_finder.load_models()
    fallacy_detector.load_models()
    topic_clusterer.load_models()
    concept_visualizer.load_models()
    neural_extractor.load_models()
    socratic_agent.load_models()