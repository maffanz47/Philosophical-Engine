import numpy as np
import random
import joblib
from pathlib import Path

class SocraticAgent:
    def __init__(self):
        self.categories = ["Ethics", "Epistemology", "Metaphysics", "Logic", "Aesthetics", "Political Philosophy"]
        self.depths = 5  # 0-4
        self.response_lengths = 3  # 0: short, 1: medium, 2: long
        self.num_states = len(self.categories) * self.depths * self.response_lengths
        self.num_actions = 30  # 30 templates
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.templates = [
            "What do you mean by {}?",
            "Can you give an example of {}?",
            "Why do you think {} is important?",
            "How does {} relate to {}?",
            "What would happen if {} were not true?",
            "How do you know that {}?",
            "What are the implications of {}?",
            "Can you explain {} in simpler terms?",
            "What is the opposite of {}?",
            "How does {} affect your life?",
            # Add more to 30
        ] * 3  # repeat for now

    def get_state(self, category, depth, response_length):
        cat_idx = self.categories.index(category) if category in self.categories else 0
        len_idx = min(response_length // 50, 2)  # bucket by word count approx
        return cat_idx * (self.depths * self.response_lengths) + depth * self.response_lengths + len_idx

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += alpha * (reward + gamma * best_next - self.q_table[state, action])

    async def train(self, db, params):  # Dummy train
        # For RL, train through interactions, but for now, random init
        self.q_table = np.random.rand(self.num_states, self.num_actions) * 0.1
        joblib.dump(self.q_table, self.models_dir / "socratic_qtable.npy")
        return {"trained": True}

    def predict(self, session_state: dict) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        category = session_state.get('category', 'Ethics')
        depth = session_state.get('depth', 0)
        response_length = session_state.get('response_length', 0)
        state = self.get_state(category, depth, response_length)
        action = self.choose_action(state, epsilon=0.1)
        question = self.templates[action % len(self.templates)].format("your idea")  # placeholder

        return {"question": question, "updated_state": {"category": category, "depth": min(depth + 1, 4), "response_length": response_length}}

    def is_trained(self) -> bool:
        return (self.models_dir / "socratic_qtable.npy").exists()

    def load_models(self):
        if self.is_trained():
            self.q_table = joblib.load(self.models_dir / "socratic_qtable.npy")

# Global instance
socratic_agent = SocraticAgent()