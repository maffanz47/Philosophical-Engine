import logging
import json
import ast
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/philosophy_corpus.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RULES_JSON_PATH = REPORTS_DIR / "association_rules.json"

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.warning(f"Data file not found at {DATA_PATH}.")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

def mine_concepts():
    df = load_data()
    if df.empty or 'top_concepts' not in df.columns:
        logger.error("Insufficient data for association mining.")
        return

    # Extract transactions
    transactions = []
    for val in df['top_concepts'].dropna():
        try:
            # Handle string representation of lists
            if isinstance(val, str):
                items = ast.literal_eval(val)
            else:
                items = val
            if isinstance(items, list):
                transactions.append(items)
        except (ValueError, SyntaxError):
            pass

    if not transactions:
        logger.warning("No valid transactions found.")
        return

    logger.info(f"Mining concepts from {len(transactions)} transactions...")
    
    # 1. Transaction Matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # 2. Apriori
    frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
    
    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found. Try lowering min_support.")
        return

    # 3. Association Rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3, num_itemsets=len(frequent_itemsets))
    rules = rules.sort_values(by='lift', ascending=False)
    
    # Format rules for JSON
    formatted_rules = []
    for _, row in rules.iterrows():
        formatted_rules.append({
            "antecedents": list(row['antecedents']),
            "consequents": list(row['consequents']),
            "support": round(row['support'], 4),
            "confidence": round(row['confidence'], 4),
            "lift": round(row['lift'], 4)
        })
        
    with open(RULES_JSON_PATH, "w") as f:
        json.dump(formatted_rules, f, indent=2)
        
    logger.info(f"Saved {len(formatted_rules)} rules to {RULES_JSON_PATH}")

def get_associations(concept: str) -> List[Dict[str, Any]]:
    if not RULES_JSON_PATH.exists():
        return [{"error": "Association rules not generated yet."}]
        
    with open(RULES_JSON_PATH, "r") as f:
        rules = json.load(f)
        
    concept = concept.lower()
    matching_rules = []
    
    for rule in rules:
        if concept in rule['antecedents'] or concept in rule['consequents']:
            matching_rules.append(rule)
            
    return matching_rules

if __name__ == "__main__":
    mine_concepts()
