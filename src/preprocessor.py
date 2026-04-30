import spacy, os, json
from pathlib import Path

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer")

def preprocess_text(text: str) -> str:
    """Lowercase, remove punct, lemmatize, remove stopwords"""
    doc = nlp(text[:1_000_000])  # spaCy limit
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return " ".join(tokens)

def chunk_text(text: str, chunk_size: int = 200) -> list:
    """Split text into chunks of ~200 words"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
            if len(words[i:i+chunk_size]) >= 50]  # skip tiny chunks

def build_dataset(data_dir: str = "data") -> list:
    dataset = []
    for label_dir in Path(data_dir).iterdir():
        if not label_dir.is_dir(): continue
        label = label_dir.name
        for txt_file in label_dir.glob("*.txt"):
            raw = txt_file.read_text(encoding="utf-8", errors="ignore")
            cleaned = preprocess_text(raw)
            chunks  = chunk_text(cleaned)
            for chunk in chunks:
                dataset.append({"text": chunk, "label": label})
    return dataset

if __name__ == "__main__":
    data = build_dataset()
    with open("data/dataset.json", "w") as f:
        json.dump(data, f)
    print(f"Built {len(data)} samples")