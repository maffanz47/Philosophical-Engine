"""
taxonomy.py — Hierarchical Data Taxonomy for the Philosophical Text Engine (v4).

Tier 1 (Branches):  Metaphysics | Epistemology | Value Theory
Tier 2 (Schools):   Idealism | Materialism | Rationalism | Empiricism |
                    Existentialism | Nihilism | Stoicism

BOOK_SOURCES now contains 10 Gutenberg IDs per Tier-2 category (70 total).
The scraper cycles through them in order until the 1,500-para quota is met.
"""

# ── Tier 1 Labels ──────────────────────────────────────────────────────────
TIER1_LABELS = ["Metaphysics", "Epistemology", "Value Theory"]

# ── Tier 2 Labels (per Tier 1 branch) ─────────────────────────────────────
TIER2_MAP = {
    "Metaphysics":   ["Idealism",       "Materialism"],
    "Epistemology":  ["Rationalism",    "Empiricism"],
    "Value Theory":  ["Existentialism", "Nihilism", "Stoicism"],
}

# Flat list — order defines the integer encoding used everywhere
TIER2_LABELS = (
    TIER2_MAP["Metaphysics"]    # idx 0, 1
    + TIER2_MAP["Epistemology"] # idx 2, 3
    + TIER2_MAP["Value Theory"] # idx 4, 5, 6
)

# ── 70-Book Gutenberg Library (10 per Tier-2 category) ────────────────────
# Scraped sequentially until STRICT_PARA_LIMIT (5000) is reached per category.
# Format: { tier2_label: [gutenberg_id, ...] }
BOOK_SOURCES: dict[str, list[int]] = {
    "Idealism":       [4280, 52821, 5683, 39064, 51635, 4723, 1323, 38427, 38428, 40868],
    "Materialism":    [3207,  7319, 40770, 8909, 61408, 61409, 1041, 27814, 73877, 73906],
    "Rationalism":    [59,    4391,   919,  920,   921,   922, 3800,  7304,  7305,  7306],
    "Empiricism":     [10615, 9662,  4705, 10574, 11074,  2759, 5827,  2529, 34763, 11690],
    "Existentialism": [600,  28054,  2554,  2197,  6853,  1399, 1727,  5200,  7849, 40745],
    "Nihilism":       [4363,  1998, 52263, 51356, 38226, 38227, 30723, 38145, 52319, 52318],
    "Stoicism":       [2680, 45109, 10661,  3774, 44822,  1308, 8769, 57271, 61460, 56164],
}

# Tier-2 → Tier-1 parent mapping (used by ingestion for labelling)
TIER2_TO_TIER1: dict[str, str] = {}
for _t1, _t2_list in TIER2_MAP.items():
    for _t2 in _t2_list:
        TIER2_TO_TIER1[_t2] = _t1

# ── Helper Encoders ────────────────────────────────────────────────────────
TIER1_TO_IDX = {l: i for i, l in enumerate(TIER1_LABELS)}
TIER2_TO_IDX = {l: i for i, l in enumerate(TIER2_LABELS)}
IDX_TO_TIER1 = {i: l for l, i in TIER1_TO_IDX.items()}
IDX_TO_TIER2 = {i: l for l, i in TIER2_TO_IDX.items()}

# ── Nihilism Class Index (used for weighted-loss penalty) ──────────────────
NIHILISM_T2_IDX = TIER2_TO_IDX["Nihilism"]   # = 5
