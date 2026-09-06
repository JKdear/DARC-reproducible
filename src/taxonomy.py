"""Fixed structural-damage taxonomy and annotation normalization."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

UNKNOWN_CATEGORY = "unknown"

CATEGORY_DEFINITIONS = {
    "crack": {
        "description": "Visible cracking or fissures on concrete, pavement, or structural surfaces.",
        "keywords": ["crack", "cracks", "cracked", "cracking", "fissure", "fissures", "fracture", "fractures"],
        "primary_priority": 10,
    },
    "pothole": {
        "description": "Localized road or pavement depression, pit, or pothole damage.",
        "keywords": ["pothole", "pot hole", "road pit"],
        "primary_priority": 4,
    },
    "void": {
        "description": "Voids, holes, pores, or cavities in concrete or surface material.",
        "keywords": ["void", "voids", "hole", "holes", "pore", "pores", "porous", "cavity", "cavities"],
        "primary_priority": 30,
    },
    "honeycomb": {
        "description": "Honeycomb-like concrete defect with dense cavities and poor compactness.",
        "keywords": ["honeycomb", "honeycombing"],
        "primary_priority": 25,
    },
    "spalling": {
        "description": "Surface spalling, flaking, material loss, crushed fragments, or broken-off layers.",
        "keywords": ["spalling", "spall", "spalled", "breakage", "broken", "material loss", "surface loss", "fragment"],
        "primary_priority": 40,
    },
    "corrosion": {
        "description": "Rusting or corrosion of steel or reinforcement.",
        "keywords": ["corrosion", "corroded", "corroding", "rust", "rusting", "rusted"],
        "primary_priority": 15,
    },
    "exposed_rebar": {
        "description": "Exposed reinforcement bars or steel bars visible through damaged concrete.",
        "keywords": ["exposed rebar", "exposed reinforcement", "reinforcing bar", "reinforcing bars", "steel bar", "steel bars", "rebar exposed"],
        "primary_priority": 12,
    },
    "efflorescence": {
        "description": "White crystalline deposits, salt deposits, or efflorescence on surface or joints.",
        "keywords": ["efflorescence", "white deposit", "white deposits", "salt deposit", "salt deposits", "white powder", "powdery crystals"],
        "primary_priority": 18,
    },
    "looseness": {
        "description": "Loose, loosened, or poorly compacted material on the surface.",
        "keywords": ["looseness", "loose", "loosened", "loosening", "flaking", "poor compactness", "insufficient compactness"],
        "primary_priority": 50,
    },
    "peeling": {
        "description": "Peeling or peeled surface layer.",
        "keywords": ["peeling", "peeled", "peel"],
        "primary_priority": 35,
    },
    "fatigue_crack": {
        "description": "Fatigue cracking or fatigue-related surface crack pattern.",
        "keywords": ["fatigue crack", "fatigue cracks", "fatigue cracking"],
        "primary_priority": 5,
    },
}

ATTRIBUTE_KEYWORDS = {
    "standing_water": ["standing water", "water inside", "water-filled", "water filled"],
    "left_to_right": ["left to right", "left-to-right", "rightward"],
    "top_right_to_bottom_left": ["top-right to bottom-left", "top right to bottom left"],
    "top_left_to_bottom_right": ["top-left to bottom-right", "top left to bottom right"],
    "diagonal": ["diagonal", "oblique"],
    "irregular": ["irregular", "uneven", "winding", "curved", "non-linear", "nonlinear"],
    "dense": ["dense", "many", "numerous", "covered with", "full of"],
    "localized": ["local", "localized", "in some areas", "in some parts", "specific areas"],
    "load_bearing_risk": ["load-bearing", "load bearing", "stability", "structural strength", "durability", "robustness"],
}

PREFIX_CATEGORY_HINTS = {
    "crack": "crack",
    "kong": "void",
    "gangf": "corrosion",
    "lou": "exposed_rebar",
    "xiu": "corrosion",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z_-]*")


def canonical_categories() -> list[str]:
    return sorted(CATEGORY_DEFINITIONS)


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword or "-" in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def normalize_categories(texts: Iterable[str], prefix: str | None = None) -> list[str]:
    combined = " ".join(text for text in texts if text).lower()
    categories = [
        category
        for category, specification in CATEGORY_DEFINITIONS.items()
        if any(_contains_keyword(combined, keyword) for keyword in specification["keywords"])
    ]
    if not categories and prefix in PREFIX_CATEGORY_HINTS:
        categories.append(PREFIX_CATEGORY_HINTS[prefix])
    return sorted(set(categories)) or [UNKNOWN_CATEGORY]


def choose_primary_category(categories: Iterable[str], prefix: str | None = None) -> str:
    unique = [category for category in dict.fromkeys(categories) if category != UNKNOWN_CATEGORY]
    if not unique:
        return UNKNOWN_CATEGORY
    if prefix and PREFIX_CATEGORY_HINTS.get(prefix) in unique:
        return PREFIX_CATEGORY_HINTS[prefix]
    return min(
        unique,
        key=lambda category: CATEGORY_DEFINITIONS.get(category, {}).get("primary_priority", 999),
    )


def extract_attributes(texts: Iterable[str]) -> list[str]:
    combined = " ".join(text for text in texts if text).lower()
    return sorted(
        attribute
        for attribute, keywords in ATTRIBUTE_KEYWORDS.items()
        if any(keyword in combined for keyword in keywords)
    )


def keyword_frequencies(texts: Iterable[str], top_k: int = 40) -> list[dict]:
    stopwords = {
        "the", "and", "with", "there", "this", "that", "from", "image", "surface", "structure", "structural",
        "damage", "damaged", "based", "picture", "appears", "visible", "present", "exists", "shows", "show",
        "seen", "using", "according", "determine", "whether", "please", "describe", "characteristics", "features",
        "concrete", "road", "has", "have", "are", "is", "in", "on", "of", "to", "a", "an", "as", "if", "yes",
    }
    counter = Counter()
    for text in texts:
        counter.update(
            token.lower()
            for token in _WORD_RE.findall(text)
            if token.lower() not in stopwords and len(token) > 2
        )
    return [{"token": token, "count": count} for token, count in counter.most_common(top_k)]


def build_taxonomy_payload(label_texts: Iterable[str] | None = None) -> dict:
    texts = list(label_texts or [])
    return {
        "name": "DARC structural damage taxonomy",
        "version": "1.0",
        "language": "en",
        "categories": [
            {
                "name": name,
                "description": specification["description"],
                "keywords": specification["keywords"],
                "prefix_hints": sorted(
                    prefix for prefix, category in PREFIX_CATEGORY_HINTS.items() if category == name
                ),
            }
            for name, specification in sorted(CATEGORY_DEFINITIONS.items())
        ],
        "attributes": [
            {"name": name, "keywords": keywords}
            for name, keywords in sorted(ATTRIBUTE_KEYWORDS.items())
        ],
        "unknown_category": UNKNOWN_CATEGORY,
        "label_token_frequencies": keyword_frequencies(texts),
    }
