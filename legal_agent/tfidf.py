"""Pure-Python TF-IDF (no scikit-learn, no compiled dependencies).

Mirrors sklearn's TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True) with its defaults
(smooth idf, l2 norm), so results are numerically the same. It exists because some Windows PCs block scikit-learn's
compiled files ("Application Control policy has blocked this file").
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

_TOKEN = re.compile(r"(?u)\b\w\w+\b")

# Same list as sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone",
    "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an",
    "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as",
    "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
    "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but",
    "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe",
    "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere",
    "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former",
    "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
    "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
    "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc",
    "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover",
    "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never",
    "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now",
    "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put",
    "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she",
    "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than",
    "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those",
    "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward",
    "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was",
    "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever",
    "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
    "yours", "yourself", "yourselves",
])

Vector = Dict[str, float]


def terms(doc: str) -> List[str]:
    toks = [t for t in _TOKEN.findall((doc or "").lower()) if t not in ENGLISH_STOP_WORDS]
    return toks + [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]


class TfidfModel:
    def __init__(self) -> None:
        self.idf: Dict[str, float] = {}

    def _vector(self, doc: str) -> Vector:
        counts = Counter(t for t in terms(doc) if t in self.idf)
        vec = {t: (1.0 + math.log(c)) * self.idf[t] for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def fit_transform(self, docs: List[str]) -> List[Vector]:
        n = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(terms(d)))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        return [self._vector(d) for d in docs]

    def transform(self, docs: List[str]) -> List[Vector]:
        return [self._vector(d) for d in docs]


def cosine(a: Vector, b: Vector) -> float:
    """Dot product of two l2-normalised vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
