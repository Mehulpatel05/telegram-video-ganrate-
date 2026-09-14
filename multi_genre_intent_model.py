"""
multi_genre_intent_model.py — Deep Genre & Content Intent AI Classifier
========================================================================

Specialized Sub-Models for:
1. Trading & Crypto Strategy (Breakout, Price action, Profit/Loss, Entry/Exit)
2. Gaming & Esports Action (Clutches, Squad wipes, Gyro/Sens settings, 1v4)
3. Storytelling & Controversy (Secret reveals, Exposed, Untold truth, Case study)
4. Comedy & Viral Entertainment (Punchlines, Laughs, Meme reactions)
5. Tech & Gadgets Gyaan (Settings, Comparisons, Pro Tips, Tricks)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from hinglish_text import romanize, fold

GENRE_VOCABULARIES = {
    'trading': {
        'breakout': 1.2, 'support': 1.1, 'resistance': 1.1, 'target': 1.2, 'stoploss': 1.1,
        'entry': 1.1, 'exit': 1.0, 'profit': 1.2, 'loss': 1.0, 'price action': 1.3,
        'candlestick': 1.1, 'risk reward': 1.2, 'scalping': 1.1, 'nifty': 1.0, 'banknifty': 1.0,
        'chart': 1.1, 'volume': 1.0, 'fake breakout': 1.2, 'level': 0.9, 'setup': 1.1,
    },
    'gaming': {
        'clutch': 1.3, 'headshot': 1.1, 'squad wipe': 1.3, '1v4': 1.3, '1v3': 1.2, '1v2': 1.1,
        'recoil': 1.2, 'sensitivity': 1.2, 'gyro': 1.2, 'rush': 1.0, 'rotation': 1.1,
        'zone': 1.0, 'loot': 0.9, 'ace': 1.2, 'no scope': 1.2, 'wwcd': 1.3, 'conqueror': 1.1,
    },
    'story_podcast': {
        'asli kahani': 1.3, 'secret': 1.2, 'exposed': 1.3, 'kisi ko nahi pata': 1.3,
        'pehli baar': 1.2, 'parde ke peeche': 1.3, 'reality': 1.2, 'shocking': 1.1,
        'controversy': 1.2, 'masterplan': 1.3, 'deal': 1.1, 'money': 1.0, 'crores': 1.1,
    },
    'comedy': {
        'kya mara': 1.1, 'bawaal': 1.2, 'gajab': 1.1, 'omg': 1.0, 'paisa vasool': 1.2,
        'mazza aa gaya': 1.2, 'choke': 1.1, 'funny': 1.1, 'pagal': 1.0, 'meme': 1.2,
    },
    'tech_tutorial': {
        'trick': 1.2, 'setting': 1.2, 'step by step': 1.2, 'kaise karein': 1.1,
        'best method': 1.2, 'hidden feature': 1.3, 'formula': 1.1, 'algorithm': 1.2,
    }
}


@dataclass
class GenreIntentResult:
    primary_genre: str
    genre_score: float
    detected_keywords: List[str]
    intent_confidence: float
    category_fit_multiplier: float


def analyze_genre_intent(transcript_text: str, target_category: str = 'default') -> GenreIntentResult:
    '''Analyze transcript text and calculate deep intent score and primary genre fit.'''
    if not transcript_text:
        return GenreIntentResult('default', 0.5, [], 0.5, 1.0)

    clean_text = romanize(transcript_text).lower()
    words = re.findall(r'[a-z0-9]+', clean_text)
    joined_text = ' '.join(words)

    genre_scores: Dict[str, float] = {}
    genre_hits: Dict[str, List[str]] = {}

    for genre, vocab in GENRE_VOCABULARIES.items():
        score = 0.0
        hits = []
        for kw, weight in vocab.items():
            if kw in joined_text:
                score += weight
                hits.append(kw)
        genre_scores[genre] = score
        genre_hits[genre] = hits

    best_genre = max(genre_scores, key=lambda g: genre_scores[g])
    best_score = genre_scores[best_genre]
    best_hits = genre_hits[best_genre]

    normalized_score = min(1.0, best_score / 3.5)

    multiplier = 1.0
    if target_category.lower() in best_genre or best_genre in target_category.lower():
        multiplier = 1.25

    confidence = min(1.0, len(best_hits) * 0.25)

    return GenreIntentResult(
        primary_genre=best_genre if best_score > 0 else 'general',
        genre_score=round(normalized_score, 3),
        detected_keywords=best_hits[:6],
        intent_confidence=round(confidence, 3),
        category_fit_multiplier=round(multiplier, 2),
    )
