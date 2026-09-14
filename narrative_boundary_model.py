"""
narrative_boundary_model.py — Narrative Completeness & Thought Boundary AI Engine
==================================================================================

Solves:
1. Mid-sentence awkward starting cuts.
2. Trailing incomplete sentences cut-offs.
3. Sudden cut without statement resolution.

Ensures every generated vertical clip is a STANDALONE, COMPLETE, SATISFYING point/story.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Words that usually start an exciting thought/story in Hindi/Hinglish
HOOK_OPENERS = {
    'dekho', 'dekh', 'suno', 'ab', 'pehle', 'sabse', 'ek', 'agar', 'jaise',
    'jab', 'maine', 'aaj', 'yeh', 'kya', 'bhai', 'actually', 'basically',
    'pata', 'socho', 'imagine', 'look', 'listen', 'first', 'real', 'truth',
    'सच', 'देखो', 'सुनो', 'अगर', 'जब', 'पहले', 'भाई', 'आज', 'ये'
}

# Words that indicate sentence continuation (NEVER end a clip on these)
INCOMPLETE_TRAILERS = {
    'aur', 'and', 'lekin', 'but', 'par', 'toh', 'then', 'kyunki', 'because',
    'matlab', 'meaning', 'yaani', 'jaise', 'like', 'ki', 'that', 'agar', 'if',
    'woh', 'jo', 'isliye', 'so', 'phir', 'और', 'लेकिन', 'पर', 'तो', 'क्योंकि',
    'मतलब', 'कि', 'अगर', 'फिर', 'इसलिए'
}

# Conclusive punchline / resolution markers (GREAT places to end a clip)
RESOLUTION_MARKERS = {
    'gaya', 'diya', 'hua', 'hai', 'karo', 'liya', 'raha', 'chahiye', 'tha',
    'done', 'win', 'finish', 'wipe', 'hit', 'profit', 'loss', 'clutch', 'bawaal',
    'हो गया', 'कर दिया', 'मिला', 'जीते', 'खत्म', 'बवाल', 'प्रॉफिट', 'लॉस', 'सपोर्ट'
}


@dataclass
class BoundaryEvaluation:
    is_complete_thought: bool
    adjusted_start: float
    adjusted_end: float
    boundary_score: float
    hook_quality: float
    resolution_quality: float
    narrative_reason: str


def evaluate_and_refine_boundary(
    segments: List[dict],
    start_time: float,
    end_time: float,
    total_duration: float,
    min_dur: float = 15.0,
    max_dur: float = 60.0,
) -> BoundaryEvaluation:
    '''Refine start and end times to snap strictly to natural thought & sentence boundaries.'''
    if not segments:
        return BoundaryEvaluation(True, start_time, end_time, 0.7, 0.7, 0.7, 'No transcript fallback')

    # Find overlapping segment indices
    active_indices = []
    for idx, seg in enumerate(segments):
        s_start = float(seg.get('start', 0))
        s_end = float(seg.get('end', 0))
        if s_end > start_time and s_start < end_time:
            active_indices.append(idx)

    if not active_indices:
        return BoundaryEvaluation(True, start_time, end_time, 0.7, 0.7, 0.7, 'Standard window')

    first_seg = segments[active_indices[0]]
    last_seg = segments[active_indices[-1]]

    # 1. Snap Start to beginning of first segment (avoid mid-word clip start)
    refined_start = max(0.0, float(first_seg.get('start', start_time)) - 0.15)

    # 2. Check and Snap End to full sentence conclusion
    refined_end = min(total_duration, float(last_seg.get('end', end_time)) + 0.20)

    # Check last segment text for trailing conjunctions
    last_text = (last_seg.get('text') or '').strip().lower()
    last_words = [w for w in re.findall(r'[A-Za-zऀ-ॿ]+', last_text)]
    
    # If last word is a dangling trailer (e.g. aur..., lekin...), try to include next segment if within max_dur
    if last_words and last_words[-1] in INCOMPLETE_TRAILERS:
        next_idx = active_indices[-1] + 1
        if next_idx < len(segments):
            next_seg = segments[next_idx]
            cand_end = float(next_seg.get('end', refined_end)) + 0.20
            if (cand_end - refined_start) <= max_dur:
                refined_end = cand_end
                last_text = (next_seg.get('text') or '').strip().lower()
                last_words = [w for w in re.findall(r'[A-Za-zऀ-ॿ]+', last_text)]

    # 3. Hook Analysis (First 4 seconds)
    first_text = (first_seg.get('text') or '').strip().lower()
    first_words = [w for w in re.findall(r'[A-Za-zऀ-ॿ]+', first_text)]
    hook_quality = 0.5
    if any(w in HOOK_OPENERS for w in first_words[:3]):
        hook_quality = 0.95
    elif len(first_words) >= 2:
        hook_quality = 0.75

    # 4. Resolution Analysis (Last 3 seconds)
    resolution_quality = 0.5
    if any(w in RESOLUTION_MARKERS for w in last_words[-3:]):
        resolution_quality = 0.95
    elif last_words and last_words[-1] not in INCOMPLETE_TRAILERS:
        resolution_quality = 0.80
    else:
        resolution_quality = 0.35

    # 5. Length constraints
    actual_dur = refined_end - refined_start
    if actual_dur < min_dur:
        refined_end = min(total_duration, refined_start + min_dur)
    elif actual_dur > max_dur:
        refined_end = refined_start + max_dur

    boundary_score = 0.5 * hook_quality + 0.5 * resolution_quality
    is_complete = (resolution_quality >= 0.70 and hook_quality >= 0.60)

    reason = f'Hook: {hook_quality:.2f}, Resolution: {resolution_quality:.2f}, Duration: {refined_end - refined_start:.1f}s'

    return BoundaryEvaluation(
        is_complete_thought=is_complete,
        adjusted_start=round(refined_start, 2),
        adjusted_end=round(refined_end, 2),
        boundary_score=round(boundary_score, 3),
        hook_quality=round(hook_quality, 3),
        resolution_quality=round(resolution_quality, 3),
        narrative_reason=reason,
    )
