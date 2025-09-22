"""Simple disaster keyword detection for disaster tweet classification system."""

import re
from typing import List, Dict, Any


class DisasterKeywordDetector:
    """Simple disaster keyword detection for tweets."""

    def __init__(self):
        """Initialize keyword detector with basic disaster keywords."""
        self.disaster_keywords = [
            'earthquake', 'flood', 'hurricane', 'tornado', 'wildfire', 'tsunami',
            'avalanche', 'drought', 'volcano', 'storm', 'cyclone', 'typhoon',
            'disaster', 'emergency', 'evacuate', 'rescue', 'damage', 'destruction',
            'casualty', 'victim', 'injured', 'fatal', 'death', 'missing', 'trapped',
            'collapse', 'explosion', 'fire', 'burning', 'smoke', 'chemical', 'spill',
            'outbreak', 'pandemic', 'quarantine', 'lockdown', 'curfew', 'riot',
            'violence', 'shooting', 'attack', 'terror', 'bomb',
            'warning', 'alert', 'watch', 'danger', 'threat', 'risk', 'hazard',
            'crisis', 'catastrophe', 'devastation', 'ruin', 'wreckage',
            'debris', 'aftermath', 'survivor', 'shelter', 'aid', 'relief', 'donation',
            'redcross', 'fema', 'emergency', 'sos', 'help', 'save', 'urgent',
            'breaking', 'breakingnews', 'news', 'update', 'situation', 'ongoing'
        ]

        self.compiled_pattern = None

    def compile_patterns(self):
        """Compile regex pattern for efficient matching."""
        patterns = []
        for keyword in self.disaster_keywords:
            escaped = re.escape(keyword)
            pattern = r'\b' + escaped + r'\b'
            patterns.append(pattern)

        self.compiled_pattern = re.compile(
            '|'.join(patterns),
            flags=re.IGNORECASE
        )

    def detect_keywords(self, text: str) -> List[str]:
        """Detect disaster keywords in text.

        Args:
            text: Input text to analyze

        Returns:
            List of matching keywords
        """
        if not text or not isinstance(text, str):
            return []

        if not self.compiled_pattern:
            self.compile_patterns()

        matches = self.compiled_pattern.findall(text)
        return list(set(match.lower() for match in matches))  # Remove duplicates and lowercase

    def get_keyword_count(self, text: str) -> int:
        """Get count of disaster keywords in text."""
        return len(self.detect_keywords(text))

    def has_disaster_keywords(self, text: str) -> bool:
        """Check if text contains any disaster keywords."""
        return len(self.detect_keywords(text)) > 0

    def get_disaster_score(self, text: str) -> float:
        """Get simple disaster relevance score (0.0 to 1.0)."""
        keyword_count = self.get_keyword_count(text)
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        # Score based on keyword density
        score = keyword_count / max(word_count, 1)
        return min(score * 2, 1.0)  # Cap at 1.0, boost weight