"""Unit tests for the DisasterKeywordDetector class."""

import pytest
from src.preprocessing.keyword_detector import DisasterKeywordDetector


class TestDisasterKeywordDetector:
    """Unit tests for DisasterKeywordDetector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DisasterKeywordDetector()

    def test_initialization(self):
        """Test detector initialization."""
        assert hasattr(self.detector, 'disaster_keywords')
        assert isinstance(self.detector.disaster_keywords, list)
        assert len(self.detector.disaster_keywords) > 0
        assert 'earthquake' in self.detector.disaster_keywords
        assert 'flood' in self.detector.disaster_keywords

    def test_compile_patterns(self):
        """Test pattern compilation."""
        self.detector.compile_patterns()
        assert self.detector.compiled_pattern is not None
        assert hasattr(self.detector.compiled_pattern, 'findall')

    def test_detect_keywords_simple(self):
        """Test keyword detection in simple text."""
        text = "There was an earthquake and flood in the area"
        keywords = self.detector.detect_keywords(text)
        assert isinstance(keywords, list)
        assert 'earthquake' in keywords
        assert 'flood' in keywords

    def test_detect_keywords_case_insensitive(self):
        """Test case insensitive keyword detection."""
        text = "EARTHQUAKE and Flood are serious disasters"
        keywords = self.detector.detect_keywords(text)
        assert 'earthquake' in keywords
        assert 'flood' in keywords

    def test_detect_keywords_no_matches(self):
        """Test keyword detection with no matches."""
        text = "This is a normal tweet about the weather"
        keywords = self.detector.detect_keywords(text)
        assert isinstance(keywords, list)
        assert len(keywords) == 0

    def test_detect_keywords_empty_text(self):
        """Test keyword detection with empty text."""
        assert self.detector.detect_keywords("") == []
        assert self.detector.detect_keywords(None) == []

    def test_get_keyword_count(self):
        """Test keyword count functionality."""
        text = "Earthquake and flood caused damage"
        count = self.detector.get_keyword_count(text)
        assert isinstance(count, int)
        assert count >= 2

    def test_has_disaster_keywords(self):
        """Test disaster keyword presence check."""
        disaster_text = "Major earthquake hits city"
        normal_text = "Just a normal day"
        assert self.detector.has_disaster_keywords(disaster_text) == True
        assert self.detector.has_disaster_keywords(normal_text) == False

    def test_get_disaster_score(self):
        """Test disaster score calculation."""
        disaster_text = "Earthquake flood disaster emergency"
        normal_text = "Hello world"
        disaster_score = self.detector.get_disaster_score(disaster_text)
        normal_score = self.detector.get_disaster_score(normal_text)

        assert isinstance(disaster_score, float)
        assert isinstance(normal_score, float)
        assert disaster_score > normal_score
        assert 0.0 <= disaster_score <= 1.0

    def test_multiple_keywords_same_type(self):
        """Test detection of multiple keywords of same type."""
        text = "Earthquake and aftershock rocked the area"
        keywords = self.detector.detect_keywords(text)
        assert 'earthquake' in keywords
        # Should not duplicate keywords
        assert len(keywords) == len(set(keywords))

    def test_edge_cases(self):
        """Test edge cases."""
        # Text with punctuation
        text = "Earthquake! Flood? Disaster."
        keywords = self.detector.detect_keywords(text)
        assert 'earthquake' in keywords
        assert 'flood' in keywords
        assert 'disaster' in keywords

        # Text with numbers
        text = "5 earthquake magnitude"
        keywords = self.detector.detect_keywords(text)
        assert 'earthquake' in keywords