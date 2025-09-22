"""Unit tests for the TextCleaner class."""

import pytest
from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Unit tests for TextCleaner functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()

    def test_clean_tweet_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello World! This is a TEST tweet."
        cleaned = self.cleaner.clean_tweet_text(text)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        assert cleaned.islower()  # Should be lowercase

    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check out this link: https://example.com and this one http://test.org"
        cleaned = self.cleaner.remove_urls(text)
        assert "https://example.com" not in cleaned
        assert "http://test.org" not in cleaned
        assert "Check out this link:" in cleaned

    def test_extract_hashtags(self):
        """Test hashtag extraction."""
        text = "This is a #test tweet with #multiple #hashtags"
        hashtags = self.cleaner.extract_hashtags(text)
        assert isinstance(hashtags, list)
        assert "#test" in hashtags
        assert "#multiple" in hashtags
        assert "#hashtags" in hashtags

    def test_extract_mentions(self):
        """Test mention extraction."""
        text = "Hey @user1 and @user2 check this out!"
        mentions = self.cleaner.extract_mentions(text)
        assert isinstance(mentions, list)
        assert "@user1" in mentions
        assert "@user2" in mentions

    def test_extract_urls(self):
        """Test URL extraction."""
        text = "Visit https://example.com and http://test.org"
        urls = self.cleaner.extract_urls(text)
        assert isinstance(urls, list)
        assert "https://example.com" in urls
        assert "http://test.org" in urls

    def test_clean_disaster_tweet(self):
        """Test disaster-specific tweet cleaning."""
        text = "RT @user: Major #earthquake hits California! https://t.co/abc123"
        cleaned = self.cleaner.clean_disaster_tweet(text)
        assert " rt " not in cleaned.lower()  # RT prefix removed (as standalone word)
        assert "https://t.co/abc123" not in cleaned  # URL removed
        assert "earthquake" in cleaned  # Hashtag text preserved

    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["this", "is", "a", "test", "tweet", "about", "earthquake"]
        filtered = self.cleaner.remove_stopwords(tokens)
        assert isinstance(filtered, list)
        assert "this" not in filtered
        assert "is" not in filtered
        assert "earthquake" in filtered
        assert "tweet" in filtered

    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "RT @user: Major #earthquake hits California! This is a TEST"
        tokens = self.cleaner.preprocess_pipeline(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should be cleaned and tokenized
        assert all(isinstance(token, str) for token in tokens)

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        assert self.cleaner.clean_tweet_text("") == ""
        assert self.cleaner.clean_tweet_text(None) == ""
        assert self.cleaner.extract_hashtags("") == []
        assert self.cleaner.extract_mentions("") == []

    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Disaster!!! Multiple???? Exclamation...marks"
        cleaned = self.cleaner.clean_tweet_text(text, remove_repeat_chars=True)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0