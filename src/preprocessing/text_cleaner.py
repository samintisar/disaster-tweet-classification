"""Simple text cleaning functions for disaster tweet classification system."""

import re
import string
from typing import List, Optional
import html


class TextCleaner:
    """Simple text cleaning utilities for tweet preprocessing."""

    def __init__(self):
        """Initialize text cleaner with default settings."""
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@[a-zA-Z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[a-zA-Z0-9_]+')
        self.whitespace_pattern = re.compile(r'\s+')

    def clean_tweet_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_numbers: bool = False,
        remove_special_chars: bool = False,
        remove_repeat_chars: bool = True,
        to_lowercase: bool = True,
        remove_extra_whitespace: bool = True,
        decode_html_entities: bool = True,
    ) -> str:
        """Clean tweet text with various preprocessing options.

        Args:
            text: Input tweet text
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            remove_numbers: Whether to remove numbers
            remove_special_chars: Whether to remove special characters
            remove_repeat_chars: Whether to remove repeated characters
            to_lowercase: Whether to convert to lowercase
            remove_extra_whitespace: Whether to remove extra whitespace
            decode_html_entities: Whether to decode HTML entities

        Returns:
            Cleaned tweet text
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned_text = text

        # Decode HTML entities
        if decode_html_entities:
            cleaned_text = html.unescape(cleaned_text)

        # Remove URLs
        if remove_urls:
            cleaned_text = self.url_pattern.sub('', cleaned_text)

        # Remove mentions
        if remove_mentions:
            cleaned_text = self.mention_pattern.sub('', cleaned_text)

        # Remove hashtags
        if remove_hashtags:
            cleaned_text = self.hashtag_pattern.sub('', cleaned_text)

        # Remove numbers
        if remove_numbers:
            cleaned_text = re.sub(r'\d+', '', cleaned_text)

        # Remove special characters
        if remove_special_chars:
            cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)

        # Remove repeated characters
        if remove_repeat_chars:
            cleaned_text = re.sub(r'(.)\1{2,}', r'\1\1', cleaned_text)

        # Convert to lowercase
        if to_lowercase:
            cleaned_text = cleaned_text.lower()

        # Remove extra whitespace
        if remove_extra_whitespace:
            cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text).strip()

        return cleaned_text

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)

    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return self.mention_pattern.sub('', text)

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        return [tag.lower() for tag in self.hashtag_pattern.findall(text)]

    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text."""
        return [mention.lower() for mention in self.mention_pattern.findall(text)]

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        return self.url_pattern.findall(text)

    def clean_disaster_tweet(self, text: str) -> str:
        """Clean text specifically for disaster tweet classification."""
        if not text or not isinstance(text, str):
            return ""

        # Remove retweet prefix first (before any other processing)
        cleaned = re.sub(r'^RT\s+@\w+\s*:\s*', '', text, flags=re.IGNORECASE)

        # Then apply comprehensive cleaning
        cleaned = self.clean_tweet_text(
            cleaned,
            remove_urls=True,
            remove_mentions=False,  # Keep mentions for context
            remove_hashtags=False,  # Keep hashtags for disaster keywords
            remove_numbers=False,   # Keep numbers for context (e.g., magnitude)
            remove_special_chars=True,
            remove_repeat_chars=True,
            to_lowercase=True,
            remove_extra_whitespace=True,
            decode_html_entities=True,
        )

        return cleaned

    def remove_stopwords(self, tokens: List[str], stopwords: Optional[List[str]] = None) -> List[str]:
        """Remove stopwords from token list."""
        if stopwords is None:
            # Basic English stopwords
            stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
            }

        return [token for token in tokens if token.lower() not in stopwords]

    def preprocess_pipeline(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Complete preprocessing pipeline.

        Args:
            text: Input tweet text
            remove_stopwords: Whether to remove stopwords

        Returns:
            List of cleaned tokens
        """
        # Clean the text
        cleaned = self.clean_disaster_tweet(text)

        # Tokenize
        tokens = cleaned.split()

        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]

        return tokens