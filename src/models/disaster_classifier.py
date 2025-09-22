"""Simple disaster tweet classifier using DistilBERT."""

import time
import logging
from typing import Dict, List, Union, Optional
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path


class DisasterTweetClassifier:
    """Simple DistilBERT-based disaster tweet classifier."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        model_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 128
    ):
        """Initialize the classifier.

        Args:
            model_name: HuggingFace model name
            model_path: Path to saved model weights
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model components
        self.tokenizer = None
        self.model = None

        # Model info
        self.model_info = {
            "model_name": model_name,
            "model_path": model_path,
            "device": self.device,
            "max_length": max_length,
            "num_labels": 2  # Binary classification
        }

        # Performance stats
        self.performance_stats = {
            "load_time": 0.0,
            "inference_count": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0
        }

        self.logger = logging.getLogger(__name__)

    def load_model(self) -> bool:
        """Load the model and tokenizer.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            start_time = time.time()
            self.logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

            # Load model
            if self.model_path and Path(self.model_path).exists():
                self.logger.info(f"Loading model from path: {self.model_path}")
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            else:
                self.logger.info(f"Loading pre-trained model: {self.model_name}")
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2
                )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Update load time
            self.performance_stats["load_time"] = time.time() - start_time

            self.logger.info(f"Model loaded successfully on {self.device}")
            self.logger.info(f"Model parameters: {self.count_parameters():,}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.tokenizer is not None and self.model is not None

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get probability predictions for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Array of probabilities with shape (n_samples, n_classes)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Preprocess text
        inputs = self.preprocess_text(texts)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Convert to numpy
        probabilities = probabilities.cpu().numpy()

        # Update performance stats
        inference_time = time.time() - start_time
        self.performance_stats["inference_count"] += len(texts) if isinstance(texts, list) else 1
        self.performance_stats["total_inference_time"] += inference_time
        self.performance_stats["average_inference_time"] = (
            self.performance_stats["total_inference_time"] / self.performance_stats["inference_count"]
        )

        return probabilities

    def predict(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Get class predictions for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Predicted class indices
        """
        probabilities = self.predict_proba(texts)
        predictions = np.argmax(probabilities, axis=1)

        if isinstance(texts, str):
            return int(predictions[0])
        else:
            return predictions.tolist()

    def predict_with_confidence(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Get predictions with confidence scores.

        Args:
            texts: Single text or list of texts

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        probabilities = self.predict_proba(texts)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        if isinstance(texts, str):
            return {
                'prediction': int(predictions[0]),
                'confidence': float(confidences[0]),
                'probabilities': {
                    'disaster': float(probabilities[0][1]),
                    'non_disaster': float(probabilities[0][0])
                }
            }
        else:
            return [
                {
                    'prediction': int(pred),
                    'confidence': float(conf),
                    'probabilities': {
                        'disaster': float(probs[1]),
                        'non_disaster': float(probs[0])
                    }
                }
                for pred, conf, probs in zip(predictions, confidences, probabilities)
            ]

    def predict_with_enhanced_features(self, texts: Union[str, List[str]], features: Dict) -> Union[Dict, List[Dict]]:
        """Get predictions with enhanced features.

        Args:
            texts: Single text or list of texts
            features: Extracted features dictionary

        Returns:
            Enhanced prediction results
        """
        # Get base predictions
        base_results = self.predict_with_confidence(texts)

        if features and isinstance(texts, str):
            # Enhance single prediction with features
            enhanced_confidence = self._adjust_confidence_with_features(
                base_results['confidence'],
                features
            )

            return {
                **base_results,
                'confidence': enhanced_confidence,
                'enhanced_features': features,
                'feature_analysis': self._analyze_feature_contribution(features)
            }
        elif features and isinstance(texts, list):
            # Enhance batch predictions with features
            enhanced_results = []
            for i, result in enumerate(base_results):
                text_features = features.get(i, {}) if isinstance(features, dict) else {}
                enhanced_confidence = self._adjust_confidence_with_features(
                    result['confidence'],
                    text_features
                )

                enhanced_results.append({
                    **result,
                    'confidence': enhanced_confidence,
                    'enhanced_features': text_features,
                    'feature_analysis': self._analyze_feature_contribution(text_features)
                })
            return enhanced_results
        else:
            # Return base predictions without enhancement
            return base_results

    def preprocess_text(self, texts: Union[str, List[str]]) -> Dict:
        """Preprocess text for model input.

        Args:
            texts: Single text or list of texts

        Returns:
            Dictionary with tokenized inputs
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return encoded

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        if not self.model:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            **self.model_info,
            'parameters': self.count_parameters(),
            'device': self.device,
            'is_loaded': self.is_model_loaded(),
            'performance_stats': self.performance_stats.copy()
        }

    def _adjust_confidence_with_features(self, base_confidence: float, features: Dict) -> float:
        """Adjust confidence score based on extracted features.

        Args:
            base_confidence: Original confidence from model
            features: Extracted features dictionary

        Returns:
            Adjusted confidence score
        """
        if not features:
            return base_confidence

        adjustment = 0.0

        # Boost confidence for disaster indicators
        if features.get('disaster_keywords', []):
            adjustment += min(0.1, len(features['disaster_keywords']) * 0.02)

        # Boost confidence for high sentiment scores (indicating urgency)
        sentiment_score = features.get('sentiment_score', 0.0)
        if abs(sentiment_score) > 0.5:
            adjustment += 0.05

        # Boost confidence for URLs (often in disaster reports)
        if features.get('url_count', 0) > 0:
            adjustment += 0.03

        # Ensure confidence stays within valid range
        adjusted_confidence = base_confidence + adjustment
        return max(0.0, min(1.0, adjusted_confidence))

    def _analyze_feature_contribution(self, features: Dict) -> Dict:
        """Analyze how features contributed to the prediction.

        Args:
            features: Extracted features dictionary

        Returns:
            Feature contribution analysis
        """
        if not features:
            return {'contribution_score': 0.0, 'factors': []}

        factors = []
        score = 0.0

        if features.get('disaster_keywords', []):
            factors.append(f"Disaster keywords: {features['disaster_keywords']}")
            score += len(features['disaster_keywords']) * 0.2

        if features.get('sentiment_score', 0.0) < -0.5:
            factors.append("Negative sentiment detected")
            score += 0.1

        if features.get('url_count', 0) > 0:
            factors.append(f"URLs present: {features['url_count']}")
            score += 0.05

        if features.get('hashtag_count', 0) > 2:
            factors.append(f"Multiple hashtags: {features['hashtag_count']}")
            score += 0.05

        return {
            'contribution_score': min(1.0, score),
            'factors': factors
        }

    def clear_cache(self):
        """Clear model cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.performance_stats = {
            'load_time': self.performance_stats['load_time'],
            'inference_count': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache()