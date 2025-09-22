"""Single-file deployment application for disaster tweet classification."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
import uvicorn

# Import all necessary components
from src.inference.predictor import TweetClassificationService
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.features import FeatureExtractor
from src.preprocessing.keyword_detector import KeywordDetector
from src.models.disaster_classifier import DisasterClassifier
from src.models.classification_result import ClassificationResult
from src.models.tweet import Tweet
from src.models.processed_tweet import ProcessedTweet


class TweetClassificationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=280, description="Tweet text to classify")
    include_features: bool = Field(True, description="Include feature analysis")
    include_keywords: bool = Field(True, description="Include keyword analysis")


class SimpleDeploymentApp:
    """Single-file deployment application for disaster tweet classification."""

    def __init__(self):
        self.app = FastAPI(
            title="Disaster Tweet Classification API",
            description="Simple API for disaster tweet classification",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        self.classification_service = TweetClassificationService()
        self.setup_routes()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_routes(self):
        """Setup all API routes."""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize the application on startup."""
            self.logger.info("Starting Disaster Tweet Classification API")
            try:
                success = self.classification_service.initialize()
                if success:
                    self.logger.info("Classification service initialized successfully")
                else:
                    self.logger.error("Failed to initialize classification service")
            except Exception as e:
                self.logger.error(f"Startup error: {str(e)}")

        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Disaster Tweet Classification API",
                "version": "1.0.0",
                "documentation": "/docs",
                "endpoints": {
                    "classify": "/api/classify",
                    "health": "/api/health"
                }
            }

        @self.app.get("/api/health")
        async def health_check():
            """Simple health check endpoint."""
            try:
                return {
                    "status": "healthy",
                    "service": "classification",
                    "model_loaded": self.classification_service.is_ready(),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                raise HTTPException(status_code=500, detail="Health check failed")

        @self.app.post("/api/classify")
        async def classify_tweet(request: TweetClassificationRequest):
            """Classify a single tweet text."""
            try:
                result = self.classification_service.classify_text(
                    text=request.text,
                    include_features=request.include_features,
                    include_keywords=request.include_keywords
                )

                # Convert result to dict for JSON response
                response = {
                    "tweet_id": result.tweet_id,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "timestamp": result.timestamp.isoformat()
                }

                # Add features if requested
                if request.include_features and hasattr(result, 'processed_tweet') and result.processed_tweet:
                    response["features"] = result.processed_tweet.features

                return JSONResponse(content=response)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                self.logger.error(f"Classification error: {str(e)}")
                raise HTTPException(status_code=500, detail="Classification failed")

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler."""
            self.logger.error(f"Unhandled exception: {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )

    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    deployment_app = SimpleDeploymentApp()
    return deployment_app.get_app()


# For direct execution
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)