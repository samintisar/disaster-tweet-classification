"""Simple FastAPI application for disaster tweet classification."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from ..inference.predictor import TweetClassificationService


# Initialize FastAPI app
app = FastAPI(
    title="Disaster Tweet Classification API",
    description="Simple API for disaster tweet classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize classification service
classification_service = TweetClassificationService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class TweetClassificationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=280, description="Tweet text to classify")
    include_features: bool = Field(True, description="Include feature analysis")
    include_keywords: bool = Field(True, description="Include keyword analysis")


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Disaster Tweet Classification API")
    try:
        success = classification_service.initialize()
        if success:
            logger.info("Classification service initialized successfully")
        else:
            logger.error("Failed to initialize classification service")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")


@app.get("/")
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


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "classification",
            "model_loaded": classification_service.is_ready(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/api/classify")
async def classify_tweet(request: TweetClassificationRequest):
    """Classify a single tweet text.

    Args:
        request: Classification request containing tweet text
    """
    try:
        result = classification_service.classify_text(
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
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification failed")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)