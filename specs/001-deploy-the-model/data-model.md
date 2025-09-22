# Data Model

## Core Entities

### Tweet
**Description**: Individual tweet object collected from X API v2

**Fields**:
- `id` (string): Unique tweet identifier
- `text` (string): Tweet content text
- `author_id` (string): Twitter user ID of author
- `created_at` (datetime): Tweet creation timestamp
- `language` (string): Language code (e.g., "en")
- `public_metrics` (object): Engagement metrics
  - `retweet_count` (integer)
  - `like_count` (integer)
  - `reply_count` (integer)
  - `quote_count` (integer)
- `entities` (object): Tweet entities (hashtags, mentions, etc.)

**Validation Rules**:
- `id` must be non-empty string
- `text` must be non-empty and <= 280 characters
- `created_at` must be valid datetime
- `language` must be valid ISO 639-1 code

### ClassificationResult
**Description**: Model prediction output for tweet classification

**Fields**:
- `tweet_id` (string): Reference to original tweet
- `prediction` (string): Binary classification ("disaster" | "non_disaster")
- `confidence` (float): Confidence score (0.0 - 1.0)
- `probabilities` (object): Class probabilities
  - `disaster` (float)
  - `non_disaster` (float)
- `timestamp` (datetime): Prediction timestamp
- `model_version` (string): Model version identifier

**Validation Rules**:
- `confidence` must be between 0.0 and 1.0
- `probabilities` must sum to 1.0
- `prediction` must match highest probability class

### ProcessedTweet
**Description**: Tweet after preprocessing pipeline

**Fields**:
- `original_tweet` (Tweet): Original tweet object
- `cleaned_text` (string): Preprocessed text
- `features` (object): Extracted features
  - `text_length` (integer)
  - `word_count` (integer)
  - `hashtag_count` (integer)
  - `mention_count` (integer)
  - `url_count` (integer)
  - `sentiment_score` (float)
  - `disaster_keywords` (array): List of disaster-related keywords found
- `processing_timestamp` (datetime): When tweet was processed

**Validation Rules**:
- `cleaned_text` must be non-empty
- All feature counts must be non-negative integers
- `sentiment_score` must be between -1.0 and 1.0

### APIStatus
**Description**: System status and health monitoring

**Fields**:
- `status` (string): System status ("healthy" | "degraded" | "error")
- `last_tweet_collected` (datetime): Last successful tweet collection
- `last_prediction_made` (datetime): Last successful prediction
- `api_rate_limit_remaining` (integer): Remaining API calls in current window
- `model_loaded` (boolean): Whether model is loaded in memory
- `error_message` (string): Current error message (if any)
- `uptime_seconds` (integer): System uptime in seconds

**Validation Rules**:
- `status` must be one of allowed values
- All timestamp fields must be valid datetimes
- `api_rate_limit_remaining` must be non-negative

### DisasterKeywords
**Description**: Configuration for disaster-related keyword filtering

**Fields**:
- `keywords` (array): List of disaster-related keywords
  - `term` (string): Keyword or phrase
  - `weight` (float): Importance weight (0.0 - 1.0)
  - `category` (string): Disaster category (e.g., "earthquake", "flood", "fire")
- `last_updated` (datetime): When keywords were last updated

**Validation Rules**:
- All terms must be non-empty strings
- Weights must be between 0.0 and 1.0
- Categories must be predefined disaster types

## Entity Relationships

```
Tweet → ProcessedTweet → ClassificationResult
    ↓
APIStatus (system-wide)
DisasterKeywords (configuration)
```

## State Transitions

### Tweet Processing Pipeline
```
Raw Tweet → Preprocessing → Feature Extraction → Model Inference → Classification Result
```

### API Status States
```
healthy → degraded → error
   ↑        ↓
   ←─────────┘
```

## Data Flow

1. **Collection**: X API v2 → Tweet objects
2. **Preprocessing**: Tweet → ProcessedTweet
3. **Inference**: ProcessedTweet → ClassificationResult
4. **Monitoring**: All components → APIStatus

## Error States

### Collection Errors
- `API_AUTH_FAILED`: X API authentication error
- `RATE_LIMIT_EXCEEDED`: API rate limit reached
- `NETWORK_ERROR`: Network connectivity issues

### Processing Errors
- `PREPROCESSING_FAILED`: Text cleaning pipeline error
- `FEATURE_EXTRACTION_ERROR`: Feature computation error
- `MODEL_ERROR`: Model loading or inference error

### Display Errors
- `UI_RENDER_ERROR`: Streamlit interface error
- `DATA_DISPLAY_ERROR`: Results visualization error

## Performance Metrics

### Collection Metrics
- Tweets collected per minute
- API call success rate
- Rate limit utilization percentage

### Processing Metrics
- Average preprocessing time per tweet
- Model inference time per tweet
- End-to-end processing time

### Accuracy Metrics
- Model accuracy (compared to ground truth)
- Confidence score distribution
- False positive/negative rates