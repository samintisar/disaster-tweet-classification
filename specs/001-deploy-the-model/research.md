# Phase 0: Research Findings

## Technical Decisions & Rationale

### X API v2 Integration
**Decision**: Use tweepy library with X API v2 for tweet collection
**Rationale**: tweepy is the most widely used and well-maintained Python library for X API integration
**Alternatives considered**:
- requests with manual OAuth 2.0 implementation (too complex)
- python-twitter (less maintained, API v1 focus)

### Preprocessing Pipeline
**Decision**: Use existing text cleaning functions from project repository
**Rationale**: Ensures consistency between training and inference, maintains feature engineering standards
**Alternatives considered**:
- NLTK preprocessing (overkill for this use case)
- spaCy pipeline (too heavy for simple portfolio project)

### Model Architecture
**Decision**: Use DistilBERT with binary classification head
**Rationale**: Good balance of performance vs speed, suitable for real-time processing
**Alternatives considered**:
- BERT-base (slower, more memory)
- RoBERTa (similar performance but larger)

### Deployment Architecture
**Decision**: Single-file Streamlit application with embedded model
**Rationale**: Simplest deployment approach for portfolio project, constitutional requirement for single-file deployment
**Alternatives considered**:
- FastAPI + React frontend (more complex)
- Microservice architecture (overkill for portfolio)

### Performance Targets
**Decision**: Target <1s end-to-end response time, >75% accuracy
**Rationale**: Reasonable for portfolio project, balances performance with development effort
**Alternatives considered**:
- Sub-100ms response time (would require more optimization)
- >90% accuracy (would need larger model/dataset)

## Unknowns Resolved

### API Rate Limiting
**Finding**: X API v2 has different rate limits for different endpoints
**Decision**: Implement exponential backoff with 15-minute window tracking
**Implementation**: Use tweepy's built-in rate limiting with custom retry logic

### Real-time Processing
**Finding**: Need to balance tweet collection frequency with API limits
**Decision**: Poll every 60 seconds with disaster-related keyword filters
**Implementation**: Background thread with configurable polling interval

### Model Loading Strategy
**Finding**: Model loading time affects user experience
**Decision**: Lazy loading with caching in Streamlit session state
**Implementation**: Load model on first prediction request, cache for subsequent requests

### Error Handling
**Finding**: Multiple failure points (API, network, model, preprocessing)
**Decision**: Graceful degradation with user-friendly error messages
**Implementation**: Try-catch blocks with specific error handling for each component

## Technology Stack Finalized

### Core Dependencies
- **Python 3.11**: Stable, well-supported version
- **tweepy 4.14**: X API v2 support with rate limiting
- **transformers 4.35**: DistilBERT model support
- **torch 2.1**: PyTorch with optimizations
- **streamlit 1.28**: Web UI framework
- **pandas 2.1**: Data processing and feature engineering

### Development Tools
- **pytest 7.4**: Unit testing framework
- **black 23.9**: Code formatting
- **flake8 6.1**: Linting

## Performance Considerations

### Inference Optimization
- Use CPU-only inference (no GPU requirements for portfolio)
- Model quantization to reduce memory footprint
- Batch processing for multiple tweets

### API Usage
- Implement smart polling to avoid rate limits
- Cache disaster keywords to minimize API calls
- Use streaming endpoints where possible

### Memory Management
- Lazy loading of model components
- Clear session state periodically
- Monitor memory usage in Streamlit

## Security & Compliance

### API Key Management
- Use Streamlit secrets management
- Never commit API keys to repository
- Provide clear documentation for setup

### Data Privacy
- Only store necessary tweet metadata
- Implement data retention policies
- Clear sensitive information from display

## Integration Strategy

### Component Communication
- Direct function calls between components
- Shared configuration object
- Standardized data formats

### Testing Approach
- Unit tests for individual components
- Integration tests for API calls
- Manual testing for Streamlit UI

## Conclusion

All technical unknowns have been resolved through research. The architecture follows constitutional principles with emphasis on simplicity and real-time processing. The single-file deployment approach is perfect for a portfolio project - it's easy to understand, quick to implement, and effectively demonstrates ML deployment concepts without unnecessary complexity.