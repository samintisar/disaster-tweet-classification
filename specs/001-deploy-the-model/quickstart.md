# Quick Start Guide

## Prerequisites

- Python 3.11 or higher
- X (Twitter) Developer Account with API access
- Git repository access

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/disaster-tweet-classification.git
cd disaster-tweet-classification
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up X API Credentials

1. Go to [X Developer Portal](https://developer.twitter.com/)
2. Create a new app and get API keys
3. Create `.streamlit/secrets.toml` file:
```toml
# X API v2 Credentials
TWITTER_BEARER_TOKEN = "your_bearer_token_here"
TWITTER_API_KEY = "your_api_key_here"
TWITTER_API_SECRET = "your_api_secret_here"
TWITTER_ACCESS_TOKEN = "your_access_token_here"
TWITTER_ACCESS_TOKEN_SECRET = "your_access_token_secret_here"
```

## Quick Test

### 1. Run the Application
```bash
streamlit run deploy/simple_deploy.py
```

### 2. Test Classification
- Open the application in your browser
- Enter a tweet text in the input field
- Click "Classify" to see the prediction

### 3. Test Real-time Streaming
- Click "Start Streaming" button
- Enter disaster-related keywords (e.g., "earthquake", "flood", "wildfire")
- Set polling interval (default: 60 seconds)
- Monitor real-time classification results

## Expected Results

### Single Tweet Classification
```
Input: "Major earthquake hits San Francisco, buildings damaged"
Output:
- Prediction: disaster
- Confidence: 0.92
- Processing Time: <1s
```

### Real-time Streaming
- System collects tweets every 60 seconds
- Displays classification results in dashboard
- Shows confidence scores and disaster categories
- Updates system status and health metrics

### Performance Metrics
- Inference time: <100ms per tweet
- End-to-end response: <1s
- Memory usage: <1GB
- API rate limit: Monitored and respected

## Troubleshooting

### Common Issues

#### API Authentication Failed
- Verify X API credentials in `.streamlit/secrets.toml`
- Check API key permissions
- Ensure API access is enabled

#### Model Loading Error
- Check if model files exist in `models/` directory
- Verify PyTorch and transformers installation
- Check available memory

#### Rate Limit Exceeded
- Reduce polling frequency
- Use more specific keywords
- Monitor API usage in dashboard

#### Streamlit Not Starting
- Verify Python version (3.11+)
- Check dependencies installation
- Run `streamlit --version` to verify installation

### Debug Mode
```bash
# Enable debug logging
export STREAMLIT_LOG_LEVEL=debug
streamlit run deploy/simple_deploy.py
```

## Next Steps

### Customization
1. **Add New Keywords**: Edit disaster keywords in configuration
2. **Adjust Model**: Replace model file in `models/` directory
3. **Modify UI**: Edit Streamlit components in `deploy/simple_deploy.py`

### Advanced Features
1. **Batch Processing**: Use `/api/batch-classify` endpoint
2. **System Monitoring**: Check `/api/health` endpoint
3. **Custom Metrics**: Add new metrics to dashboard

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the data model documentation
3. Examine API contracts
4. Check system logs for error details

## Performance Optimization

### For Better Performance
- Use GPU if available
- Reduce model size with quantization
- Increase polling interval
- Use more specific keywords

### For Higher Accuracy
- Fine-tune model on domain-specific data
- Add more disaster keywords
- Implement ensemble methods
- Use feature engineering techniques

---

**Note**: This is a portfolio project designed for demonstration purposes. For production use, additional security, monitoring, and scaling considerations would be needed.