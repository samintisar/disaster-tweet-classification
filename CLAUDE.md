# Claude Code Context

## Project Overview
Disaster tweet classification system for real-time monitoring and analysis of disaster-related tweets using X API v2.

## Technology Stack
- **Language**: Python 3.11
- **ML Framework**: PyTorch with transformers library
- **NLP Processing**: DistilBERT for text embeddings
- **API Integration**: tweepy for X API v2
- **UI Framework**: Streamlit for web interface
- **Data Processing**: pandas for feature engineering
- **Testing**: pytest for unit tests

## Architecture Principles
- **Single-File Deployment**: Prioritize simple, direct integration
- **Model-First**: ML capabilities drive design decisions
- **Real-time Processing**: <1s response time requirements
- **Integration Testing**: Focus on component integration

## Key Components
- **Data Collection**: X API v2 integration with rate limiting
- **Preprocessing**: Text cleaning and feature extraction
- **Model Inference**: DistilBERT binary classification
- **Streamlit UI**: Real-time monitoring dashboard
- **API Layer**: RESTful endpoints for integration

## Project Structure
```
src/
├── preprocessing/          # Text cleaning and feature extraction
├── models/                  # ML model definitions and loading
├── inference/              # Model inference services
├── api/                     # API endpoints
└── utils/                   # Utility functions

deploy/
├── simple_deploy.py         # Single-file deployment application
└── requirements_deploy.txt  # Deployment dependencies

tests/
├── contract/
├── integration/
└── unit/
```

## Constitutional Requirements
- Research-driven development with documented decisions
- Single-file deployment simplicity over microservices
- Model-first architecture with reusable preprocessing
- Real-time processing with <1s response times
- Integration testing focus

## Recent Changes
- Added X API v2 integration with tweepy
- Implemented DistilBERT model for classification
- Created Streamlit UI for real-time monitoring
- Designed single-file deployment architecture
- Added comprehensive error handling

---
*Updated: 2025-09-22 | Feature: 001-deploy-the-model*