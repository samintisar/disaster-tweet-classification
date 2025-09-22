# Disaster Tweet Classification Constitution

## Core Principles

### I. Research-Driven Development
All features MUST start with comprehensive research phase. Unknowns MUST be identified and resolved before implementation. Research findings MUST be documented with decisions, rationale, and alternatives considered. Technical choices MUST be validated against domain best practices.

### II. Single-File Deployment Simplicity
Deployment components MUST prioritize simplicity over complexity. Single-file applications preferred over microservices. Direct integration over message queues. Manual configuration over automation for simple deployments. Each deployment component MUST be independently runnable.

### III. Model-First Architecture
Machine learning model capabilities MUST drive architecture decisions. Preprocessing pipelines MUST be reusable between training and inference. Feature extraction MUST be consistent across all environments. Model performance MUST be validated in production-like conditions.

### IV. Real-time Processing
Tweet classification MUST support real-time processing. Stream processing MUST be prioritized over batch processing. Latency MUST be measured and optimized. API responses MUST be under 1 second for individual predictions.

### V. Integration Testing Focus
Testing MUST focus on integration between components. Model inference tests MUST use real preprocessing pipelines. API tests MUST validate end-to-end workflows. Performance tests MUST simulate production load patterns.

## Technical Standards

### Technology Stack Requirements
- **ML Framework**: PyTorch with transformers library
- **NLP Processing**: DistilBERT for text embeddings
- **Feature Engineering**: Pandas-based meta-feature extraction
- **API Layer**: Streamlit for user interface
- **External APIs**: X API v2 for tweet collection
- **Data Processing**: Real-time streaming with tweepy

### Performance Standards
- **Inference Speed**: <100ms per tweet on CPU
- **Memory Usage**: <1GB for complete application
- **API Response**: <1 second for classification results
- **Stream Processing**: Handle 100+ tweets/minute
- **Model Accuracy**: Maintain >80% F1 score

## Development Workflow

### Feature Development Process
1. **Research Phase**: Document technical decisions and alternatives
2. **Design Phase**: Create data models and API contracts
3. **Test Phase**: Write failing tests before implementation
4. **Implementation**: Build features to make tests pass
5. **Validation**: End-to-end testing and performance validation

### Code Quality Standards
- All preprocessing functions MUST be reusable
- Model loading MUST be lazy and efficient
- Error handling MUST be comprehensive
- Configuration MUST be environment-based
- Dependencies MUST be minimal and justified

## Governance

### Amendment Process
- Amendments require documented research and justification
- Changes MUST be backward compatible
- Version numbers MUST follow semantic versioning
- All team members MUST review and approve amendments

### Compliance Requirements
- All code MUST follow constitutional principles
- Architecture decisions MUST be documented
- Performance MUST be continuously monitored
- Security MUST be validated for external API integrations

**Version**: 1.0.0 | **Ratified**: 2025-09-22 | **Last Amended**: 2025-09-22