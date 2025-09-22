# Models Directory

This directory contains trained model files for the disaster tweet classification system.

## Expected Structure

```
models/
├── distilbert_classifier/      # DistilBERT model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
├── trained_models/             # Other trained models
│   ├── disaster_classifier.pkl
│   └── feature_extractor.pkl
└── model_metadata/             # Model metadata and configs
    ├── model_config.json
    └── training_params.json
```

## Usage

Model files should be organized by type and include appropriate metadata for reproducibility.

- **distilbert_classifier/**: Fine-tuned DistilBERT model for text classification
- **trained_models/**: Traditional ML models and feature extractors
- **model_metadata/**: Configuration files and training parameters