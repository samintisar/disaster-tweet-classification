# Modular Notebooks for Disaster Tweet Classification

This directory contains the modularized version of the original monolithic notebook, broken down into logical components for better maintainability and reproducibility.

## Notebook Structure

### 🔄 **Sequential Notebooks** (Must run in order)

1. **01_data_exploration.ipynb** - Data Loading & Initial Analysis
   - Load raw datasets
   - Perform exploratory data analysis
   - Generate visualizations and statistics
   - Save enhanced datasets

2. **02_feature_engineering.ipynb** - Meta-Feature Extraction
   - Extract 10 engineered features (linguistic, syntactic, social, sentiment)
   - Feature correlation analysis
   - Feature normalization and scaling
   - Save datasets with meta-features

3. **03_text_preprocessing.ipynb** - Text Cleaning & Processing
   - Advanced text preprocessing pipeline
   - URL removal, mention cleaning, hashtag processing
   - Optional spell correction
   - Save cleaned datasets for model training

4. **04_model_training.ipynb** - Model Architecture & Training
   - Build hybrid DistilBERT + meta-features model
   - Set up training pipeline
   - Train model with validation monitoring
   - Save trained model weights

5. **05_evaluation_prediction.ipynb** - Evaluation & Submission
   - Evaluate model performance
   - Generate test predictions
   - Create submission file
   - Comprehensive performance analysis

## Directory Structure

```
disaster-tweet-classification/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_prediction.ipynb
│   └── README.md
├── Data/                          # Raw and processed data
├── models/                        # Trained model weights
├── config/
│   └── hyperparameters.json       # Configuration file
├── results/
│   ├── metrics/                   # Performance metrics
│   └── visualizations/           # Generated plots
└── disaster_tweets_nlp.ipynb      # Original monolithic notebook
```

## Running the Pipeline

### Sequential Execution (Recommended)

Run notebooks in numerical order:

```bash
# 1. Start with data exploration
jupyter notebook 01_data_exploration.ipynb

# 2. After completion, run feature engineering
jupyter notebook 02_feature_engineering.ipynb

# 3. Continue with text preprocessing
jupyter notebook 03_text_preprocessing.ipynb

# 4. Train the model
jupyter notebook 04_model_training.ipynb

# 5. Generate final predictions
jupyter notebook 05_evaluation_prediction.ipynb
```

### Independent Execution

Some notebooks can be run independently for specific tasks:

- **01_data_exploration.ipynb** - Can run standalone for data analysis
- **02_feature_engineering.ipynb** - Can run if you have raw data
- **03_text_preprocessing.ipynb** - Depends on output from notebook 02
- **04_model_training.ipynb** - Depends on output from notebook 03
- **05_evaluation_prediction.ipynb** - Depends on output from notebook 04

## Configuration

All notebooks use the shared configuration file `config/hyperparameters.json` which contains:

- Data paths
- Model hyperparameters
- Feature engineering settings
- Training configuration

Modify this file to change experiment parameters across all notebooks.

## Dependencies

Make sure you have all required dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy matplotlib seaborn
pip install nltk vaderSentiment
pip install symspellpy  # Optional for spell correction
```

## Data Dependencies

### Input Files
- `Data/train.csv` - Raw training data
- `Data/test.csv` - Raw test data

### Generated Files
- `Data/train_enhanced.csv` - Enhanced training data
- `Data/test_enhanced.csv` - Enhanced test data
- `Data/train_with_meta.csv` - Training data with meta-features
- `Data/test_with_meta.csv` - Test data with meta-features
- `Data/train_cleaned.csv` - Final cleaned training data
- `Data/test_cleaned.csv` - Final cleaned test data
- `Data/submission.csv` - Final predictions

## Advantages of Modular Approach

1. **Maintainability**: Easier to update individual components
2. **Debugging**: Isolate issues to specific notebooks
3. **Experimentation**: Modify specific steps without affecting others
4. **Reproducibility**: Clear data flow and dependencies
5. **Collaboration**: Team members can work on different notebooks
6. **Resource Management**: Less memory usage per notebook
7. **Version Control**: Easier to track changes in specific components

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure previous notebooks have completed successfully
2. **Memory issues**: Restart kernel between notebooks
3. **Import errors**: Install all required dependencies
4. **Configuration errors**: Check `hyperparameters.json` file

### Recovery

If a notebook fails, you can:
1. Fix the issue in the current notebook
2. Re-run from the beginning of that notebook
3. Continue with the next notebook

## Future Enhancements

The modular structure allows for easy addition of:

- New feature engineering notebooks
- Alternative model architectures
- Ensemble methods
- Advanced evaluation techniques
- Real-time inference notebooks

## Original vs Modular

| Aspect | Original | Modular |
|--------|----------|---------|
| Size | 2.8MB monolithic | ~500KB each (5 notebooks) |
| Loading | Slow startup | Fast loading |
| Memory | High usage | Lower usage |
| Debugging | Difficult | Easy |
| Updates | Risky | Safe |
| Collaboration | Challenging | Easy |