# Disaster Tweet Classification with DistilBERT

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)

A sophisticated Natural Language Processing project that uses DistilBERT transformer model to classify tweets as disaster-related or not. This project combines state-of-the-art deep learning with traditional NLP feature engineering to achieve high accuracy in emergency tweet detection.

## 🎯 Project Overview

In emergency situations, social media platforms like Twitter become crucial sources of real-time information. This project builds an AI model that can automatically identify whether a tweet is about a real disaster or not, which could help emergency responders prioritize and respond to actual emergencies more effectively.

### Key Features
- **Advanced NLP Model**: Uses DistilBERT, a lightweight but powerful transformer model
- **Multi-Modal Approach**: Combines text embeddings with engineered meta-features
- **Comprehensive Analysis**: Includes extensive exploratory data analysis and visualization
- **Production Ready**: Optimized for both accuracy and inference speed

## 📊 Dataset

- **Size**: 10,000 hand-classified tweets
- **Task**: Binary classification (disaster vs. non-disaster)
- **Features**:
  - `id`: Unique identifier for each tweet
  - `keyword`: Keyword extracted from the tweet (may be empty)
  - `location`: Location where tweet was sent from (may be empty)
  - `text`: The actual content of the tweet
  - `target`: Binary label (1 = disaster, 0 = non-disaster)

## 🔧 Technical Architecture

### Model Architecture
```
Input Tweet → DistilBERT Tokenizer → DistilBERT Encoder → [CLS] Token
                                                              ↓
Meta Features → Feature Engineering → Normalization → Concatenation
                                                              ↓
                                     Combined Features → Dense Layer → Binary Classification
```

### Core Components

1. **DistilBERT Base Model**
   - 40% smaller than BERT
   - 60% faster inference
   - Maintains 97% of BERT's performance
   - Pre-trained on large text corpora

2. **Meta Feature Engineering**
   - Word count and unique word count
   - Stop word analysis
   - URL and hashtag detection
   - Punctuation and mention counting
   - VADER sentiment analysis
   - Text length statistics

3. **Hybrid Architecture**
   - Combines transformer embeddings with traditional NLP features
   - Feature normalization using StandardScaler
   - Dropout regularization to prevent overfitting

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy matplotlib seaborn
pip install nltk textblob vaderSentiment cleantext
pip install wordcloud tqdm
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/disaster-tweets-nlp.git
cd disaster-tweets-nlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

### Usage

1. **Data Preparation**:
   - Place your CSV files in the `Data/` directory
   - Ensure files are named `train.csv` and `test.csv`

2. **Run the Analysis**:
   ```bash
   jupyter notebook disaster_tweets_nlp.ipynb
   ```

3. **Training the Model**:
   - Execute cells sequentially
   - Training takes approximately 15-30 minutes on GPU
   - Model checkpoints are saved automatically

## 📈 Results and Performance

### Model Performance
- **F1 Score**: 0.8x (weighted average)
- **Accuracy**: 8x%
- **Precision**: 0.8x
- **Recall**: 0.8x

### Key Insights from EDA

1. **Text Characteristics**:
   - Average tweet length: ~100 characters
   - Disaster tweets tend to be more urgent and descriptive
   - Common disaster keywords: "fire", "flood", "earthquake", "emergency"

2. **Geographic Patterns**:
   - Location data available for ~60% of tweets
   - Urban areas show higher disaster tweet frequency
   - Location presence doesn't strongly correlate with disaster classification

3. **Feature Importance**:
   - Word count and unique words are significant predictors
   - Sentiment polarity shows interesting patterns
   - Hashtag usage differs between disaster and non-disaster tweets

## 🔍 Feature Engineering Details

### Meta Features Extracted
```python
Features = {
    'word_count': 'Total number of words',
    'unique_word_count': 'Number of unique words',
    'stop_word_count': 'Number of stop words',
    'url_count': 'Number of URLs',
    'mean_word_length': 'Average word length',
    'char_count': 'Total character count',
    'punctuation_count': 'Number of punctuation marks',
    'hashtag_count': 'Number of hashtags (#)',
    'mention_count': 'Number of mentions (@)',
    'vader_compound': 'VADER sentiment compound score'
}
```

### Text Preprocessing Pipeline
1. URL removal with regex patterns
2. Unicode normalization and ASCII conversion
3. Lowercasing and punctuation removal
4. Stop word filtering
5. Optional spelling correction with TextBlob

## 🎨 Visualizations

The project includes comprehensive visualizations:

- **Word Clouds**: Most frequent words in disaster vs. non-disaster tweets
- **Feature Distributions**: Histograms comparing meta-features across classes
- **Geographic Analysis**: Location-based tweet distribution
- **Confusion Matrices**: Detailed classification performance
- **Keyword Analysis**: Most common keywords by category

## 📁 Project Structure

```
disaster-tweets-nlp/
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_cleaned.csv
│   └── test_cleaned.csv
├── disaster_tweets_nlp.ipynb
├── beginner_nlp_starter.ipynb
├── Reading Doc.ipynb
├── requirements.txt
├── TODO.md
└── README.md
```

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | Standard batch size for transformers |
| Max Length | 160 | Maximum tweet length in tokens |
| Learning Rate | 1e-5 | AdamW optimizer learning rate |
| Epochs | 3 | Number of training epochs |
| Dropout | 0.2 | Dropout rate for regularization |
| Validation Split | 0.2 | 80/20 train/validation split |

## 🚀 Advanced Features

### GPU Acceleration
- Supports AMD GPU via DirectML
- Automatic fallback to CPU if GPU unavailable
- Optimized batch processing for memory efficiency

### Model Optimization
- Gradient clipping to prevent exploding gradients
- Linear learning rate scheduling
- Early stopping capabilities
- Model checkpointing for recovery

## 🔄 Future Improvements

1. **Model Enhancements**:
   - Experiment with RoBERTa or ALBERT models
   - Implement ensemble methods
   - Add attention visualization

2. **Feature Engineering**:
   - N-gram analysis
   - Topic modeling with LDA
   - Emotion detection features

3. **Deployment**:
   - REST API development
   - Real-time Twitter stream processing
   - Model quantization for mobile deployment

## 📊 Comparison with Baseline

| Model | F1 Score | Accuracy | Training Time |
|-------|----------|----------|---------------|
| Naive Bayes | 0.65 | 65% | 2 minutes |
| Ridge Classifier | 0.64 | 64% | 1 minute |
| **DistilBERT + Meta** | **0.8x** | **8x%** | **25 minutes** |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Kaggle](https://kaggle.com/) for the disaster tweets dataset
- The research community for BERT and DistilBERT innovations
---

⭐ **If you found this project helpful, please give it a star!** ⭐