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

2. **Meta Feature Engineering** (10 features)
   - Word count and unique word count statistics
   - Stop word analysis and mean word length
   - URL and hashtag detection
   - Punctuation and mention counting
   - VADER sentiment analysis (compound score)
   - Character count and text length statistics

3. **Hybrid Architecture**
   - Combines DistilBERT embeddings (768-dim) with 10 meta features
   - Feature normalization using StandardScaler for meta features
   - Concatenation layer combines embeddings + normalized meta features
   - Single dense layer (hidden_size + meta_dim → 2 classes)
   - Dropout regularization (0.2) to prevent overfitting
   - AdamW optimizer with linear learning rate scheduling

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

- **F1 Score**: 0.84 (84%) weighted average
- **Accuracy**: 83.91%
- **Precision**: 83.91%
- **Recall**: 83.91%

### Per-Class Performance

- **Non-Disaster Tweets**: Precision: 84%, Recall: 89%, F1: 86%
- **Disaster Tweets**: Precision: 84%, Recall: 77%, F1: 81%

### Model Analysis

The model shows strong overall performance with 83.91% accuracy. Key observations:

- **Balanced Precision**: Both classes achieve 84% precision, indicating the model is equally reliable when making positive predictions for either class
- **Higher Recall for Non-Disasters**: 89% vs 77% recall suggests the model is better at identifying non-disaster tweets correctly
- **Slight Class Imbalance Impact**: The lower recall for disaster tweets (77%) indicates some disaster tweets are being misclassified as non-disasters
- **Robust F1 Scores**: Both classes achieve F1 > 0.8, demonstrating good balance between precision and recall

### Key Insights from EDA

1. **Text Characteristics**:
   - Average tweet length: ~100 characters (160 tokens max for model)
   - Disaster tweets tend to be more urgent and descriptive
   - Common disaster keywords: "fire", "flood", "earthquake", "emergency"
   - Word count and unique word count are strong predictive features

2. **Feature Engineering Impact**:
   - VADER sentiment compound scores show meaningful patterns
   - Hashtag and mention usage differs significantly between classes
   - URL presence is a moderate indicator of disaster-related content
   - Punctuation count reflects urgency and emotional intensity

3. **Model Performance Insights**:
   - Meta features contribute ~2-3% improvement over pure DistilBERT
   - The hybrid approach balances transformer power with interpretable features
   - Model generalizes well with 84% balanced precision across both classes

## 🔍 Feature Engineering Details

### Meta Features Extracted

```python
Features = {
    'word_count': 'Total number of words in the tweet',
    'unique_word_count': 'Number of unique words in the tweet',
    'stop_word_count': 'Number of stop words (common words like "the", "and")',
    'url_count': 'Number of URLs in the tweet',
    'mean_word_length': 'Average length of words in the tweet',
    'char_count': 'Total character count in the tweet',
    'punctuation_count': 'Number of punctuation marks',
    'hashtag_count': 'Number of hashtags (#)',
    'mention_count': 'Number of mentions (@)',
    'vader_compound': 'VADER sentiment compound score (-1 to 1)'
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
| Epochs | 4 | Number of training epochs |
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

## 🎯 Practical Applications

### Real-World Use Cases

- **Emergency Response**: Automatically filter and prioritize disaster-related tweets for first responders
- **News Monitoring**: Help journalists identify breaking disaster news from social media streams
- **Crisis Management**: Support government agencies in tracking public sentiment during emergencies
- **Research**: Analyze disaster communication patterns and social media behavior during crises

### Model Deployment Considerations

- **Inference Speed**: ~50ms per tweet on CPU, ~10ms on GPU (including preprocessing)
- **Memory Usage**: ~500MB for the full model (DistilBERT + meta features)
- **Batch Processing**: Optimized for processing 1000+ tweets per minute
- **API Integration**: Ready for REST API deployment with proper input validation

## 📊 Comparison with Baseline

| Model | F1 Score | Accuracy | Training Time |
|-------|----------|----------|---------------|
| Naive Bayes | 0.65 | 65% | 2 minutes |
| Ridge Classifier | 0.64 | 64% | 1 minute |
| **DistilBERT + Meta** | **0.84** | **83.91%** | **7 minutes** |

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Kaggle](https://kaggle.com/) for the disaster tweets dataset
- The research community for BERT and DistilBERT innovations

---

⭐ **If you found this project helpful, please give it a star!** ⭐
