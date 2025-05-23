# natural-language-processing

1. list of the keywords used
2. variation of the keywords and the spelling errors
3. location
4. sentimental score

5. clean some data to get a better look
6. EDA!!
7. sort out tweets by location and look at clusters of key words

## Here's a breakdown of the main components of Keras_NLP_Starter

Project Goal:
Build a model that predicts whether a tweet is about a real disaster (1) or not (0)
Uses a dataset of 10,000 hand-classified tweets
Model Choice:
Uses DistilBERT, which is a lighter version of BERT (Bidirectional Encoder Representations from Transformers)
DistilBERT is 40% smaller and 60% faster than BERT while maintaining 97% of its performance
Pre-trained on large text datasets and fine-tuned for this specific task
Key Steps in the Notebook:
a) Data Loading & Exploration:

Loads train and test data from CSV files
Each tweet has an ID, keyword, location, text, and target label (1 for disaster, 0 for not)
Analyzes tweet lengths and basic statistics
b) Data Preprocessing:

Splits training data into train/validation sets (80%/20%)
Tokenizes the text using DistilBERT's tokenizer
Converts data into PyTorch datasets and dataloaders
c) Model Setup:

Initializes DistilBERT model with a classification head
Sets up on GPU if available (falls back to CPU if not)
Uses CrossEntropyLoss for binary classification
Uses AdamW optimizer (standard for transformers)
d) Training Loop:

Runs for 2 epochs
For each batch:
Moves data to GPU/CPU
Computes model predictions
Calculates loss
Updates model weights
Tracks and prints average loss per epoch
e) Evaluation:

Includes helper functions to:
Get model predictions
Create confusion matrices
Calculate F1 scores
The notebook is using modern NLP practices:

Using transformers (state-of-the-art for NLP)
Proper data splitting for validation
Batch processing for efficiency
GPU support for faster training
Evaluation metrics appropriate for classification