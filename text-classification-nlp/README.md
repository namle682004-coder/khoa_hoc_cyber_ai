README.md
# TODO: Bài tập về nhà - Text Classification with NLP

## Mục tiêu
Xây dựng một hệ thống phân loại văn bản (text classification) hoàn chỉnh, so sánh 3 approaches: TF-IDF + ML, LSTM, và Transformer (BERT), bao gồm training, inference và Gradio demo.

## Yêu cầu

### 1. Cấu trúc Project
Tạo folder `text-classification-nlp/` với cấu trúc:

```
text-classification-nlp/
├── README.md
├── requirements.txt
├── config.py
├── models.py           # Định nghĩa TF-IDF, LSTM, BERT models
├── train.py            # Training script cho cả 3 models
├── inference.py        # Inference script
├── gradio_app.py       # Gradio demo
├── utils.py            # Text preprocessing, visualization
├── checkpoints/
│   ├── tfidf_model.pkl
│   ├── lstm_model.pth
│   └── bert_model.pth
└── samples/
    └── test_sentences.txt
```

### 2. Dataset
Chọn một trong các dataset sau:
- **IMDB Movie Reviews** (sentiment analysis - 2 classes: positive/negative)
- **AG News** (topic classification - 4 classes: World, Sports, Business, Sci/Tech)
- **Twitter Sentiment** (3 classes: positive, negative, neutral)
- **20 Newsgroups** (multi-class topic classification)

**Hoặc** tạo custom dataset cho domain của bạn (customer reviews, support tickets, etc.)

### 3. Các File Cần Implement

#### 3.1. `config.py`
```python
# TODO: Tạo Config class

class Config:
    # Dataset settings
    DATASET_NAME = ...
    DATA_PATH = ...
    NUM_CLASSES = ...
    CLASS_NAMES = [...]
    
    # Text preprocessing
    MAX_VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 128
    MIN_WORD_FREQ = 2
    
    # TF-IDF settings
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    
    # LSTM settings
    LSTM_EMBEDDING_DIM = 128
    LSTM_HIDDEN_DIM = 64
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # BERT settings
    BERT_MODEL_NAME = 'distilbert-base-uncased'
    BERT_MAX_LENGTH = 128
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    # Checkpointing
    CHECKPOINT_DIR = './checkpoints'
    
    # TODO: Add more configurations
```

#### 3.2. `models.py`
```python
# TODO: Định nghĩa 3 model architectures

class TFIDFClassifier:
    """
    TODO: Wrapper cho TF-IDF + Logistic Regression
    """
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        # TODO: Initialize TfidfVectorizer và LogisticRegression
        pass
    
    def fit(self, X_train, y_train):
        """TODO: Train TF-IDF + classifier"""
        pass
    
    def predict(self, X_test):
        """TODO: Predict classes"""
        pass
    
    def predict_proba(self, X_test):
        """TODO: Predict probabilities"""
        pass


class LSTMClassifier(nn.Module):
    """
    TODO: LSTM model cho text classification
    Architecture:
    - Embedding layer
    - LSTM layers (bidirectional optional)
    - Dropout
    - Fully connected layer
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_classes, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        # TODO: Define layers
        pass
    
    def forward(self, x):
        """TODO: Forward pass"""
        pass


class BERTClassifier:
    """
    TODO: Wrapper cho DistilBERT fine-tuning
    """
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        # TODO: Load pretrained BERT và add classification head
        pass
    
    def train(self, train_loader, val_loader, epochs=3):
        """TODO: Fine-tune BERT"""
        pass
    
    def predict(self, texts):
        """TODO: Predict with BERT"""
        pass
```

#### 3.3. `utils.py`
```python
# TODO: Text preprocessing và helper functions

def clean_text(text):
    """
    TODO: Clean text
    - Lowercase
    - Remove URLs, mentions, hashtags
    - Remove special characters
    - Remove extra whitespace
    """
    pass

def tokenize_texts(texts, tokenizer_type='basic'):
    """
    TODO: Tokenize texts
    - For TF-IDF: return cleaned strings
    - For LSTM: return sequences of token IDs
    - For BERT: return BERT token IDs + attention masks
    """
    pass

def build_vocab(texts, max_vocab_size=10000, min_freq=2):
    """TODO: Build vocabulary from texts"""
    pass

def plot_training_history(history, model_name, save_path=None):
    """TODO: Plot loss và accuracy curves"""
    pass

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """TODO: Plot confusion matrix"""
    pass

def compare_models_performance(results_dict):
    """
    TODO: So sánh performance của 3 models
    Input: {'TF-IDF': {...}, 'LSTM': {...}, 'BERT': {...}}
    Plot bar charts cho accuracy, precision, recall, F1
    """
    pass
```

#### 3.4. `train.py`
```python
# TODO: Training script cho cả 3 models

def train_tfidf_model(X_train, y_train, X_val, y_val, config):
    """
    TODO:
    1. Initialize TF-IDF vectorizer
    2. Transform train và validation data
    3. Train Logistic Regression
    4. Evaluate and save model
    """
    pass

def train_lstm_model(train_loader, val_loader, config):
    """
    TODO:
    1. Initialize LSTM model
    2. Training loop với validation
    3. Save best checkpoint
    4. Plot training curves
    """
    pass

def train_bert_model(train_texts, train_labels, val_texts, val_labels, config):
    """
    TODO:
    1. Load pretrained BERT
    2. Tokenize texts
    3. Fine-tune với training loop
    4. Save fine-tuned model
    """
    pass

def main():
    """
    TODO: Main function
    1. Load và preprocess dataset
    2. Train/test split
    3. Train tất cả 3 models
    4. Compare results
    5. Save best model
    """
    pass

# TODO: Command line arguments:
# --dataset, --model (tfidf/lstm/bert/all), --epochs, --batch-size
```

**Key Points:**
- Train cả 3 models trên cùng train/test split
- Track metrics: accuracy, precision, recall, F1
- Save checkpoints cho từng model
- Compare training time và inference speed

#### 3.5. `inference.py`
```python
# TODO: Inference script

def load_model(model_type, checkpoint_path, config):
    """TODO: Load trained model (TF-IDF/LSTM/BERT)"""
    pass

def predict_single_text(model, text, model_type, config):
    """
    TODO:
    1. Preprocess text
    2. Run inference
    3. Return predicted class và probabilities
    """
    pass

def predict_batch(model, texts, model_type, config):
    """TODO: Batch prediction"""
    pass

def main():
    """
    TODO: Main inference function
    - Load model
    - Accept text input (stdin or file)
    - Print predictions với confidence scores
    - Support batch processing
    """
    pass

# TODO: Command line arguments:
# --model-type, --checkpoint, --text, --input-file
```

#### 3.6. `gradio_app.py`
```python
# TODO: Gradio interface

class TextClassifierApp:
    """TODO: Wrapper class cho all 3 models"""
    
    def __init__(self, tfidf_path, lstm_path, bert_path, config):
        """TODO: Load all models"""
        pass
    
    def predict_with_model(self, text, model_type):
        """TODO: Predict với model được chọn"""
        pass
    
    def predict_all_models(self, text):
        """TODO: Predict với cả 3 models và compare"""
        pass

def create_gradio_interface(app):
    """
    TODO: Create interface với:
    
    Tab 1: Single Model Prediction
    - Text input box
    - Dropdown để chọn model (TF-IDF/LSTM/BERT)
    - Output: predicted class + confidence scores
    - Bar chart cho class probabilities
    
    Tab 2: Compare Models
    - Text input box
    - Run prediction trên cả 3 models
    - Side-by-side comparison
    - Highlight differences
    - Show inference time
    
    Tab 3: Batch Prediction
    - Upload text file
    - Choose model
    - Download predictions as CSV
    """
    pass

# TODO: Launch app
```

### 4. README.md
TODO: Viết README với:
- Project overview
- Dataset description và stats
- Installation steps
- Usage examples (train, inference, demo)
- Model comparison results (table)
- Sample predictions
- Trade-offs analysis (speed vs accuracy)
- Challenges và solutions
- Future improvements

### 5. Requirements
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=4.0.0
tqdm>=4.65.0
nltk>=3.8.0  # For text preprocessing
# TODO: Add more
```

## Tiêu chí Đánh giá

### Cơ bản (7 điểm)
- [ ] Cả 3 models (TF-IDF, LSTM, BERT) train thành công
- [ ] Training scripts với proper checkpointing
- [ ] Inference script hoạt động
- [ ] README đầy đủ với results comparison
- [ ] Code có comments và docstrings
- [ ] Text preprocessing được implement đúng

### Nâng cao (3 điểm)
- [ ] Gradio app với full features (3 tabs)
- [ ] Comprehensive model comparison (accuracy, speed, memory)
- [ ] Training curves và confusion matrices
- [ ] Error analysis (examples where models fail)
- [ ] Hyperparameter tuning experiments
- [ ] Try ensemble methods (combine 3 models)
- [ ] Deploy app to Hugging Face Spaces

## Hints và Resources

### Load Dataset
```python
# IMDB
from datasets import load_dataset
dataset = load_dataset('imdb')

# AG News
dataset = load_dataset('ag_news')

# Twitter
# Download từ Kaggle hoặc tạo custom
```

### Text Preprocessing
```python
import re
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters
    # Remove stopwords (optional)
    return text.strip()
```

### BERT Fine-tuning
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=num_classes
)

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, 
                   max_length=128, return_tensors='pt')
```

### Model Comparison Table Example
```
| Model          | Accuracy | Train Time | Inference Time | Memory  |
|----------------|----------|------------|----------------|---------|
| TF-IDF + LR    | 87.5%    | 10s        | 0.5ms/sample   | 50 MB   |
| LSTM           | 89.2%    | 5 min      | 2ms/sample     | 200 MB  |
| DistilBERT     | 93.8%    | 15 min     | 10ms/sample    | 500 MB  |
```

### Common Issues
1. **OOM with BERT**: Giảm batch size hoặc max_length
2. **Overfitting LSTM**: Add dropout, reduce model complexity
3. **Poor TF-IDF results**: Try different ngram_range, increase max_features
4. **Slow BERT inference**: Use DistilBERT thay vì BERT base

## Tham khảo
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Scikit-learn Text Processing](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- Lab 1.4: `baselines_text_classification_nlp.ipynb`
- [PyTorch Text Classification Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

## Deadline
TODO: Thêm deadline

## Nộp bài
TODO: Instructions (GitHub repo link, zip file, Hugging Face Space link)

---

**Good luck! 📝**

*Tips: Bắt đầu với TF-IDF (đơn giản nhất), sau đó LSTM, cuối cùng BERT. So sánh results sau mỗi model.*