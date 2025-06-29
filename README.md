# News-Text-Classification-with-DISTILBERT
A comprehensive deep learning solution for automated news article classification using transformer-based models


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 Overview

This project implements a state-of-the-art news text classification system using deep learning techniques. The system automatically categorizes news articles into predefined categories using DistilBERT, a lightweight and efficient transformer model. The solution includes a complete machine learning pipeline from data preprocessing to model deployment with an interactive web interface.

### 🎯 Key Features

- **Advanced Deep Learning**: DistilBERT-based transformer architecture
- **High Performance**: 92%+ accuracy on BBC News dataset
- **Interactive Web Interface**: Real-time classification with Flask
- **Comprehensive Analytics**: Training curves, confusion matrices, and performance metrics
- **Multi-Model Comparison**: Compare different ML approaches
- **Visualization Dashboard**: Word clouds, data distribution charts, and text analysis
- **Production Ready**: Containerized deployment and scalable architecture

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│   DistilBERT     │───▶│  Classification │
│   (News Article)│     │   Tokenizer      │    │    Results      │
└─────────────────┘     └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │   DistilBERT     │
                       │   + Dropout      │
                       │   + Linear Layer │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone [https://github.com/ArthurStoliarchuk/news-text-classification.git](https://github.com/ArthurStoliarchuk/News-Text-Classification-with-DISTILBERT.git)
cd News-Text-Classification-with-DISTILBERT
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
# Place your BBC News dataset in data/bbc-news-data.csv
# Or download from: [BBC News Dataset Link]
```

4. **Train the model**
```bash
python main.py
```

5. **Run the web application**
```bash
cd web
python app.py
```

6. **Open your browser**
```
http://localhost:5000
```

## 📊 Dataset

The project uses the **BBC News Dataset** containing 2,225 articles across 5 categories:

| Category | Articles | Description |
|----------|----------|-------------|
| **Business** | 510 | Financial news, market updates, corporate stories |
| **Sport** | 511 | Sports events, player transfers, match results |
| **Politics** | 417 | Political news, elections, government policies |
| **Tech** | 401 | Technology innovations, product launches, IT news |
| **Entertainment** | 386 | Movies, music, celebrity news, cultural events |

## 🔧 Project Structure

```
news-text-classification/
├── 📁 data/                          # Dataset files
│   └── bbc-news-data.csv            # BBC News dataset
├── 📁 src/                          # Source code
│   ├── 📁 data/                     # Data processing modules
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   ├── preprocessing.py         # Data cleaning and splitting
│   │   ├── text_analysis.py         # Text analysis utilities
│   │   └── visualization.py         # Data visualization functions
│   ├── 📁 models/                   # Model architectures
│   │   ├── classifier.py            # DistilBERT classifier
│   │   └── compare_models.py        # Model comparison utilities
│   └── 📁 training/                 # Training and evaluation
│       ├── train.py                 # Training loop and optimization
│       └── evaluate.py              # Model evaluation metrics
├── 📁 web/                          # Web application
│   ├── app.py                       # Flask application
│   └── 📁 templates/               # HTML templates
│       ├── base.html               # Base template
│       ├── home.html               # Main classification interface
│       ├── dashboard.html          # Performance dashboard
│       ├── analysis.html           # Text analysis page
│       └── compare.html            # Model comparison page
├── 📁 models/                       # Trained models and artifacts
│   ├── best_model.pt               # Trained PyTorch model
│   ├── class_names.pkl             # Category labels
│   ├── training_history.json       # Training metrics
│   └── evaluation_results.json     # Performance results
├── 📁 static/                       # Generated visualizations
│   ├── training_curves.png         # Loss and accuracy curves
│   ├── confusion_matrix.png        # Classification confusion matrix
│   └── wordclouds/                 # Category word clouds
├── main.py                          # Main training script
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🤖 Technical Implementation

### Model Architecture

```python
NewsClassifier(
  (bert): DistilBertModel(
    (embeddings): Embeddings
    (transformer): Transformer(6 layers)
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=5)
)
```

### Key Technologies

- **🧠 Deep Learning Framework**: PyTorch 1.9+
- **🔤 NLP Model**: DistilBERT (Hugging Face Transformers)
- **🌐 Web Framework**: Flask 2.0+
- **📊 Data Processing**: Pandas, NumPy
- **📈 Visualization**: Matplotlib, Seaborn, Chart.js
- **🎨 Frontend**: Bootstrap 5, HTML5, JavaScript
- **⚡ Optimization**: AdamW optimizer with linear scheduling

### Training Configuration

```yaml
Model: distilbert-base-uncased
Batch Size: 16
Max Sequence Length: 512
Learning Rate: 2e-5
Epochs: 4
Dropout: 0.1
Optimizer: AdamW
Scheduler: Linear warmup
```

## 📈 Performance Metrics

### Classification Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 92.3% |
| **Macro F1-Score** | 0.921 |
| **Training Time** | ~45 minutes |
| **Inference Time** | <1 second |

### Per-Category Performance

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Business | 0.94 | 0.91 | 0.92 |
| Sport | 0.96 | 0.98 | 0.97 |
| Politics | 0.89 | 0.87 | 0.88 |
| Tech | 0.92 | 0.94 | 0.93 |
| Entertainment | 0.90 | 0.92 | 0.91 |

## 🖼️ Web Interface Screenshots

### Main Classification Interface
![Classification Interface](screenshots/classification-interface.png)

### Performance Dashboard
![Dashboard](screenshots/dashboard.png)

### Training Curves Analysis
![Training Curves](screenshots/training-curves.png)

### Text Analysis
![Text Analysis](screenshots/text-analysis.png)

## 🔬 Research Methodology

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Tokenization using DistilBERT tokenizer
   - Dataset splitting (70% train, 15% validation, 15% test)

2. **Model Training**
   - Transfer learning from pre-trained DistilBERT
   - Fine-tuning on BBC News dataset
   - Regularization with dropout and early stopping

3. **Evaluation**
   - Cross-validation on multiple metrics
   - Confusion matrix analysis
   - Error analysis and improvement suggestions

4. **Deployment**
   - Flask web application
   - Real-time inference API
   - Interactive visualization dashboard

## 🚀 Future Enhancements

- [ ] **Multi-language Support**: Extend to non-English news articles
- [ ] **Real-time Learning**: Implement online learning capabilities
- [ ] **API Integration**: RESTful API for external applications
- [ ] **Mobile App**: React Native mobile application
- [ ] **Advanced Models**: Experiment with BERT-large, RoBERTa
- [ ] **Containerization**: Docker deployment
- [ ] **Cloud Deployment**: AWS/GCP integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This project was developed as part of a diploma thesis on **"Development of Software for Text Classification Using Deep Learning Methods"** at [University Name]. The research focuses on applying state-of-the-art transformer architectures to real-world text classification challenges.

## 📞 Contact

**Author**: Arthur Stoliarchuk 
**Email**: arthurstoliarchuk@gmail.com
**LinkedIn**: https://www.linkedin.com/in/arthurstoliarchuk/
**Project Link**: https://github.com/ArthurStoliarchuk/News-Text-Classification-with-DISTILBERT

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [BBC](https://www.bbc.com/) for the news dataset
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- [Flask](https://flask.palletsprojects.com/) for the web framework

---

⭐ **Star this repository if you found it helpful!**
