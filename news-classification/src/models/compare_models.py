# src/models/compare_models.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.data.dataset import NewsDataset
from src.models.classifier import NewsClassifier

def train_and_evaluate_transformer(model_name, train_dataloader, val_dataloader, test_dataloader, 
                                  num_classes, epochs=3, device='cuda'):
    """Train and evaluate a transformer model"""
    start_time = time.time()
    
    # Initialize model
    model = NewsClassifier(model_name, num_classes)
    model.to(device)
    
    # Training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    training_time = time.time() - start_time
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time
    }

def train_and_evaluate_traditional(model_name, train_texts, train_labels, test_texts, test_labels):
    """Train and evaluate a traditional ML model"""
    start_time = time.time()
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Initialize model
    if model_name == 'naive_bayes':
        model = MultinomialNB()
    elif model_name == 'svm':
        model = LinearSVC()
    
    # Train
    model.fit(X_train, train_labels)
    
    # Predict
    preds = model.predict(X_test)
    
    training_time = time.time() - start_time
    accuracy = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='weighted')
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time
    }

def compare_models(train_df, test_df, class_names):
    """Compare different models and return results"""
    # Prepare data for transformer models
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create label mapping
    label_dict = {label: i for i, label in enumerate(class_names)}
    
    # Create datasets
    train_dataset = NewsDataset(
        texts=train_df['text'].tolist(),
        labels=[label_dict[label] for label in train_df['category']],
        tokenizer=tokenizer
    )
    
    test_dataset = NewsDataset(
        texts=test_df['text'].tolist(),
        labels=[label_dict[label] for label in test_df['category']],
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    
    # Model comparisons
    results = []
    
    # DistilBERT
    distilbert_results = train_and_evaluate_transformer(
        "distilbert-base-uncased",
        train_dataloader,
        test_dataloader,
        test_dataloader,
        len(class_names),
        epochs=2,
        device=device
    )
    results.append(distilbert_results)
    
    # BERT
    bert_results = train_and_evaluate_transformer(
        "bert-base-uncased",
        train_dataloader,
        test_dataloader,
        test_dataloader,
        len(class_names),
        epochs=2,
        device=device
    )
    results.append(bert_results)
    
    # Traditional models with raw data
    train_texts = train_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    train_labels = [label_dict[label] for label in train_df['category']]
    test_labels = [label_dict[label] for label in test_df['category']]
    
    # Naive Bayes
    nb_results = train_and_evaluate_traditional(
        'naive_bayes',
        train_texts,
        train_labels,
        test_texts,
        test_labels
    )
    results.append(nb_results)
    
    # SVM
    svm_results = train_and_evaluate_traditional(
        'svm',
        train_texts,
        train_labels,
        test_texts,
        test_labels
    )
    results.append(svm_results)
    
    # Plot results
    plot_comparison_results(results)
    
    return results

def plot_comparison_results(results):
    """Plot model comparison results"""
    df = pd.DataFrame(results)
    
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(df['model_name'], df['accuracy'], color='blue')
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Training time comparison
    plt.subplot(1, 2, 2)
    plt.bar(df['model_name'], df['training_time'], color='green')
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('static/model_comparison.png')
    
    # F1 Score comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['model_name'], df['f1_score'], color='orange')
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/f1_comparison.png')