# main.py
import torch
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.preprocessing import load_bbc_news_dataset
from src.data.dataset import NewsDataset
from src.models.classifier import NewsClassifier
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.data.text_analysis import analyze_text_length, generate_wordclouds, get_top_words

def create_training_curves_from_history():
    """
    Creates training curves from saved training history
    """
    history_path = 'models/training_history.json'
    if not os.path.exists(history_path):
        print("Warning: No training history found. Model needs to be trained first.")
        return
    
    # Load actual training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accuracies = history['train_accuracies']
    val_accuracies = history['val_accuracies']
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=6)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=6)
    ax1.set_title('Training and Validation Loss', fontsize=16, pad=20, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Add value annotations on the last point
    ax1.annotate(f'{train_losses[-1]:.3f}', 
                xy=(len(train_losses), train_losses[-1]), 
                xytext=(10, 10), textcoords='offset points', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax1.annotate(f'{val_losses[-1]:.3f}', 
                xy=(len(val_losses), val_losses[-1]), 
                xytext=(10, -15), textcoords='offset points', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Plot Accuracy curves
    ax2.plot(epochs, [acc * 100 for acc in train_accuracies], 'b-', 
             label='Training Accuracy', linewidth=2, marker='o', markersize=6)
    ax2.plot(epochs, [acc * 100 for acc in val_accuracies], 'r-', 
             label='Validation Accuracy', linewidth=2, marker='s', markersize=6)
    ax2.set_title('Training and Validation Accuracy', fontsize=16, pad=20, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(50, 100)  # Better range for accuracy
    
    # Add value annotations on the last point
    ax2.annotate(f'{train_accuracies[-1]*100:.1f}%', 
                xy=(len(train_accuracies), train_accuracies[-1]*100), 
                xytext=(10, 10), textcoords='offset points', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax2.annotate(f'{val_accuracies[-1]*100:.1f}%', 
                xy=(len(val_accuracies), val_accuracies[-1]*100), 
                xytext=(10, -15), textcoords='offset points', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Save the plot with high quality
    plt.savefig('static/training_curves.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("Training curves saved to static/training_curves.png")

def main():
    # Configuration
    model_name = "distilbert-base-uncased"
    batch_size = 16
    max_length = 512
    epochs = 4
    learning_rate = 2e-5
    data_path = "data/bbc-news-data.csv"
    
    # Create needed directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_bbc_news_dataset(data_path)
    
    # Get class names and mapping
    class_names = train_df['category'].unique().tolist()
    label_dict = {label: i for i, label in enumerate(class_names)}
    
    # Save class names for the web app
    with open('models/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("Creating datasets...")
    # Convert labels to numeric
    train_labels = [label_dict[label] for label in train_df['category']]
    val_labels = [label_dict[label] for label in val_df['category']]
    test_labels = [label_dict[label] for label in test_df['category']]
    
    train_dataset = NewsDataset(train_df['text'].tolist(), train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_df['text'].tolist(), val_labels, tokenizer, max_length)
    test_dataset = NewsDataset(test_df['text'].tolist(), test_labels, tokenizer, max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = NewsClassifier(model_name, len(class_names), dropout=0.1)
    
    # Train model
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trained_model = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        epochs=epochs, 
        lr=learning_rate, 
        device=device
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(trained_model, test_dataloader, device, class_names)
    
    # Save evaluation results
    with open('models/evaluation_results.json', 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report']
        }, f, indent=2)
    
    print(f"Training completed! Final accuracy: {results['accuracy']:.3f}")
    
    #disable since visualisation doesn´t work: 

    # Generate dataset statistics and visualizations
    #print("Generating dataset statistics and visualizations...")
    
    # Class distribution
    #class_distribution = train_df['category'].value_counts().to_dict()
    #with open('models/class_distribution.json', 'w') as f:
    #    json.dump(class_distribution, f)
        
    # Text length analysis
    #length_stats = analyze_text_length(train_df)
    #with open('models/length_stats.json', 'w') as f:
    #    json.dump(length_stats, f)
    
    # Generate wordclouds
    #generate_wordclouds(train_df)
    
    # Get top words
    #top_words = get_top_words(train_df)
    #with open('models/top_words.json', 'w') as f:
    #    json.dump(top_words, f)
    
    # После обучения создать графики с реальными данными
    #print("Creating training curves with real data...")
    #create_training_curves_from_history()

if __name__ == "__main__":
    main()