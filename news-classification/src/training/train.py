# src/training/train.py
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def train_model(model, train_dataloader, val_dataloader, epochs=4, lr=2e-5, device='cuda'):
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Lists to store training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
    
    # Create training curves plots
    create_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save training history for later visualization
    import json
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/training_history.json', 'w') as f:
        json.dump(training_history, f)
    
    print("Training history saved to models/training_history.json")
            
    return model

def create_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Creates training curves showing loss and accuracy over epochs
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Training and Validation Loss', fontsize=16, pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Add value annotations on the last point
    ax1.annotate(f'{train_losses[-1]:.3f}', 
                xy=(len(train_losses), train_losses[-1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.annotate(f'{val_losses[-1]:.3f}', 
                xy=(len(val_losses), val_losses[-1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Plot Accuracy curves
    ax2.plot(epochs, [acc * 100 for acc in train_accuracies], 'b-', 
             label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(epochs, [acc * 100 for acc in val_accuracies], 'r-', 
             label='Validation Accuracy', linewidth=2, marker='s')
    ax2.set_title('Training and Validation Accuracy', fontsize=16, pad=20)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add value annotations on the last point
    ax2.annotate(f'{train_accuracies[-1]*100:.1f}%', 
                xy=(len(train_accuracies), train_accuracies[-1]*100), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax2.annotate(f'{val_accuracies[-1]*100:.1f}%', 
                xy=(len(val_accuracies), val_accuracies[-1]*100), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Save the plot
    plt.savefig('static/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training curves saved to static/training_curves.png")