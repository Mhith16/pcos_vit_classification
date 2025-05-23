import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

from config import Config
from models import get_model
from utils import (get_data_loaders, calculate_metrics, plot_training_history, 
                   plot_confusion_matrix, save_results, EarlyStopping)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            prob = torch.softmax(output, dim=1)[:, 1]  # Probability of PCOS class
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics, all_labels, all_preds, all_probs

def train_model(model_type):
    """Train a specific model"""
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*50}")
    
    # Initialize model
    model = get_model(model_type, Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        Config.DATA_PATH, 
        Config.BATCH_SIZE, 
        val_split=0.2
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    print(f"Dataset size: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Device: {Config.DEVICE}")
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss, metrics, val_labels, val_preds, val_probs = validate_epoch(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(metrics['accuracy'])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {metrics['accuracy']:.4f}")
        print(f"Val AUC-ROC: {metrics['auc_roc']:.4f}, Val F1: {metrics['f1']:.4f}")
        
        # Save best model
        if metrics['accuracy'] > best_val_acc:
            best_val_acc = metrics['accuracy']
            os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'epoch': epoch
            }, os.path.join(Config.MODEL_SAVE_PATH, f'best_{model_type}.pth'))
            print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation
    print(f"\n{'='*30}")
    print("FINAL RESULTS")
    print(f"{'='*30}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(Config.MODEL_SAVE_PATH, f'best_{model_type}.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    _, final_metrics, final_labels, final_preds, final_probs = validate_epoch(
        model, val_loader, criterion, Config.DEVICE
    )
    
    # Print all metrics
    print("Final Validation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Generate visualizations
    plot_training_history(train_losses, val_losses, train_accs, val_accs, model_type)
    plot_confusion_matrix(final_labels, final_preds, model_type)
    
    # Save results
    results = {
        'model_type': model_type,
        'final_metrics': final_metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        },
        'best_epoch': checkpoint['epoch']
    }
    
    save_results(results, model_type, Config.RESULTS_SAVE_PATH)
    
    return final_metrics

def main():
    """Main training function"""
    print("PCOS Classification Training")
    print(f"Device: {Config.DEVICE}")
    
    # Train all models
    all_results = {}
    
    for model_type in Config.MODELS_TO_TRAIN:
        try:
            metrics = train_model(model_type)
            all_results[model_type] = metrics
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue
    
    # Compare all models
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results).T
        print(df.round(4))
        
        # Save comparison
        os.makedirs(Config.RESULTS_SAVE_PATH, exist_ok=True)
        df.to_csv(os.path.join(Config.RESULTS_SAVE_PATH, 'model_comparison.csv'))
        
        # Find best model
        best_model = max(all_results.keys(), key=lambda x: all_results[x]['auc_roc'])
        print(f"\nBest Model: {best_model} (AUC-ROC: {all_results[best_model]['auc_roc']:.4f})")

if __name__ == "__main__":
    main()