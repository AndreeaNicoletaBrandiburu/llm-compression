"""
Training script for Knowledge Distillation.
Teacher-student training on transformer models.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import build_bert_model, build_distilbert_model, count_parameters
from src.distillation import DistillationLoss, distill_step, evaluate_distillation


def prepare_imdb_data(tokenizer, max_length=128, batch_size=16):
    """Prepare IMDB dataset for sentiment analysis."""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)
    
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_distillation(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    epochs=3,
    lr=2e-5,
    temperature=3.0,
    alpha=0.5,
    device='cuda'
):
    """Train student model using knowledge distillation."""
    
    # Move models to device
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Setup
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = Adam(student_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    print(f"\nTeacher Parameters: {count_parameters(teacher_model)['total']:,}")
    print(f"Student Parameters: {count_parameters(student_model)['total']:,}")
    print(f"Compression Ratio: {count_parameters(teacher_model)['total'] / count_parameters(student_model)['total']:.2f}x")
    print(f"\nStarting distillation training...")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Training
        student_model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            loss_dict = distill_step(
                teacher_model, student_model, inputs, labels,
                criterion, optimizer, device
            )
            train_losses.append(loss_dict['total_loss'])
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        eval_results = evaluate_distillation(
            teacher_model, student_model, val_loader, device
        )
        
        scheduler.step(avg_train_loss)
        
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Teacher Accuracy: {eval_results['teacher_accuracy']:.4f}")
        print(f"  Student Accuracy: {eval_results['student_accuracy']:.4f}")
        print(f"  Accuracy Gap: {eval_results['accuracy_gap']:.4f}")
        
        if eval_results['student_accuracy'] > best_val_acc:
            best_val_acc = eval_results['student_accuracy']
            torch.save(student_model.state_dict(), 'best_student_model.pth')
            print(f"  ✓ New best student accuracy: {best_val_acc:.4f}")
    
    print(f"\nTraining complete!")
    print(f"Best Student Accuracy: {best_val_acc:.4f}")
    
    return student_model


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    teacher_model, teacher_tokenizer = build_bert_model("bert-base-uncased", num_labels=2)
    student_model, student_tokenizer = build_distilbert_model(num_labels=2)
    
    # Prepare data
    train_loader, test_loader = prepare_imdb_data(
        teacher_tokenizer, max_length=128, batch_size=16
    )
    
    # Train
    trained_student = train_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=1,  # Reduced to 1 epoch for testing
        lr=2e-5,
        temperature=3.0,
        alpha=0.5,
        device=device
    )
    
    print("\nModel saved to: best_student_model.pth")

