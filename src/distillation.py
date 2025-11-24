"""
Knowledge Distillation implementation for transformer models.
Teacher-student training with temperature scaling and KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from tqdm import tqdm


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    Combines task loss (cross-entropy) with distillation loss (KL divergence).
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for distillation loss (1-alpha for task loss)
            reduction: Loss reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits [batch_size, num_classes]
            teacher_logits: Teacher model logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        
        Returns:
            Dictionary with total loss, task loss, and distillation loss
        """
        # Task loss (hard targets)
        task_loss = self.ce_loss(student_logits, labels)
        
        # Distillation loss (soft targets)
        # Softmax with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence
        distillation_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distillation_loss': distillation_loss
        }


def distill_step(
    teacher_model: nn.Module,
    student_model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Single distillation training step.
    
    Args:
        teacher_model: Teacher model (frozen)
        student_model: Student model (trainable)
        inputs: Model inputs (varies by model type)
        labels: Ground truth labels
        criterion: Distillation loss function
        optimizer: Optimizer for student model
        device: Device to run on
    
    Returns:
        Dictionary with loss values
    """
    teacher_model.eval()
    student_model.train()
    
    # Move to device
    labels = labels.to(device)
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = inputs.to(device)
    
    # Forward pass
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs) if isinstance(inputs, dict) else teacher_model(inputs)
        teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
    
    student_outputs = student_model(**inputs) if isinstance(inputs, dict) else student_model(inputs)
    student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
    
    # Compute loss
    loss_dict = criterion(student_logits, teacher_logits, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss_dict['total_loss'].backward()
    optimizer.step()
    
    # Return loss values
    return {
        'total_loss': loss_dict['total_loss'].item(),
        'task_loss': loss_dict['task_loss'].item(),
        'distillation_loss': loss_dict['distillation_loss'].item()
    }


def evaluate_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate teacher and student models.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataloader: Evaluation dataloader
        device: Device to run on
    
    Returns:
        Dictionary with accuracy for both models
    """
    teacher_model.eval()
    student_model.eval()
    
    teacher_correct = 0
    student_correct = 0
    total = 0
    
    print("Evaluating models on validation set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Handle dictionary batch (from datasets library)
            if isinstance(batch, dict):
                inputs_dict = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                labels = batch['label'].to(device)
                inputs = None
            elif len(batch) == 2:
                inputs, labels = batch
                inputs_dict = None
                labels = labels.to(device)
                if inputs is not None:
                    inputs = inputs.to(device)
            else:
                inputs_dict, labels = batch
                inputs = None
                labels = labels.to(device)
                if inputs_dict is not None:
                    inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            
            # Teacher predictions
            if inputs_dict is not None:
                teacher_outputs = teacher_model(**inputs_dict)
            else:
                teacher_outputs = teacher_model(inputs)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
            teacher_preds = torch.argmax(teacher_logits, dim=1)
            
            # Student predictions
            if inputs_dict is not None:
                student_outputs = student_model(**inputs_dict)
            else:
                student_outputs = student_model(inputs)
            student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
            student_preds = torch.argmax(student_logits, dim=1)
            
            # Count correct
            teacher_correct += (teacher_preds == labels).sum().item()
            student_correct += (student_preds == labels).sum().item()
            total += labels.size(0)
    
    return {
        'teacher_accuracy': teacher_correct / total,
        'student_accuracy': student_correct / total,
        'accuracy_gap': (teacher_correct - student_correct) / total
    }

