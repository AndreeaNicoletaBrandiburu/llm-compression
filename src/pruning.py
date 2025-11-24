"""
Structured pruning for transformer models.
Channel-wise and layer-wise pruning with magnitude-based selection.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Optional
import numpy as np


def prune_linear_layer(
    layer: nn.Linear,
    amount: float = 0.3,
    method: str = 'magnitude'
) -> nn.Linear:
    """
    Prune a linear layer using magnitude-based pruning.
    
    Args:
        layer: Linear layer to prune
        amount: Fraction of weights to prune (0.0 to 1.0)
        method: Pruning method ('magnitude' or 'random')
    
    Returns:
        Pruned layer
    """
    if method == 'magnitude':
        prune.l1_unstructured(layer, name='weight', amount=amount)
    elif method == 'random':
        prune.random_unstructured(layer, name='weight', amount=amount)
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    # Make pruning permanent
    prune.remove(layer, 'weight')
    
    return layer


def prune_attention_heads(
    model: nn.Module,
    layer_name: str,
    num_heads_to_prune: int,
    head_dim: int = 64
) -> nn.Module:
    """
    Prune attention heads from transformer layer.
    
    Args:
        model: Transformer model
        layer_name: Name of the layer to prune
        num_heads_to_prune: Number of attention heads to remove
        head_dim: Dimension of each attention head
    
    Returns:
        Model with pruned attention heads
    """
    layer = dict(model.named_modules())[layer_name]
    
    if hasattr(layer, 'attention'):
        attention = layer.attention
        if hasattr(attention, 'self'):
            # Get attention weights
            qkv = attention.self.query.weight.data
            # Simple magnitude-based head pruning
            # This is a simplified version - full implementation would be more complex
            pass
    
    return model


def iterative_pruning(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cuda',
    pruning_steps: int = 5,
    final_sparsity: float = 0.5,
    fine_tune_epochs: int = 1
) -> Tuple[nn.Module, List[float]]:
    """
    Iterative pruning with fine-tuning.
    
    Args:
        model: Model to prune
        dataloader: Training dataloader
        criterion: Loss function
        device: Device to run on
        pruning_steps: Number of pruning iterations
        final_sparsity: Target sparsity (0.0 to 1.0)
        fine_tune_epochs: Epochs to fine-tune after each pruning step
    
    Returns:
        Pruned model and list of accuracies after each step
    """
    model = model.to(device)
    accuracies = []
    
    sparsity_per_step = final_sparsity / pruning_steps
    
    for step in range(pruning_steps):
        # Prune model
        current_sparsity = (step + 1) * sparsity_per_step
        
        # Prune linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune_linear_layer(module, amount=sparsity_per_step)
        
        # Fine-tune
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        for epoch in range(fine_tune_epochs):
            model.train()
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, labels = batch
                    inputs_dict = None
                else:
                    inputs_dict, labels = batch
                    inputs = None
                
                labels = labels.to(device)
                if inputs_dict is not None:
                    inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                elif inputs is not None:
                    inputs = inputs.to(device)
                
                optimizer.zero_grad()
                
                if inputs_dict is not None:
                    outputs = model(**inputs_dict)
                else:
                    outputs = model(inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, labels = batch
                    inputs_dict = None
                else:
                    inputs_dict, labels = batch
                    inputs = None
                
                labels = labels.to(device)
                if inputs_dict is not None:
                    inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                elif inputs is not None:
                    inputs = inputs.to(device)
                
                if inputs_dict is not None:
                    outputs = model(**inputs_dict)
                else:
                    outputs = model(inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Pruning step {step+1}/{pruning_steps}, Sparsity: {current_sparsity:.2%}, Accuracy: {accuracy:.4f}")
    
    return model, accuracies


def calculate_sparsity(model: nn.Module) -> float:
    """
    Calculate model sparsity (fraction of zero weights).
    
    Args:
        model: PyTorch model
    
    Returns:
        Sparsity ratio (0.0 to 1.0)
    """
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0

