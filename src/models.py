"""
Transformer model definitions for NLP and Vision tasks.
Supports BERT, DistilBERT, and Vision Transformers.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)
from torchvision import models
import timm


def build_bert_model(model_name: str = "bert-base-uncased", num_labels: int = 2):
    """
    Build BERT model for sequence classification.
    
    Args:
        model_name: Hugging Face model name
        num_labels: Number of classification labels
    
    Returns:
        Model and tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def build_distilbert_model(num_labels: int = 2):
    """
    Build DistilBERT model (smaller student model).
    
    Args:
        num_labels: Number of classification labels
    
    Returns:
        Model and tokenizer
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer


def build_vit_model(num_classes: int = 10, pretrained: bool = True):
    """
    Build Vision Transformer model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Vision Transformer model
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def build_mobilevit_model(num_classes: int = 10, pretrained: bool = True):
    """
    Build MobileViT model (smaller student model for vision).
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        MobileViT model
    """
    model = timm.create_model(
        "mobilevit_s",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

