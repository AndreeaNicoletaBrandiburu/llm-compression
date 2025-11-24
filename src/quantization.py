"""
Quantization utilities for transformer models.
INT8 quantization for inference acceleration.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
from typing import Optional


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic INT8 quantization to transformer model.
    
    Args:
        model: PyTorch model to quantize
        dtype: Quantization dtype (torch.qint8 or torch.float16)
    
    Returns:
        Quantized model
    """
    model.eval()
    
    # Quantize only linear layers (embedding layers require special config)
    # Embedding layers are typically left in FP32 for dynamic quantization
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype
    )
    
    return quantized_model


def apply_static_quantization(
    model: nn.Module,
    calibration_dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> nn.Module:
    """
    Apply static quantization with calibration.
    More accurate but requires calibration data.
    
    Args:
        model: PyTorch model to quantize
        calibration_dataloader: DataLoader for calibration
        device: Device to run on
    
    Returns:
        Quantized model
    """
    model.eval()
    model.qconfig = default_qconfig
    
    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate
    with torch.no_grad():
        for batch in calibration_dataloader:
            if len(batch) == 2:
                inputs, _ = batch
                inputs_dict = None
            else:
                inputs_dict, _ = batch
                inputs = None
            
            if inputs_dict is not None:
                inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                _ = model(**inputs_dict)
            elif inputs is not None:
                inputs = inputs.to(device)
                _ = model(inputs)
    
    # Convert to quantized
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    return quantized_model


def compare_quantized_vs_original(
    original_model: nn.Module,
    quantized_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> dict:
    """
    Compare original and quantized model performance.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized INT8 model
        dataloader: Test dataloader
        device: Device to run on
    
    Returns:
        Dictionary with accuracy and size comparison
    """
    original_model.eval()
    quantized_model.eval()
    
    original_correct = 0
    quantized_correct = 0
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
            
            # Original model
            if inputs_dict is not None:
                orig_outputs = original_model(**inputs_dict)
            else:
                orig_outputs = original_model(inputs)
            orig_logits = orig_outputs.logits if hasattr(orig_outputs, 'logits') else orig_outputs
            orig_preds = torch.argmax(orig_logits, dim=1)
            
            # Quantized model
            if inputs_dict is not None:
                quant_outputs = quantized_model(**inputs_dict)
            else:
                quant_outputs = quantized_model(inputs)
            quant_logits = quant_outputs.logits if hasattr(quant_outputs, 'logits') else quant_outputs
            quant_preds = torch.argmax(quant_logits, dim=1)
            
            original_correct += (orig_preds == labels).sum().item()
            quantized_correct += (quant_preds == labels).sum().item()
            total += labels.size(0)
    
    # Model sizes
    from src.models import get_model_size_mb
    orig_size = get_model_size_mb(original_model)
    quant_size = get_model_size_mb(quantized_model)
    
    return {
        'original_accuracy': original_correct / total,
        'quantized_accuracy': quantized_correct / total,
        'accuracy_drop': (original_correct - quantized_correct) / total,
        'original_size_mb': orig_size,
        'quantized_size_mb': quant_size,
        'size_reduction': (orig_size - quant_size) / orig_size * 100
    }

