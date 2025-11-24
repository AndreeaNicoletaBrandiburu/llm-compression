"""
Comprehensive benchmarking for transformer models.
Compares teacher, student, pruned, and quantized models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, List, Optional
from src.models import get_model_size_mb, count_parameters


def benchmark_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False
) -> Dict:
    """
    Benchmark model inference performance.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for inference
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark runs
        use_amp: Whether to use mixed precision
    
    Returns:
        Dictionary with performance metrics
    """
    model.eval()
    model.to(device)
    
    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_warmup:
                break
            
            # Handle dictionary batch (from datasets library)
            if isinstance(batch, dict):
                inputs_dict = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                inputs = None
            elif len(batch) == 2:
                inputs, _ = batch
                inputs_dict = None
                if inputs is not None:
                    inputs = inputs.to(device)
            else:
                inputs_dict, _ = batch
                inputs = None
                if inputs_dict is not None:
                    inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    if inputs_dict is not None:
                        _ = model(**inputs_dict)
                    else:
                        _ = model(inputs)
            else:
                if inputs_dict is not None:
                    _ = model(**inputs_dict)
                else:
                    _ = model(inputs)
    
    # Synchronize GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    total_samples = 0
    
    with torch.no_grad():
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            if i >= num_runs:
                break
            
            # Handle dictionary batch (from datasets library)
            if isinstance(batch, dict):
                inputs_dict = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                batch_size = batch['input_ids'].size(0)
                inputs = None
            elif len(batch) == 2:
                inputs, labels = batch
                inputs_dict = None
                batch_size = inputs.size(0) if inputs is not None else labels.size(0)
                if inputs is not None:
                    inputs = inputs.to(device)
            else:
                inputs_dict, labels = batch
                inputs = None
                batch_size = inputs_dict[list(inputs_dict.keys())[0]].size(0) if inputs_dict is not None else labels.size(0)
                if inputs_dict is not None:
                    inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            
            # Measure latency
            if device == "cuda":
                torch.cuda.synchronize()
            
            iter_start = time.time()
            
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    if inputs_dict is not None:
                        _ = model(**inputs_dict)
                    else:
                        _ = model(inputs)
            else:
                if inputs_dict is not None:
                    _ = model(**inputs_dict)
                else:
                    _ = model(inputs)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            iter_end = time.time()
            latency_ms = (iter_end - iter_start) * 1000
            latencies.append(latency_ms)
            total_samples += batch_size
        
        end_time = time.time()
    
    total_time = end_time - start_time
    throughput = total_samples / total_time  # samples per second
    
    metrics = {
        'throughput_samples_per_sec': throughput,
        'avg_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'total_samples': total_samples,
        'total_time_sec': total_time,
        'device': device,
        'use_amp': use_amp
    }
    
    return metrics


def get_gpu_memory_usage(device: int = 0) -> Dict:
    """Get GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    
    return {
        'available': True,
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated
    }


def benchmark_model(
    model: nn.Module,
    dataloader: DataLoader,
    model_name: str = "Model",
    device: str = 'cuda',
    use_amp: bool = False,
    num_runs: int = 100
) -> Dict:
    """
    Comprehensive model benchmarking.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for benchmarking
        model_name: Name of the model
        device: Device to run on
        use_amp: Whether to use mixed precision
        num_runs: Number of benchmark iterations
    
    Returns:
        Dictionary with all benchmark metrics
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*60}")
    
    # Model size and parameters
    model_size_mb = get_model_size_mb(model)
    param_counts = count_parameters(model)
    
    print(f"\nModel Size: {model_size_mb:.2f} MB")
    print(f"Total Parameters: {param_counts['total']:,}")
    print(f"Trainable Parameters: {param_counts['trainable']:,}")
    
    # GPU memory (before)
    if device == "cuda":
        gpu_mem_before = get_gpu_memory_usage()
        print(f"\nGPU Memory (before):")
        print(f"  Allocated: {gpu_mem_before['allocated_gb']:.2f} GB")
        print(f"  Reserved: {gpu_mem_before['reserved_gb']:.2f} GB")
    
    # Inference benchmark
    inference_metrics = benchmark_inference(
        model, dataloader, device=device,
        num_runs=num_runs, use_amp=use_amp
    )
    
    # GPU memory (after)
    if device == "cuda":
        gpu_mem_after = get_gpu_memory_usage()
        print(f"\nGPU Memory (after):")
        print(f"  Allocated: {gpu_mem_after['allocated_gb']:.2f} GB")
        print(f"  Reserved: {gpu_mem_after['reserved_gb']:.2f} GB")
        print(f"  Max Allocated: {gpu_mem_after['max_allocated_gb']:.2f} GB")
    
    # Combine all metrics
    all_metrics = {
        'model_name': model_name,
        'model_size_mb': model_size_mb,
        'parameters': param_counts,
        'inference': inference_metrics
    }
    
    if device == "cuda":
        all_metrics['gpu_memory'] = gpu_mem_after
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"Throughput: {inference_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Average Latency: {inference_metrics['avg_latency_ms']:.2f} ms")
    print(f"P95 Latency: {inference_metrics['p95_latency_ms']:.2f} ms")
    print(f"P99 Latency: {inference_metrics['p99_latency_ms']:.2f} ms")
    print(f"{'='*60}\n")
    
    return all_metrics


def compare_models(
    models: List[nn.Module],
    model_names: List[str],
    dataloader: DataLoader,
    device: str = 'cuda',
    use_amp: bool = False,
    num_runs: int = 100
) -> Dict:
    """
    Compare multiple models side by side.
    
    Args:
        models: List of models to compare
        model_names: List of model names
        dataloader: DataLoader for benchmarking
        device: Device to run on
        use_amp: Whether to use mixed precision
        num_runs: Number of benchmark iterations
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"{'='*80}")
    
    all_metrics = []
    
    for model, name in zip(models, model_names):
        metrics = benchmark_model(
            model, dataloader, model_name=name,
            device=device, use_amp=use_amp, num_runs=num_runs
        )
        all_metrics.append(metrics)
    
    # Print comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Size (MB)':<12} {'Params':<15} {'Throughput':<15} {'Avg Latency':<15} {'P95 Latency':<15}")
    print("-"*100)
    
    for metrics in all_metrics:
        name = metrics['model_name']
        size = metrics['model_size_mb']
        params = metrics['parameters']['total']
        throughput = metrics['inference']['throughput_samples_per_sec']
        avg_lat = metrics['inference']['avg_latency_ms']
        p95_lat = metrics['inference']['p95_latency_ms']
        
        print(f"{name:<20} {size:>8.2f} MB  {params:>12,}  {throughput:>10.2f} img/s  {avg_lat:>10.2f} ms  {p95_lat:>10.2f} ms")
    
    print(f"{'='*100}\n")
    
    return {
        'models': all_metrics,
        'comparison': {
            'size_reduction': calculate_reduction(all_metrics, 'model_size_mb'),
            'speedup': calculate_speedup(all_metrics, 'throughput_samples_per_sec'),
            'latency_reduction': calculate_reduction(all_metrics, 'avg_latency_ms', inverse=True)
        }
    }


def calculate_reduction(metrics_list: List[Dict], key: str, inverse: bool = False) -> Dict:
    """Calculate reduction percentages between models."""
    if len(metrics_list) < 2:
        return {}
    
    baseline = metrics_list[0]
    baseline_value = baseline[key] if key in baseline else baseline['inference'][key]
    
    reductions = {}
    for i, metrics in enumerate(metrics_list[1:], 1):
        current_value = metrics[key] if key in metrics else metrics['inference'][key]
        if inverse:
            reduction = ((baseline_value - current_value) / baseline_value) * 100
        else:
            reduction = ((baseline_value - current_value) / baseline_value) * 100
        reductions[f"{baseline['model_name']} -> {metrics['model_name']}"] = reduction
    
    return reductions


def calculate_speedup(metrics_list: List[Dict], key: str) -> Dict:
    """Calculate speedup factors between models."""
    if len(metrics_list) < 2:
        return {}
    
    baseline = metrics_list[0]
    baseline_value = baseline['inference'][key]
    
    speedups = {}
    for metrics in metrics_list[1:]:
        current_value = metrics['inference'][key]
        speedup = current_value / baseline_value
        speedups[f"{baseline['model_name']} -> {metrics['model_name']}"] = speedup
    
    return speedups

