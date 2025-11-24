"""
Simple example demonstrating model compression techniques.
"""

import torch
from src.models import build_bert_model, build_distilbert_model, count_parameters, get_model_size_mb
from src.distillation import DistillationLoss
from src.pruning import prune_linear_layer, calculate_sparsity
from src.quantization import apply_dynamic_quantization

def example_knowledge_distillation():
    """Example of knowledge distillation setup."""
    print("="*60)
    print("KNOWLEDGE DISTILLATION EXAMPLE")
    print("="*60)
    
    # Load models
    teacher, _ = build_bert_model("bert-base-uncased", num_labels=2)
    student, _ = build_distilbert_model(num_labels=2)
    
    # Compare sizes
    teacher_size = get_model_size_mb(teacher)
    student_size = get_model_size_mb(student)
    teacher_params = count_parameters(teacher)['total']
    student_params = count_parameters(student)['total']
    
    print(f"\nTeacher (BERT-base):")
    print(f"  Parameters: {teacher_params:,}")
    print(f"  Size: {teacher_size:.2f} MB")
    
    print(f"\nStudent (DistilBERT):")
    print(f"  Parameters: {student_params:,}")
    print(f"  Size: {student_size:.2f} MB")
    
    print(f"\nCompression:")
    print(f"  Parameter Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    print(f"  Size Reduction: {(1 - student_size/teacher_size)*100:.1f}%")
    
    # Distillation loss
    criterion = DistillationLoss(temperature=3.0, alpha=0.5)
    print(f"\nDistillation Loss configured:")
    print(f"  Temperature: {criterion.temperature}")
    print(f"  Alpha (distillation weight): {criterion.alpha}")


def example_pruning():
    """Example of model pruning."""
    print("\n" + "="*60)
    print("PRUNING EXAMPLE")
    print("="*60)
    
    model, _ = build_distilbert_model(num_labels=2)
    
    print(f"\nOriginal model:")
    print(f"  Parameters: {count_parameters(model)['total']:,}")
    print(f"  Size: {get_model_size_mb(model):.2f} MB")
    
    # Prune some layers
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'classifier' in name:
            prune_linear_layer(module, amount=0.3, method='magnitude')
            pruned_count += 1
    
    sparsity = calculate_sparsity(model)
    
    print(f"\nAfter pruning ({pruned_count} layers, 30% each):")
    print(f"  Parameters: {count_parameters(model)['total']:,}")
    print(f"  Size: {get_model_size_mb(model):.2f} MB")
    print(f"  Overall Sparsity: {sparsity:.2%}")


def example_quantization():
    """Example of model quantization."""
    print("\n" + "="*60)
    print("QUANTIZATION EXAMPLE")
    print("="*60)
    
    model, _ = build_distilbert_model(num_labels=2)
    
    print(f"\nOriginal model (FP32):")
    print(f"  Size: {get_model_size_mb(model):.2f} MB")
    
    # Quantize
    quantized_model = apply_dynamic_quantization(model)
    
    print(f"\nQuantized model (INT8):")
    print(f"  Size: {get_model_size_mb(quantized_model):.2f} MB")
    print(f"  Size Reduction: {(1 - get_model_size_mb(quantized_model)/get_model_size_mb(model))*100:.1f}%")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MODEL COMPRESSION TECHNIQUES - EXAMPLES")
    print("="*60)
    
    example_knowledge_distillation()
    example_pruning()
    example_quantization()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)

