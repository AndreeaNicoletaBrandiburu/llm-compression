"""
Comprehensive benchmarking script.
Compares teacher, student, pruned, and quantized models.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from src.models import build_bert_model, build_distilbert_model, get_model_size_mb
from src.pruning import iterative_pruning, calculate_sparsity
from src.quantization import apply_dynamic_quantization, compare_quantized_vs_original
from src.benchmark import benchmark_model, compare_models


def prepare_imdb_data(tokenizer, max_length=128, batch_size=16):
    """Prepare IMDB dataset."""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    test_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False)
    return test_loader


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    teacher_model, teacher_tokenizer = build_bert_model("bert-base-uncased", num_labels=2)
    student_model, student_tokenizer = build_distilbert_model(num_labels=2)
    
    # Load student weights if available
    try:
        student_model.load_state_dict(torch.load('best_student_model.pth', map_location=device))
        print("Loaded trained student model")
    except FileNotFoundError:
        print("Student model not found, using pretrained weights")
    
    # Prepare data
    test_loader = prepare_imdb_data(teacher_tokenizer, max_length=128, batch_size=16)
    
    # Benchmark original models
    print("\n" + "="*80)
    print("BENCHMARKING ORIGINAL MODELS")
    print("="*80)
    
    teacher_metrics = benchmark_model(
        teacher_model, test_loader,
        model_name="Teacher (BERT-base)",
        device=device,
        num_runs=50
    )
    
    student_metrics = benchmark_model(
        student_model, test_loader,
        model_name="Student (DistilBERT)",
        device=device,
        num_runs=50
    )
    
    # Pruning
    print("\n" + "="*80)
    print("APPLYING PRUNING")
    print("="*80)
    
    pruned_model = student_model
    # Note: Full iterative pruning would take time, so we'll do a simple example
    print("Pruning student model...")
    from src.pruning import prune_linear_layer
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune_linear_layer(module, amount=0.3, method='magnitude')
    
    sparsity = calculate_sparsity(pruned_model)
    print(f"Model sparsity: {sparsity:.2%}")
    
    pruned_metrics = benchmark_model(
        pruned_model, test_loader,
        model_name="Pruned Student",
        device=device,
        num_runs=50
    )
    
    # Quantization
    print("\n" + "="*80)
    print("APPLYING QUANTIZATION")
    print("="*80)
    
    quantized_model = apply_dynamic_quantization(student_model)
    print("Quantized model created")
    
    quantized_metrics = benchmark_model(
        quantized_model, test_loader,
        model_name="Quantized Student",
        device=device,
        num_runs=50
    )
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    comparison = compare_models(
        models=[teacher_model, student_model, pruned_model, quantized_model],
        model_names=["Teacher (BERT)", "Student (DistilBERT)", "Pruned", "Quantized"],
        dataloader=test_loader,
        device=device,
        num_runs=50
    )
    
    print("\nBenchmarking complete!")

