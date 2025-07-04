#!/usr/bin/env python3
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os

# Set cache directory
os.environ['HF_DATASETS_CACHE'] = '/content/cache'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", type=str, default="results/raw")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_simple_dataset(tokenizer, num_samples=100):
    """Create a simple dataset for testing"""
    # Create simple prompts
    prompts = [f"Hello, this is example {i}. The quick brown fox jumps over the lazy dog." for i in range(num_samples)]
    
    # Tokenize
    encodings = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=128,  # Shorter for faster testing
        return_tensors="pt"
    )
    
    # Create dataset
    import torch
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}
        
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    return SimpleDataset(encodings)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"\nBenchmarking config: {config['name']}")
    print("="*50)
    
    # Initialize metrics
    metrics = {
        "config": config,
        "model": args.model_name,
        "start_time": datetime.now().isoformat(),
    }
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    metrics["model_load_time"] = time.time() - start_time
    
    if torch.cuda.is_available():
        metrics["gpu_available"] = torch.cuda.get_device_name()
        torch.cuda.empty_cache()
    
    # Create simple dataset for testing
    print("Creating dataset...")
    num_samples = config["dataset"].get("max_samples", 100)
    train_dataset = create_simple_dataset(tokenizer, num_samples)
    print(f"Dataset size: {len(train_dataset)} samples")
    
    # Setup training
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{config['name']}",
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Run training
    print("Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    train_start = time.time()
    train_result = trainer.train()
    metrics["training_time"] = time.time() - train_start
    
    # Collect metrics
    metrics["final_loss"] = float(train_result.training_loss)
    metrics["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["total_parameters"] = sum(p.numel() for p in model.parameters())
    
    if torch.cuda.is_available():
        metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    
    # Calculate throughput
    metrics["samples_per_second"] = num_samples / metrics["training_time"]
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Benchmark completed!")
    print("="*50)
    print(f"Configuration: {config['name']}")
    print(f"Model: {args.model_name}")
    print(f"LoRA rank: {config['lora']['r']}")
    print(f"Training time: {metrics['training_time']:.1f}s")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Peak GPU memory: {metrics.get('peak_gpu_memory_gb', 0):.2f}GB")
    print(f"Samples/sec: {metrics['samples_per_second']:.2f}")
    print(f"Trainable params: {metrics['trainable_parameters']:,}")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
