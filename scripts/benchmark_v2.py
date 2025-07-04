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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", type=str, default="results/raw")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"Benchmarking config: {config['name']}")
    
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
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config["dataset"]["name"], split="train")
    dataset = dataset.select(range(min(config["dataset"]["max_samples"], len(dataset))))
    
    def tokenize_function(examples):
        if "instruction" in examples:
            texts = examples["instruction"]
        elif "text" in examples:
            texts = examples["text"]
        else:
            texts = [str(x) for x in examples["input"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(f"Dataset size: {len(tokenized_dataset)} samples")
    
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
        train_dataset=tokenized_dataset,
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
    total_tokens = len(tokenized_dataset) * 512
    metrics["tokens_per_second"] = total_tokens / metrics["training_time"]
    metrics["samples_per_second"] = len(tokenized_dataset) / metrics["training_time"]
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\nBenchmark completed!")
    print(f"Time: {metrics['training_time']:.1f}s")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Peak GPU memory: {metrics.get('peak_gpu_memory_gb', 0):.2f}GB")
    print(f"Tokens/sec: {metrics['tokens_per_second']:.0f}")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
