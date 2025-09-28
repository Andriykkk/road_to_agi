#!/usr/bin/env python3
"""
Benchmark script to calculate parameters and test training speed/memory for all models
"""

import sys
import os
sys.path.append('/home/andriy/hobbies/programming/road to agi/mymodel')

import yaml
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from src.models.model_factory import create_model
from src.training.pretraining_regression import train_step

class FakeDataset(Dataset):
    """Fake dataset for benchmarking"""
    def __init__(self, seq_len, vocab_size, num_samples=1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random token sequences
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory():
    """Get GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0

def calculate_model_parameters(model_config):
    """Calculate theoretical number of parameters"""
    arch = model_config['architecture']
    
    n_layers = arch['n_layers']
    n_heads = arch['n_heads']
    d_model = arch['d_model']
    d_ff = arch['d_ff']
    vocab_size = arch['vocab_size']
    max_seq_len = arch['max_seq_len']
    
    # Token embedding
    token_emb = vocab_size * d_model
    
    # Position embedding
    pos_emb = max_seq_len * d_model
    
    # Transformer layers
    # Attention: Q, K, V projections + output projection
    attn_params_per_layer = 4 * d_model * d_model
    
    # Feed forward: two linear layers
    ff_params_per_layer = d_model * d_ff + d_ff * d_model
    
    # Layer norms (2 per layer: pre-attn and pre-ff)
    ln_params_per_layer = 2 * d_model * 2  # weight + bias
    
    transformer_params = n_layers * (attn_params_per_layer + ff_params_per_layer + ln_params_per_layer)
    
    # Final layer norm
    final_ln = d_model * 2
    
    # Output head (usually tied with input embedding, but count separately)
    output_head = d_model * vocab_size
    
    total_params = token_emb + pos_emb + transformer_params + final_ln + output_head
    
    return {
        'token_embedding': token_emb,
        'position_embedding': pos_emb,
        'transformer_layers': transformer_params,
        'final_ln': final_ln,
        'output_head': output_head,
        'total': total_params
    }

def benchmark_model(model_name, model_config, device='cpu', num_steps=10):
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING {model_name.upper()}")
    print(f"{'='*60}")
    
    arch = model_config['architecture']
    
    # Calculate theoretical parameters
    theoretical_params = calculate_model_parameters(model_config)
    print(f"📊 Theoretical Parameters:")
    print(f"   Token embedding:     {theoretical_params['token_embedding']:,}")
    print(f"   Position embedding:  {theoretical_params['position_embedding']:,}")
    print(f"   Transformer layers:  {theoretical_params['transformer_layers']:,}")
    print(f"   Final layer norm:    {theoretical_params['final_ln']:,}")
    print(f"   Output head:         {theoretical_params['output_head']:,}")
    print(f"   Total (theoretical): {theoretical_params['total']:,}")
    
    # Calculate embedding percentage
    total_embeddings = theoretical_params['token_embedding'] + theoretical_params['position_embedding'] + theoretical_params['output_head']
    embedding_percentage = (total_embeddings / theoretical_params['total']) * 100
    transformer_percentage = (theoretical_params['transformer_layers'] / theoretical_params['total']) * 100
    
    print(f"\n📈 Parameter Distribution:")
    print(f"   Embeddings (tok+pos+out): {total_embeddings:,} ({embedding_percentage:.1f}%)")
    print(f"   Transformer layers:       {theoretical_params['transformer_layers']:,} ({transformer_percentage:.1f}%)")
    print(f"   Other (layer norms):      {theoretical_params['final_ln']:,} ({100-embedding_percentage-transformer_percentage:.1f}%)")
    
    # Create actual model
    try:
        print(f"\n🔧 Creating model...")
        model, _ = create_model(model_name, 'config/models.yaml')
        actual_params = model.get_num_params()
        print(f"   Actual parameters:   {actual_params:,}")
        
        # Calculate difference
        diff = actual_params - theoretical_params['total']
        diff_pct = (diff / theoretical_params['total']) * 100
        print(f"   Difference:          {diff:+,} ({diff_pct:+.1f}%)")
        
        model = model.to(device)
        
        # Create fake dataset
        seq_len = arch['max_seq_len']
        vocab_size = arch['vocab_size']
        batch_size = model_config.get('batch_size', 8)
        
        fake_dataset = FakeDataset(seq_len, vocab_size)
        dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)
        
        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        # Memory before training
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory()
        
        print(f"\n🏃 Benchmarking training speed...")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Device: {device}")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        if device == 'cuda':
            print(f"   Initial GPU memory: {initial_gpu_memory:.1f} MB")
        
        model.train()
        step_times = []
        losses = []
        
        # Warmup step (not timed)
        batch = next(iter(dataloader))
        optimizer.zero_grad()
        loss, accuracy = train_step(model, batch, device)
        loss.backward()
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark steps
        for step in range(num_steps):
            batch = next(iter(dataloader))
            
            start_time = time.time()
            
            optimizer.zero_grad()
            loss, accuracy = train_step(model, batch, device)
            loss.backward()
            optimizer.step()
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            step_time = time.time() - start_time
            step_times.append(step_time)
            losses.append(loss.item())
            
            if step % 5 == 0:
                print(f"   Step {step+1}/{num_steps}: {step_time:.3f}s, loss={loss.item():.4f}")
        
        # Calculate statistics
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)
        steps_per_second = 1.0 / avg_step_time
        
        # Tokens per second
        tokens_per_step = batch_size * seq_len
        tokens_per_second = tokens_per_step / avg_step_time
        
        # Memory after training
        final_memory = get_memory_usage()
        final_gpu_memory = get_gpu_memory()
        memory_usage = final_memory - initial_memory
        
        if device == 'cuda':
            gpu_memory_usage = final_gpu_memory - initial_gpu_memory
        
        print(f"\n📈 Performance Results:")
        print(f"   Average step time:   {avg_step_time:.3f} ± {std_step_time:.3f} seconds")
        print(f"   Steps per second:    {steps_per_second:.2f}")
        print(f"   Tokens per step:     {tokens_per_step:,}")
        print(f"   Tokens per second:   {tokens_per_second:,.0f}")
        print(f"   Average loss:        {np.mean(losses):.4f}")
        
        print(f"\n💾 Memory Usage:")
        print(f"   Memory usage:        {memory_usage:.1f} MB")
        print(f"   Final memory:        {final_memory:.1f} MB")
        if device == 'cuda':
            print(f"   GPU memory usage:    {gpu_memory_usage:.1f} MB")
            print(f"   Final GPU memory:    {final_gpu_memory:.1f} MB")
        
        # Calculate memory per parameter
        memory_per_param = memory_usage / actual_params * 1024 * 1024  # bytes per param
        print(f"   Memory per param:    {memory_per_param:.2f} bytes")
        
        return {
            'model_name': model_name,
            'theoretical_params': theoretical_params['total'],
            'actual_params': actual_params,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second,
            'tokens_per_second': tokens_per_second,
            'memory_usage_mb': memory_usage,
            'memory_per_param_bytes': memory_per_param,
            'avg_loss': np.mean(losses),
            'gpu_memory_mb': gpu_memory_usage if device == 'cuda' else 0
        }
        
    except Exception as e:
        print(f"❌ Error benchmarking {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main benchmarking function"""
    print("🚀 Model Parameter Calculation and Training Benchmark")
    print("="*60)
    
    # Load model configurations
    with open('config/models.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Benchmark each model
    results = []
    
    for model_name, model_config in config['models'].items():
        result = benchmark_model(model_name, model_config, device, num_steps=10)
        if result:
            results.append(result)
    
    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Model':<10} {'Params':<12} {'Step/s':<8} {'Tokens/s':<10} {'Memory':<10} {'Loss':<8}")
        print(f"{'-'*80}")
        
        for result in results:
            params_str = f"{result['actual_params']/1e6:.1f}M"
            memory_str = f"{result['memory_usage_mb']:.0f}MB"
            
            print(f"{result['model_name']:<10} {params_str:<12} {result['steps_per_second']:<8.2f} "
                  f"{result['tokens_per_second']:<10,.0f} {memory_str:<10} {result['avg_loss']:<8.4f}")
        
if __name__ == "__main__":
    main()