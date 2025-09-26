#!/usr/bin/env python3
"""
Test script for memory-efficient chunked data loading
"""

import sys
import os
sys.path.append('/home/andriy/hobbies/programming/road to agi/mymodel')

from src.data.data_loader import ChunkedC4Dataset, create_full_dataset
from src.data.sampler import create_sampled_dataset
import torch
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_chunked_dataset():
    """Test the chunked dataset implementation"""
    print("🔧 Testing ChunkedC4Dataset...")
    
    # Initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Create a small chunked dataset for testing
    print("\n📊 Creating chunked dataset...")
    dataset = ChunkedC4Dataset(
        split='train',
        max_length=1024,
        tokenizer_name='gpt2',
        target_tokens_billions=0.01,  # Small test - 10M tokens
        cache_dir='data_cache',
        max_file_size_gb=0.05,  # 50MB max per file for testing
        chunk_size_tokens=1_000_000  # 1M tokens per chunk
    )
    
    after_creation = get_memory_usage()
    print(f"Memory after dataset creation: {after_creation:.1f} MB (+{after_creation - initial_memory:.1f} MB)")
    
    # Test basic functionality
    print(f"\n📏 Dataset length: {len(dataset):,} tokens")
    print(f"Number of chunks: {len(dataset.chunk_files)}")
    print(f"Chunk sizes: {dataset.chunk_sizes}")
    
    # Test token access
    print("\n🔍 Testing token access...")
    token_0 = dataset[0]
    token_1000 = dataset[1000]
    print(f"Token at index 0: {token_0}")
    print(f"Token at index 1000: {token_1000}")
    
    # Test slice access
    print("\n✂️ Testing slice access...")
    slice_data = dataset.get_token_slice(100, 200)
    print(f"Slice [100:200] shape: {slice_data.shape}, dtype: {slice_data.dtype}")
    print(f"First 10 tokens: {slice_data[:10]}")
    
    memory_after_access = get_memory_usage()
    print(f"Memory after token access: {memory_after_access:.1f} MB (+{memory_after_access - after_creation:.1f} MB)")
    
    # Test with sampler
    print("\n🎯 Testing with sampler...")
    model_config = {
        'target_tokens_billions': 0.005,  # 5M tokens
        'architecture': {'max_seq_len': 1024}
    }
    
    sampled_dataset = create_sampled_dataset(dataset, model_config)
    print(f"Sampled dataset length: {len(sampled_dataset)} sequences")
    
    # Test batch creation
    print("\n📦 Testing batch creation...")
    sample = sampled_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Input dtype: {sample['input_ids'].dtype}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    final_memory = get_memory_usage()
    print(f"\n💾 Final memory usage: {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB total)")
    
    # Clean up
    del dataset, sampled_dataset
    gc.collect()
    
    cleanup_memory = get_memory_usage()
    print(f"Memory after cleanup: {cleanup_memory:.1f} MB")
    
    print("\n✅ Chunked dataset test completed successfully!")

def test_integration():
    """Test full integration with config"""
    print("\n🔧 Testing integration with config...")
    
    try:
        train_dataset, val_dataset, data_config = create_full_dataset()
        print(f"✅ Integration test successful!")
        print(f"Train dataset type: {type(train_dataset).__name__}")
        print(f"Val dataset type: {type(val_dataset).__name__}")
        
        # Test stats
        train_stats = train_dataset.get_token_stats()
        val_stats = val_dataset.get_token_stats()
        
        print(f"Train tokens: {train_stats['actual_tokens']:,}")
        print(f"Val tokens: {val_stats['actual_tokens']:,}")
        
        if 'num_chunks' in train_stats:
            print(f"Train chunks: {train_stats['num_chunks']}")
            print(f"Val chunks: {val_stats['num_chunks']}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting memory-efficient data loading tests...\n")
    
    try:
        test_chunked_dataset()
        test_integration()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()